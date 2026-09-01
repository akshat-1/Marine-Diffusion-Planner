"""Dp-VLA backbone -- pure architecture.

Ported from official HDP-navsim repository.
Stateless backbone exposing:
- encode
- decode
- forward
- generate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from .configuration_dp_vla import DpVlaConfig
from .dit.decoder import CustomDiT
from .diffusion_utils.dpm_solver_pytorch import model_wrapper

logger = logging.getLogger(__name__)


@dataclass
class DpVlaEncoderOutput:
    last_hidden_state: torch.FloatTensor
    attention_mask: torch.LongTensor


@dataclass
class DpVlaModelOutput:
    prediction: torch.FloatTensor

    @property
    def noise_pred(self) -> torch.FloatTensor:
        return self.prediction

    encoder_hidden_states: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.LongTensor] = None


class LightweightContextEncoder(nn.Module):
    """Lightweight context encoder fallback for simple baseline inputs."""

    def __init__(self, hidden_size: int = 1024, in_dim: int = 10):
        super().__init__()
        self.ego_proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, hidden_size),
        )
        self.agent_proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, hidden_size),
        )
        self.map_proj = nn.Sequential(
            nn.Linear(2, 256),
            nn.GELU(),
            nn.Linear(256, hidden_size),
        )

    def forward(self, ego, agents=None, map_lines=None, agent_mask=None, map_mask=None):
        B = ego.shape[0]
        # Summarize ego trajectory over time
        ego_tokens = self.ego_proj(ego).mean(dim=1, keepdim=True)  # (B, 1, hidden_size)

        tokens = [ego_tokens]
        masks = [torch.ones((B, 1), dtype=torch.bool, device=ego.device)]

        if agents is not None and agents.numel() > 0:
            ag_tokens = self.agent_proj(agents).mean(dim=2)  # (B, N_ag, hidden_size)
            tokens.append(ag_tokens)
            if agent_mask is not None:
                masks.append(~agent_mask)
            else:
                masks.append(torch.ones(ag_tokens.shape[:2], dtype=torch.bool, device=ego.device))

        if map_lines is not None and map_lines.numel() > 0:
            map_tokens = self.map_proj(map_lines).mean(dim=2)  # (B, N_map, hidden_size)
            tokens.append(map_tokens)
            if map_mask is not None:
                masks.append(~map_mask)
            else:
                masks.append(torch.ones(map_tokens.shape[:2], dtype=torch.bool, device=ego.device))

        cat_tokens = torch.cat(tokens, dim=1)
        cat_masks = torch.cat(masks, dim=1)
        return DpVlaEncoderOutput(last_hidden_state=cat_tokens, attention_mask=cat_masks)


class HighCapacityVectorSceneEncoder(nn.Module):
    """SOTA High-Capacity Vector Scene Transformer Encoder for non-image AIS + Map datasets.

    Replaces simple fallback encoders with a deep, hierarchical spatial-temporal
    Transformer backbone designed to match the representation capacity of VLM visual encoders.

    Key Innovations:
    1. Temporal 1D Transformer for Ego & Agent trajectory dynamic encoding.
    2. Point-Polyline Sub-graph Transformer for Coastline & Shallow map geometry.
    3. Type & Spatial Modality Embeddings.
    4. Multimodal Deep Scene Transformer (Heterogeneous Self-Attention over all scene tokens).
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_obs_frames: int = 20,
        map_points: int = 20,
        in_dim: int = 10,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # --- 1. Ego Trajectory Sub-Encoder ---
        self.ego_feature_proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_size),
        )
        self.ego_pos_embed = nn.Parameter(torch.zeros(1, max_obs_frames, hidden_size))

        # --- 2. Agent Trajectory Sub-Encoder ---
        self.agent_feature_proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_size),
        )
        self.agent_pos_embed = nn.Parameter(torch.zeros(1, max_obs_frames, hidden_size))
        agent_temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.agent_temporal_transformer = nn.TransformerEncoder(
            agent_temporal_layer, num_layers=2, enable_nested_tensor=False
        )
        self.agent_temporal_pool = nn.AdaptiveAvgPool1d(4)

        # --- 3. Map Polyline Sub-Encoder ---
        self.map_feature_proj = nn.Sequential(
            nn.Linear(5, 128),
            nn.GELU(),
            nn.Linear(128, hidden_size),
        )
        self.map_pos_embed = nn.Parameter(torch.zeros(1, map_points, hidden_size))
        map_point_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.map_point_transformer = nn.TransformerEncoder(
            map_point_layer, num_layers=2, enable_nested_tensor=False
        )
        self.map_point_pool = nn.AdaptiveAvgPool1d(4)

        # --- 4. Modality / Type Embeddings ---
        # 0: Ego, 1: Agent, 2: Map
        self.type_embed = nn.Embedding(3, hidden_size)

        # --- 5. Multimodal Scene Transformer Backbone ---
        scene_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.scene_transformer = nn.TransformerEncoder(
            scene_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.ego_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.agent_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.map_pos_embed, std=0.02)

    def forward(
        self,
        ego: torch.Tensor,               # (B, T_obs, 6)
        agents: torch.Tensor = None,     # (B, N_ag, T_obs, 6)
        map_lines: torch.Tensor = None,  # (B, N_map, N_pts, 2)
        agent_mask: torch.Tensor = None, # (B, N_ag) True = padded/ignore
        map_mask: torch.Tensor = None,   # (B, N_map) True = padded/ignore
    ) -> DpVlaEncoderOutput:
        B, T_obs, _ = ego.shape
        device = ego.device

        # 1. Process Ego Tokens (T_obs tokens)
        ego_feat = self.ego_feature_proj(ego) + self.ego_pos_embed[:, :T_obs, :]
        ego_tokens = ego_feat + self.type_embed(torch.tensor(0, device=device))
        ego_mask = torch.ones((B, T_obs), dtype=torch.bool, device=device)

        tokens_list = [ego_tokens]
        masks_list = [ego_mask]

        # 2. Process Agent Tokens (N_ag * 4 sub-tokens)
        if agents is not None and agents.numel() > 0:
            N_ag, D_ag = agents.shape[1], agents.shape[-1]
            ag_flat = agents.view(B * N_ag, T_obs, D_ag)
            ag_feat = self.agent_feature_proj(ag_flat) + self.agent_pos_embed[:, :T_obs, :]
            ag_trans = self.agent_temporal_transformer(ag_feat)
            ag_pooled = self.agent_temporal_pool(ag_trans.transpose(1, 2)).transpose(1, 2)
            ag_tokens = ag_pooled.view(B, N_ag * 4, self.hidden_size) + self.type_embed(torch.tensor(1, device=device))

            if agent_mask is not None:
                ag_valid = (~agent_mask).repeat_interleave(4, dim=1)
            else:
                ag_valid = torch.ones((B, N_ag * 4), dtype=torch.bool, device=device)

            tokens_list.append(ag_tokens)
            masks_list.append(ag_valid)

        # 3. Process Map / Coastline Tokens (N_map * 4 sub-tokens)
        if map_lines is not None and map_lines.numel() > 0:
            N_map, N_pts, _ = map_lines.shape[1], map_lines.shape[2], map_lines.shape[3]
            map_flat = map_lines.view(B * N_map, N_pts, 2)

            delta = torch.zeros_like(map_flat)
            delta[:, 1:, :] = map_flat[:, 1:, :] - map_flat[:, :-1, :]
            dist = torch.norm(map_flat, dim=-1, keepdim=True)
            
            map_feats_5d = torch.cat([map_flat, delta, dist], dim=-1)
            map_proj = self.map_feature_proj(map_feats_5d) + self.map_pos_embed[:, :N_pts, :]
            map_trans = self.map_point_transformer(map_proj)
            map_pooled = self.map_point_pool(map_trans.transpose(1, 2)).transpose(1, 2)
            map_tokens = map_pooled.view(B, N_map * 4, self.hidden_size) + self.type_embed(torch.tensor(2, device=device))

            if map_mask is not None:
                map_valid = (~map_mask).repeat_interleave(4, dim=1)
            else:
                map_valid = torch.ones((B, N_map * 4), dtype=torch.bool, device=device)

            tokens_list.append(map_tokens)
            masks_list.append(map_valid)

        # 4. Multimodal Scene Transformer Backbone
        cat_tokens = torch.cat(tokens_list, dim=1)
        cat_valid = torch.cat(masks_list, dim=1)

        padding_mask = ~cat_valid
        all_padded = padding_mask.all(dim=1)
        if all_padded.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padded, 0] = False

        scene_tokens = self.scene_transformer(cat_tokens, src_key_padding_mask=padding_mask)
        scene_tokens = self.norm(scene_tokens)

        return DpVlaEncoderOutput(last_hidden_state=scene_tokens, attention_mask=cat_valid)


class DpVlaModel(nn.Module):
    """Pure backbone of the Dp-VLA planner (official architecture)."""

    config_class = DpVlaConfig

    def __init__(self, config: DpVlaConfig) -> None:
        super().__init__()
        self.config = config

        # High-capacity Vector Scene Transformer Encoder for non-image AIS dataset pipeline
        self.fallback_encoder = HighCapacityVectorSceneEncoder(
            hidden_size=config.hidden_size,
            num_layers=6,
            num_heads=config.num_heads,
        )

        self.decoder = CustomDiT(
            num_actions=config.num_actions,
            dim_action=config.dim_action,
            dim_y=config.dim_y,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
        )

    def decode(
        self,
        action_with_noise: torch.Tensor,
        time: torch.Tensor,
        proprio: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        adapter: Optional[str] = None,
    ) -> torch.Tensor:
        """Single-step noise/x_start prediction."""
        return self.decoder(
            action_with_noise, time, proprio,
            encoder_hidden_states, attention_mask,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_obs: Optional[torch.FloatTensor] = None,
        action_with_noise: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        adapter: Optional[str] = None,
    ) -> DpVlaModelOutput:
        """One-shot encode + decode."""
        if encoder_hidden_states is None:
            encoder_outputs = self.fallback_encoder(input_ids)
            encoder_hidden_states = encoder_outputs.last_hidden_state
            attention_mask = encoder_outputs.attention_mask
        elif attention_mask is None:
            attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                dtype=torch.bool, device=encoder_hidden_states.device,
            )

        prediction = self.decode(
            action_with_noise, time, proprio,
            encoder_hidden_states, attention_mask,
            adapter=adapter,
        )
        return DpVlaModelOutput(
            prediction=prediction,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        diffusion_sde,
        encoder_hidden_states: torch.Tensor,
        proprio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_actions: Optional[int] = None,
        dim_action: Optional[int] = None,
        steps: int = 10,
        sample_temperature: float = 0.5,
        cfg_scale: Optional[float] = None,
        use_base: bool = False,
    ) -> torch.Tensor:
        """Sample an action trajectory via iterative denoising using official DPM-Solver SDE."""
        if attention_mask is None:
            attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                dtype=torch.bool, device=encoder_hidden_states.device,
            )
        num_actions = num_actions or self.config.num_actions
        dim_action = dim_action or self.config.dim_action

        def decoder_fn(x, t, **kwargs):
            return self.decode(
                x, t, kwargs["y"], kwargs["c"], kwargs["c_mask"],
                adapter=None,
            )

        wrapped = model_wrapper(
            decoder_fn,
            diffusion_sde.sde,
            model_type=self.config.model_type,
            guidance_type="uncond",
            model_kwargs={
                "y": proprio,
                "c": encoder_hidden_states,
                "c_mask": attention_mask,
            },
        )

        B = encoder_hidden_states.shape[0]
        x_init = torch.randn(
            B, num_actions, dim_action, device=encoder_hidden_states.device,
        ) * sample_temperature
        return diffusion_sde.generate(x_init, wrapped, steps)
