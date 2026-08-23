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
    """Lightweight context encoder fallback for when pretrained VLM backbone
    (e.g., Florence-2) is not loaded or for custom lightweight dataset inputs.
    """

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.ego_proj = nn.Sequential(
            nn.Linear(6, 256),
            nn.GELU(),
            nn.Linear(256, hidden_size),
        )
        self.agent_proj = nn.Sequential(
            nn.Linear(6, 256),
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


class DpVlaModel(nn.Module):
    """Pure backbone of the Dp-VLA planner (official architecture)."""

    config_class = DpVlaConfig

    def __init__(self, config: DpVlaConfig) -> None:
        super().__init__()
        self.config = config

        # Built-in lightweight fallback encoder for non-VLM pipeline environments
        self.fallback_encoder = LightweightContextEncoder(hidden_size=config.hidden_size)

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
