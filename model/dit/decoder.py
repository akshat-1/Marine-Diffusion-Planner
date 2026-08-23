"""Diffusion-transformer decoder used as the action head of DpVlaModel.

Ported from official HDP-navsim repository.
The decoder is a stack of DiT blocks with adaLN-zero conditioning.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .DiT import DiTBlock, FinalLayer, TimestepEmbedder, MlpBlock


def sinusoidal_positional_encoding(pos_tensor: torch.Tensor, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding.
    Frequency base is 100 to match official checkpoints.
    """
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for sinusoidal positional encoding")

    pos = pos_tensor.float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=pos.device).float()
        * (-math.log(100.0) / d_model)
    )
    pos_encodings = pos.unsqueeze(-1) * div_term

    pe = torch.zeros(*pos.shape, d_model, device=pos.device)
    pe[..., 0::2] = torch.sin(pos_encodings)
    pe[..., 1::2] = torch.cos(pos_encodings)
    return pe


class CustomDiT(nn.Module):
    """DiT-style diffusion decoder."""

    def __init__(
        self,
        num_actions: int = 8,
        dim_action: int = 4,
        dim_y: int = 12,
        hidden_size: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.t_embedder = TimestepEmbedder(hidden_size)
        pos_emb = sinusoidal_positional_encoding(
            torch.arange(0, num_actions, dtype=torch.int32), hidden_size
        )
        self.pos_emb = nn.Parameter(pos_emb.unsqueeze(0))

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )

        self.action_in_proj = MlpBlock(
            in_features=dim_action, hidden_features=512,
            out_features=hidden_size, drop=0.0,
        )
        self.y_in_proj = MlpBlock(
            in_features=dim_y, hidden_features=512,
            out_features=hidden_size, drop=0.0,
        )
        self.final_layer = FinalLayer(hidden_size, dim_action)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.final_layer.proj[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor,
        c_mask: torch.Tensor,
    ) -> torch.Tensor:
        y = self.t_embedder(t) + self.y_in_proj(y)
        x = self.action_in_proj(x) + self.pos_emb
        for block in self.blocks:
            x = block(x, y, c, c_mask)
        return self.final_layer(x, y)
