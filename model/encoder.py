#!/usr/bin/env python3
"""
Scene Context Encoder for Diffusion Transformer conditioning.

Pipeline:
    raw tensors (from preparedataset.py)
        -> PolylineMLPMixer (encode ego + agents + map polylines)
        -> ContextCrossAttention (fuse using ego as query)
        -> context vector `c` (conditions the diffusion process)

Feature dim note:
    preparedataset.py emits 6 features per state: [x, y, vx, vy, heading, yaw_rate].
    map_lines emit 2 features per point: [x, y]. The map mixer is built with
    feature_dim=2 to match, while the agent/ego mixers use feature_dim=6.
"""

import torch
import torch.nn as nn


class PolylineMLPMixer(nn.Module):
    """
    Encodes a set of polylines (agents over time, or map lines over points)
    into one vector per polyline via alternating token/channel mixing.
    """

    def __init__(self, num_points, in_features, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        # Token mixing: aggregates across the sequence/time/point dimension
        self.token_norm = nn.LayerNorm(in_features)
        self.token_mixing = nn.Sequential(
            nn.Linear(num_points, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_points),
            nn.Dropout(dropout)
        )

        # Channel mixing: aggregates across the feature dimension
        self.channel_norm = nn.LayerNorm(in_features)
        self.channel_mixing = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

        # Collapse the sequence into a single vector per polyline
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x shape: (Batch, Num_Elements, Num_Points, In_Features)
        B, N, S, F = x.shape
        x_flat = x.reshape(B * N, S, F)

        # Token Mixing (operate on the sequence dimension)
        normed = self.token_norm(x_flat)
        mix_token = self.token_mixing(normed.transpose(-1, -2)).transpose(-1, -2)
        x_flat = x_flat + mix_token  # skip connection

        # Channel Mixing (operate on the feature dimension)
        mix_channel = self.channel_mixing(self.channel_norm(x_flat))

        # Pool across the sequence dimension -> one vector per element
        pooled = self.pool(mix_channel.transpose(-1, -2)).squeeze(-1)

        return pooled.view(B, N, -1)


class ContextCrossAttention(nn.Module):
    """Fuses environment features into the ego query via cross-attention."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, context, mask=None):
        # query:   (B, 1, embed_dim)
        # context: (B, N, embed_dim)
        # mask:    (B, N) with True = ignore/pad
        attn_out, _ = self.attention(
            query, context, context, key_padding_mask=mask
        )
        query = self.norm(query + attn_out)
        ffn_out = self.ffn(query)
        return self.norm2(query + ffn_out)


class SceneContextEncoder(nn.Module):
    """
    Encodes ego + agents + map into a single context vector `c`
    used to condition the Diffusion Transformer.
    """

    def __init__(self, hist_steps=20, map_points=20, feature_dim=6,
                 map_feature_dim=2, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        
        # 1. Ego Mixer: Treat the Ego just like an agent to preserve spatial-temporal bias
        self.ego_mixer = PolylineMLPMixer(
            num_points=hist_steps, in_features=feature_dim,
            hidden_dim=embed_dim * 2, out_dim=embed_dim, dropout=dropout
        )

        # 2. Agent Mixer: Mix over time (6 features per state)
        self.agent_mixer = PolylineMLPMixer(
            num_points=hist_steps, in_features=feature_dim,
            hidden_dim=embed_dim * 2, out_dim=embed_dim, dropout=dropout
        )

        # 3. Map Mixer: Mix over points (2 features per point: x, y)
        self.map_mixer = PolylineMLPMixer(
            num_points=map_points, in_features=map_feature_dim,
            hidden_dim=embed_dim * 2, out_dim=embed_dim, dropout=dropout
        )

        self.cross_attention = ContextCrossAttention(
            embed_dim, num_heads=num_heads, dropout=dropout
        )

    def forward(self, ego, agents, map_lines, agent_mask=None, map_mask=None):
        """
        ego:        (B, hist_steps, feature_dim)
        agents:     (B, max_agents, hist_steps, feature_dim)
        map_lines:  (B, max_polylines, map_points, map_feature_dim)
        agent_mask: (B, max_agents)     True = pad/ignore
        map_mask:   (B, max_polylines)  True = pad/ignore
        returns:    (B, embed_dim) context vector `c`
        """
        
        # 1. Encode Ego -> (B, 1, embed_dim)
        # Add an artificial "Num_Elements" dimension of 1 for the PolylineMLPMixer
        ego_embed = self.ego_mixer(ego.unsqueeze(1)) 

        # 2. Mix Agents and Map -> (B, N, embed_dim)
        agent_embeds = self.agent_mixer(agents)
        map_embeds = self.map_mixer(map_lines)

        # 3. Concatenate into a single environment context
        env_context = torch.cat([agent_embeds, map_embeds], dim=1)

        # Combine masks (True = ignore/pad)
        env_mask = None
        if agent_mask is not None and map_mask is not None:
            env_mask = torch.cat([agent_mask, map_mask], dim=1)

        # Guard: cross-attention breaks if a row masks every key.
        # Force at least one valid key per sample (unmask slot 0).
        # Use vectorized ops (no data-dependent if) for ONNX export compatibility.
        if env_mask is not None:
            all_masked = env_mask.all(dim=1)
            env_mask = env_mask.clone()
            env_mask[:, 0] = env_mask[:, 0] & ~all_masked

        # 4. Fuse via cross-attention
        context_vector = self.cross_attention(
            query=ego_embed, context=env_context, mask=env_mask
        )

        return context_vector.squeeze(1)  # (B, embed_dim)


if __name__ == "__main__":
    # Smoke test against the shapes emitted by preparedataset.py
    B, A, T, F, P, L = 4, 10, 20, 6, 20, 20
    enc = SceneContextEncoder(
        hist_steps=T, map_points=P, feature_dim=F, map_feature_dim=2, embed_dim=256, dropout=0.1
    )
    ego = torch.randn(B, T, F)
    agents = torch.randn(B, A, T, F)
    map_lines = torch.randn(B, L, P, 2)
    agent_mask = torch.zeros(B, A, dtype=torch.bool)
    map_mask = torch.zeros(B, L, dtype=torch.bool)

    c = enc(ego, agents, map_lines, agent_mask, map_mask)
    print("Context vector shape:", c.shape)  # expected (4, 256)