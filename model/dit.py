#!/usr/bin/env python3
"""
Diffusion Transformer (DiT) Trajectory Decoder.
Implements HDP paper's τ₀-prediction (direct velocity prediction).
Based on: "τ₀-prediction model with τ₀-loss yields both fast convergence and high-quality generation"
"""

import math
import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into high-dimensional vectors.
    Uses sinusoidal positional embeddings followed by an MLP.
    """
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class AISDiffusionTransformer(nn.Module):
    """
    Diffusion Transformer (DiT) for Ego Trajectory Decoding.
    
    Key HDP paper insights:
    - Uses τ₀-prediction: directly predicts clean velocity trajectory x_0 (not noise ε)
    - Outputs 4D velocity: [vx, vy, theta, yaw_rate] as per paper
    - Conditioned on Scene Context Vector and diffusion timestep
    - Uses hybrid loss (velocity + waypoint) during training
    """
    def __init__(self, pred_frames=20, feature_dim=4, embed_dim=256, num_layers=6, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.pred_frames = pred_frames
        self.feature_dim = feature_dim  # 4 for velocity: [vx, vy, theta, yaw_rate]

        # 1. Diffusion Timestep Embedder
        self.time_embedder = TimestepEmbedder(embed_dim)

        # 2. Input Projection for Noisy Trajectory
        self.input_proj = nn.Linear(feature_dim, embed_dim)

        # 3. Position Embeddings for Future Frames
        self.pos_embed = nn.Parameter(torch.zeros(1, pred_frames, embed_dim))

        # 4. Decoder Blocks (Cross-Attention based conditioning)
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])

        # 5. Final Output Layer - predicts x_0 (clean velocity trajectory)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_layer = nn.Linear(embed_dim, feature_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize input projection and output layers
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, z_t, t, context):
        """
        z_t:     (Batch, pred_frames, feature_dim)  - Noisy future velocity trajectory
        t:       (Batch,)                           - Diffusion timesteps
        context: (Batch, embed_dim)                 - Latent scene context vector
        
        Returns:
            Predicted clean velocity trajectory x_0 (τ₀-prediction)
            Shape: (Batch, pred_frames, feature_dim)
        """
        # 1. Embed Timestep
        t_emb = self.time_embedder(t)  # (Batch, embed_dim)

        # 2. Add conditioning context and time
        # We project context + time to form the conditioning source (memory)
        condition = (context + t_emb).unsqueeze(1)  # (Batch, 1, embed_dim)

        # 3. Project input trajectory and add spatial-temporal position embeddings
        x = self.input_proj(z_t) + self.pos_embed  # (Batch, pred_frames, embed_dim)

        # 4. Process through Transformer Decoder blocks
        for block in self.blocks:
            x = block(tgt=x, memory=condition)

        # 5. Output projection - predicts x_0 directly (τ₀-prediction)
        x = self.final_norm(x)
        return self.final_layer(x)  # Shape: (Batch, pred_frames, feature_dim)


if __name__ == "__main__":
    # Quick shape check
    B, T, F, D = 4, 20, 4, 256
    model = AISDiffusionTransformer(pred_frames=T, feature_dim=F, embed_dim=D)
    
    z_t = torch.randn(B, T, F)
    t = torch.randint(0, 1000, (B,))
    context = torch.randn(B, D)
    
    out = model(z_t, t, context)
    print("Decoder output shape:", out.shape)  # Expected: (4, 20, 4)
    print("Model predicts clean velocity trajectory x_0 (τ₀-prediction)")