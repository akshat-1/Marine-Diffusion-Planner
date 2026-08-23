#!/usr/bin/env python3
"""
Diffusion Transformer Training for Trajectory Prediction.
Aligned with official HDP-navsim DP-VLA base architecture and diffusion pipeline.

- Model: DpVlaModel (CustomDiT decoder + adaLN-zero blocks + CustomCrossAttention)
- Sampling: DiffusionSDE + NoiseScheduleVP + DPM_Solver
- Loss: Prediction loss + Hybrid waypoint loss with detached_integral
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preparedataset import AISScenarioDataset
from model import (
    DpVlaConfig,
    DpVlaModel,
    DiffusionSDE,
    NoiseScheduleVP,
    TimeSampler,
)
from utils import detached_integral, hybrid_loss


def waypoint_to_diff(actions: torch.Tensor) -> torch.Tensor:
    """Official waypoint to step-diff conversion."""
    xy = actions[..., :2]
    origin = torch.zeros_like(xy[..., :1, :])
    prev_xy = torch.cat([origin, xy[..., :-1, :]], dim=-2)
    return torch.cat([xy - prev_xy, actions[..., 2:4]], dim=-1)


def diff_to_waypoint(actions: torch.Tensor) -> torch.Tensor:
    """Official step-diff back to waypoint conversion."""
    xy = torch.cumsum(actions[..., :2], dim=-2)
    return torch.cat([xy, actions[..., 2:4]], dim=-1)


def build_loader(scenario_dir, batch_size=32, obs_frames=20, pred_frames=20,
                 max_agents=10, max_polylines=20, num_workers=4):
    dataset = AISScenarioDataset(
        scenario_dir=scenario_dir,
        obs_frames=obs_frames,
        pred_frames=pred_frames,
        max_agents=max_agents,
        max_polylines=max_polylines,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def main():
    SCENARIO_DIR = "/run/media/akshat/Akshat_USB/generated_scenarios3"
    OBS_FRAMES = 20
    PRED_FRAMES = 20
    MAX_AGENTS = 10
    MAX_POLYLINES = 20
    BATCH_SIZE = 32
    EPOCHS = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Build Data Pipeline
    dataset, loader = build_loader(
        SCENARIO_DIR,
        batch_size=BATCH_SIZE,
        obs_frames=OBS_FRAMES,
        pred_frames=PRED_FRAMES,
        max_agents=MAX_AGENTS,
        max_polylines=MAX_POLYLINES,
    )

    # 2. Build Model & Diffusion SDE (matching official DP-VLA config)
    config = DpVlaConfig(
        with_encoder=False,      # Using lightweight context encoder for ego+agents+map
        hidden_size=512,
        depth=6,
        num_heads=8,
        num_actions=PRED_FRAMES,
        dim_action=4,
        dim_y=12,
        model_type="noise",
        kinematic_type="diff",   # Predict velocity/diff actions
    )

    model = DpVlaModel(config).to(device)

    # Official VP SDE & Noise Schedule setup
    alphas_cumprod = torch.cos(((torch.linspace(0, 1000, 1001) / 1000) + 0.008) / 1.008 * 3.141592653589793 * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)

    sde = NoiseScheduleVP(schedule="discrete", betas=betas)
    time_sampler = TimeSampler(sample_method="uniform", eps=1e-3)
    diffusion_sde = DiffusionSDE(
        sde=sde,
        time_sampler=time_sampler,
        sample_order=2,
        sample_skip_type="time_uniform",
        sample_method="multistep",
        denoise_to_zero=False,
    )

    # 3. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.01,
    )

    print(f"Device: {device}")
    print(f"Training samples: {len(dataset)}")
    print(f"Batches per epoch: {len(loader)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Kinematic type: {config.kinematic_type}, Model type: {config.model_type}")

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            ego_hist = batch["ego_history"].to(device, non_blocking=True)
            target_full = batch["ego_target"].to(device, non_blocking=True)
            agents = batch["agents_history"].to(device, non_blocking=True)
            map_lines = batch["map_lines"].to(device, non_blocking=True)
            agent_mask = batch["agent_mask"].to(device, non_blocking=True)
            map_mask = batch["map_mask"].to(device, non_blocking=True)

            if not (torch.isfinite(ego_hist).all() and torch.isfinite(target_full).all()):
                continue

            # Convert waypoints to target actions (diff/velocity representation)
            target_actions = target_full[:, :, 2:6]  # [vx, vy, theta, yaw_rate]
            if config.kinematic_type == "diff":
                model_actions = waypoint_to_diff(target_actions)
            else:
                model_actions = target_actions

            # Sample noise and timesteps via official DiffusionSDE
            action_with_noise, t, target_dict = diffusion_sde.sample(model_actions)

            # Construct proprioception conditioning (ego latest status)
            proprio = torch.cat([
                ego_hist[:, -1, :],
                torch.zeros((ego_hist.shape[0], config.dim_y - 6), device=device)
            ], dim=-1)

            # Encode context
            enc_out = model.fallback_encoder(ego_hist, agents, map_lines, agent_mask, map_mask)

            # Predict noise
            output = model(
                action_with_noise=action_with_noise,
                time=t,
                proprio=proprio,
                encoder_hidden_states=enc_out.last_hidden_state,
                attention_mask=enc_out.attention_mask,
            )

            # Supervised diffusion loss
            loss_noise = F.mse_loss(output.prediction, target_dict["noise"])

            # Hybrid waypoint loss with detached integral if kinematic_type is diff
            if config.kinematic_type == "diff":
                pred_x0 = (action_with_noise - diffusion_sde.sde.marginal_std(t)[:, None, None] * output.prediction) / diffusion_sde.sde.marginal_alpha(t)[:, None, None]
                pred_wpt = detached_integral(pred_x0[..., :2], detach_window_size=3)
                target_wpt = torch.cumsum(model_actions[..., :2], dim=-2)
                loss_wpt = F.mse_loss(pred_wpt, target_wpt)
                loss = loss_noise + 0.1 * loss_wpt
            else:
                loss = loss_noise

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{EPOCHS} | loss={avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            }, f"checkpoint_epoch_{epoch + 1}.pt")

    print("Training complete!")

    # 5. Inference / Sampling Test via generate()
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        ego_hist = batch["ego_history"][:1].to(device)
        agents = batch["agents_history"][:1].to(device)
        map_lines = batch["map_lines"][:1].to(device)
        agent_mask = batch["agent_mask"][:1].to(device)
        map_mask = batch["map_mask"][:1].to(device)

        proprio = torch.cat([
            ego_hist[:, -1, :],
            torch.zeros((1, config.dim_y - 6), device=device)
        ], dim=-1)

        enc_out = model.fallback_encoder(ego_hist, agents, map_lines, agent_mask, map_mask)

        gen_actions = model.generate(
            diffusion_sde=diffusion_sde,
            encoder_hidden_states=enc_out.last_hidden_state,
            proprio=proprio,
            attention_mask=enc_out.attention_mask,
            steps=6,
        )
        print(f"Generated actions shape (DPM-Solver): {gen_actions.shape}")


if __name__ == "__main__":
    main()
