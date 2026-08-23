#!/usr/bin/env python3
"""
Diffusion Transformer (DiT) Training for AIS Trajectory Prediction.
Implements HDP paper: VP noise schedule, τ₀-prediction, hybrid loss with detach, DPM-Solver sampling.

Key HDP paper insights implemented:
1. VP (variance-preserving) noise schedule following Zheng et al. (2025)
2. τ₀-prediction (direct velocity prediction) instead of ε-prediction
3. Hybrid loss with detached integral (Algorithm 1, Appendix D.3)
4. DPM-Solver with 6 steps for fast sampling (Appendix D.4)
5. Velocity representation [vx, vy, theta, yaw_rate] with waypoint supervision
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preparedataset import AISScenarioDataset
from model import SceneContextEncoder, AISDiffusionTransformer
from utils import GaussianDiffusion, hybrid_loss, detached_integral


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
    # Configuration matching HDP paper
    SCENARIO_DIR = "/run/media/akshat/Akshat_USB/generated_scenarios3"
    OBS_FRAMES = 20
    PRED_FRAMES = 20
    MAX_AGENTS = 10
    MAX_POLYLINES = 20
    FEATURE_DIM = 6        # Input feature dim (x, y, vx, vy, theta, yaw_rate)
    TARGET_DIM = 4         # Target feature dim (vx, vy, theta, yaw_rate) - velocity representation
    EMBED_DIM = 256
    BATCH_SIZE = 32
    EPOCHS = 100
    DIFFUSION_STEPS = 1000
    
    # HDP paper hyperparameters (Table 6)
    HYBRID_LOSS_WEIGHT = 0.1      # ω = 0.1 from paper
    DETACH_WINDOW = 3             # W = 3 from paper (L-1 where L=6, but we use L=20, so W=3 is reasonable)
    DT = 10.0                     # dt = 10s for AIS pipeline
    VP_BETA_START = 0.0001
    VP_BETA_END = 0.02
    LR = 5e-4                     # Learning rate from paper
    WEIGHT_DECAY = 0.01           # Weight decay from paper

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

    # 2. Build Model Pipeline
    scene_encoder = SceneContextEncoder(
        hist_steps=OBS_FRAMES,
        map_points=20,
        feature_dim=FEATURE_DIM,
        map_feature_dim=2,
        embed_dim=EMBED_DIM,
        num_heads=4,
    ).to(device)

    dit_model = AISDiffusionTransformer(
        pred_frames=PRED_FRAMES,
        feature_dim=TARGET_DIM,  # 4D velocity: [vx, vy, theta, yaw_rate]
        embed_dim=EMBED_DIM,
        num_layers=6,            # 6 layers as per paper (Table 6)
        num_heads=8              # 8 heads as per paper (Table 6)
    ).to(device)

    # Use VP schedule as per paper Appendix D.4
    diffusion = GaussianDiffusion(
        timesteps=DIFFUSION_STEPS, 
        beta_start=VP_BETA_START, 
        beta_end=VP_BETA_END, 
        schedule="vp"
    )

    # 3. Optimization (AdamW with paper's hyperparameters)
    optimizer = torch.optim.AdamW(
        list(scene_encoder.parameters()) + list(dit_model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    print(f"Device: {device}")
    print(f"Training samples: {len(dataset)}")
    print(f"Batches per epoch: {len(loader)}")
    print(f"HDP Configuration:")
    print(f"  - VP noise schedule (β_start={VP_BETA_START}, β_end={VP_BETA_END})")
    print(f"  - τ₀-prediction (direct velocity prediction)")
    print(f"  - Hybrid loss weight ω={HYBRID_LOSS_WEIGHT}")
    print(f"  - Detach window W={DETACH_WINDOW}")
    print(f"  - DT={DT}s")
    print(f"  - DPM-Solver with 6 steps for sampling")

    # 4. Training Loop
    for epoch in range(EPOCHS):
        scene_encoder.train()
        dit_model.train()
        epoch_loss = 0.0
        epoch_vel_loss = 0.0
        epoch_waypoint_loss = 0.0

        for batch in loader:
            # Prepare Inputs
            ego_hist = batch["ego_history"].to(device, non_blocking=True)
            # Target contains: [x, y, vx, vy, theta, yaw_rate]
            target_full = batch["ego_target"].to(device, non_blocking=True)
            
            # Innovation 1: Target Velocity Representation (HDP paper Section 4.2)
            # We predict the kinematics (velocities and yaw rate)
            target_vel = target_full[:, :, 2:6]  # [vx, vy, theta, yaw_rate]
            target_pos = target_full[:, :, 0:2]  # [x, y] for reference

            agents = batch["agents_history"].to(device, non_blocking=True)
            map_lines = batch["map_lines"].to(device, non_blocking=True)
            agent_mask = batch["agent_mask"].to(device, non_blocking=True)
            map_mask = batch["map_mask"].to(device, non_blocking=True)

            # Check for finite batch
            if not torch.isfinite(ego_hist).all() or not torch.isfinite(target_full).all():
                print("Skipping non-finite batch")
                continue

            # A. Encode Context
            context = scene_encoder(
                ego=ego_hist,
                agents=agents,
                map_lines=map_lines,
                agent_mask=agent_mask,
                map_mask=map_mask,
            )

            # B. Diffusion: Add noise to velocity target
            t = torch.randint(0, DIFFUSION_STEPS, (BATCH_SIZE,), device=device).long()
            z_t, noise = diffusion.q_sample(target_vel, t)

            # C. Model: Predict the clean velocity x_0 (τ₀-prediction)
            pred_vel_0 = dit_model(z_t, t, context)

            # D. Compute Hybrid Loss (Algorithm 1 from Appendix D.3)
            # Velocity loss: ||v_θ - v_0||^2
            loss_vel = F.mse_loss(pred_vel_0, target_vel)
            
            # Waypoint loss with detached integral
            # Only use [vx, vy] for waypoint integration (first 2 dims)
            pred_vel_xy = pred_vel_0[:, :, :2]
            target_vel_xy = target_vel[:, :, :2]
            
            loss_waypoint = hybrid_loss(
                pred_vel_xy, 
                target_vel_xy, 
                W=DETACH_WINDOW, 
                omega=HYBRID_LOSS_WEIGHT, 
                dt=DT
            ) - loss_vel  # hybrid_loss returns l_v + ω*l_wpt, so subtract l_v to get just l_wpt
            
            # Actually, let's compute it properly using the function
            # hybrid_loss already includes both terms, so we use it directly
            total_loss = hybrid_loss(
                pred_vel_xy, 
                target_vel_xy, 
                W=DETACH_WINDOW, 
                omega=HYBRID_LOSS_WEIGHT, 
                dt=DT
            )
            
            # Also compute individual components for logging
            with torch.no_grad():
                l_v = F.mse_loss(pred_vel_xy, target_vel_xy)
                pred_wpt = detached_integral(pred_vel_xy, DETACH_WINDOW, DT)
                gt_wpt = torch.cumsum(target_vel_xy, dim=1) * DT
                l_wpt = F.mse_loss(pred_wpt, gt_wpt)

            # F. Step
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(scene_encoder.parameters()) + list(dit_model.parameters()), 1.0
            )
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_vel_loss += l_v.item()
            epoch_waypoint_loss += l_wpt.item()

        avg_loss = epoch_loss / max(len(loader), 1)
        avg_vel = epoch_vel_loss / max(len(loader), 1)
        avg_wp = epoch_waypoint_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{EPOCHS} | total={avg_loss:.6f} | vel={avg_vel:.6f} | wp={avg_wp:.6f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "scene_encoder": scene_encoder.state_dict(),
                "dit_model": dit_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            }, f"checkpoint_epoch_{epoch + 1}.pt")

    print("Training complete!")
    
    # Test DPM-Solver sampling
    print("\nTesting DPM-Solver sampling...")
    scene_encoder.eval()
    dit_model.eval()
    
    with torch.no_grad():
        # Get a sample batch
        batch = next(iter(loader))
        ego_hist = batch["ego_history"].to(device, non_blocking=True)
        agents = batch["agents_history"].to(device, non_blocking=True)
        map_lines = batch["map_lines"].to(device, non_blocking=True)
        agent_mask = batch["agent_mask"].to(device, non_blocking=True)
        map_mask = batch["map_mask"].to(device, non_blocking=True)
        
        context = scene_encoder(
            ego=ego_hist,
            agents=agents,
            map_lines=map_lines,
            agent_mask=agent_mask,
            map_mask=map_mask,
        )
        
        # Sample using DPM-Solver (6 steps as per paper)
        shape = (1, PRED_FRAMES, TARGET_DIM)
        sampled_vel = diffusion.sample(
            dit_model, 
            context[:1], 
            shape, 
            use_dpm_solver=True, 
            steps=6
        )
        print(f"DPM-Solver sampled velocity shape: {sampled_vel.shape}")
        
        # Integrate to get waypoints
        sampled_waypoints = torch.cumsum(sampled_vel[:, :, :2], dim=1) * DT
        print(f"Sampled waypoints shape: {sampled_waypoints.shape}")


if __name__ == "__main__":
    main()