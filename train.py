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
from torch.utils.data import DataLoader, WeightedRandomSampler

from preparedataset import AISScenarioDataset
from model import (
    DpVlaConfig,
    DpVlaModel,
    DiffusionSDE,
    NoiseScheduleVP,
    TimeSampler,
)
from utils import (
    detached_integral,
    hybrid_loss,
    ExponentialMovingAverage,
    apply_maritime_augmentations,
)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


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
                 max_agents=10, max_polylines=20, num_workers=4, is_distributed=False,
                 use_stratified_sampling=True):
    dataset = AISScenarioDataset(
        scenario_dir=scenario_dir,
        obs_frames=obs_frames,
        pred_frames=pred_frames,
        max_agents=max_agents,
        max_polylines=max_polylines,
    )
    if is_distributed:
        sampler = DistributedSampler(dataset)
    elif use_stratified_sampling:
        weights = []
        for item in dataset.index_map:
            cat = item.get('category', 'coastal')
            if cat == 'opensea':
                weights.append(2.0)   # Upweight rare open-sea scenarios
            elif cat == 'congested':
                weights.append(1.0)   # Standard weight for port scenarios
            else:
                weights.append(1.2)   # Coastal transit
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader, sampler


def main():
    SCENARIO_DIR = "/run/media/akshat/Akshat_USB/all_scenerios"
    OBS_FRAMES = 20
    PRED_FRAMES = 20
    MAX_AGENTS = 10
    MAX_POLYLINES = 20
    BATCH_SIZE = 32
    EPOCHS = 100

    # DDP & CPU Cluster Configuration
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed:
        backend = "gloo" if not torch.cuda.is_available() else "nccl"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Build Data Pipeline
    dataset, loader, sampler = build_loader(
        SCENARIO_DIR,
        batch_size=BATCH_SIZE,
        obs_frames=OBS_FRAMES,
        pred_frames=PRED_FRAMES,
        max_agents=MAX_AGENTS,
        max_polylines=MAX_POLYLINES,
        is_distributed=is_distributed,
    )

    # 2. Build Model & Diffusion SDE (matching official HDP / DP-VLA config)
    config = DpVlaConfig(
        with_encoder=False,      # Using lightweight context encoder for ego+agents+map
        hidden_size=512,
        depth=6,
        num_heads=8,
        num_actions=PRED_FRAMES,
        dim_action=4,
        dim_y=12,
        model_type="x_start",    # HDP Paper: tau0-prediction (x_start) with tau0-loss
        kinematic_type="diff",   # Velocity-based trajectory representation
    )

    model = DpVlaModel(config).to(device)

    if is_distributed:
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)

    raw_model = model.module if is_distributed else model
    ema = ExponentialMovingAverage(raw_model, decay=0.999)

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

    # 3. Optimizer, LR Scheduler & AMP Scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.01,
    )

    total_steps = EPOCHS * len(loader)
    warmup_steps = int(total_steps * 0.05)  # 5% warmup steps per paper
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        total_steps=max(total_steps, 1),
        pct_start=0.05,
        anneal_strategy="cos",
    )

    use_amp = torch.cuda.is_available()
    device_type = "cuda" if use_amp else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Device: {device}")
    print(f"Training samples: {len(dataset)}")
    print(f"Batches per epoch: {len(loader)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Kinematic type: {config.kinematic_type}, Model type: {config.model_type}, AMP enabled: {use_amp}")

    start_epoch = 0
    checkpoint_path = "weight/checkpoint_epoch_10.pt"  # Set to None or empty string to train from scratch

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from '{checkpoint_path}'...")
        # map_location=device prevents GPU/CPU memory conflicts if hardware changed
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        raw_model.load_state_dict(checkpoint["model"])
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        start_epoch = checkpoint["epoch"]
        # Use .get() to prevent crashes if an older checkpoint didn't save 'loss'
        previous_loss = checkpoint.get("loss", "Unknown")
        
        print(f"Resuming training from epoch {start_epoch} with previous loss: {previous_loss}")
    else:
        print("No valid checkpoint found. Starting training from scratch (Epoch 0).")

    for epoch in range(start_epoch, EPOCHS):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        for raw_batch in loader:
            batch = apply_maritime_augmentations(raw_batch) if model.training else raw_batch

            ego_hist = batch["ego_history"].to(device, non_blocking=True)
            target_full = batch["ego_target"].to(device, non_blocking=True)
            agents = batch["agents_history"].to(device, non_blocking=True)
            map_lines = batch["map_lines"].to(device, non_blocking=True)
            agent_mask = batch["agent_mask"].to(device, non_blocking=True)
            map_mask = batch["map_mask"].to(device, non_blocking=True)

            if not (torch.isfinite(ego_hist).all() and torch.isfinite(target_full).all()):
                continue

            # Target selection: [x, y, vx, vy, theta, yaw_rate]
            target_wpt = target_full[:, :, :2]       # Spatial waypoints [x, y] in meters
            model_actions = target_full[:, :, 2:6]   # Velocity actions [vx, vy, theta, yaw_rate]

            # Sample noise and timesteps via official DiffusionSDE
            action_with_noise, t, target_dict = diffusion_sde.sample(model_actions)

            # Construct proprioception conditioning (ego latest status)
            proprio = torch.cat([
                ego_hist[:, -1, :],
                torch.zeros((ego_hist.shape[0], config.dim_y - 6), device=device)
            ], dim=-1)

            # Encode context
            enc_out = model.fallback_encoder(ego_hist, agents, map_lines, agent_mask, map_mask)

            # Predict x_start (clean velocity actions tau0_v) directly with mixed precision
            with torch.amp.autocast(device_type, enabled=use_amp):
                output = model(
                    action_with_noise=action_with_noise,
                    time=t,
                    proprio=proprio,
                    encoder_hidden_states=enc_out.last_hidden_state,
                    attention_mask=enc_out.attention_mask,
                )

                # 1. Supervised tau0 diffusion loss on velocity representation (HDP Paper Eq. 3)
                loss_vel = F.mse_loss(output.prediction.float(), target_dict["x_start"].float())

                # 2. Hybrid waypoint loss with detached integral (HDP Paper Eq. 3, 4 & Algorithm 1)
                # Integrate predicted velocity (vx, vy) with DT_SECONDS=10.0 to get spatial waypoints
                if config.kinematic_type == "diff":
                    pred_v = output.prediction.float()[:, :, :2]
                    pred_wpt = detached_integral(pred_v, detach_window_size=3) * 10.0  # dt = 10.0s
                    loss_wpt = F.mse_loss(pred_wpt, target_wpt.float())
                    loss = loss_vel + 0.1 * loss_wpt
                else:
                    loss = loss_vel

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_before <= scale_after:
                scheduler.step()

            ema.update(raw_model)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(loader), 1)
        if rank == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | loss={avg_loss:.6f} | lr={scheduler.get_last_lr()[0]:.6e}")

        if (epoch + 1) % 10 == 0 and rank == 0:
            os.makedirs("weight", exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model": raw_model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            }, f"weight/checkpoint_epoch_{epoch + 1}.pt")

    print("Training complete!")

    

if __name__ == "__main__":
    main()
