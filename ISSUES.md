# Codebase Issues & Fix Status

This document tracks identified codebase issues, architectural alignment with the HDP paper (arXiv:2602.22801v2), and their resolution status.

---

## 🟢 Resolved Issues

#### 1. `epsilon`-Prediction vs `tau0`-Loss (Critical)
- **Status**: FIXED
- **Details**: Updated `model_type` to `"x_start"` in `DpVlaConfig` and `train.py` (matching HDP Paper Section 4.1).

#### 2. Velocity Transformation Double-Rotation (Critical)
- **Status**: FIXED
- **Details**: `preparedataset.py` converted body-frame surge/sway velocities to world-frame velocities prior to egocentric transformation:
  $$\begin{pmatrix} v_{x,\mathrm{world}} \\\\ v_{y,\mathrm{world}} \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} u \\\\ v \end{pmatrix}$$

#### 3. Hybrid Waypoint Loss Timestep Scaling (High)
- **Status**: FIXED
- **Details**: Added explicit $\Delta t = 10.0$s scaling to `detached_integral` in `train.py` to correctly integrate velocities into spatial meters:
  $$\hat{\tau}_0^x = \mathrm{detached\_integral}(\hat{\tau}_0^v, W=3) \cdot \Delta t \quad (\text{where } \Delta t = 10.0\text{s})$$

#### 4. Target Action Feature Indexing (High)
- **Status**: FIXED
- **Details**: Separated spatial position waypoints `target_wpt = target_full[:, :, :2]` from velocity action targets `model_actions = target_full[:, :, 2:6]` in `train.py`.

#### 5. Coastline Vertex Sampling Bias (Medium)
- **Status**: FIXED
- **Details**: Added `_resample_vertices(vertices, n_points=20)` using uniform arc-length interpolation in `preparedataset.py` instead of taking the first 20 raw vertices.

#### 6. Ego First-Frame Outliers (Medium)
- **Status**: FIXED
- **Details**: Filtered sliding windows in `_build_index_map()` where displacement between $t_{start}$ and $t_{anchor}$ exceeds $8000.0$ meters.

#### 7. Exponential Moving Average (EMA) Weights & Checkpointing (High)
- **Status**: FIXED
- **Details**: Added `ExponentialMovingAverage` (decay=0.999 per paper/timm) in `utils/__init__.py`. `train.py` continuously updates EMA shadow weights during training and saves `ema` state dictionary in checkpoints.

#### 8. CPU Cluster Distributed Training (DDP / Gloo) (High)
- **Status**: FIXED
- **Details**: Added PyTorch DDP launcher support in `train.py`. Automatically uses the `gloo` backend for CPU cluster nodes (and `nccl` for CUDA GPUs), with `DistributedSampler` managing data partition across cluster ranks.

#### 9. LR Scheduler & Warmup (Medium)
- **Status**: FIXED
- **Details**: Added `torch.optim.lr_scheduler.OneCycleLR` with 5% warmup and cosine decay in `train.py`.

#### 8. Mixed Precision (AMP) Training (Low)
- **Status**: FIXED
- **Details**: Enabled modern `torch.amp.autocast("cuda")` and `torch.amp.GradScaler("cuda")` in `train.py`.

---

## 🟡 Remaining / Planned Enhancements

#### 1. Reinforcement Learning (HDP-RL) Post-Training Pipeline
- **Description**: HDP paper achieves an 84.07 overall score via RL post-training. The weighted diffusion loss is in `utils/__init__.py`, but the pseudo-closed-loop simulator and reward functions ($r_{safety}, r_{risk}, r_{follow}, r_{lane}$) are pending implementation.

#### 2. Multi-GPU Distributed Data Parallel (DDP)
- **Description**: Add PyTorch DDP launcher support to scale training across multiple GPUs.

#### 3. Config Externalization
- **Description**: Move hardcoded parameters and dataset paths to YAML configuration files.
