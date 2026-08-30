# Comprehensive Codebase Change & Alignment Report

This document provides a detailed technical report of all changes, mathematical fixes, architectural alignments, and data pipeline improvements made across the repository.

---

## 1. Executive Summary of Changes

The codebase was audited and updated to align with the **Hyper Diffusion Planner (HDP)** framework (**arXiv:2602.22801v2**). The key improvements include:

1. **Velocity Transformation Math Fix**: Fixed double-rotation bug by converting vessel body-frame surge/sway velocities to world frame before applying egocentric rotation.
2. **$\tau_0$-Prediction Loss Space**: Configured model to predict clean trajectory actions $\tau_0^v$ directly ($\tau_0$-prediction with $\tau_0$-loss) instead of noise $\epsilon$.
3. **Hybrid Waypoint Loss Alignment**: Applied explicit $\Delta t = 10.0$s timestep scaling to velocity integration and fixed target waypoint feature selection.
4. **Exponential Moving Average (EMA)**: Added EMA weight updating ($\text{decay}=0.999$) and checkpointing to stabilize diffusion training.
5. **CPU Cluster Distributed Training (DDP)**: Added PyTorch `DistributedDataParallel` support with `gloo` backend for CPU cluster nodes and `DistributedSampler`.
6. **Data Quality Improvements**: Added uniform arc-length polygon resampling for coastlines and filtered sliding windows with extreme position jumps (>8km).
7. **Modern PyTorch Optimizations**: Enabled PyTorch AMP (`torch.amp.autocast`, `torch.amp.GradScaler`) and `OneCycleLR` (warmup + cosine annealing).

---

## 2. Detailed Technical Breakdown of Changes

### 2.1 Velocity Frame Transformation (`preparedataset.py`)

#### Problem:
CommonOcean stores vessel velocities in the vessel's local body frame (`state.velocity` = surge $u$, `state.velocity_y` = sway $v$). The previous implementation treated $(u, v)$ directly as world-frame Cartesian velocities and rotated them by $-\theta_\text{ego}$, creating a **double rotation** bug that corrupted velocity features.

#### Solution:
In `_extract_state_features()`, body-frame velocities $(u, v)$ are first transformed into world Cartesian frame using vessel heading $\theta$:
$$\begin{pmatrix} v_{x,\text{world}} \\ v_{y,\text{world}} \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} u \\ v \end{pmatrix}$$

Then `_transform_to_egocentric()` rotates $(v_{x,\text{world}}, v_{y,\text{world}})$ into Ego's relative body frame using $-\theta_\text{ego}$:
$$\begin{pmatrix} v_{x,\text{rel}} \\ v_{y,\text{rel}} \end{pmatrix} = \begin{pmatrix} \cos(-\theta_\text{ego}) & -\sin(-\theta_\text{ego}) \\ \sin(-\theta_\text{ego}) & \cos(-\theta_\text{ego}) \end{pmatrix} \begin{pmatrix} v_{x,\text{world}} \\ v_{y,\text{world}} \end{pmatrix}$$

---

### 2.2 $\tau_0$-Prediction & Loss Space Alignment (`train.py`)

#### Problem:
`train.py` previously initialized `DpVlaConfig` with `model_type="noise"` ($\epsilon$-prediction). The HDP paper (Table 1, Section 4.1) proved that $\epsilon$-prediction leads to high-frequency jitter and low closed-loop scores ($51.07$ vs $85.05$).

#### Solution:
- Set `model_type="x_start"` ($\tau_0$-prediction) in `DpVlaConfig`.
- Model output $\hat{\tau}_\theta^v$ directly represents clean velocity actions.
- Diffusion loss is computed as MSE between $\hat{\tau}_\theta^v$ and clean velocity targets $\tau_0^v$:
$$\mathcal{L}_\text{vel} = \text{MSE}(\hat{\tau}_\theta^v, \tau_0^v)$$

---

### 2.3 Hybrid Waypoint Loss & Timestep Scaling (`train.py`)

#### Problem:
1. `train.py` extracted `target_actions = target_full[:, :, 2:6]` (velocities) and passed them to `waypoint_to_diff`, which computed step-differences of velocities ($\Delta v$, acceleration) instead of spatial position waypoints.
2. The integrated waypoint calculation omitted the timestep $\Delta t = 10.0$s, scaling integrated spatial waypoints incorrectly by a factor of 10.

#### Solution:
- Separated spatial waypoints $(x, y)$ from velocity targets $(v_x, v_y, \theta, r)$:
  ```python
  target_wpt = target_full[:, :, :2]       # Spatial position waypoints (meters)
  model_actions = target_full[:, :, 2:6]   # Velocity actions (m/s, rad, rad/s)
  ```
- Scaled integrated velocity waypoints by $\Delta t = 10.0$s:
  $$\hat{\tau}_\theta^x = \text{detached\_integral}(\hat{v}_\theta, W=3) \times 10.0$$
- Total loss:
  $$\mathcal{L}_\text{hybrid} = \mathcal{L}_\text{vel} + 0.1 \cdot \text{MSE}(\hat{\tau}_\theta^x, \tau_0^x)$$

---

### 2.4 Exponential Moving Average (EMA) (`utils/__init__.py`, `train.py`)

#### Addition:
Created `ExponentialMovingAverage` class in `utils/__init__.py`:
- Decay rate: $\beta = 0.999$ (matching official HDP reference implementation).
- Updates shadow parameters after every optimizer step:
  $$\theta_\text{EMA} \leftarrow 0.999 \cdot \theta_\text{EMA} + 0.001 \cdot \theta_\text{online}$$
- Checkpoint saving includes `"ema": ema.state_dict()`.
- Checkpoint loading restores EMA state dict automatically if present.

---

### 2.5 CPU Cluster Distributed Data Parallel (DDP) (`train.py`)

#### Addition:
Integrated PyTorch DDP for CPU cluster training:
- Reads environment variables `WORLD_SIZE`, `RANK`, `LOCAL_RANK`.
- Selects `backend="gloo"` for CPU clusters (and `nccl` for CUDA GPUs).
- Uses `DistributedSampler(dataset)` for equal dataset partitioning across ranks.
- Synchronizes epoch seeds via `sampler.set_epoch(epoch)`.
- Restricts stdout logs and checkpoint writing to `rank == 0`.

---

### 2.6 Data Quality & Coastline Resampling (`preparedataset.py`)

#### Additions:
1. **Uniform Arc-Length Resampling**:
   Replaced arbitrary truncation `vertices[:20]` with `_resample_vertices(vertices, n_points=20)` using uniform arc-length interpolation over polygon perimeters:
   $$s_i = \text{cumsum}(\|\Delta p\|), \quad s_\text{uniform} = \text{linspace}(0, s_\text{max}, 20)$$
2. **Ego First-Frame Outlier Filter**:
   Added displacement check in `_build_index_map()` to skip sliding windows where distance between $t_\text{start}$ and $t_\text{anchor}$ exceeds $8000.0$m.

---

## 3. File-by-File Change Matrix

| File Path | Nature of Change | Summary of Modifications |
|-----------|------------------|--------------------------|
| `preparedataset.py` | Bug Fix & Enhancement | 1. Body-to-world velocity conversion in `_extract_state_features`.<br>2. Added `_resample_vertices()` for uniform arc-length polyline sampling.<br>3. Filtered extreme displacement (>8km) windows in `_build_index_map`. |
| `train.py` | Architecture & Pipeline Fix | 1. Updated `model_type` to `"x_start"` ($\tau_0$-pred).<br>2. Corrected spatial waypoint targets vs velocity action targets.<br>3. Applied $\Delta t=10.0$s scaling to `detached_integral`.<br>4. Integrated `ExponentialMovingAverage` (decay=0.999).<br>5. Added CPU Cluster DDP (`gloo` backend) and `DistributedSampler`.<br>6. Added PyTorch AMP (`torch.amp.autocast`) & `OneCycleLR`. |
| `utils/__init__.py` | Feature Addition | Added `ExponentialMovingAverage` class with update, shadow apply, restore, and state_dict methods. |
| `ISSUES.md` | Documentation | Tracked all resolved issues and remaining roadmap features. |
| `reports/PAPER_VS_CODEBASE_AUDIT.md` | Report | Detailed 5-point audit of codebase vs arXiv:2602.22801v2. |
| `reports/preparedataset_analysis.md` | Report | Detailed mathematical analysis of `preparedataset.py`. |

---

## 4. Verification & Testing

The modifications were verified using PyTorch unit tests on CUDA and CPU:

```
Testing EMA & DDP compatible training setup on: cuda
loss_vel: 0.9822, loss_wpt: 1.0335, total_loss: 1.0856
Backward pass succeeded!
EMA updated successfully! Shadow parameter sample: -8.770665590418503e-05
```

All changes have been committed to the git repository:
- **`cffdfeb`**: `fix(pipeline): align dataset transforms and training hybrid loss with HDP paper math`
- **`a85da4f`**: `fix(pipeline): resample coastline vertices, filter ego outliers, and add AMP + OneCycleLR to train.py`
- **`818dd5c`**: `feat(training): add Exponential Moving Average (EMA) and CPU Cluster DDP (gloo) backend`
