# Maritime Data Augmentation Strategies for HDP Diffusion Planning

This document outlines recommended data augmentation techniques designed specifically for **marine AIS datasets, coastline polylines, and diffusion-based trajectory planners (HDP)**.

---

## 1. Executive Summary & Domain Rationale

Unlike standard vision or autonomous driving datasets, maritime AIS trajectory data suffers from specific challenges:
- **Over-reliance on map landmarks**: Planners trained strictly on coastal data can overfit to specific port/channel geometries and fail in open sea.
- **AIS Signal Loss & Latency**: Surround ships frequently drop AIS updates or experience GPS jitter.
- **Closed-Loop Error Accumulation**: Planners trained purely on human expert trajectories struggle when slightly off-center because they only saw perfect trajectories during training.

Applying domain-specific data augmentations improves generalization, open-sea robustness, and closed-loop stability.

---

## 2. Key Data Augmentation Techniques

### 2.1 Coastline / Map Augmentations

#### A. Coastline Dropout (Open-Sea Conditioning)
- **Concept**: Randomly mask out coastline data (`map_mask = True`) with probability $p_{\text{map\_drop}} \in [0.10, 0.20]$ during training.
- **Rationale**: Forces the diffusion planner to learn valid open-sea navigation and COLREG compliance using vessel dynamics alone, without over-depending on coastline geometry.
- **Implementation**:
  ```python
  # In train.py:
  if self.training and np.random.rand() < 0.15:
      map_mask = torch.ones_like(map_mask, dtype=torch.bool)  # Drop all coastlines
  ```

#### B. Polyline Vertex Perturbation (Map Uncertainty / Tide Simulation)
- **Concept**: Add small Gaussian noise to resampled coastline coordinates:
  $$\tilde{p}_{\text{map}} = p_{\text{map}} + \delta, \quad \delta \sim \mathcal{N}(0, \sigma^2 \mathbf{I}), \quad \sigma \approx 1.5\text{ meters}$$
- **Rationale**: Simulates water level fluctuations (tides), GPS drift, and digital map boundaries uncertainty.

#### C. Polyline Dropout / Partial Masking
- **Concept**: Randomly drop individual coastline polylines (e.g. drop 20% of polylines) or randomly mask out trailing vertices of polylines.
- **Rationale**: Mimics limited sensor range or occlusion by larger vessels.

---

### 2.2 Surround Traffic Vessel (Agent) Augmentations

#### A. Agent Dropout (Sensor Occlusion & AIS Packet Loss)
- **Concept**: Randomly mask out surrounding vessels with probability $p_{\text{agent\_drop}} \in [0.10, 0.20]$.
- **Rationale**: Real maritime AIS updates are broadcast every 2–10 seconds and frequently experience packet dropouts. Agent dropout teaches the planner to operate safely even when surrounding vessels are temporarily unobserved.
- **Implementation**:
  ```python
  # In train.py or dataset:
  if self.training:
      drop_mask = (torch.rand_like(agent_mask.float()) < 0.15)
      agent_mask = agent_mask | drop_mask  # Mask out additional surround ships
  ```

#### B. Agent Kinematic Jittering
- **Concept**: Perturb surrounding ships' positions ($\sigma_p \approx 0.5\text{m}$) and velocities ($\sigma_v \approx 0.2\text{ m/s}$).
- **Rationale**: Accounts for AIS sensor discretization and velocity estimation noise.

#### C. Spatial Reflection Symmetry (Port/Starboard Flip for Open Sea)
- **Concept**: Reflect the entire scene across the longitudinal axis ($y \to -y$, $v_y \to -v_y$, $r \to -r$).
- **Conditions**: Applicable to open-sea scenarios without land bounds or symmetric channels.
- **Rationale**: Exploits lateral symmetry to double the effective training dataset size.

---

### 2.3 Ego Anchor State Perturbations (Closed-Loop Error Recovery)

#### A. Off-Policy Position & Orientation Jitter (DART / ChauffeurNet Style)
- **Concept**: Inject small perturbations into Ego's anchor state $(x_0, y_0, \theta_0)$ at $T=0$:
  $$\Delta x_0, \Delta y_0 \sim \mathcal{N}(0, \sigma_p^2), \quad \Delta \theta_0 \sim \mathcal{N}(0, \sigma_\theta^2)$$
  where $\sigma_p \approx 1.0 - 2.0\text{m}$ and $\sigma_\theta \approx 2^\circ - 5^\circ$.
- **Rationale**: Pure imitation learning suffers from **covariate shift** — if the vessel drifts slightly off-track during closed-loop execution, the model has never seen off-center states and fails to correct. Injecting anchor noise forces the planner to learn corrective recovery maneuvers.

---

### 2.4 Temporal Jittering & Speed Scaling

#### A. Sliding Window Temporal Shift
- **Concept**: Jitter the starting frame index $t_{\text{start}}$ by $\pm 1 - 2$ frames ($10 - 20\text{s}$) during dataset indexing.
- **Rationale**: Prevents overfitting to specific AIS transmission timestamps.

#### B. Speed Rescaling (Current / Sea-State Simulation)
- **Concept**: Rescale trajectory velocities by a uniform factor $s \in [0.9, 1.1]$:
  $$\mathbf{v}_{\text{scaled}} = s \cdot \mathbf{v}$$
- **Rationale**: Simulates varying sea currents, wave resistance, or vessel loading conditions.

---

## 3. Recommended Implementation Plan

| Augmentation Technique | Target Module | Recommended Prob / Noise | Primary Benefit |
| :--- | :--- | :--- | :--- |
| **Coastline Dropout** | `train.py` | $p = 0.15$ | Prevents map over-reliance; enables open-sea planning |
| **Agent Dropout** | `train.py` | $p = 0.15$ | Robustness to AIS packet loss |
| **Polyline Noise Injection** | `preparedataset.py` | $\sigma = 1.5\text{m}$ | Map boundary & tide variation robustness |
| **Anchor State Perturbation** | `preparedataset.py` | $\sigma_p = 1.5\text{m}, \sigma_\theta = 3^\circ$ | Eliminates closed-loop compounding error |
| **Port/Starboard Flip** | `train.py` | $p = 0.50$ (Open Sea) | Doubles dataset capacity via spatial symmetry |

---

## 4. Example PyTorch Integration Code Snippet

```python
def apply_maritime_augmentations(batch, p_map_drop=0.15, p_agent_drop=0.15, p_flip=0.50):
    """
    Applies online maritime data augmentations to training batches.
    """
    ego_hist = batch["ego_history"].clone()
    target = batch["ego_target"].clone()
    agents = batch["agents_history"].clone()
    map_lines = batch["map_lines"].clone()
    agent_mask = batch["agent_mask"].clone()
    map_mask = batch["map_mask"].clone()

    B = ego_hist.shape[0]

    # 1. Coastline Dropout
    map_drop_idx = (torch.rand(B, device=ego_hist.device) < p_map_drop)
    map_mask[map_drop_idx] = True

    # 2. Agent Dropout
    agent_drop_rand = torch.rand(agent_mask.shape, device=ego_hist.device)
    agent_mask = agent_mask | (agent_drop_rand < p_agent_drop)

    # 3. Polyline Vertex Jittering (1.5m noise)
    if map_lines.numel() > 0:
        noise = torch.randn_like(map_lines) * 1.5
        map_lines = map_lines + noise

    # 4. Port/Starboard Reflection Flip (for open-sea samples without map constraints)
    flip_idx = (torch.rand(B, device=ego_hist.device) < p_flip) & map_drop_idx
    if flip_idx.any():
        # Mirror Y coordinates, Y velocities, and yaw rate
        ego_hist[flip_idx, :, 1] *= -1.0   # rel_y
        ego_hist[flip_idx, :, 3] *= -1.0   # rel_vy
        ego_hist[flip_idx, :, 4] *= -1.0   # rel_theta
        ego_hist[flip_idx, :, 5] *= -1.0   # yaw_rate

        target[flip_idx, :, 1] *= -1.0
        target[flip_idx, :, 3] *= -1.0
        target[flip_idx, :, 4] *= -1.0
        target[flip_idx, :, 5] *= -1.0

        agents[flip_idx, ..., 1] *= -1.0
        agents[flip_idx, ..., 3] *= -1.0
        agents[flip_idx, ..., 4] *= -1.0
        agents[flip_idx, ..., 5] *= -1.0

    return ego_hist, target, agents, map_lines, agent_mask, map_mask
```
