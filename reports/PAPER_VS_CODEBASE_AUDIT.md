# Codebase vs HDP Paper (arXiv:2602.22801v2) Discrepancy & Bug Analysis Report

This document presents a comprehensive review of the current codebase against the core findings, formulas, and architecture proposed in **"Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving" (Hyper Diffusion Planner / HDP)**.

---

## Executive Summary

The paper introduces **HDP (Hyper Diffusion Planner)**, establishing key design principles for diffusion-based trajectory planners:
1. **$\tau_0$-prediction with $\tau_0$-loss** is optimal for low-dimensional trajectory manifolds.
2. **Velocity representation with Hybrid Loss** (velocity MSE + integrated waypoint MSE with stop-gradient/detach) yields smooth and geometrically precise trajectories without altering score-matching optima.
3. **RL Post-training** via reward-weighted diffusion regression.

Our codebase audit revealed **several critical mathematical errors, configuration mismatches, and structural discrepancies** between the paper recommendations and the actual implementation.

---

## Critical Discrepancies & Issues Found

### Issue 1: Model Output & Diffusion Loss Space Mismatch (CRITICAL)

#### Paper Specification (Section 4.1, Table 1):
- **Finding:** $\tau_0$-prediction (directly outputting the clean trajectory $\tau_0$) supervised by $\tau_0$-loss achieves the highest performance (85.05 open-loop score vs 51.07 for $\epsilon$-pred).
- $\epsilon$-pred with $\epsilon$-loss generates noticeable non-smoothness, irregular jitters, and high-frequency artifacts during denoising.

#### Codebase Implementation:
- In `DEFAULT_MODEL_CONFIG` (`model/configuration_dp_vla.py`, line 29):
  ```python
  "model_type": "noise",  # <-- Uses noise/epsilon prediction!
  ```
- In `train.py` (lines 91, 190):
  ```python
  model_type="noise"
  ...
  loss_noise = F.mse_loss(output.prediction, target_dict["noise"])
  ```
- The model is configured and trained as **$\epsilon$-prediction with $\epsilon$-loss**, which the paper explicitly demonstrated to be sub-optimal (producing jerky trajectories and achieving much lower closed-loop performance).

#### Fix:
Change `model_type` to `"x_start"` (or `"tau_0"`) in configuration and compute loss against target $\tau_0$ (`x_data`) rather than noise $\epsilon$:
```python
# In train.py:
loss_tau0 = F.mse_loss(output.prediction, model_actions)
```

---

### Issue 2: Incorrect $\tau_0$-Reconstruction in Hybrid Loss (CRITICAL)

#### Paper Specification (Section 4.2 & Appendix D.3):
In velocity-represented diffusion, $\tau_0^v$ is predicted directly by the model $\tau_\theta^v(\tau_t^v, t, C)$. The hybrid loss combines velocity MSE and integrated waypoint MSE on the predicted clean velocity $\hat{\tau}_0^v$.

#### Codebase Implementation (`train.py`, line 194):
```python
if config.kinematic_type == "diff":
    pred_x0 = (action_with_noise - diffusion_sde.sde.marginal_std(t)[:, None, None] * output.prediction) / diffusion_sde.sde.marginal_alpha(t)[:, None, None]
    pred_wpt = detached_integral(pred_x0[..., :2], detach_window_size=3)
    target_wpt = torch.cumsum(model_actions[..., :2], dim=-2)
    loss_wpt = F.mse_loss(pred_wpt, target_wpt)
    loss = loss_noise + 0.1 * loss_wpt
```

#### Bug Analysis:
1. `output.prediction` is the model's output. Since `model_type="noise"`, the formula `(x_t - \sigma_t \cdot \epsilon_\theta) / \alpha_t` reconstructs `pred_x0`.
2. **Dimension mismatch / Feature selection error:**
   - `model_actions` shape is `(B, T, 4)`: `[delta_vx, delta_vy, theta, yaw_rate]`.
   - `pred_x0[..., :2]` extracts `[delta_vx, delta_vy]`.
   - `target_wpt = torch.cumsum(model_actions[..., :2], dim=-2)` integrates `model_actions`.
   - **HOWEVER:** If `model_type` were correctly set to `x_start` ($\tau_0$), then `output.prediction` IS ALREADY `pred_x0` directly! The noisy inversion `(action_with_noise - std * pred) / alpha` adds numerical instability when $\alpha_t \to 0$ (at high noise levels $t \to 1$).
3. **Double conversion flaw:**
   - `model_actions` in `train.py` line 164 is computed via `waypoint_to_diff(target_actions)`.
   - `waypoint_to_diff` computes `xy - prev_xy` (step differences).
   - Then `torch.cumsum(model_actions[..., :2], dim=-2)` sums step differences to recover original waypoints `xy`.
   - BUT `target_actions` were extracted as `target_full[:, :, 2:6]` — which are `[vx, vy, theta, yaw_rate]` (velocities, NOT waypoints!).
   - So `waypoint_to_diff` treats velocities `[vx, vy]` as if they were position coordinates `[x, y]`, computing `delta_vx = vx[t] - vx[t-1]`, and then `cumsum` computes `vx[t]` — NOT position `x[t]`!

#### Fix:
To properly integrate velocity to position waypoints:
1. Target waypoints should come from `ego_target[:, :, :2]` (position coordinates $[x, y]$).
2. Or, if predicting velocity $v = (v_x, v_y)$, position waypoints are integrated via $\hat{x}_k = \sum_{j=1}^k v_j \cdot \Delta t$.

---

### Issue 3: Inconsistent Kinematic Representation Mismatch

#### Paper Specification (Section 4.2):
The model outputs velocity $\tau_0^v = \{(v_x^l, v_y^l)\}_{l=1}^L$, and position waypoints $\tau_0^x = \{(x^l, y^l)\}_{l=1}^L$ are obtained by lower-triangular integration matrix $M$: $\tau_0^x = M \tau_0^v \cdot \Delta t$.

#### Codebase Implementation (`train.py` vs `preparedataset.py`):

In `train.py` (lines 162-166):
```python
target_actions = target_full[:, :, 2:6]  # [vx, vy, theta, yaw_rate]
if config.kinematic_type == "diff":
    model_actions = waypoint_to_diff(target_actions)
```

And in `preparedataset.py`:
- `ego_target` stores 6 features: `[x, y, vx, vy, theta, yaw_rate]`.
- `target_full[:, :, 2:6]` selects indices 2, 3, 4, 5 — which are `[vx, vy, theta, yaw_rate]`.
- `waypoint_to_diff` receives `[vx, vy, theta, yaw_rate]` and computes:
  ```python
  xy = actions[..., :2]  # [vx, vy]
  prev_xy = torch.cat([origin, xy[..., :-1, :]], dim=-2)
  return torch.cat([xy - prev_xy, actions[..., 2:4]], dim=-1)
  ```
  This computes velocity differences $\Delta v_x = v_x(t) - v_x(t-1)$ (i.e. acceleration), NOT velocity $v_x$!
- Then in `train.py` line 196:
  ```python
  target_wpt = torch.cumsum(model_actions[..., :2], dim=-2)
  ```
  `cumsum(\Delta v)` reconstructs `v` (velocity), NOT position $x$!

#### Impact:
The code confuses:
- Positions $x, y$
- Velocities $v_x, v_y$
- Velocity step-differences $\Delta v_x, \Delta v_y$

As a result, `loss_wpt` compares reconstructed velocities against target velocities, completely failing to perform the paper's spatial waypoint integration loss!

---

### Issue 4: Stop-Gradient Detach Window Implementation Flaw

#### Paper Specification (Section 4.2 & Appendix D.3):
The paper defines the detached integration operator to balance temporal gradient flow:
$$\hat{\tau}_\theta^x = M_W \tau_\theta^v \Delta t + \text{sg}((M - M_W)\tau_\theta^v \Delta t)$$
where $M_W$ truncates integration to a sliding window of $W$ steps (default $W=3$).

#### Codebase Implementation (`utils/__init__.py`, lines 9-24):
```python
def detached_integral(u: torch.Tensor, detach_window_size: int = 1) -> torch.Tensor:
    cum_detach = torch.cumsum(u.detach(), dim=-2)
    cum_normal = torch.cumsum(u, dim=-2)

    shifted = torch.roll(cum_normal, shifts=detach_window_size, dims=-2)
    shifted[..., :detach_window_size, :] = 0
    sum_recent = cum_normal - shifted

    cum_detach_shifted = torch.roll(cum_detach, shifts=detach_window_size, dims=-2)
    cum_detach_shifted[..., :detach_window_size, :] = 0

    cumulative_sum = cum_detach_shifted + sum_recent
    return cumulative_sum
```

#### Bug Analysis:
In `train.py` (line 195):
`pred_wpt = detached_integral(pred_x0[..., :2], detach_window_size=3)`

While the `detached_integral` implementation in `utils/__init__.py` is mathematically equivalent to $M_W u + \text{sg}((M - M_W)u)$ for positive $W$, default values in `utils.hybrid_loss` default $W=3$ and $\omega=0.1$. However:
1. `train.py` does NOT call `utils.hybrid_loss`! Instead, it reimplements a buggy version inline (lines 193-198).
2. In `train.py`, `target_wpt` is computed via plain `torch.cumsum` without multiplying by time interval $\Delta t$ ($DT\_SECONDS = 10.0s$).
   - Positions in CommonOcean are in meters, velocity in m/s, timestep $\Delta t = 10s$.
   - Integrating velocity $\sum v \cdot \Delta t$ requires multiplying by $\Delta t = 10.0$.
   - `train.py` omits $\Delta t$, resulting in integrated positions being off by a factor of 10!

---

### Issue 5: RL Post-Training Discrepancies

#### Paper Specification (Section 5, Eq. 9 & Appendix D.4, D.5):
- Policy optimization uses reward-weighted diffusion loss:
  $$\mathcal{L}_{RL} = \mathbb{E}_{t, \epsilon, (s, \tau_0) \sim \mathcal{D}} \left[ \exp\left(\beta \cdot \frac{r - \bar{r}}{\sigma_r}\right) \|\tau_\theta^v - \tau_0^v\|_P^2 \right]$$
- Group normalization per candidate batch ($G=32$, $\beta=1.0$).
- Multi-reward weighting: $\lambda_{risk} r_{risk} + \lambda_{follow} r_{follow} + \lambda_{lane} r_{lane}$.
- EMA policy updates ($\alpha_{EMA} = 0.05$).

#### Codebase Implementation:
- `utils/__init__.py` defines `reward_weighted_diffusion_loss` (lines 36-70).
- **HOWEVER**, there is **no RL training script** in the repository!
  - `train.py` only implements imitation pre-training.
  - No pseudo-closed-loop simulator, reward evaluation functions ($r_{safety}, r_{risk}, r_{follow}, r_{lane}$), or candidate sampling pipeline.

---

## Summary Table of Issues

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| 1 | Model trained as `noise` ($\epsilon$-pred) instead of `x_start` ($\tau_0$-pred) | Lowers open-loop score (75 → 51), produces jittery paths | ❌ High Priority Bug |
| 2 | Incorrect $\tau_0$-reconstruction formula in hybrid loss | Adds noise/instability during diffusion training | ❌ High Priority Bug |
| 3 | Misaligned feature indices in `train.py` (`target_actions`) | Integrates velocities instead of positions | ❌ High Priority Bug |
| 4 | Omission of $\Delta t = 10s$ scaling in waypoint integration | Waypoints scale wrong by factor of 10 | ❌ Medium Priority Bug |
| 5 | Missing RL training pipeline ($r_{risk}, r_{follow}, r_{lane}$) | Cannot perform HDP-RL post-training | ⚠️ Missing Feature |

---

## Recommended Action Plan

1. **Fix `train.py` Data Selection & Loss:**
   - Update config: `model_type="x_start"`, `kinematic_type="diff"`.
   - Correctly extract velocity targets $v = (v_x, v_y)$ and position targets $(x, y)$ from `ego_target`.
   - Apply $\Delta t = 10.0$ in waypoint integration: $\hat{x} = \text{detached\_integral}(\hat{v}, W) \cdot \Delta t$.

2. **Fix `preparedataset.py` Velocity Extraction:**
   - Convert body-frame surge/sway velocities to world frame before applying egocentric rotation (as detailed in previous dataset report).

3. **Implement RL Post-Training Pipeline:**
   - Implement reward components ($r_{safety}, r_{risk}, r_{follow}, r_{lane}$).
   - Add RL fine-tuning script utilizing `reward_weighted_diffusion_loss`.
