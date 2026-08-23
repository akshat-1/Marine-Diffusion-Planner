# HDP Paper Implementation Summary

This document summarizes the changes made to align the codebase with the **Hyper Diffusion Planner (HDP)** paper (arXiv:2602.22801v2).

## Key HDP Paper Concepts Implemented

### 1. VP (Variance-Preserving) Noise Schedule
**Paper Reference:** Appendix D.4 - "We adopt the variance-preserving(VP) noise schedule following Zheng et al. (2025)"

**Implementation:** `utils/diffusion.py` - `_vp_beta_schedule()` method
- Continuous VP schedule: β(t) = β_min + t × (β_max - β_min)
- α_t = exp(-½ ∫₀ᵗ β(s) ds) = exp(-(β_min×t + (β_max-β_min)×t²/2)/2)
- β_min = 0.0001, β_max = 0.02 (matching paper's Table 6)

### 2. τ₀-Prediction (Direct Velocity Prediction)
**Paper Reference:** Section 4.1 - "τ₀-prediction model with τ₀-loss yields both fast convergence and high-quality generation"

**Implementation:** 
- `utils/diffusion.py`: `q_sample()` uses x_t = α_t × x_0 + σ_t × ε
- `model/dit.py`: DiT outputs clean velocity trajectory x_0 directly (not noise ε)
- `train.py`: Training loss compares predicted x_0 with ground truth velocity

### 3. Velocity Representation with Hybrid Loss
**Paper Reference:** Section 4.2 - "Hybrid Loss" + Appendix D.3 Algorithm 1

**Implementation:** `utils/diffusion.py` - `hybrid_loss()` and `detached_integral()`
- Velocity loss: ||v_θ - v_0||²
- Waypoint loss with detached integral (window W=3):
  ```
  ŵ = M_W v_θ Δt + sg((M - M_W) v_θ Δt)
  L_waypoints = ||ŵ - w_0||²
  ```
- Total loss: L_hybrid = L_velocity + ω × L_waypoints (ω = 0.1 from Table 6)

### 4. DPM-Solver for Fast Sampling (6 Steps)
**Paper Reference:** Appendix D.4 - "employs the DPM-Solver (Lu et al., 2022) to accelerate the sampling process, achieving a final inference speed that easily meets the 10Hz requirement"

**Implementation:** `utils/diffusion.py` - `dpm_solver_sample()` method
- 2nd order DPM-Solver with 6 sampling steps
- Uses log-SNR (λ) space for adaptive step sizes
- τ₀-prediction compatible (model outputs x_0 directly)

### 5. Model Architecture Updates
**Paper Reference:** Table 6 - "Num. block: 6, Dim. hidden layer: 256, Num. multi-head: 8"

**Implementation:** `model/dit.py`
- 6 transformer decoder layers (was 4)
- 8 attention heads (unchanged)
- 256 embedding dim (unchanged)
- Output: 4D velocity [vx, vy, theta, yaw_rate] (was 6D including position)

### 6. Training Hyperparameters (Table 6)
**Implementation:** `train.py`
- Learning rate: 5×10⁻⁴ (was 1×10⁻⁴)
- Weight decay: 0.01 (was 1×10⁻⁴)
- Hybrid loss weight ω: 0.1
- Detach window W: 3
- DT: 10.0s (AIS pipeline)

## Files Modified

| File | Changes |
|------|---------|
| `utils/diffusion.py` | Complete rewrite: VP schedule, τ₀-prediction, DPM-Solver, hybrid loss |
| `model/dit.py` | τ₀-prediction output, 6 layers, 4D velocity output |
| `train.py` | Updated to use VP schedule, τ₀-prediction, hybrid loss, DPM-Solver |
| `utils/__init__.py` | Export new functions |
| `model/__init__.py` | Fixed duplicate import |

## Verification

All components tested and verified:
- ✅ Forward diffusion with VP schedule
- ✅ τ₀-prediction model forward pass
- ✅ Hybrid loss with detached integral (W=3)
- ✅ Backward pass through hybrid loss
- ✅ DPM-Solver sampling (6 steps)
- ✅ Full training pipeline integration
- ✅ Data loader compatibility

## Expected Benefits

1. **Speed:** 6-step DPM-Solver vs 1000-step DDPM → ~166× faster sampling
2. **Quality:** τ₀-prediction → smoother, kinematically consistent trajectories
3. **Stability:** Hybrid loss with detach → balanced gradients across time steps
4. **Real-world:** 10Hz inference capability for real-vehicle deployment