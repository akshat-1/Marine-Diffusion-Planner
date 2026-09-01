# Second-Order Mean Wave Drift Force: Physical Mechanics & Model Conditioning

This document provides a detailed explanation of **Second-Order Mean Wave Drift Forces ($F_{y,\text{wave}}$)**, how they are calculated, and how they are used within the Hyper Diffusion Planner (HDP) pipeline.

---

## 1. Physical Mechanics: First-Order vs. Second-Order Wave Forces

When ocean waves interact with a ship hull, the hydrodynamic forces are split into two orders of magnitude:

```
                    Ocean Wave Action on Vessel Hull
                    /                              \
                   /                                \
  First-Order Oscillatory Forces        Second-Order Mean Wave Drift Forces
  ------------------------------        ------------------------------------
  • Oscillates at wave frequency        • Non-linear wave reflection/diffraction
  • Period: 5 - 10 seconds              • Steady, non-zero mean lateral push
  • Causes roll, pitch, heave           • Causes steady sideways leeway drift
  • Averages to ZERO over 200s          • DOES NOT AVERAGE TO ZERO!
```

- **First-Order Forces**: High-frequency wave oscillations cause instantaneous roll, pitch, and heave. Over a 200-second trajectory planning horizon ($T_{\text{obs}} = 200\text{s}$), first-order forces average to zero net displacement.
- **Second-Order Mean Wave Drift Forces ($F_{\text{wave}}$)**: Caused by wave reflections and non-linear pressure dynamics. $F_{\text{wave}}$ exerts a **steady, non-zero lateral drift force** that constantly pushes the ship off its course.

---

## 2. Mathematical Formulation of Second-Order Wave Drift Force

The second-order mean wave drift force acting laterally ($y$-axis perpendicular to ship heading) is computed via Maruo's wave drift formulation:

$$F_{y,\text{wave}} = \frac{1}{2} \rho_w g H_s^2 L \cdot C_{Dwy}(\chi, \omega_p)$$

where:
- $\rho_w = 1025\text{ kg/m}^3$: Seawater density.
- $g = 9.81\text{ m/s}^2$: Acceleration due to gravity.
- $H_s$: Significant wave height (meters).
- $L$: Vessel length (meters).
- $\chi$: Relative wave heading angle ($\chi = \theta_{\text{wave}} - \psi$).
- $\omega_p$: Peak wave frequency ($\text{rad/s}$).
- $C_{Dwy}(\chi, \omega_p) \in [0.05, 0.40]$: Non-dimensional lateral wave drift force coefficient based on hull geometry and wave angle.

---

## 3. How We Use It in the HDP Pipeline

### Step 1: Total Hydrodynamic Force Residual Decomposition (`preparedataset.py`)
In `preparedataset.py`, the inverse MMG dynamics residue extracts the total external lateral force $F_{y,\text{ext}}$ acting on the vessel:

$$F_{y,\text{ext}} = (m + m_y) \dot{v} + (m + m_x) u r - F_{y,\text{hull}}$$

This total force $F_{y,\text{ext}}$ is the combined sum of aerodynamic wind drag and **second-order mean wave drift**:
$$F_{y,\text{ext}} = F_{y,\text{wind}} + F_{y,\text{wave}}$$

---

### Step 2: Mass-Normalization to Specific Acceleration ($a_{y,\text{ext}}$)
To ensure the neural network converges quickly across different vessel sizes (from 20m tugboats to 300m container ships), we divide by vessel mass $m$:

$$a_{y,\text{ext}} = \frac{F_{y,\text{ext}}}{m} = \frac{F_{y,\text{wind}} + F_{y,\text{wave}}}{m} \quad (\text{units of }\text{m/s}^2)$$

This specific lateral acceleration $a_{y,\text{ext}}$ is placed directly into channel 7 of the **10D State Feature Vector**:

$$\mathbf{x}_{\text{state}} = [x, y, v_x, v_y, \theta, r, a_{x,\text{ext}}, \mathbf{a_{y,\text{ext}}}, \alpha_{\text{ext}}, \beta_{\text{drift}}]$$

---

### Step 3: DiT Model Conditioning & Inference Strategy (`train.py`)

1. **Scene Encoder (`modeling_dp_vla.py`)**:
   - The 10D state vector containing $a_{y,\text{ext}}$ (which includes wave drift $F_{y,\text{wave}}/m$) is processed by `HighCapacityVectorSceneEncoder`.

2. **Proprioception Conditioning Vector ($y \in \mathbb{R}^{B \times 12}$)**:
   - The latest state's lateral wave acceleration $a_{y,\text{ext}}$ conditions the `CustomDiT` decoder via `adaLN-Zero` modulation blocks.

3. **Behavior Learned by the Model**:

```
                                  DiT Trajectory Output
                                 /                     \
                                /                       \
      Calm Water (Fy_wave ≈ 0)                     Heavy Waves (Fy_wave >> 0)
      ------------------------                     --------------------------
      • No lateral wave push                       • Lateral wave push detected (ay_ext > 0)
      • Predicts straight, unperturbed             • Predicts CRAB-ANGLE COMPENSATION
        hydrodynamic path                            & COUNTER-STEERING into the waves
      • Heading aligns with COG                    • Heading angles into waves to keep channel
```

---

## 4. Summary of Benefits

1. **Explains Sideways Drift**: The model understands that sideways movement under heavy seas is caused by wave force $F_{y,\text{wave}}$, preventing it from learning false steering behaviors.
2. **Adaptive Counter-Steering**: Allows the planner to generate crab-angle compensation when wave drift is present, maintaining the intended channel centerline.
3. **Scale Invariance**: Mass-normalization converts Newtons of wave force into $\text{m/s}^2$ specific acceleration, ensuring stable gradient descent during training.
