# External Force Estimation & Vessel Drift Physics Modeling Guide

This document presents a comprehensive mathematical formulation and implementation plan for estimating **External Environmental Forces (Wind, Wave, and Current Drift)** directly from AIS trajectory kinematics.

---

## 1. Executive Summary & Problem Rationale

### The Drifting Dilemma in Autonomous Vessel Planning
In real maritime operations, vessels are subject to strong environmental disturbances:
- **Aerodynamic Wind Forces** ($F_{\text{wind}}$) acting on the vessel's superstructure.
- **Second-Order Mean Wave Drift Forces** ($F_{\text{wave}}$) pushing the hull laterally.
- **Ocean Current Velocity** ($V_{\text{current}}$) causing leeway drift.

When a ship is drifting due to strong crosswinds or wave forces, its **Heading ($\psi$)** points in one direction while its **Course Over Ground ($\text{COG}$)** moves in a different direction.

If a machine learning model is trained purely on position/heading data **without knowing the external forces**, it suffers from a fatal flaw:
> **The model mistakes environmental drift for intentional steering behavior.**

As a result, during inference in calm water, the model might execute random sideways drifting maneuvers because it "learned" that ships sometimes move sideways at an angle.

### The Solution: Physics-Informed Force Residual Conditioning
By estimating the **External Force Vector** $\mathbf{F}_{\text{ext}} = [F_{x,\text{ext}}, F_{y,\text{ext}}, N_{\text{ext}}, \beta_{\text{drift}}]$ directly from AIS kinematic differentials:
1. We condition the Diffusion Planner on $\mathbf{F}_{\text{ext}}$.
2. Under calm conditions ($\mathbf{F}_{\text{ext}} \approx 0$), the model learns pure, unperturbed thrust and rudder control.
3. Under heavy weather ($\mathbf{F}_{\text{ext}} \neq 0$), the model understands that lateral motion is caused by external forces and learns appropriate counter-steering / crab-angle compensation.

---

## 2. Mathematical Kinematics & Drift Angle Definition

From standard AIS broadcasts, we receive:
- **Latitude & Longitude** ($\phi, \lambda$)
- **Speed Over Ground** ($\text{SOG}$, in m/s)
- **Course Over Ground** ($\text{COG}$, in radians CCW from East)
- **Heading** ($\psi$, in radians CCW from East)
- **Yaw Rate** ($r = \dot{\psi}$, in rad/s)
- **Vessel Dimensions**: Length $L$, Width $B$, Draft $d$ (meters)

```
                       North
                         ^
                         |     / SOG Vector (COG direction)
                         |    /
                         |   /  Drift Angle β = COG - ψ
                         |  / 
                         | /_____ Heading Vector (ψ)
                         |/      /
                         +------/-------> East
                               / (Ship Hull Axis)
```

### 2.1 Drift Angle (Crab Angle / Leeway Angle $\beta$)
The drift angle $\beta$ is defined as the angle between the vessel's longitudinal centerline ($\psi$) and its actual velocity vector ($\text{COG}$):
$$\beta = \text{wrap\_to\_pi}(\text{COG} - \psi)$$

### 2.2 Surge & Sway Velocity Decomposition
- **Surge Velocity ($u$, longitudinal velocity along hull)**:
  $$u = \text{SOG} \cdot \cos(\beta)$$
- **Sway Velocity ($v$, transverse velocity perpendicular to hull)**:
  $$v = \text{SOG} \cdot \sin(\beta)$$

- **Surge Acceleration ($\dot{u}$)**:
  $$\dot{u} = \frac{\Delta u}{\Delta t}$$
- **Sway Acceleration ($\dot{v}$)**:
  $$\dot{v} = \frac{\Delta v}{\Delta t}$$

---

## 3. MMG 3-DOF Hydrodynamic Inverse Dynamics Model

We model vessel horizontal motion using the Maneuvering Modeling Group (MMG) 3-DOF (Surge, Sway, Yaw) non-linear equations of motion.

### 3.1 Equations of Motion
$$\begin{aligned}
(m + m_x) \dot{u} - (m + m_y) v r &= F_{x,\text{hull}} + F_{x,\text{prop}} + F_{x,\text{ext}} \\
(m + m_y) \dot{v} + (m + m_x) u r &= F_{y,\text{hull}} + F_{y,\text{ext}} \\
(I_z + J_z) \dot{r} &= N_{\text{hull}} + N_{\text{ext}}
\end{aligned}$$

where:
- $m$: Vessel mass (displacement tonnage).
- $m_x, m_y$: Added mass in surge and sway due to entrained water.
- $I_z, J_z$: Yaw moment of inertia and added moment of inertia.
- $F_{x,\text{hull}}, F_{y,\text{hull}}, N_{\text{hull}}$: Hydrodynamic hull damping & resistance forces.
- $F_{x,\text{prop}}$: Main engine propeller thrust.
- $F_{x,\text{ext}}, F_{y,\text{ext}}, N_{\text{ext}}$: **External environmental forces and yaw moment**.

---

### 3.2 Mass & Added Mass Parametric Approximations

Using standard empirical formulas for surface commercial hulls (Norrbin / Clarke approximations):

- **Vessel Mass ($m$)**:
  $$m = \rho_w \cdot C_b \cdot L \cdot B \cdot d$$
  where $\rho_w = 1025\text{ kg/m}^3$ (seawater density) and $C_b \approx 0.70$ (block coefficient).

- **Surge Added Mass ($m_x$)**:
  $$m_x = 0.05 \cdot m$$

- **Sway Added Mass ($m_y$)**:
  $$m_y = \rho_w \cdot \frac{\pi}{2} \cdot d^2 \cdot L \cdot \left[1 + 0.4 \frac{B}{L}\right]$$

- **Yaw Moment of Inertia ($I_z$)**:
  $$I_z = m \cdot \left(0.25 L\right)^2$$

- **Yaw Added Moment of Inertia ($J_z$)**:
  $$J_z = 0.025 \cdot m \cdot L^2$$

---

### 3.3 Baseline Hydrodynamic Hull Forces ($F_{\text{hull}}$)

In calm water, hull resistance and sway damping are modeled via cross-flow drag and Taylor expansion damping coefficients:

#### A. Longitudinal Resistance ($F_{x,\text{hull}}$):
$$F_{x,\text{hull}} = -\frac{1}{2} \rho_w C_d A_w u |u|$$
where $A_w = B \cdot d$ is underwater frontal area, and $C_d \approx 0.03$ is friction coefficient.

#### B. Lateral Cross-Flow Sway Resistance ($F_{y,\text{hull}}$):
$$F_{y,\text{hull}} = -\frac{1}{2} \rho_w C_y (L \cdot d) v |v|$$
where $C_y \approx 0.6 - 1.0$ is the cross-flow drag coefficient.

#### C. Yaw Damping Moment ($N_{\text{hull}}$):
$$N_{\text{hull}} = -\frac{1}{16} \rho_w C_y (L^2 \cdot d) r |r|$$

---

### 3.4 Inverse Force Residue Equation (Extracting $F_{\text{ext}}$)

By rearranging the MMG equations, we directly compute the **unexplained external force residual** $\mathbf{F}_{\text{ext}}$ from measured AIS kinematics:

$$\begin{aligned}
F_{x,\text{ext}} &= (m + m_x) \dot{u} - (m + m_y) v r - F_{x,\text{hull}} - F_{x,\text{prop}} \\
F_{y,\text{ext}} &= (m + m_y) \dot{v} + (m + m_x) u r - F_{y,\text{hull}} \\
N_{\text{ext}} &= (I_z + J_z) \dot{r} - N_{\text{hull}}
\end{aligned}$$

*Note: For equilibrium/steady-state transit ($\dot{u} \approx 0, \dot{v} \approx 0$), the external sway force simplifies directly to the sway Coriolis plus hull drag balance:*
$$F_{y,\text{ext}} \approx (m + m_x) u r + \frac{1}{2} \rho_w C_y (L \cdot d) v |v|$$

---

## 4. Empirical Aerodynamic Wind & Wave Force Decomposition

To separate wind and wave contributions if environmental sensors are present or estimated:

### 4.1 Aerodynamic Wind Forces (Isherwood Formulation)
$$\begin{aligned}
F_{x,\text{wind}} &= \frac{1}{2} \rho_a V_{rw}^2 A_T C_{Xw}(\gamma_w) \\
F_{y,\text{wind}} &= \frac{1}{2} \rho_a V_{rw}^2 A_L C_{Yw}(\gamma_w) \\
N_{\text{wind}} &= \frac{1}{2} \rho_a V_{rw}^2 A_L L C_{Nw}(\gamma_w)
\end{aligned}$$
where:
- $\rho_a = 1.225\text{ kg/m}^3$ (air density).
- $V_{rw}$: Relative wind speed.
- $\gamma_w$: Relative wind direction ($\gamma_w = \theta_w - \psi$).
- $A_T \approx 0.8 \cdot B^2$: Frontal windage area above waterline.
- $A_L \approx L \cdot d_{\text{freeboard}}$: Lateral windage area.
- $C_{Xw}, C_{Yw}, C_{Nw}$: Empirical aerodynamic coefficients based on vessel profile.

### 4.2 Mean Wave Drift Force (2nd Order Wave Action)
$$F_{y,\text{wave}} = \frac{1}{2} \rho_w g H_s^2 L C_{Dwy}(\chi)$$
where $H_s$ is significant wave height, $g = 9.81\text{ m/s}^2$, and $C_{Dwy}(\chi)$ is the directional wave drift coefficient.

---

## 5. Codebase Integration Plan

### Step 5.1: Extend `preparedataset.py` Feature Vector
Currently, `_extract_state_features()` extracts 6 features:
`[x, y, vx_world, vy_world, theta, yaw_rate]`

We extend it to **10 features** by computing and appending external forces:
`[x, y, vx_world, vy_world, theta, yaw_rate, F_x_ext, F_y_ext, N_ext, beta_drift]`

#### Implementation Code Snippet (`preparedataset.py`):
```python
def compute_external_forces(u, v, r, du_dt, dv_dt, dr_dt, heading, cog, length, width, draft):
    """
    Computes external hydrodynamic force residuals (Fx_ext, Fy_ext, N_ext, beta_drift).
    """
    rho_w = 1025.0  # Seawater density kg/m^3
    Cb = 0.70       # Block coefficient
    
    # 1. Mass & Added Mass
    m = rho_w * Cb * length * width * draft
    mx = 0.05 * m
    my = rho_w * (np.pi / 2.0) * (draft ** 2) * length * (1.0 + 0.4 * (width / length))
    Iz = m * (0.25 * length) ** 2
    Jz = 0.025 * m * (length ** 2)

    # 2. Hull Damping Forces
    C_dx = 0.03
    C_dy = 0.80
    Ax = width * draft
    Ay = length * draft
    
    Fx_hull = -0.5 * rho_w * C_dx * Ax * u * abs(u)
    Fy_hull = -0.5 * rho_w * C_dy * Ay * v * abs(v)
    N_hull = -(1.0 / 16.0) * rho_w * C_dy * (length ** 2) * draft * r * abs(r)

    # 3. Inverse Dynamics Residuals
    Fx_ext = (m + mx) * du_dt - (m + my) * v * r - Fx_hull
    Fy_ext = (m + my) * dv_dt + (m + mx) * u * r - Fy_hull
    N_ext = (Iz + Jz) * dr_dt - N_hull

    # 4. Drift Angle
    beta_drift = (cog - heading + np.pi) % (2.0 * np.pi) - np.pi

    # Normalize forces by mass for numerical stability (force per unit mass m/s^2)
    return Fx_ext / m, Fy_ext / m, N_ext / (m * length), beta_drift
```

---

### Step 5.2: Proprioception Conditioning in `train.py` & `modeling_dp_vla.py`
In `train.py`, pass the latest external force status $\mathbf{F}_{\text{ext}}$ into the proprioception vector $y$:

```python
# Construct proprioception vector (6 kinematic states + 4 force/drift states)
proprio = torch.cat([
    ego_hist[:, -1, :],  # latest ego state including [Fx_ext, Fy_ext, N_ext, beta]
    torch.zeros((ego_hist.shape[0], config.dim_y - 10), device=device)
], dim=-1)
```

---

## 6. Summary of Benefits

1. **Eliminates False Steering Inferences**: The model stops learning "phantom drifting turns" because lateral movement is explicitly attributed to external force conditioning.
2. **Physics-Consistent Control**: In calm water ($F_{\text{ext}} \approx 0$), the planner outputs clean hydrodynamic vessel trajectories.
3. **Adaptive Weather Compensation**: In heavy weather ($F_{\text{ext}} \neq 0$), the model learns to maintain proper crab angle and counter-steering to keep the desired channel path.
