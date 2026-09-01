# Theoretical Analysis & Choice: External Force vs. External Velocity for Diffusion Trajectory Planning

This report provides a rigorous theoretical comparison between using **Total External Forces ($\mathbf{F}_{\text{ext}}$)** vs. **External Drift Velocity ($\mathbf{V}_{\text{ext}}$)** for conditioning the Hyper Diffusion Planner (HDP), and details our final design choice.

---

## 1. Executive Summary & Design Choice

### The Architectural Question
When modeling environmental disturbances (wind, wave drift, leeway currents), should the trajectory diffusion planner be conditioned on:
1. **Raw External Forces & Yaw Moments** ($\mathbf{F}_{\text{ext}} = [F_{x,\text{ext}}, F_{y,\text{ext}}, N_{\text{ext}}]$, in Newtons and N$\cdot$m), OR
2. **External Drift Velocity & Crab Angle** ($\mathbf{V}_{\text{ext}} = [v_{x,\text{drift}}, v_{y,\text{drift}}, \beta_{\text{drift}}]$, in m/s and radians), OR
3. **Mass-Normalized Specific Force-Kinematic Residual Vector** ($\mathbf{f}_{\text{ext}} = [a_{x,\text{ext}}, a_{y,\text{ext}}, \alpha_{\text{ext}}, \beta_{\text{drift}}]$)?

---

### Our Final Choice: **Mass-Normalized Specific Force-Kinematic Vector ($\mathbf{f}_{\mathrm{ext}}$)**

$$\mathbf{f}_{\mathrm{ext}} = \left[ \underbrace{\frac{F_{x,\mathrm{ext}}}{m}}_{a_{x,\mathrm{ext}} \ (\text{m/s}^2)}, \; \underbrace{\frac{F_{y,\mathrm{ext}}}{m}}_{a_{y,\mathrm{ext}} \ (\text{m/s}^2)}, \; \underbrace{\frac{N_{\mathrm{ext}}}{m \cdot L}}_{\alpha_{\mathrm{ext}} \ (\text{rad/s}^2)}, \; \underbrace{\beta_{\mathrm{drift}}}_{\text{Drift Angle (rad)}} \right]$$

### Why This Choice Wins on Both Convergence & Physics:
- **Fastest Neural Network Convergence**: By dividing forces by vessel mass $m$, we eliminate vessel scale disparities ($10^3\text{ N}$ vs $10^6\text{ N}$ across vessel sizes). All feature channels fall cleanly into unit scale $\sim [-1.0, 1.0]$.
- **Zero Loss of Acceleration Information**: Unlike pure velocity drift (which discards $\dot{u}, \dot{v}, \dot{r}$), specific force residuals ($a_{x,\text{ext}}, a_{y,\text{ext}}, \alpha_{\text{ext}}$) preserve full second-order acceleration dynamics caused by wind gusts and wave impacts.
- **Dual Cause-and-Effect Representation**: The model receives both the **cause** (external acceleration $\mathbf{a}_{\text{ext}}$) and the **effect** (kinematic crab angle $\beta_{\text{drift}}$).

---

## 2. In-Depth Comparative Analysis

| Criteria | Option A: Raw Forces ($\mathbf{F}_{\text{ext}}$) | Option B: Pure Drift Velocity ($\mathbf{V}_{\text{ext}}$) | Option C (OUR CHOICE): Specific Force Residual ($\mathbf{f}_{\text{ext}}$) |
| :--- | :--- | :--- | :--- |
| **Physical Quantity** | Absolute Forces ($F_x, F_y, N$) | Kinematic Velocity Drift ($v_{x,\text{drift}}, v_{y,\text{drift}}$) | Mass-Normalized Acceleration ($a_{x,\text{ext}}, a_{y,\text{ext}}$) + Drift Angle ($\beta$) |
| **SI Units** | Newtons ($\text{N}$), $\text{N}\cdot\text{m}$ | $\text{m/s}$, radians | $\text{m/s}^2$, $\text{rad/s}^2$, radians |
| **Acceleration Information** | **Preserved** ($\dot{u}, \dot{v}, \dot{r}$) | ❌ **Lost** (1st-order velocity only) | **Preserved** ($\dot{u}, \dot{v}, \dot{r}$) |
| **Mass Independence** | ❌ **No** (Vessel length/mass dependent) | **Yes** (Pure kinematics) | **Yes** (Divided by mass $m$) |
| **Numerical Range** | Extreme variance ($10^3$ to $10^6\text{ N}$) | Moderate ($[-3, 3]\text{ m/s}$) | **Bounded** ($[-1, 1]\text{ m/s}^2$) |
| **NN Convergence Speed** | 🐌 **Slow** (Vessel size imbalance) | ⚡ **Fast** | ⚡⚡ **Fastest & Most Stable** |
| **Noise Sensitivity** | High (Derivative noise $\cdot$ large mass) | Low | Low (Normalized derivative noise) |

---

## 3. Mathematical Proof & Detailed Analysis

### 3.1 Why Raw Force ($\mathbf{F}_{\text{ext}}$) Causes Slow Neural Network Convergence

Consider two vessels operating in the same sea state with identical wind/wave acceleration ($a_{\text{env}} = 0.2\text{ m/s}^2$):
1. **Small Tugboat**: Length $L = 20\text{m}$, Mass $m_1 = 200\text{ tonnes} = 2 \times 10^5\text{ kg}$.
   $$F_{y,\text{ext},1} = m_1 \cdot a_{\text{env}} = 40,000\text{ N}$$
2. **Container Ship**: Length $L = 300\text{m}$, Mass $m_2 = 100,000\text{ tonnes} = 10^8\text{ kg}$.
   $$F_{y,\text{ext},2} = m_2 \cdot a_{\text{env}} = 20,000,000\text{ N}$$

#### Consequence for Neural Network Gradient Descent:
If raw forces are fed into the neural network, the linear projections ($\mathbf{W} \cdot \mathbf{F}_{\text{ext}}$) experience **massive gradient scale imbalances**:
- Gradient updates for large container ships are $500\times$ larger than for small tugboats!
- This causes gradient explosion, unstable loss spikes, and **slow network convergence**.

---

### 3.2 Why Pure Velocity Drift ($\mathbf{V}_{\text{ext}}$) Loses Critical Acceleration Physics

If we condition the model *only* on drift velocity $v_{y,\text{drift}} = \text{SOG} \cdot \sin(\beta)$:

$$\mathbf{V}_{\text{ext}} = \begin{pmatrix} v_{x,\text{drift}} \\ v_{y,\text{drift}} \end{pmatrix}$$

#### The Physical Deficit:
- Drift velocity $\mathbf{V}_{\text{ext}}$ represents the **first-order kinematic result** of environmental push, but contains **zero information about transient accelerations ($\dot{u}, \dot{v}, \dot{r}$)**.
- Example: A sudden wind gust or squall exerts an immediate lateral force spike ($F_{y,\text{ext}}$) before the vessel's massive hull has time to build up lateral velocity $v_{y,\text{drift}}$.
- If conditioned only on velocity, the model cannot sense the onset of a wind gust until after the ship has already drifted off course!

---

### 3.3 The Specific Force Residual Solution ($\mathbf{f}_{\text{ext}}$)

By defining the specific force residual per unit mass:

$$\begin{aligned}
a_{x,\text{ext}} &= \frac{F_{x,\text{ext}}}{m} = (1 + \frac{m_x}{m}) \dot{u} - (1 + \frac{m_y}{m}) v r - \frac{F_{x,\text{hull}}}{m} \\
a_{y,\text{ext}} &= \frac{F_{y,\text{ext}}}{m} = (1 + \frac{m_y}{m}) \dot{v} + (1 + \frac{m_x}{m}) u r - \frac{F_{y,\text{hull}}}{m} \\
\alpha_{\text{ext}} &= \frac{N_{\text{ext}}}{m \cdot L} = \frac{(I_z + J_z) \dot{r} - N_{\text{hull}}}{m \cdot L}
\end{aligned}$$

#### Key Mathematical Properties:
1. **Scale Invariance**: $a_{x,\text{ext}}$ and $a_{y,\text{ext}}$ have units of acceleration ($\text{m/s}^2$). Both the 20m tugboat and the 300m container ship yield specific force features in the range $a \in [-1.0, 1.0]\text{ m/s}^2$.
2. **Preserves 2nd-Order Physics**: $\dot{u}, \dot{v}, \dot{r}$ are retained inside $a_{x,\text{ext}}, a_{y,\text{ext}}, \alpha_{\text{ext}}$, allowing the model to sense sudden force spikes instantly.
3. **Paired with Drift Angle $\beta_{\text{drift}}$**: Appending $\beta_{\text{drift}} = \text{wrap\_to\_pi}(\text{COG} - \psi)$ provides the first-order kinematic drift result alongside the second-order specific force cause.

---

## 4. Implementation in Codebase

This choice is fully implemented across the codebase:

### 1. Feature Extraction (`preparedataset.py`)
`compute_external_forces()` computes mass-normalized specific forces:
```python
# Normalize forces by vessel mass m (m/s^2 acceleration scale)
return Fx_ext / m, Fy_ext / m, N_ext / (m * length), beta_drift
```

### 2. Feature Vector Structure (10D State Vector)
$$\mathbf{x}_{\text{state}} = [x, y, v_{x,\text{world}}, v_{y,\text{world}}, \theta, r, \underbrace{a_{x,\text{ext}}, a_{y,\text{ext}}, \alpha_{\text{ext}}, \beta_{\text{drift}}}_{\text{Specific Force & Drift Vector}}]$$

### 3. Z-Score Normalization (`train.py` & `utils/__init__.py`)
`ZScoreNormalizer` normalizes all 10 feature channels:
```python
feature_mean = [0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
feature_std  = [100.0, 100.0, 3.0, 1.0, 1.0, 0.05, 1.0, 1.0, 0.5, 0.5]
state_normalizer = ZScoreNormalizer(feature_mean, feature_std).to(device)
```

---

## 5. Summary Matrix

| Metric | Raw Force | Pure Velocity | Specific Force Vector (OUR CHOICE) |
| :--- | :---: | :---: | :---: |
| **NN Convergence Speed** | 🔴 Slow | 🟡 Fast | 🟢 **Fastest & Most Stable** |
| **Mass Scale Independence** | 🔴 Bad | 🟢 Excellent | 🟢 **Excellent** |
| **Acceleration Transient Physics** | 🟢 Retained | 🔴 Lost | 🟢 **Retained** |
| **Kinematic Drift Representation** | 🟡 Indirect | 🟢 Direct | 🟢 **Direct (via $\beta$)** |
| **Overall Recommendation** | ❌ Rejected | ❌ Incomplete | ✅ **SELECTED** |
