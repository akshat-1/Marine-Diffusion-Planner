# Mathematical Modeling of Maritime Trajectory Diffusion Planner

## 1. Foundational Paradigm: Continuous Variance-Preserving SDE and $x_0$-Matching

In generative diffusion modeling, four related paradigms exist:
1. Discrete Denoising Diffusion Probabilistic Models (DDPM)
2. Score-Based Stochastic Differential Equations (Score SDE)
3. Flow Matching and Continuous Normalizing Flows
4. Direct Clean-State Parameterization ($x_0$-Prediction / $\tau_0$-Matching)

### 1.1 Paradigm Classification
Our system operates under a **Continuous-Time Variance-Preserving SDE (VP-SDE)** backbone parameterized via **Direct Clean-State ($x_0$) Prediction**, with sampling executed through a **Probability Flow Ordinary Differential Equation (PF-ODE)** using an analytical exponential integrator (DPM-Solver++).

Unlike standard image diffusion models that predict added Gaussian noise $\boldsymbol{\epsilon}$, trajectory generation requires enforcing physical kinematics, bound constraints, and geometric obstacle boundaries. Predicting the clean trajectory action $\hat{\mathbf{x}}_0$ at every diffusion timestep allows immediate projection through differential kinematic operators and geometric clearance fields, enabling end-to-end multi-objective supervision.

---

## 2. Forward Diffusion Process Formulation

Let $\mathbf{x}_0 \in \mathbb{R}^{H \times d_a}$ denote the ground-truth trajectory action sequence across horizon $H = 15$ with action dimension $d_a = 4$ representing $[v_x, v_y, \theta, r]$ in the ego vessel body-fixed coordinate frame.

### 2.1 Itô Stochastic Differential Equation
The forward process perturbs $\mathbf{x}_0$ continuously over diffusion time $t \in [0, 1]$ according to the linear Itô SDE:

$$\mathrm{d}\mathbf{x}_t = -\frac{1}{2} \beta(t) \mathbf{x}_t \,\mathrm{d}t + \sqrt{\beta(t)} \,\mathrm{d}\mathbf{w}_t$$

where $\beta(t) > 0$ denotes the continuous noise schedule and $\mathbf{w}_t \in \mathbb{R}^{H \times d_a}$ is standard multidimensional Brownian motion satisfying:

$$\mathbb{E}[\mathrm{d}\mathbf{w}_t] = \mathbf{0}, \quad \mathbb{E}[\mathrm{d}\mathbf{w}_t \mathrm{d}\mathbf{w}_t^\top] = \mathbf{I} \,\mathrm{d}t$$

### 2.2 Analytical Solution of the Linear Forward SDE
To solve this SDE analytically, define the integrating factor:

$$\Phi(t) = \exp\left( \frac{1}{2} \int_0^t \beta(s) \,\mathrm{d}s \right)$$

Applying Itô's lemma to the transformed variable $\mathbf{y}_t = \Phi(t) \mathbf{x}_t$:

$$\mathrm{d}\mathbf{y}_t = \mathrm{d}\Phi(t) \mathbf{x}_t + \Phi(t) \mathrm{d}\mathbf{x}_t + \mathrm{d}\Phi(t) \mathrm{d}\mathbf{x}_t$$

Since $\Phi(t)$ is deterministically differentiable:

$$\mathrm{d}\Phi(t) = \frac{1}{2} \beta(t) \Phi(t) \,\mathrm{d}t, \quad \mathrm{d}\Phi(t) \mathrm{d}\mathbf{x}_t = \mathbf{0}$$

Substituting $\mathrm{d}\mathbf{x}_t$:

$$\mathrm{d}\mathbf{y}_t = \frac{1}{2} \beta(t) \Phi(t) \mathbf{x}_t \,\mathrm{d}t + \Phi(t) \left[ -\frac{1}{2} \beta(t) \mathbf{x}_t \,\mathrm{d}t + \sqrt{\beta(t)} \,\mathrm{d}\mathbf{w}_t \right]$$

The deterministic drift terms cancel identically:

$$\mathrm{d}\mathbf{y}_t = \Phi(t) \sqrt{\beta(t)} \,\mathrm{d}\mathbf{w}_t$$

Integrating both sides from $0$ to $t$:

$$\mathbf{y}_t - \mathbf{y}_0 = \int_0^t \Phi(s) \sqrt{\beta(s)} \,\mathrm{d}\mathbf{w}_s$$

Since $\mathbf{y}_0 = \Phi(0) \mathbf{x}_0 = \mathbf{x}_0$:

$$\Phi(t) \mathbf{x}_t = \mathbf{x}_0 + \int_0^t \Phi(s) \sqrt{\beta(s)} \,\mathrm{d}\mathbf{w}_s$$

Multiplying by $\Phi(t)^{-1}$:

$$\mathbf{x}_t = \exp\left(-\frac{1}{2} \int_0^t \beta(s)\,\mathrm{d}s\right) \mathbf{x}_0 + \int_0^t \exp\left(-\frac{1}{2} \int_s^t \beta(u)\,\mathrm{d}u\right) \sqrt{\beta(s)} \,\mathrm{d}\mathbf{w}_s$$

### 2.3 Transition Kernel Distribution
Because Itô integrals of deterministic functions with respect to Brownian motion are Gaussian, $\mathbf{x}_t$ conditioned on $\mathbf{x}_0$ is normally distributed:

$$p_{t|0}(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t;\, \alpha_t \mathbf{x}_0,\, \sigma_t^2 \mathbf{I}\right)$$

The mean coefficient $\alpha_t$ and variance $\sigma_t^2$ are derived as follows:

$$\alpha_t = \mathbb{E}\left[\frac{\mathbf{x}_t}{\mathbf{x}_0}\right] = \exp\left(-\frac{1}{2} \int_0^t \beta(s)\,\mathrm{d}s\right)$$

By the Itô isometry, the variance of the stochastic integral is:

$$\sigma_t^2 = \mathrm{Var}(\mathbf{x}_t \mid \mathbf{x}_0) = \int_0^t \exp\left(-\int_s^t \beta(u)\,\mathrm{d}u\right) \beta(s) \,\mathrm{d}s$$

Let $v(s) = \int_s^t \beta(u)\,\mathrm{d}u$. Then $\mathrm{d}v(s) = -\beta(s)\,\mathrm{d}s$. The integral becomes:

$$\sigma_t^2 = \int_{v(0)}^{v(t)} \exp(-v) (-\mathrm{d}v) = \int_0^{v(0)} \exp(-v) \,\mathrm{d}v = 1 - \exp(-v(0))$$

Noticing that $\exp(-v(0)) = \exp\left(-\int_0^t \beta(u)\,\mathrm{d}u\right) = \alpha_t^2$, we obtain:

$$\sigma_t^2 = 1 - \alpha_t^2$$

Thus, the total variance $\alpha_t^2 + \sigma_t^2 = 1$ is strictly preserved for all $t \in [0, 1]$, validating the Variance-Preserving designation.

Given $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, the reparameterized sample is:

$$\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \mathbf{z}$$

### 2.4 Cosine Noise Schedule
In the codebase, $\alpha_t^2 = \bar{\alpha}(t)$ follows the Nichol-Dhariwal cosine schedule:

$$f(t) = \cos^2\left(\frac{t + s}{1 + s} \cdot \frac{\pi}{2}\right), \quad s = 0.008$$

$$\bar{\alpha}(t) = \frac{f(t)}{f(0)}, \quad \alpha_t = \sqrt{\bar{\alpha}(t)}, \quad \sigma_t = \sqrt{1 - \bar{\alpha}(t)}$$

---

## 3. Reverse Diffusion and Probability Flow ODE

### 3.1 Reverse-Time SDE
By Anderson's reverse-time theorem (1982), any forward diffusion SDE possesses an equivalent reverse-time SDE moving backward from $t = 1$ to $t = 0$:

$$\mathrm{d}\mathbf{x}_t = \left[ -\frac{1}{2}\beta(t)\mathbf{x}_t - \beta(t) \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \right] \mathrm{d}t + \sqrt{\beta(t)} \,\mathrm{d}\bar{\mathbf{w}}_t$$

where $\bar{\mathbf{w}}_t$ is standard Brownian motion flowing backward in time, and $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$ is the Stein score of the perturbed marginal distribution.

### 3.2 Probability Flow ODE Derivation
Song et al. (2020) demonstrated that for any linear SDE of the form:

$$\mathrm{d}\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t)\,\mathrm{d}t + g(t)\,\mathrm{d}\mathbf{w}_t$$

there exists a deterministic Ordinary Differential Equation whose trajectory distributions $p_t(\mathbf{x}_t)$ match those of the SDE for all $t$:

$$\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \mathbf{f}(\mathbf{x}_t, t) - \frac{1}{2} g(t)^2 \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$$

Substituting our drift $\mathbf{f}(\mathbf{x}_t, t) = -\frac{1}{2}\beta(t)\mathbf{x}_t$ and diffusion $g(t) = \sqrt{\beta(t)}$:

$$\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = -\frac{1}{2}\beta(t)\mathbf{x}_t - \frac{1}{2}\beta(t) \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = -\frac{1}{2}\beta(t)\left[ \mathbf{x}_t + \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \right]$$

---

## 4. Tweedie's Identity and $x_0$-Parameterization

### 4.1 Derivation of Tweedie's Formula for Gaussian Perturbation
Consider the marginal distribution $p_t(\mathbf{x}_t) = \int p_{t|0}(\mathbf{x}_t \mid \mathbf{x}_0) p(\mathbf{x}_0)\,\mathrm{d}\mathbf{x}_0$.
Since $p_{t|0}(\mathbf{x}_t \mid \mathbf{x}_0) = (2\pi \sigma_t^2)^{-D/2} \exp\left(-\frac{\|\mathbf{x}_t - \alpha_t \mathbf{x}_0\|^2}{2\sigma_t^2}\right)$, we differentiate with respect to $\mathbf{x}_t$:

$$\nabla_{\mathbf{x}_t} p_t(\mathbf{x}_t) = \int \nabla_{\mathbf{x}_t} p_{t|0}(\mathbf{x}_t \mid \mathbf{x}_0) p(\mathbf{x}_0)\,\mathrm{d}\mathbf{x}_0$$

$$\nabla_{\mathbf{x}_t} p_{t|0}(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\mathbf{x}_t - \alpha_t \mathbf{x}_0}{\sigma_t^2} p_{t|0}(\mathbf{x}_t \mid \mathbf{x}_0)$$

Dividing both sides by $p_t(\mathbf{x}_t)$:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = \frac{\nabla_{\mathbf{x}_t} p_t(\mathbf{x}_t)}{p_t(\mathbf{x}_t)} = -\int \frac{\mathbf{x}_t - \alpha_t \mathbf{x}_0}{\sigma_t^2} \frac{p_{t|0}(\mathbf{x}_t \mid \mathbf{x}_0) p(\mathbf{x}_0)}{p_t(\mathbf{x}_t)} \,\mathrm{d}\mathbf{x}_0$$

Recognizing Bayes' rule $p(\mathbf{x}_0 \mid \mathbf{x}_t) = \frac{p_{t|0}(\mathbf{x}_t \mid \mathbf{x}_0) p(\mathbf{x}_0)}{p_t(\mathbf{x}_t)}$:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = -\frac{\mathbf{x}_t - \alpha_t \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]}{\sigma_t^2}$$

Rearranging gives Tweedie's formula:

$$\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t] = \frac{\mathbf{x}_t + \sigma_t^2 \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)}{\alpha_t}$$

### 4.2 Parameterization Equivalences
From the forward equation $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \mathbf{z}$, the Gaussian noise sample is:

$$\mathbf{z} = \frac{\mathbf{x}_t - \alpha_t \mathbf{x}_0}{\sigma_t}$$

Therefore, the exact mathematical mappings between representations are:
- **Score Representation**:
  $$\mathbf{s}_\theta(\mathbf{x}_t, t) = -\frac{\mathbf{z}}{\sigma_t} = \frac{\alpha_t \hat{\mathbf{x}}_0 - \mathbf{x}_t}{\sigma_t^2}$$
- **Noise Representation**:
  $$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) = -\sigma_t \mathbf{s}_\theta(\mathbf{x}_t, t) = \frac{\mathbf{x}_t - \alpha_t \hat{\mathbf{x}}_0}{\sigma_t}$$
- **Clean Action Representation ($x_0$-prediction)**:
  $$\hat{\mathbf{x}}_0 = f_\theta(\mathbf{x}_t, t, \mathbf{c}) = \frac{\mathbf{x}_t - \sigma_t \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\alpha_t}$$

### 4.3 Why $x_0$-Parameterization Is Used
In trajectory generation:
1. Physical marine boundaries (maximum velocity, acceleration, turning radius) apply directly to physical action $\mathbf{x}_0$, not to arbitrary Gaussian noise $\mathbf{z}$.
2. Differentiable downstream objective functions (such as obstacle clearance distances and goal anchor penalties) require spatial trajectories computed from $\hat{\mathbf{x}}_0$. With $x_0$-prediction, these losses are computed during every forward training step.

---

## 5. Derivation of the Evidence Lower Bound and Loss Function

### 5.1 Continuous-Time Variational Bound
Let $q(\mathbf{x}_{0:1})$ denote the forward diffusion path measure and $p_\theta(\mathbf{x}_{0:1})$ denote the reverse path measure. By Girsanov's theorem, the continuous-time Evidence Lower Bound (ELBO) on log marginal likelihood is:

$$\log p_\theta(\mathbf{x}_0) \ge -\mathbb{E}_q \left[ \int_0^1 \frac{1}{2} \beta(t) \left\| \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) - \mathbf{s}_\theta(\mathbf{x}_t, t, \mathbf{c}) \right\|_2^2 \,\mathrm{d}t \right] + C$$

Substituting the score estimator $\mathbf{s}_\theta(\mathbf{x}_t, t, \mathbf{c}) = \frac{\alpha_t f_\theta(\mathbf{x}_t, t, \mathbf{c}) - \mathbf{x}_t}{\sigma_t^2}$ and true score $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{x}_0) = \frac{\alpha_t \mathbf{x}_0 - \mathbf{x}_t}{\sigma_t^2}$:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{x}_0) - \mathbf{s}_\theta(\mathbf{x}_t, t, \mathbf{c}) = \frac{\alpha_t}{\sigma_t^2} \left[ \mathbf{x}_0 - f_\theta(\mathbf{x}_t, t, \mathbf{c}) \right]$$

The integrand becomes:

$$\frac{1}{2} \beta(t) \left( \frac{\alpha_t}{\sigma_t^2} \right)^2 \left\| \mathbf{x}_0 - f_\theta(\mathbf{x}_t, t, \mathbf{c}) \right\|_2^2 = \frac{1}{2} w(t) \left\| \mathbf{x}_0 - \hat{\mathbf{x}}_0 \right\|_2^2$$

where $w(t) = \beta(t) \frac{\alpha_t^2}{\sigma_t^4}$.

Following standard practice in continuous diffusion planners (Ho et al., Kingma et al.), unweighted score matching ($w(t) = 1$) is adopted to prevent gradient instability near $t \to 0$ where $\sigma_t \to 0$:

$$\mathcal{L}_{\text{vel}} = \mathbb{E}_{t \sim \mathcal{U}[0, 1],\, \mathbf{x}_0,\, \mathbf{z}} \left[ \frac{1}{H d_a} \sum_{k=1}^H \|\hat{\mathbf{x}}_{0, k} - \mathbf{x}_{0, k}\|_2^2 \right]$$

---

## 6. Kinematic Operator and Downstream Loss Derivations

### 6.1 Discrete Kinematic Waypoint Operator
The action output contains velocity components:

$$\hat{\mathbf{v}}_k = [\hat{v}_{x, k}, \hat{v}_{y, k}]^\top \in \mathbb{R}^2, \quad k \in \{1, \dots, H\}$$

The discrete integration operator $\mathcal{I}: \mathbb{R}^{H \times 2} \to \mathbb{R}^{H \times 2}$ maps body velocities to egocentric spatial waypoints over sampling interval $\Delta t = 10.0\,\text{s}$:

$$\hat{\mathbf{p}}_k = \begin{bmatrix} \hat{x}_k \\ \hat{y}_k \end{bmatrix} = \sum_{j=1}^k \hat{\mathbf{v}}_j \Delta t, \quad k \in \{1, \dots, H\}$$

#### Gradient Truncation Scheme (Detached Integral)
Direct backpropagation through 15 cumulative additions yields compound Jacobian products:

$$\frac{\partial \hat{\mathbf{p}}_k}{\partial \hat{\mathbf{v}}_j} = \Delta t \cdot \mathbf{I}_{2 \times 2} \quad \forall j \le k$$

This accumulation causes gradient variance to scale quadratically with horizon length $k^2$. To stabilize optimization, moving-window gradient detachment with window size $W = 3$ is used:

$$\hat{\mathbf{p}}_k = \left[ \hat{\mathbf{p}}_{\max(0, k-W)} \right]_{\text{no\_grad}} + \sum_{j=\max(1, k-W+1)}^k \hat{\mathbf{v}}_j \Delta t$$

This limits backpropagation paths to local 3-step horizons while preserving exact global positional values during the forward pass.

### 6.2 Waypoint Trajectory Loss ($\mathcal{L}_{\text{wpt}}$)
Given ground-truth waypoints $\mathbf{p}_k^{\text{GT}} = [x_k^{\text{GT}}, y_k^{\text{GT}}]^\top$:

$$\mathcal{L}_{\text{wpt}} = \frac{1}{H} \sum_{k=1}^H \|\hat{\mathbf{p}}_k - \mathbf{p}_k^{\text{GT}}\|_2^2$$

### 6.3 Goal Anchor Penalty ($\mathcal{L}_{\text{goal}}$)
Let $\mathbf{g} = \mathbf{p}_H^{\text{GT}}$ denote the desired destination at final step $H = 15$ ($t = 150\,\text{s}$). The endpoint penalty is:

$$\mathcal{L}_{\text{goal}} = \|\hat{\mathbf{p}}_H - \mathbf{g}\|_2^2 = (\hat{x}_H - x_H^{\text{GT}})^2 + (\hat{y}_H - y_H^{\text{GT}})^2$$

### 6.4 Imitation-Consistent Relative Clearance Deficit Loss ($\mathcal{L}_{\text{obs}}$)

#### Formulation
Let $\mathcal{M} = \{\mathbf{m}_l\}_{l=1}^{N_{\text{map}}} \subset \mathbb{R}^2$ denote all active polyline vertices representing coastlines and obstacles.

At each prediction step $k \in \{1, \dots, H\}$, the Euclidean clearance distance to the nearest land point is:

$$d_k^{\text{pred}} = \min_{l \in \{1, \dots, N_{\text{map}}\}} \|\hat{\mathbf{p}}_k - \mathbf{m}_l\|_2$$

$$d_k^{\text{GT}} = \min_{l \in \{1, \dots, N_{\text{map}}\}} \|\mathbf{p}_k^{\text{GT}} - \mathbf{m}_l\|_2$$

Define the clearance deficit with safety buffer $\delta_{\text{margin}} = 10.0\,\text{m}$:

$$\Delta_k = d_k^{\text{GT}} - d_k^{\text{pred}} - \delta_{\text{margin}}$$

The loss is defined as:

$$\mathcal{L}_{\text{obs}} = \frac{\sum_{b=1}^B \mathbb{I}(\text{has\_coastline}_b) \cdot \frac{1}{H} \sum_{k=1}^H \left[\mathrm{ReLU}(\Delta_k)\right]^2}{\sum_{b=1}^B \mathbb{I}(\text{has\_coastline}_b) + \epsilon}$$

#### Mathematical Proof of Gradient Alignment
Traditional repulsive obstacle potentials add an absolute penalty:

$$\mathcal{L}_{\text{repulsive}} = \frac{1}{d_k^2}$$

This creates a conflicting gradient: when expert human vessels operate in narrow harbor channels, $d_k^{\text{GT}}$ is small. A pure repulsive penalty pushes the model away from the channel, directly conflicting with the imitation loss $\mathcal{L}_{\text{vel}}$.

Under IC-RCDL:
1. When $\hat{\mathbf{p}}_k = \mathbf{p}_k^{\text{GT}}$, we have $d_k^{\text{pred}} = d_k^{\text{GT}}$. Then:
   $$\Delta_k = -\delta_{\text{margin}} = -10.0 < 0$$
   $$\mathrm{ReLU}(\Delta_k) = 0 \implies \mathcal{L}_{\text{obs}} = 0, \quad \nabla_{\theta}\mathcal{L}_{\text{obs}} = \mathbf{0}$$
2. The global minimum of the expert imitation objective lies in the zero-penalty basin of $\mathcal{L}_{\text{obs}}$.
3. Penalties activate if and only if the predicted vessel strays closer to shore than the expert demonstration by more than $\delta_{\text{margin}}$:
   $$\frac{\partial \mathcal{L}_{\text{obs}}}{\partial \hat{\mathbf{p}}_k} = -2 \mathrm{ReLU}(\Delta_k) \cdot \frac{\hat{\mathbf{p}}_k - \mathbf{m}_{l^*}}{\|\hat{\mathbf{p}}_k - \mathbf{m}_{l^*}\|_2}$$
   where $\mathbf{m}_{l^*}$ is the nearest obstacle point. The restoring gradient points directly along the outward surface normal away from land.

---

## 7. Total Multi-Objective Optimization Formulation

The complete training objective is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{vel}} + \lambda_{\text{wpt}} \mathcal{L}_{\text{wpt}} + \lambda_{\text{goal}} \mathcal{L}_{\text{goal}} + \lambda_{\text{obs}} \mathcal{L}_{\text{obs}}$$

with configured hyperparameters:
- $\lambda_{\text{wpt}} = 0.1$
- $\lambda_{\text{goal}} = 1.0$
- $\lambda_{\text{obs}} = 0.5$
- $\delta_{\text{margin}} = 10.0\,\text{m}$

Parameter updates use AdamW with cosine learning rate scheduling:

$$\theta_{k+1} = \theta_k - \eta_k \cdot \mathrm{AdamW}\left(\nabla_\theta \mathcal{L}_{\text{total}}\right)$$

Exponential Moving Average (EMA) weights are updated with decay rate $\mu = 0.999$:

$$\theta_{\text{EMA}} \leftarrow \mu \theta_{\text{EMA}} + (1 - \mu) \theta$$

---

## 8. Inference Sampling via DPM-Solver++ (Fast PF-ODE)

### 8.1 Log-SNR Coordinate Transformation
Define the log signal-to-noise ratio:

$$\lambda_t = \log\left(\frac{\alpha_t}{\sigma_t}\right) = \frac{1}{2} \log\left(\frac{\bar{\alpha}(t)}{1 - \bar{\alpha}(t)}\right)$$

Since $\bar{\alpha}(t)$ decreases monotonically from $1$ to $0$ as $t$ increases from $0$ to $1$, $\lambda_t$ decreases monotonically from $+\infty$ to $-\infty$. Thus, $\lambda$ provides a well-behaved coordinate system for integration.

### 8.2 Semi-Linear ODE Form
From Tweedie's formula, the Probability Flow ODE:

$$\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = -\frac{1}{2}\beta(t)\left[ \mathbf{x}_t + \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \right]$$

can be rewritten in terms of $\hat{\mathbf{x}}_0$:

$$\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t) = \frac{\alpha_t \hat{\mathbf{x}}_0 - \mathbf{x}_t}{\sigma_t^2}$$

Substituting into the ODE:

$$\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = -\frac{1}{2}\beta(t)\left[ \mathbf{x}_t + \frac{\alpha_t \hat{\mathbf{x}}_0 - \mathbf{x}_t}{\sigma_t^2} \right] = -\frac{1}{2}\beta(t)\frac{\alpha_t}{\sigma_t^2}\left[ \alpha_t \mathbf{x}_t - \sigma_t^2 \frac{\alpha_t}{\sigma_t^2} \dots \right]$$

Transforming time derivative $\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}\lambda}$:

$$\frac{\mathrm{d}\mathbf{x}_\lambda}{\mathrm{d}\lambda} = \frac{\mathrm{d}t}{\mathrm{d}\lambda} \frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t}$$

Noticing $\frac{\mathrm{d}\lambda}{\mathrm{d}t} = -\frac{1}{2}\frac{\beta(t)}{\sigma_t^2}$, this simplifies to:

$$\frac{\mathrm{d}\left(e^{-\lambda} \mathbf{x}_\lambda\right)}{\mathrm{d}\lambda} = -e^{-\lambda} \hat{\mathbf{x}}_0(\mathbf{x}_\lambda, \lambda)$$

### 8.3 Exact First-Order DPM-Solver Update (Single Step)
Integrating from step $t_{i-1}$ to $t_i$ ($h_i = \lambda_{t_i} - \lambda_{t_{i-1}}$) assuming $\hat{\mathbf{x}}_0$ is constant over the interval yields the exact exponential solution:

$$\mathbf{x}_{t_i} = \frac{\sigma_{t_i}}{\sigma_{t_{i-1}}} \mathbf{x}_{t_{i-1}} - \alpha_{t_i} \left(e^{-h_i} - 1\right) f_\theta\left(\mathbf{x}_{t_{i-1}}, t_{i-1}\right)$$

### 8.4 Second-Order DPM-Solver++ (2M)
Using linear Taylor interpolation for $\hat{\mathbf{x}}_0$ between steps $t_{i-1}$ and $t_{i-2}$ yields the second-order multistep update:

$$\mathbf{x}_{t_i} = \frac{\sigma_{t_i}}{\sigma_{t_{i-1}}} \mathbf{x}_{t_{i-1}} - \alpha_{t_i} \left(e^{-h_i} - 1\right) \hat{\mathbf{x}}_0^{(i-1)} - \frac{\alpha_{t_i} \left(e^{-h_i} - 1 + h_i\right)}{h_i / h_{i-1}} \left( \hat{\mathbf{x}}_0^{(i-1)} - \hat{\mathbf{x}}_0^{(i-2)} \right)$$

This allows exact trajectory generation in only $N = 15$ function evaluations, reducing inference latency while maintaining path smoothness and dynamic feasibility.
