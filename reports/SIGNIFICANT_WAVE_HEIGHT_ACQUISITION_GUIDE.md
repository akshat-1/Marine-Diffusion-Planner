# Significant Wave Height ($H_s$) Acquisition & Estimation Guide

This document details how **Significant Wave Height ($H_s$, in meters)** is acquired, estimated from AIS trajectory differentials, and used within the Hyper Diffusion Planner (HDP) pipeline.

---

## 1. Definition of Significant Wave Height ($H_s$)

In oceanography and naval architecture, **Significant Wave Height ($H_s$ or $H_{1/3}$)** is defined as the mean height (from trough to crest) of the highest one-third of waves in a given sea state:

$$H_s = 4 \cdot \sigma_z = 4 \cdot \sqrt{m_0}$$

where $m_0 = \int_0^\infty S(\omega) d\omega$ is the zeroth moment (variance) of the wave elevation energy spectrum $S(\omega)$.

---

## 2. Three Methods for Acquiring $H_s$ in Maritime Navigation

```
                       Significant Wave Height (Hs)
                       /            |            \
                      /             |             \
        Method 1: NOAA/ERA5    Method 2: Inverse MMG   Method 3: Onboard Sensors
        Marine Reanalysis     Hydrodynamic Residue      (Radar / IMU Heave)
        ------------------     -------------------      -------------------
        • Spatiotemporal       • Extracted from AIS     • X-band Wave Radar
          Grid Lookup            differential drift     • IMU Heave Spectrum
        • Global coverage      • Zero external sensor   • Real-vehicle deployment
```

---

### Method 1: Global Marine Weather Reanalysis (NOAA WaveWatch III / ECMWF ERA5)
During offline dataset creation, $H_s$ is looked up from global hindcast ocean wave models using the scenario timestamp and GPS coordinates:

- **Data Sources**:
  - **NOAA WaveWatch III**: Global wave forecasts and archives ($0.5^\circ \times 0.5^\circ$ resolution, hourly).
  - **ECMWF ERA5 Marine Reanalysis**: Historical global wave height grids.
- **Lookup Formula**:
  Given AIS scenario timestamp $T$ and vessel origin $(\phi, \lambda)$:
  $$H_s = \mathrm{BilinearInterpolate}(\mathrm{WaveWatchIII}, \phi, \lambda, T)$$

---

### Method 2: Inverse Hydrodynamic Residual Estimation (Our AIS Dataset Method)
When external wave buoys or ERA5 grids are unavailable, $H_s$ is **derived directly from observed AIS trajectory drift differentials**:

From our inverse MMG hydrodynamic residual equation (`preparedataset.py`):

$$F_{y,\text{ext}} = (m + m_y) \dot{v} + (m + m_x) u r - F_{y,\text{hull}}$$

Subtracting aerodynamic wind drag $F_{y,\text{wind}}$ isolates the wave drift force:

$$F_{y,\text{wave}} = F_{y,\text{ext}} - F_{y,\text{wind}}$$

Applying Maruo's second-order wave drift formula allows solving for the **effective equivalent wave height $H_s$**:

$$H_s \approx \sqrt{\frac{2 \cdot |F_{y,\text{ext}} - F_{y,\text{wind}}|}{\rho_w \cdot g \cdot L \cdot C_{Dwy}(\chi)}}$$

where:
- $\rho_w = 1025\text{ kg/m}^3$ (seawater density).
- $g = 9.81\text{ m/s}^2$ (gravity).
- $L$: Vessel length.
- $C_{Dwy}(\chi)$: Directional wave drift coefficient based on relative wave heading $\chi$.

> **Key Advantage**: Produces an **effective equivalent sea-state wave height $H_s$** directly from observed trajectory leeway drift without requiring physical sensors!

---

### Method 3: Real-Time Onboard Sensors (Real-Vehicle ASV Deployment)
During real-vehicle closed-loop execution on an Autonomous Surface Vessel (ASV):

1. **Marine X-Band Radar Wave Spectrum (e.g., WaMoS II / Miros WaveRadar)**:
   - Processes spatial sea surface radar clutter to output $H_s$, peak wave period $T_p$, and wave direction $\theta_{\text{wave}}$ in real-time.
2. **Inertial Measurement Unit (IMU) Vertical Heave Spectrum**:
   - Integrates vertical acceleration $a_z$ to derive heave displacement $z(t)$ and computes wave variance:
     $$H_s = 4 \cdot \sqrt{\text{Variance}(z_{\text{heave}})}$$

---

## 3. Integration into the HDP Diffusion Pipeline

```
  Wave Height (Hs) & Direction (χ)
              │
              ▼
  Second-Order Wave Force: Fy_wave = 1/2 ρw g Hs² L C_Dwy(χ)
              │
              ▼
  Specific Force Residual: ay_ext = (Fy_wind + Fy_wave) / m   [m/s²]
              │
              ▼
  10D State Feature Tensor: [x, y, vx, vy, θ, r, ax_ext, ay_ext, α_ext, β_drift]
              │
              ▼
  CustomDiT Decoder Conditioning (y ∈ R^{B × 12})
```

### Benefit to Model Performance
Conditioning on $H_s$ via $a_{y,\text{ext}}$ enables the diffusion planner to:
1. Distinguish between calm seas ($H_s < 0.5\text{m}$) and heavy seas ($H_s > 2.5\text{m}$).
2. Predict proactive **crab-angle steering into incoming waves** before the vessel is pushed off course.
