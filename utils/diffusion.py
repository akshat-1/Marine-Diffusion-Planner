#!/usr/bin/env python3
"""
Gaussian Diffusion Utilities with DPM-Solver.
Implements HDP paper's VP noise schedule, τ₀-prediction, and fast sampling.
Based on Appendix D.4: "We adopt the variance-preserving(VP) noise schedule 
following Zheng et al. (2025) and use 6 sampling steps for efficient generation."
"""

import torch
import torch.nn.functional as F
import numpy as np


class GaussianDiffusion:
    """
    Gaussian Diffusion with VP schedule and DPM-Solver.
    
    Key changes from standard DDPM:
    - Uses VP (variance-preserving) noise schedule
    - Uses τ₀-prediction (direct velocity prediction) instead of ε-prediction
    - Implements DPM-Solver for fast sampling (6 steps)
    """
    
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule="vp"):
        self.timesteps = timesteps
        
        if schedule == "vp":
            self.betas = self._vp_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            self.betas = self._cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # VP schedule uses continuous time formulation
        # α_t = exp(-1/2 ∫_0^t β(s) ds), σ_t = √(1 - α_t²)
        # For discrete: α_t = √(ᾱ_t), σ_t = √(1 - ᾱ_t)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For τ₀-prediction: x_t = α_t * x_0 + σ_t * ε
        # where α_t = √(ᾱ_t), σ_t = √(1 - ᾱ_t)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        # Precompute for DPM-Solver
        self._precompute_dpm_solver()

    def _vp_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """
        VP (Variance-Preserving) noise schedule.
        Uses discrete linear schedule as commonly used in diffusion literature
        (e.g., DDPM, Improved DDPM). The beta_start and beta_end are the 
        discrete beta values for the first and last timestep.
        
        This matches the standard VP schedule used in practice.
        """
        return torch.linspace(beta_start, beta_end, timesteps)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _precompute_dpm_solver(self):
        """Precompute values needed for DPM-Solver."""
        # For DPM-Solver, we need:
        # - lambda_t = log(α_t / σ_t) = 0.5 * log(ᾱ_t / (1 - ᾱ_t))
        # - α_t = √(ᾱ_t), σ_t = √(1 - ᾱ_t)
        self.alphas_cumprod = self.alphas_cumprod.to(dtype=torch.float64)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(dtype=torch.float64)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(dtype=torch.float64)
        
        # λ_t = log(α_t / σ_t)
        self.lambdas = torch.log(self.sqrt_alphas_cumprod / self.sqrt_one_minus_alphas_cumprod)
        
        # Store as float32 for efficiency
        self.lambdas = self.lambdas.float()
        self.alphas_cumprod = self.alphas_cumprod.float()
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.float()
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.float()

    def _get_index(self, vals, t, x_shape):
        """
        Extract values at index t and reshape for broadcasting.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: Add noise to the sample x_0 at timestep t.
        x_t = α_t * x_0 + σ_t * ε
        where α_t = √(ᾱ_t), σ_t = √(1 - ᾱ_t)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self._get_index(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def reconstruct_x0(self, x_t, t, noise_pred):
        """
        Reconstruct x_0 (original signal) from x_t and predicted noise at timestep t.
        For ε-prediction: x_0 = (x_t - σ_t * ε_pred) / α_t
        """
        sqrt_alphas_cumprod_t = self._get_index(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t

    def predict_x0_from_xt(self, x_t, t, v_t):
        """
        Reconstruct x_0 from x_t and predicted velocity v_t (τ₀-prediction).
        x_t = α_t * x_0 + σ_t * ε
        v_t = x_0 (since model predicts x_0 directly)
        """
        # For τ₀-prediction, the model directly outputs x_0
        return v_t

    def get_alpha_sigma(self, t):
        """Get α_t and σ_t for given timestep t."""
        alpha_t = self._get_index(self.sqrt_alphas_cumprod, t, (1,))
        sigma_t = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, (1,))
        return alpha_t, sigma_t

    def get_lambda(self, t, x_shape=None):
        """Get λ_t = log(α_t / σ_t) for given timestep t."""
        if x_shape is None:
            x_shape = (1,)
        return self._get_index(self.lambdas, t, x_shape)

    @torch.no_grad()
    def dpm_solver_sample(self, model, context, shape, steps=6, order=2):
        """
        DPM-Solver fast sampling adapted for τ₀-prediction (x₀-prediction).
        
        Paper: "we follow the approach used by Zheng et al. (2025), which employs 
        the DPM-Solver (Lu et al., 2022) to accelerate the sampling process, 
        achieving a final inference speed that easily meets the 10Hz requirement."
        
        For τ₀-prediction (model outputs x₀ directly), we use the DDIM-style 
        update which is the correct first-order solver for x₀-prediction.
        Higher-order solvers for x₀-prediction require DPM-Solver++ formulation.
        
        Args:
            model: The denoising model (predicts x₀ directly)
            context: Conditioning context
            shape: Shape of the output (B, T, F)
            steps: Number of sampling steps (default 6 as per paper)
            order: Order of solver (1 or 2; 3 requires DPM-Solver++)
        
        Returns:
            Sampled trajectories
        """
        device = next(model.parameters()).device
        B = shape[0]
        
        # Start from pure Gaussian noise (x_T ~ N(0, I))
        x = torch.randn(shape, device=device)
        
        # Time steps for sampling (from T to 0)
        t_steps = torch.linspace(self.timesteps - 1, 0, steps + 1, device=device).long()
        
        # Store previous model outputs for higher-order solvers
        model_outputs = []
        t_prev_list = []
        
        for i in range(steps):
            t = t_steps[i]
            t_next = t_steps[i + 1]
            
            t_batch = torch.full((B,), t.item(), device=device, dtype=torch.long)
            t_next_batch = torch.full((B,), t_next.item(), device=device, dtype=torch.long)
            
            # Model predicts x₀ (τ₀-prediction)
            model_output = model(x, t_batch, context)  # This is predicted x₀ (v₀)
            
            # Get α and σ for current and next timestep
            alpha_t = self._get_index(self.sqrt_alphas_cumprod, t_batch, x.shape)
            sigma_t = self._get_index(self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
            alpha_t_next = self._get_index(self.sqrt_alphas_cumprod, t_next_batch, x.shape)
            sigma_t_next = self._get_index(self.sqrt_one_minus_alphas_cumprod, t_next_batch, x.shape)
            
            if order == 1:
                # First order: DDIM update for x₀-prediction
                # x_{t_next} = α_{t_next} * x₀_pred + σ_{t_next} * ε
                # where ε = (x_t - α_t * x₀_pred) / σ_t
                # = (σ_{t_next}/σ_t) * x_t + (α_{t_next} - σ_{t_next}*α_t/σ_t) * x₀_pred
                if t_next == 0:
                    # Final step: return x₀ directly
                    x = model_output
                else:
                    x = (sigma_t_next / sigma_t) * x + (alpha_t_next - sigma_t_next * alpha_t / sigma_t) * model_output
                
            elif order == 2:
                # Second order: Use improved DDIM with previous prediction
                # This is a simplified version - for true second-order, use DPM-Solver++
                if i == 0:
                    # First step: first order
                    if t_next == 0:
                        x = model_output
                    else:
                        x = (sigma_t_next / sigma_t) * x + (alpha_t_next - sigma_t_next * alpha_t / sigma_t) * model_output
                else:
                    # Second step: use previous prediction for momentum
                    model_output_prev = model_outputs[-1]
                    
                    # DDIM-style with momentum correction
                    if t_next == 0:
                        x = model_output
                    else:
                        # Use average of current and previous prediction
                        x = (sigma_t_next / sigma_t) * x + (alpha_t_next - sigma_t_next * alpha_t / sigma_t) * \
                            (0.5 * model_output + 0.5 * model_output_prev)
            
            model_outputs.append(model_output)
            t_prev_list.append(t_batch)
            
            # Keep only last 2 outputs for order 2
            if len(model_outputs) > 2:
                model_outputs.pop(0)
                t_prev_list.pop(0)
        
        return x

    @torch.no_grad()
    def sample(self, model, context, shape, use_dpm_solver=True, steps=6):
        """
        Complete sampling loop.
        
        Args:
            model: The denoising model
            context: Conditioning context
            shape: Shape of the output (B, T, F)
            use_dpm_solver: Whether to use DPM-Solver (default True)
            steps: Number of sampling steps for DPM-Solver (default 6 as per paper)
        """
        if use_dpm_solver:
            return self.dpm_solver_sample(model, context, shape, steps=steps)
        else:
            # Fallback to standard DDPM sampling
            return self._ddpm_sample(model, context, shape)

    @torch.no_grad()
    def _ddpm_sample(self, model, context, shape):
        """Standard DDPM sampling (slow, for comparison)."""
        device = next(model.parameters()).device
        B = shape[0]
        
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, context, i)
            
        return img

    @torch.no_grad()
    def p_sample(self, model, x, t, context, t_index):
        """
        One step of denoising using the model (DDPM).
        For τ₀-prediction: model outputs x_0 directly.
        """
        betas_t = self._get_index(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._get_index(torch.sqrt(1.0 / self.alphas), t, x.shape)
        
        # Model predicts x_0 (τ₀-prediction)
        model_output = model(x, t, context)  # This is predicted x_0
        
        # DDPM update with x_0 prediction
        # x_{t-1} = √ᾱ_{t-1} * x_0_pred + √(1 - ᾱ_{t-1}) * ε
        # where ε = (x_t - √ᾱ_t * x_0_pred) / √(1 - ᾱ_t)
        alpha_t = self._get_index(self.sqrt_alphas_cumprod, t, x.shape)
        alpha_t_prev = self._get_index(self.sqrt_alphas_cumprod, 
                                        torch.clamp(t - 1, min=0), x.shape)
        sigma_t = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sigma_t_prev = self._get_index(self.sqrt_one_minus_alphas_cumprod, 
                                        torch.clamp(t - 1, min=0), x.shape)
        
        # Compute noise from x_0 prediction
        eps = (x - alpha_t * model_output) / sigma_t
        
        if t_index == 0:
            return model_output
        else:
            posterior_variance_t = self._get_index(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return alpha_t_prev * model_output + torch.sqrt(posterior_variance_t) * noise


def detached_integral(v, W, dt):
    """
    Detached integral as per Algorithm 1 in HDP paper.
    
    Args:
        v: velocity of future trajectory (B, T, 2) - [vx, vy]
        W: gradient detach window size
        dt: time interval
    
    Returns:
        Integrated waypoints with detached gradients beyond window W
    """
    # v shape: (B, T, 2)
    B, T, _ = v.shape
    
    # Standard cumulative integration (with gradients)
    wpt = torch.cumsum(v, dim=1) * dt  # (B, T, 2)
    
    # Detached cumulative integration
    v_detached = v.detach()
    wpt_sg = torch.cumsum(v_detached, dim=1) * dt
    
    # Shift by W
    shift_sg = torch.roll(wpt_sg, shifts=W, dims=1)
    shift_sg[:, :W] = 0
    
    shift = torch.roll(wpt, shifts=W, dims=1)
    shift[:, :W] = 0
    
    # Return: wpt + shift_sg - shift
    # This gives gradient only from the last W waypoints
    return wpt + shift_sg - shift


def hybrid_loss(pred_v, gt_v, W, omega, dt=1.0):
    """
    Hybrid loss as per Algorithm 1 in HDP paper.
    
    Args:
        pred_v: predicted future velocity (B, T, 2) - [vx, vy]
        gt_v: ground truth future velocity (B, T, 2)
        W: gradient detach window size
        omega: loss balancing weight
        dt: time interval
    
    Returns:
        Total hybrid loss
    """
    # Velocity loss
    l_v = F.mse_loss(pred_v, gt_v)
    
    # Waypoint loss with detached integral
    pred_wpt = detached_integral(pred_v, W, dt)
    gt_wpt = torch.cumsum(gt_v, dim=1) * dt
    l_wpt = F.mse_loss(pred_wpt, gt_wpt)
    
    return l_v + omega * l_wpt


if __name__ == "__main__":
    # Smoke test
    B, T, F = 4, 20, 4  # 4 features: vx, vy, theta, yaw_rate
    diff = GaussianDiffusion(timesteps=1000, schedule="vp")
    x_0 = torch.randn(B, T, F)
    t = torch.randint(0, 1000, (B,))
    x_t, noise = diff.q_sample(x_0, t)
    
    print("Forward diffusion x_t shape:", x_t.shape)
    print("Noise shape:", noise.shape)
    
    # Test DPM-Solver sampling
    class DummyModel(torch.nn.Module):
        def forward(self, x, t, context):
            return torch.randn_like(x)
    
    model = DummyModel()
    context = torch.randn(B, 256)
    sampled = diff.dpm_solver_sample(model, context, (B, T, F), steps=6)
    print("DPM-Solver sampled shape:", sampled.shape)
    
    # Test hybrid loss
    pred_v = torch.randn(B, T, 2)
    gt_v = torch.randn(B, T, 2)
    loss = hybrid_loss(pred_v, gt_v, W=3, omega=0.1, dt=10.0)
    print("Hybrid loss:", loss.item())