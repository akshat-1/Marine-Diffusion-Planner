"""Diffusion utilities forwarding to official model/diffusion_utils modules."""

import torch
import torch.nn.functional as F
from model.diffusion_utils.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP, model_wrapper
from model.diffusion_utils.diffusion_sde import DiffusionSDE, TimeSampler


def detached_integral(u: torch.Tensor, detach_window_size: int = 1) -> torch.Tensor:
    """Official detached integral implementation from dp_vla_agent.py.
    u: (B, T, D)
    """
    cum_detach = torch.cumsum(u.detach(), dim=-2)
    cum_normal = torch.cumsum(u, dim=-2)

    shifted = torch.roll(cum_normal, shifts=detach_window_size, dims=-2)
    shifted[..., :detach_window_size, :] = 0
    sum_recent = cum_normal - shifted

    cum_detach_shifted = torch.roll(cum_detach, shifts=detach_window_size, dims=-2)
    cum_detach_shifted[..., :detach_window_size, :] = 0

    cumulative_sum = cum_detach_shifted + sum_recent
    return cumulative_sum


def hybrid_loss(pred_actions: torch.Tensor, target_actions: torch.Tensor, W: int = 3, omega: float = 0.1) -> torch.Tensor:
    """Official hybrid loss calculation using detached_integral."""
    l_action = F.mse_loss(pred_actions, target_actions)
    pred_wpt = detached_integral(pred_actions[..., :2], detach_window_size=W)
    target_wpt = torch.cumsum(target_actions[..., :2], dim=-2)
    l_wpt = F.mse_loss(pred_wpt, target_wpt)
    return l_action + omega * l_wpt


def reward_weighted_diffusion_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    rewards: torch.Tensor,
    beta: float = 1.0,
    W: int = 3,
    omega: float = 0.1,
) -> torch.Tensor:
    """Official reward-weighted diffusion loss for HDP RL posttraining (DpVlaRlAgent).

    Computes group-normalized exponential reward weights:
        weight = exp(beta * (r - mean) / (std + 1e-6))
    and applies them to the per-sample loss.
    """
    # Group-relative normalization if 2D (B, G), or standard normalization if 1D (B*G)
    if rewards.dim() == 2:
        r_norm = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-6)
        r_norm = r_norm.reshape(-1)
    else:
        r_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    weights = torch.exp(beta * r_norm).detach()

    # Per-sample MSE
    per_sample_mse = F.mse_loss(pred, target, reduction="none").mean(dim=[-1, -2])

    if W > 0 and omega > 0:
        pred_wpt = detached_integral(pred[..., :2], detach_window_size=W)
        target_wpt = torch.cumsum(target[..., :2], dim=-2)
        per_sample_wpt = F.mse_loss(pred_wpt, target_wpt, reduction="none").mean(dim=[-1, -2])
        per_sample_loss = per_sample_mse + omega * per_sample_wpt
    else:
        per_sample_loss = per_sample_mse

    return (weights * per_sample_loss).mean()


__all__ = [
    "DPM_Solver",
    "NoiseScheduleVP",
    "model_wrapper",
    "DiffusionSDE",
    "TimeSampler",
    "detached_integral",
    "hybrid_loss",
    "reward_weighted_diffusion_loss",
]
