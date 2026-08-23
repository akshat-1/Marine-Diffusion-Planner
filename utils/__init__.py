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


__all__ = [
    "DPM_Solver",
    "NoiseScheduleVP",
    "model_wrapper",
    "DiffusionSDE",
    "TimeSampler",
    "detached_integral",
    "hybrid_loss",
]
