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


def apply_maritime_augmentations(
    batch: dict,
    p_map_drop: float = 0.15,
    p_agent_drop: float = 0.15,
    p_flip: float = 0.50,
    jitter_std: float = 1.5,
) -> dict:
    """Applies online domain-specific data augmentations for maritime diffusion training.

    1. Coastline Dropout: Randomly masks map_mask with probability p_map_drop.
    2. Agent Dropout: Randomly masks surround agents to simulate AIS packet drops.
    3. Polyline Jitter: Adds Gaussian noise (1.5m) to coastline polyline vertices.
    4. Reflection Flip: Mirrors port/starboard coordinates (y, vy, theta, yaw_rate).
    """
    ego_hist = batch["ego_history"].clone()
    target_full = batch["ego_target"].clone()
    agents = batch["agents_history"].clone()
    map_lines = batch["map_lines"].clone()
    agent_mask = batch["agent_mask"].clone()
    map_mask = batch["map_mask"].clone()

    B = ego_hist.shape[0]
    device = ego_hist.device

    # 1. Coastline Dropout (Open Sea conditioning)
    map_drop = (torch.rand(B, device=device) < p_map_drop)
    map_mask[map_drop] = True

    # 2. Agent Dropout (AIS Packet Loss simulation)
    agent_drop_rand = torch.rand(agent_mask.shape, device=device)
    agent_mask = agent_mask | (agent_drop_rand < p_agent_drop)

    # 3. Polyline Vertex Jittering (Tide & Map Noise)
    if map_lines.numel() > 0 and jitter_std > 0:
        map_lines = map_lines + torch.randn_like(map_lines) * jitter_std

    # 4. Port/Starboard Reflection Flip (Open sea batches)
    flip_mask = (torch.rand(B, device=device) < p_flip) & map_drop
    if flip_mask.any():
        for tensor in [ego_hist, target_full]:
            tensor[flip_mask, :, 1] *= -1.0  # y
            tensor[flip_mask, :, 3] *= -1.0  # vy
            tensor[flip_mask, :, 4] *= -1.0  # theta
            tensor[flip_mask, :, 5] *= -1.0  # yaw_rate

        agents[flip_mask, ..., 1] *= -1.0
        agents[flip_mask, ..., 3] *= -1.0
        agents[flip_mask, ..., 4] *= -1.0
        agents[flip_mask, ..., 5] *= -1.0

    return {
        "ego_history": ego_hist,
        "ego_target": target_full,
        "agents_history": agents,
        "map_lines": map_lines,
        "agent_mask": agent_mask,
        "map_mask": map_mask,
    }


class ZScoreNormalizer(torch.nn.Module):
    """Z-Score Standard Normalizer: z = (x - mean) / (std + eps).
    Normalizes dataset feature tensors to zero mean and unit variance per HDP paper specification.
    """

    def __init__(self, mean, std, eps: float = 1e-6):
        super().__init__()
        mean_t = torch.tensor(mean, dtype=torch.float32) if not isinstance(mean, torch.Tensor) else mean
        std_t = torch.tensor(std, dtype=torch.float32) if not isinstance(std, torch.Tensor) else std
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Z-Score normalization: (x - mean) / (std + eps)."""
        m = self.mean.to(device=x.device, dtype=x.dtype)
        s = self.std.to(device=x.device, dtype=x.dtype)
        return (x - m) / (s + self.eps)

    def unnormalize(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse Z-Score transform: z * std + mean."""
        m = self.mean.to(device=z.device, dtype=z.dtype)
        s = self.std.to(device=z.device, dtype=z.dtype)
        return z * (s + self.eps) + m


class ExponentialMovingAverage:
    """Exponential Moving Average (EMA) of model weights.
    Matches timm / HDP paper implementation (decay=0.999).
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = {}

    def update(self, model: torch.nn.Module) -> None:
        """Update shadow weights after an optimizer step."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    new_val = param.detach()
                    self.shadow[name].copy_(
                        self.decay * self.shadow[name] + (1.0 - self.decay) * new_val
                    )

    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Substitute model weights with shadow EMA weights (for eval / save)."""
        self.backup = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    param.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        """Restore original training weights after eval."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.backup:
                    param.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


__all__ = [
    "DPM_Solver",
    "NoiseScheduleVP",
    "model_wrapper",
    "DiffusionSDE",
    "TimeSampler",
    "detached_integral",
    "hybrid_loss",
    "reward_weighted_diffusion_loss",
    "apply_maritime_augmentations",
    "ExponentialMovingAverage",
    "ZScoreNormalizer",
]
