"""Shared utility functions for lean meditation model."""

from typing import Dict, Iterable, Union

import numpy as np
import torch


def to_float(value: Union[float, int, torch.Tensor]) -> float:
    """Convert scalar-like values (including 0-dim tensors) to float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def clip_probability(value: Union[float, int, torch.Tensor]) -> float:
    """Clamp scalar-like values to [0, 1]."""
    return float(np.clip(to_float(value), 0.0, 1.0))


def clamp_for_log(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Clamp tensor to open interval used in log-probability terms."""
    return torch.clamp(x, eps, 1.0 - eps)


def clamp_activation(x: torch.Tensor, clip_min: float, clip_max: float) -> torch.Tensor:
    """Clamp activations to configured model bounds."""
    return torch.clamp(x, clip_min, clip_max)


def bernoulli_kl(q: torch.Tensor, p: torch.Tensor, eps: float) -> torch.Tensor:
    """Elementwise Bernoulli KL averaged over dimensions."""
    q_safe = clamp_for_log(q, eps)
    p_safe = clamp_for_log(p, eps)
    return torch.mean(
        q_safe * torch.log(q_safe / p_safe)
        + (1.0 - q_safe) * torch.log((1.0 - q_safe) / (1.0 - p_safe))
    )


def networks_to_tensor(
    network_values: Dict[str, Union[float, torch.Tensor]],
    networks: Iterable[str],
    device: torch.device = None,
    default: float = 0.0,
    detach: bool = False,
) -> torch.Tensor:
    """Convert ordered network dict values to a single tensor."""
    values = []
    for net in networks:
        value = network_values.get(net)
        if value is None:
            tensor = torch.tensor(default, dtype=torch.float32, device=device)
        elif isinstance(value, torch.Tensor):
            tensor = value.to(device) if device is not None else value
        else:
            tensor = torch.tensor(value, dtype=torch.float32, device=device)

        if detach:
            tensor = tensor.detach()
        values.append(tensor)

    return torch.stack(values)
