"""Math utility functions for lean meditation model."""

from typing import Dict, Iterable, Union
import numpy as np
import torch

from utils.config import EPS

def to_float(value: Union[float, int, torch.Tensor]) -> float:
    """Convert scalar-like values (including 0-dim tensors) to float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)

def clip_probability(value: Union[float, int, torch.Tensor]) -> float:
    """Clamp scalar-like values to [0, 1]."""
    return float(np.clip(to_float(value), 0.0, 1.0))

def precision_weight(value: Union[float, int, torch.Tensor]) -> float:
    """Map a precision-like scalar to a bounded [0, 1] weight."""
    return clip_probability(value)

def precision_from_variance(variance: Union[float, int, torch.Tensor], eps: float = EPS) -> float:
    """Precision as inverse variance (act-inf convention)."""
    v = max(0.0, to_float(variance))
    return float(1.0 / (v + eps))

def ema_update(value: Union[float, int, torch.Tensor], mean: float, var: float, beta: float = 0.9) -> tuple[float, float]:
    """Exponential moving mean/variance update."""
    v = to_float(value)
    new_mean = beta * mean + (1.0 - beta) * v
    diff = v - new_mean
    new_var = beta * var + (1.0 - beta) * (diff * diff)
    return new_mean, new_var

def softmax(logits: np.ndarray) -> np.ndarray:
    """Stable softmax for 1D arrays."""
    if logits.size == 0:
        return logits
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    exp_sum = exp.sum()
    if exp_sum <= 0.0:
        return np.full_like(exp, 1.0 / max(exp.size, 1))
    return exp / exp_sum

def entropy(probabilities: np.ndarray, eps: float = EPS) -> float:
    """Shannon entropy for a probability vector."""
    if probabilities.size == 0:
        return 0.0
    p = np.clip(probabilities, eps, 1.0)
    return float(-np.sum(p * np.log(p)))

def policy_posterior(log_prior: np.ndarray, g_vals: np.ndarray, gamma: float) -> np.ndarray:
    """Active-inference policy posterior: softmax(log_prior - gamma * G)."""
    return softmax(log_prior - gamma * g_vals)

def policy_confidence(pi: np.ndarray, eps: float = EPS) -> float:
    """Policy confidence = 1 - normalized entropy."""
    if pi.size == 0:
        return 0.0
    if pi.size == 1:
        return 1.0
    denom = np.log(len(pi) + eps)
    if denom <= eps:
        return 0.0
    return float(np.clip(1.0 - (entropy(pi, eps) / denom), 0.0, 1.0))

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


def bernoulli_entropy(p: torch.Tensor, eps: float) -> torch.Tensor:
    """Elementwise Bernoulli entropy averaged over dimensions."""
    p_safe = clamp_for_log(p, eps)
    return torch.mean(
        -p_safe * torch.log(p_safe) - (1.0 - p_safe) * torch.log(1.0 - p_safe)
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
