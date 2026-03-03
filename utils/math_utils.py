"""Math utility functions for lean meditation model."""

from typing import Dict, Iterable, Optional, Union
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

def policy_posterior(log_prior: np.ndarray, g_vals: np.ndarray, gamma: float) -> np.ndarray:
    """Active-inference policy posterior: softmax(log_prior - gamma * G)."""
    return softmax(log_prior - gamma * g_vals)

def normalize_scores(values: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Z-score normalize if variance is non-trivial."""
    if values.size == 0:
        return values
    std = float(np.std(values))
    if std <= eps:
        return values
    return (values - np.mean(values)) / (std + eps)

def _to_array(
    belief: Union[Dict[str, float], np.ndarray, list, tuple, None],
    keys: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Helper to convert dictionary or sequence to 1D float array."""
    if belief is None:
        return np.empty(0, dtype=float)
    if isinstance(belief, dict):
        if keys is None:
            return np.array(list(belief.values()), dtype=float)
        return np.array([float(belief.get(k, 0.0)) for k in keys], dtype=float)
    return np.array(belief, dtype=float)

def normalize_belief(
    belief: Union[Dict[str, float], np.ndarray, list, tuple, None],
    keys: Optional[Iterable[str]] = None,
    eps: float = EPS,
) -> np.ndarray:
    """Normalize belief weights (uniform if missing or degenerate)."""
    values = _to_array(belief, keys)
    n = max(values.size, 1) if values.size > 0 else (len(keys) if keys else 1)
    total = float(np.sum(values))
    if total <= eps or values.size == 0:
        return np.full(n, 1.0 / n, dtype=float)
    return np.clip(values / total, 0.0, 1.0)

def belief_entropy(
    belief: Union[Dict[str, float], np.ndarray, list, tuple],
    keys: Optional[Iterable[str]] = None,
    eps: float = EPS,
) -> float:
    """Entropy of a discrete belief (in nats) with safe normalization."""
    values = np.clip(normalize_belief(belief, keys, eps), eps, 1.0)
    return float(-np.sum(values * np.log(values)))

def state_confidence(
    belief: Union[Dict[str, float], np.ndarray, list, tuple, None],
    keys: Optional[Iterable[str]] = None,
    eps: float = EPS,
) -> float:
    """Entropy-based confidence in a discrete belief (0..1)."""
    if belief is None:
        return 0.0
    entropy = belief_entropy(belief, keys=keys, eps=eps)
    n = len(keys) if keys is not None else max(len(_to_array(belief)), 1)
    max_entropy = float(np.log(max(n, 1)))
    if max_entropy <= eps:
        return 0.0
    return float(np.clip(1.0 - (entropy / max_entropy), 0.0, 1.0))

def policy_entropy(
    probs: Union[np.ndarray, list, tuple, None],
    eps: float = EPS,
) -> float:
    """Entropy of a categorical policy distribution (in nats)."""
    if probs is None or len(probs) == 0:
        return 0.0
    values = np.clip(normalize_belief(probs, eps=eps), eps, 1.0)
    return float(-np.sum(values * np.log(values)))

def clamp_activation(x: torch.Tensor, clip_min: float, clip_max: float) -> torch.Tensor:
    """Clamp activations to configured model bounds."""
    return torch.clamp(x, clip_min, clip_max)


def ou_step_scalar(
    value: torch.Tensor,
    target: torch.Tensor,
    dt: float,
    tau: float,
    noise_level: float,
    clip_min: float,
    clip_max: float,
) -> torch.Tensor:
    """Single-step scalar OU update for latent variables.

    dZ = -(1/tau) * (Z - target) dt + sqrt(noise_level) dW
    """
    theta = 1.0 / max(tau, dt)
    value = clamp_activation(value.detach(), clip_min, clip_max)
    drift = -theta * (value - target.detach())
    noise_std = float(np.sqrt(noise_level))
    if noise_std > 0.0:
        noise = torch.randn_like(value) * noise_std * np.sqrt(dt)
    else:
        noise = torch.zeros_like(value)
    updated = value + drift * dt + noise
    return clamp_activation(updated, clip_min, clip_max)


def mse_error(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean squared error averaged over dimensions."""
    return torch.mean((x_hat - x) ** 2)


def recon_error(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reconstruction error in network space (MSE)."""
    return mse_error(x_hat, x)


def prior_error(z: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """Prior-matching error in thoughtseed space (MSE)."""
    return mse_error(z, prior)


def forward_error(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Forward-model prediction error (MSE)."""
    return mse_error(x_hat, x)


def networks_to_tensor(
    network_values: Dict[str, Union[float, torch.Tensor]],
    networks: Iterable[str],
    default: float = 0.0,
    detach: bool = False,
) -> torch.Tensor:
    """Convert ordered network dict values to a single tensor."""
    values = []
    for net in networks:
        value = network_values.get(net)
        if value is None:
            tensor = torch.tensor(default, dtype=torch.float32)
        elif isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.tensor(value, dtype=torch.float32)

        if detach:
            tensor = tensor.detach()
        values.append(tensor)

    return torch.stack(values)
