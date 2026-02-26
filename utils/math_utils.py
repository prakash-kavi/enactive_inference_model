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

def precision_from_surprisal(
    surprisal: Union[float, int, torch.Tensor],
    eps: float = EPS,
) -> float:
    """Precision from surprisal via exponential mapping."""
    s = max(0.0, to_float(surprisal))
    return clip_probability(np.exp(-s))

def integrate_precision_logit(
    base_precision: Union[float, int, torch.Tensor],
    meta_precision: Union[float, int, torch.Tensor],
    eps: float = EPS,
) -> float:
    """Integrate two precision signals via logit-add (odds multiplication).
    Base from forward surprisal; meta from L3 meta-awareness.
    Yields a single sensory precision in [0,1]."""
    b = float(np.clip(to_float(base_precision), eps, 1.0 - eps))
    m = float(np.clip(to_float(meta_precision), eps, 1.0 - eps))
    logit = np.log(b / (1.0 - b)) + np.log(m / (1.0 - m))
    return float(1.0 / (1.0 + np.exp(-logit)))


def compute_precision_sensory(
    base_precision: Union[float, int, torch.Tensor],
    meta_precision: Union[float, int, torch.Tensor],
    eps: float = EPS,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> float:
    """Compute integrated sensory precision from base + meta signals."""
    b = float(np.clip(to_float(base_precision), clip_min, clip_max))
    m = float(np.clip(to_float(meta_precision), clip_min, clip_max))
    return integrate_precision_logit(b, m, eps)

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

def clamp_activation(x: torch.Tensor, clip_min: float, clip_max: float) -> torch.Tensor:
    """Clamp activations to configured model bounds."""
    return torch.clamp(x, clip_min, clip_max)


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
