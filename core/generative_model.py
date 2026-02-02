"""Explicit probabilistic model components used by training and policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from utils.meditation_config import NETWORK_PROFILES, STATES
from utils.meditation_utils import get_exit_transition_probs


@dataclass(frozen=True)
class EFETerms:
    """Container for expected free-energy decomposition."""

    risk: float
    ambiguity: float
    total: float


class ObservationModel:
    """Observation and latent terms used in the VFE proxy."""

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def reconstruction_nll(self, recon_x: torch.Tensor, observed_x: torch.Tensor) -> torch.Tensor:
        # Gaussian likelihood proxy with fixed variance reduces to MSE up to constants.
        return F.mse_loss(recon_x, observed_x, reduction='mean')

    def latent_bernoulli_kl(self, z: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        z = torch.clamp(z, eps, 1.0 - eps)
        prior = torch.clamp(prior, eps, 1.0 - eps)
        return torch.mean(
            z * torch.log(z / prior)
            + (1.0 - z) * torch.log((1.0 - z) / (1.0 - prior))
        )


class PolicyEnergyModel:
    """Explicit transition + preference model used for EFE terms."""

    def __init__(
        self,
        experience_level: str,
        ambiguity_weight: float = 0.4,
        cycle_strength: float = 0.35,
    ):
        self.experience_level = experience_level
        self.ambiguity_weight = float(ambiguity_weight)
        self.cycle_strength = float(cycle_strength)

    def _base_distribution(self, current_state: str) -> Dict[str, float]:
        return get_exit_transition_probs(self.experience_level, current_state)

    def _preference_distribution(self, current_state: str, base: Dict[str, float]) -> Dict[str, float]:
        if not base:
            return {}
        try:
            next_state = STATES[(STATES.index(current_state) + 1) % len(STATES)]
        except ValueError:
            next_state = None

        if next_state in base:
            return {
                s: ((1.0 - self.cycle_strength) * base.get(s, 0.0)
                    + (self.cycle_strength * (1.0 if s == next_state else 0.0)))
                for s in base
            }
        return dict(base)

    def _predicted_distribution(
        self,
        base: Dict[str, float],
        pref: Dict[str, float],
        transition_pressure: float,
    ) -> Dict[str, float]:
        pred = dict(base)
        drive = float(np.clip(transition_pressure, 0.0, 1.0))
        if drive > 0.0:
            blend = drive * self.cycle_strength
            for s in pred:
                pred[s] = (1.0 - blend) * base.get(s, 0.0) + blend * pref.get(s, 0.0)
        return pred

    def compute_efe_terms(self, current_state: str, transition_pressure: float) -> EFETerms:
        base = self._base_distribution(current_state)
        if not base:
            return EFETerms(risk=0.0, ambiguity=0.0, total=0.0)

        pref = self._preference_distribution(current_state, base)
        pred = self._predicted_distribution(base, pref, transition_pressure)

        eps = 1e-8
        risk = 0.0
        for state in STATES:
            p_s = max(eps, float(pred.get(state, 0.0)))
            q_s = max(eps, float(pref.get(state, 0.0)))
            risk += p_s * (np.log(p_s) - np.log(q_s))

        ambiguity = 0.0
        for state, p_s in pred.items():
            profile = NETWORK_PROFILES.get(state, {}).get(self.experience_level, {})
            if not profile:
                continue
            for val in profile.values():
                p = float(np.clip(val, eps, 1.0 - eps))
                ambiguity += float(p_s) * (-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))

        total = risk + (self.ambiguity_weight * ambiguity)
        return EFETerms(risk=float(risk), ambiguity=float(ambiguity), total=float(total))
