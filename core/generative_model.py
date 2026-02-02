"""Explicit probabilistic model components used by training and policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core.layer1.layer1_config import NETWORK_PROFILES
from utils.meditation_utils import get_exit_transition_probs


@dataclass(frozen=True)
class EFETerms:
    """Container for expected free-energy decomposition."""

    risk: float
    ambiguity: float
    total: float


@dataclass(frozen=True)
class PolicyInferenceResult:
    """Policy inference output q(pi) with EFE decomposition."""

    posterior: Dict[str, float]
    efe_by_policy: Dict[str, float]
    risk_by_policy: Dict[str, float]
    ambiguity_by_policy: Dict[str, float]
    expected_risk: float
    expected_ambiguity: float
    expected_total: float
    selected_policy: str
    selected_confidence: float
    posterior_entropy: float


@dataclass(frozen=True)
class LatentVFETerms:
    """Decomposition for the latent variational update objective."""

    reconstruction: torch.Tensor
    prior_kl: torch.Tensor
    sensory_consistency: torch.Tensor
    temporal_consistency: torch.Tensor
    total: torch.Tensor


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

    def latent_variational_terms(
        self,
        z: torch.Tensor,
        prior: torch.Tensor,
        recon_x: torch.Tensor,
        observed_x: torch.Tensor,
        precision: float,
        *,
        sensory_target: Optional[torch.Tensor] = None,
        temporal_target: Optional[torch.Tensor] = None,
        obs_weight: float = 1.0,
        prior_weight: float = 1.0,
        sensory_weight: float = 0.0,
        temporal_weight: float = 0.0,
    ) -> LatentVFETerms:
        """Build a weighted latent-VFE objective used in Layer-2 inference updates."""
        recon = self.reconstruction_nll(recon_x, observed_x)
        prior_kl = self.latent_bernoulli_kl(z, prior)
        prior_term = float(np.clip(precision, 0.0, 1.0)) * prior_kl

        sensory_term = torch.zeros((), device=z.device, dtype=z.dtype)
        if sensory_target is not None:
            sensory_target = torch.clamp(sensory_target, self.eps, 1.0 - self.eps)
            sensory_term = F.mse_loss(z, sensory_target, reduction='mean')

        temporal_term = torch.zeros((), device=z.device, dtype=z.dtype)
        if temporal_target is not None:
            temporal_target = torch.clamp(temporal_target, self.eps, 1.0 - self.eps)
            temporal_term = F.mse_loss(z, temporal_target, reduction='mean')

        total = (
            (float(obs_weight) * recon)
            + (float(prior_weight) * prior_term)
            + (float(sensory_weight) * sensory_term)
            + (float(temporal_weight) * temporal_term)
        )
        return LatentVFETerms(
            reconstruction=recon,
            prior_kl=prior_term,
            sensory_consistency=sensory_term,
            temporal_consistency=temporal_term,
            total=total,
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

    def _efe_from_pred_pref(self, pred: Dict[str, float], pref: Dict[str, float]) -> EFETerms:
        eps = 1e-8
        risk = 0.0
        for state, p_s in pred.items():
            p_s = max(eps, float(p_s))
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

    @staticmethod
    def _posterior_from_energies(efe_by_policy: Dict[str, float], policy_temperature: float) -> Dict[str, float]:
        temp = float(max(1e-6, policy_temperature))
        policies = list(efe_by_policy.keys())
        if not policies:
            return {}
        energies = np.array([efe_by_policy[p] for p in policies], dtype=np.float64)
        logits = -energies / temp
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / max(np.sum(probs), 1e-12)
        return {policies[i]: float(probs[i]) for i in range(len(policies))}

    def _infer_policies_one_step(
        self,
        current_state: str,
        transition_pressure: float,
        policy_temperature: float,
    ) -> PolicyInferenceResult:
        base = self._base_distribution(current_state)
        if not base:
            return PolicyInferenceResult(
                posterior={},
                efe_by_policy={},
                risk_by_policy={},
                ambiguity_by_policy={},
                expected_risk=0.0,
                expected_ambiguity=0.0,
                expected_total=0.0,
                selected_policy=current_state,
                selected_confidence=1.0,
                posterior_entropy=0.0,
            )

        candidates = list(base.keys())
        efe_by_policy: Dict[str, float] = {}
        risk_by_policy: Dict[str, float] = {}
        ambiguity_by_policy: Dict[str, float] = {}

        for candidate in candidates:
            pref = {s: (1.0 if s == candidate else 0.0) for s in candidates}
            pred = self._predicted_distribution(base, pref, transition_pressure)
            terms = self._efe_from_pred_pref(pred, pref)
            efe_by_policy[candidate] = float(terms.total)
            risk_by_policy[candidate] = float(terms.risk)
            ambiguity_by_policy[candidate] = float(terms.ambiguity)

        posterior = self._posterior_from_energies(efe_by_policy, policy_temperature)
        selected_policy = max(posterior, key=posterior.get)
        selected_confidence = float(posterior[selected_policy])
        posterior_entropy = float(-sum(
            p * np.log(max(p, 1e-12)) for p in posterior.values()
        ))
        expected_risk = float(sum(posterior[p] * risk_by_policy[p] for p in posterior))
        expected_ambiguity = float(sum(posterior[p] * ambiguity_by_policy[p] for p in posterior))
        expected_total = float(sum(posterior[p] * efe_by_policy[p] for p in posterior))

        return PolicyInferenceResult(
            posterior=posterior,
            efe_by_policy=efe_by_policy,
            risk_by_policy=risk_by_policy,
            ambiguity_by_policy=ambiguity_by_policy,
            expected_risk=expected_risk,
            expected_ambiguity=expected_ambiguity,
            expected_total=expected_total,
            selected_policy=selected_policy,
            selected_confidence=selected_confidence,
            posterior_entropy=posterior_entropy,
        )

    def _expected_future_total(
        self,
        start_state: str,
        remaining_horizon: int,
        policy_temperature: float,
        policy_horizon_discount: float,
        cache: Dict[Tuple[str, int], float],
    ) -> float:
        if remaining_horizon <= 0:
            return 0.0
        key = (start_state, remaining_horizon)
        if key in cache:
            return cache[key]

        one_step = self._infer_policies_one_step(
            current_state=start_state,
            transition_pressure=0.0,
            policy_temperature=policy_temperature,
        )
        immediate = float(one_step.expected_total)
        if remaining_horizon == 1 or not one_step.posterior:
            cache[key] = immediate
            return immediate

        continuation = 0.0
        for next_state, prob in one_step.posterior.items():
            continuation += float(prob) * self._expected_future_total(
                start_state=next_state,
                remaining_horizon=remaining_horizon - 1,
                policy_temperature=policy_temperature,
                policy_horizon_discount=policy_horizon_discount,
                cache=cache,
            )

        total = immediate + (float(policy_horizon_discount) * continuation)
        cache[key] = total
        return total

    def infer_policies(
        self,
        current_state: str,
        transition_pressure: float,
        policy_horizon: int = 1,
        policy_temperature: float = 1.0,
        policy_horizon_discount: float = 0.6,
    ) -> PolicyInferenceResult:
        """Infer q(pi) over candidate next-state policies with optional short horizon."""
        horizon = max(1, int(policy_horizon))
        discount = float(np.clip(policy_horizon_discount, 0.0, 1.0))
        one_step = self._infer_policies_one_step(
            current_state=current_state,
            transition_pressure=transition_pressure,
            policy_temperature=policy_temperature,
        )
        if horizon == 1 or not one_step.posterior:
            return one_step

        cache: Dict[Tuple[str, int], float] = {}
        adjusted_efe = {}
        for policy_state, efe_value in one_step.efe_by_policy.items():
            future = self._expected_future_total(
                start_state=policy_state,
                remaining_horizon=horizon - 1,
                policy_temperature=policy_temperature,
                policy_horizon_discount=discount,
                cache=cache,
            )
            adjusted_efe[policy_state] = float(efe_value + (discount * future))

        posterior = self._posterior_from_energies(adjusted_efe, policy_temperature)
        selected_policy = max(posterior, key=posterior.get)
        selected_confidence = float(posterior[selected_policy])
        posterior_entropy = float(-sum(
            p * np.log(max(p, 1e-12)) for p in posterior.values()
        ))

        expected_risk = float(sum(
            posterior[p] * one_step.risk_by_policy.get(p, 0.0)
            for p in posterior
        ))
        expected_ambiguity = float(sum(
            posterior[p] * one_step.ambiguity_by_policy.get(p, 0.0)
            for p in posterior
        ))
        expected_total = float(sum(posterior[p] * adjusted_efe.get(p, 0.0) for p in posterior))

        return PolicyInferenceResult(
            posterior=posterior,
            efe_by_policy=adjusted_efe,
            risk_by_policy=one_step.risk_by_policy,
            ambiguity_by_policy=one_step.ambiguity_by_policy,
            expected_risk=expected_risk,
            expected_ambiguity=expected_ambiguity,
            expected_total=expected_total,
            selected_policy=selected_policy,
            selected_confidence=selected_confidence,
            posterior_entropy=posterior_entropy,
        )

    def compute_efe_terms(self, current_state: str, transition_pressure: float) -> EFETerms:
        result = self.infer_policies(current_state, transition_pressure)
        return EFETerms(
            risk=float(result.expected_risk),
            ambiguity=float(result.expected_ambiguity),
            total=float(result.expected_total),
        )
