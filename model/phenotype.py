"""Phenotype configuration for expert/novice meditation agents.

All phenotype-specific constants live here. The rest of the model code
is phenotype-agnostic — it reads fields from PhenotypeConfig rather than
branching on a string label.
"""

from dataclasses import dataclass
from utils.config import LEARNING_RATES, EPS

# Fixed policy precision (Eq. 7). Kept here for centralized tuning.
POLICY_GAMMA = 1.0


@dataclass(frozen=True)
class PhenotypeConfig:
    """Single source of truth for all per-phenotype parameters.

    Fields
    ------
    level : str
        Identifier used to index config tables (DWELL_TIMES, NETWORK_PROFILES,
        STATE_TRANSITION_PROBS). One of 'expert' | 'novice'.
    learning_rate : float
        Adam optimiser step size for L2 VAE parameters.
    alpha_rec : float
        Recognition-loss weight in the composite loss (Eq. 9).
        Expert=1.0 (full amortised inference); novice is proportionally weaker.
    theta_boost : bool
        Whether to apply expert-specific Theta(s) adjustments in Layer 1
        (stronger self-stabilisation in BF, amplified DMN-DAN inhibition in RA,
        VAN-FPN synergy in MA).
    label : str
        Human-readable label used in console output and saved result files.
    """
    level:         str
    learning_rate: float
    alpha_rec:     float
    theta_boost:   bool
    label:         str


# ---------------------------------------------------------------------------
# Pre-built configs — import these rather than constructing ad-hoc strings
# ---------------------------------------------------------------------------

EXPERT_PHENOTYPE = PhenotypeConfig(
    level='expert',
    learning_rate=LEARNING_RATES['expert'],
    alpha_rec=1.0,
    theta_boost=True,
    label='EXPERT',
)

NOVICE_PHENOTYPE = PhenotypeConfig(
    level='novice',
    learning_rate=LEARNING_RATES['novice'],
    alpha_rec=LEARNING_RATES['novice'] / max(LEARNING_RATES['expert'], EPS),
    theta_boost=False,
    label='NOVICE',
)

# ---------------------------------------------------------------------------

def phenotype_from_str(level: str) -> PhenotypeConfig:
    """Convenience: resolve a level string to the matching PhenotypeConfig."""
    if level == 'expert':
        return EXPERT_PHENOTYPE
    if level == 'novice':
        return NOVICE_PHENOTYPE
    raise ValueError(f"Unknown phenotype level '{level}'. Use 'expert' or 'novice'.")
