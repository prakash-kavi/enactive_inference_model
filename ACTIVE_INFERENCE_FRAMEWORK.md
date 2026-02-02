# Active Inference Framework for Meditation State Modeling

## Overview

This document provides a reference for the **Differentiable Hierarchical Engine **, a control-theoretic Active Inference model of meditation dynamics. It combines:
- **Biological Priors** (state-specific network profiles)
- **Stability Guarantees** (spectral scaling of the OU dynamics)
- **Active Inference Control** (policy modulation + precision weighting)
- **Hierarchical Message Passing** (L1 dynamics, L2 beliefs, L3 policy)

---

# Part I: Empirical Foundation (v3.1)

## 1. Network Activation Profiles

These profiles define the "Biological Ground Truth" used in the Generative Process (Layer 1).
They are configured in `utils/meditation_config.py` and used as state-specific attractors.

### Breath Focus (Tonic Maintenance)

| State | Profile | DMN | VAN | DAN | FPN |
|-------|---------|-----|-----|-----|-----|
| Breath Focus | Expert | 0.40 | 0.45 | 0.60 | 0.65 |
| Breath Focus | Novice | 0.50 | 0.45 | 0.58 | 0.60 |

**Dynamics:**
- **Expert**: Efficient maintenance using **DAN** (0.60) and **FPN** (0.65). DMN is suppressed (0.40) but not zero. VAN is low (0.45) as errors are rare.
- **Novice**: Higher DMN (0.55) reflecting lower stability. Lower DAN (0.50) indicates weaker focus.

### Mind Wandering (Default Mode)

| State | Profile | DMN | VAN | DAN | FPN |
|-------|---------|-----|-----|-----|-----|
| Mind Wandering | Expert | 0.65 | 0.50 | 0.40 | 0.50 |
| Mind Wandering | Novice | 0.82 | 0.35 | 0.30 | 0.33 |

**Dynamics:**
- **Universal**: DMN is the dominant network.
- **Expert**: Less deep wandering (0.65 vs 0.75) and higher "background monitoring" (VAN 0.50 vs 0.40).

### Meta-Awareness (Detection of Mind Wandering)

| State | Profile | DMN | VAN | DAN | FPN |
|-------|---------|-----|-----|-----|-----|
| Meta-Awareness | Expert | 0.40 | 0.78 | 0.42 | 0.55 |
| Meta-Awareness | Novice | 0.45 | 0.85 | 0.42 | 0.56 |

**Dynamics:**
- **Salience Spike**: VAN peaks (0.70) in both groups to signal "Error/Lapse".
- **Transition**: DMN begins dropping. FPN begins rising for evaluation.

### Redirect Attention (Phasic Control)

| State | Profile | DMN | VAN | DAN | FPN |
|-------|---------|-----|-----|-----|-----|
| Redirect | Expert | 0.35 | 0.50 | 0.78 | 0.70 |
| Redirect | Novice | 0.40 | 0.50 | 0.78 | 0.74 |

**Dynamics:**
- **Control Burst**: DAN and FPN peak here (higher than in Breath Focus). This is the active "re-orienting" effort.
- **Expert Efficiency**: Sharper control burst (DAN 0.70/FPN 0.75) compared to Novice.

---

# Part II: Control Theoretic Architecture (v3)

## 2. Top-Down Modulation Signals

Unlike passive predictive coding, this engine uses **policy modulation** signals:

1) **Agent bias**: L2 projects its state-conditioned intent into network space and sends it to L1.
2) **Enactive bias (l2tol1_enactive_bias)**: L1 blends agent bias into the biological drift.
3) **Noise reduction**: L3 modulates L1 diffusion based on opacity/precision.
4) **Transition drive**: L3 biases the state machine toward transitions when policy pressure is high.

## 3. Automatic Spectral Scaling (Stability Guarantee)

To prevent "Runaway Excitation" (DMN $\to$ 1.0) or "Attractor Collapse", Layer 1 implements **Automatic Spectral Scaling**.
- **Transformation**: The connectivity matrix $\Theta$ is dynamically renormalized at every timestep.
- **Logic**: It enforces $\sum |OffDiagonal| < Diagonal$.
- **Result**: We can set *any* theoretical coupling strengths (e.g., +500% Inhibition), and the physics engine will "squash" them into the maximum allowable stable region without breaking the sign/direction of the interaction. This guarantees mathematical stability for any parameter set.

## 4. Precision-Weighted Belief Updates

The Agent maintains its internal state ($z$) using a mechanism analogous to a Kalman Filter.

$$ \mu_{posterior} = K \cdot \mu_{likelihood} + (1 - K) \cdot \mu_{prior} $$

Where $K$ (Kalman Gain) is derived from **Meta-Awareness**:
- **High Meta-Awareness** -> High prior precision -> agent sticks to intent.
- **Low Meta-Awareness** -> Low prior precision -> agent more driven by bottom-up inputs.

### 4.1 Meta-Awareness and Opacity
Meta-awareness is computed from thoughtseed activations using state-specific weights.
Opacity is defined as `1 - meta_awareness` and gates policy precision in L3.

---

# Part III: Implementation

## 5. Policy Objective (Expected Free Energy)
Policy selection uses **Expected Free Energy (EFE)**:
- **Risk** term uses a cycle-aligned preference (BF -> MW -> MA -> RB -> BF).
- **Ambiguity** term penalizes uncertain observations.
EFE modulates transition_drive but does not backprop into L1 dynamics.

## 6. Naming Map (Code -> Act-Inference Term)
- `free_energy` -> Variational Free Energy (VFE)
- `efe_value` -> Expected Free Energy (policy objective)
- `transition_drive` -> Policy pressure on state transitions
- `precision_modulation` -> Precision weighting of beliefs
- `noise_reduction` -> Sensory noise attenuation
- `l2tol1_enactive_bias` -> Enactive bias (policy-driven drift bias)

---

*Last Updated: January 2026*
*Framework Version: 3.1 (Control Theoretic Upgrade + Biological Tuning)*
