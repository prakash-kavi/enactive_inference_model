# Active Inference Framework for Meditation State Modeling

## Overview

This document provides a comprehensive reference for the **Differentiable Hierarchical Engine (v3.1)**, a control-theoretic Active Inference model of meditation dynamics. It combines:
- **Rigorous Biological Priors** (v3.2: Systematically Derived Expert Profiles)
- **Guaranteed Stability** (Automatic Spectral Scaling via Gershgorin Disc Theorem)
- **Control Theoretic Inference** (Continuous Top-Down Steering + Precision Weighting)
- **Deep Learning Optimization** (Contrastive Regularization for "Parsimony")

---

# Part I: Empirical Foundation (v3.1)

## 1. Network Activation Profiles

These profiles define the "Biological Ground Truth" used in the Generative Process (Layer 1). verify
**Note (v3.2 Update)**: Expert profiles are no longer static text files. They are **Systematically Derived** from the Novice baseline using functional transformations (e.g., `Expert_Theta = Novice_Theta + Inhibition_Delta`). This ensures expertise is modeled as a *structural optimization* of the same underlying biology.

### Breath Focus (Tonic Maintenance)

| State | Profile | DMN | VAN | DAN | FPN |
|-------|---------|-----|-----|-----|-----|
| Breath Focus | Expert | 0.40 | 0.45 | 0.60 | 0.65 |
| Breath Focus | Novice | 0.55 | 0.50 | 0.50 | 0.55 |

**Dynamics:**
- **Expert**: Efficient maintenance using **DAN** (0.60) and **FPN** (0.65). DMN is suppressed (0.40) but not zero. VAN is low (0.45) as errors are rare.
- **Novice**: Higher DMN (0.55) reflecting lower stability. Lower DAN (0.50) indicates weaker focus.

### Mind Wandering (Default Mode)

| State | Profile | DMN | VAN | DAN | FPN |
|-------|---------|-----|-----|-----|-----|
| Mind Wandering | Expert | 0.65 | 0.50 | 0.40 | 0.50 |
| Mind Wandering | Novice | 0.75 | 0.40 | 0.35 | 0.40 |

**Dynamics:**
- **Universal**: DMN is the dominant network.
- **Expert**: Less deep wandering (0.65 vs 0.75) and higher "background monitoring" (VAN 0.50 vs 0.40).

### Meta-Awareness (Detection of Mind Wandering)

| State | Profile | DMN | VAN | DAN | FPN |
|-------|---------|-----|-----|-----|-----|
| Meta-Awareness | Expert | 0.40 | 0.70 | 0.45 | 0.50 |
| Meta-Awareness | Novice | 0.50 | 0.70 | 0.40 | 0.45 |

**Dynamics:**
- **Salience Spike**: VAN peaks (0.70) in both groups to signal "Error/Lapse".
- **Transition**: DMN begins dropping. FPN begins rising for evaluation.

### Redirect Breath (Phasic Control)

| State | Profile | DMN | VAN | DAN | FPN |
|-------|---------|-----|-----|-----|-----|
| Redirect | Expert | 0.35 | 0.55 | 0.70 | 0.75 |
| Redirect | Novice | 0.45 | 0.50 | 0.65 | 0.70 |

**Dynamics:**
- **Control Burst**: DAN and FPN peak here (higher than in Breath Focus). This is the active "re-orienting" effort.
- **Expert Efficiency**: Sharper control burst (DAN 0.70/FPN 0.75) compared to Novice.

---

# Part II: Control Theoretic Architecture (v3)

## 2. Continuous Top-Down Control

Unlike standard predictive coding which is often passive, this engine uses **Active Control**.

1.  **Agent Bias ($\beta$)**: Layer 2 (Agent) calculates a continuous control vector by projecting its intent ($\mu_{target}$) into the network space via the Generative Model ($W$).
    $$ \beta = \mu_{target} \cdot W $$
2.  **Somatic Compliance ($\alpha$)**: Layer 1 (Biology) receives this bias and blends it into its drift dynamics based on the agent's expertise.
    $$ \mu_{effective} = (1 - \alpha)\mu_{bio} + \alpha \beta $$
    - **Expert ($\alpha \approx 0.6$)**: Strong somatic compliance. The body obeys the mind.
    - **Novice ($\alpha \approx 0.1$)**: Weak compliance. The body follows biological defaults.

## 3. Automatic Spectral Scaling (Stability Guarantee)

To prevent "Runaway Excitation" (DMN $\to$ 1.0) or "Attractor Collapse", Layer 1 implements **Automatic Spectral Scaling**.
- **Transformation**: The connectivity matrix $\Theta$ is dynamically renormalized at every timestep.
- **Logic**: It enforces $\sum |OffDiagonal| < Diagonal$.
- **Result**: We can set *any* theoretical coupling strengths (e.g., +500% Inhibition), and the physics engine will "squash" them into the maximum allowable stable region without breaking the sign/direction of the interaction. This guarantees mathematical stability for any parameter set.

## 3. Precision-Weighted Kalman Updates

The Agent maintains its internal state ($z$) using a mechanism analogous to a Kalman Filter.

$$ \mu_{posterior} = K \cdot \mu_{likelihood} + (1 - K) \cdot \mu_{prior} $$

Where $K$ (Kalman Gain) is derived from **Meta-Awareness**:
- **High Meta-Awareness** $\rightarrow$ High Prior Precision $\rightarrow$ Agent sticks to goal (ignores sensory noise).
- **Low Meta-Awareness** $\rightarrow$ Low Prior Precision $\rightarrow$ Agent biased by bottom-up sensation (more easily distracted).

### 3.1 Stochastic VFE Generation (The "Spark")
To prevent stagnation in highly stable attractors (Mind Wandering), Layer 2 implements **Stochastic Monitoring**.
- Random "probes" of awareness accumulate during distraction.
- This creates a conflict: Agent expects "Aha" (Wake up), Biology says "Stuck" (DMN).
- Result: **High VFE**. This acts as the "Error Signal" that triggers the transition out of the sticky attractor.

---

# Part III: Implementation

## 4. Russian Doll Architecture

replace with 

## 5. Contrastive Regularization ("Parsimony Penalty")

To ensure the Agent learns meaningful logical structures rather than just "averaging" the noisy biology, we implement a **Contrastive Aux Loss** during training.

$$ Loss_{total} = VFE + \lambda \cdot \text{ReLU}(\text{Gap}_{target} - (\text{DAN} - \text{DMN})) $$

This forces the Agent to learn internal models ($\theta, \mu$) that exhibit a strong separation between Focused (DAN) and Wandering (DMN) states, even if the novice training data is mushy. This "hallucinated ideal" is then used to steer the biology in the correct direction.

---

*Last Updated: January 2026*
*Framework Version: 3.1 (Control Theoretic Upgrade + Biological Tuning)*
