# Active Inference Framework for Meditation State Modeling

## Overview

This document provides a comprehensive reference for the Active Inference meditation model, combining:
- **Empirical neuroscience foundation** (network profiles, coupling patterns, APA citations)
- **Current technical implementation** (MVOU dynamics, 2-file architecture with Markov Blanket)
- **Future architectural evolution** (Russian Doll nesting proposal)

The framework is based on empirical neuroimaging studies and provides ground-truth reference for algorithm development, state detection, and neurofeedback applications.

---

# Part I: Empirical Foundation

## 1. Network Activation Profiles

These values (scaled 0.0 to 1.0) reflect the "Neural Efficiency" hypothesis, where experts show lower executive effort (FPN) during sustained attention while maintaining superior monitoring capabilities.

**Note:** The activation values (0.0–1.0) and correlation coefficients specified in this document are heuristic priors chosen to reproduce relative expert–novice and state differences reported in the literature, rather than direct empirical estimates from any single study (Brewer et al., 2011; Hasenkamp et al., 2012).

### Breath Focus (Sustained Attention)

| State | Profile | DMN | VAN | DAN | FPN |
|-------|---------|-----|-----|-----|-----|
| Breath Focus | Expert | 0.30 | 0.45 | 0.60 | 0.45 |
| Breath Focus | Novice | 0.55 | 0.50 | 0.60 | 0.65 |

**Key Distinguishing Pattern:**
- **Expert Efficiency**: Lower DMN (0.30) and FPN (0.45) compared to novices reflects reduced DMN activity during meditation and attentional tasks in experienced meditators (Brewer et al., 2011; Hasenkamp et al., 2012; Taren et al., 2015).
- **Similar DAN engagement** (0.60) in both groups is consistent with dorsal attention engagement during FA meditation and attention tasks (Hasenkamp et al., 2012; Tello et al., 2023).
- **Higher Novice FPN** (0.65 vs 0.45 in experts) reflects greater executive effort, compatible with "neural efficiency" ideas and findings that long-term practice is associated with more efficient prefrontal recruitment rather than maximal activation (Brefczynski-Lewis et al., 2007; Kral et al., 2022; Lin et al., 2020).

### Mind Wandering

| State | Profile | DMN | VAN | DAN | FPN |
|-------|---------|-----|-----|-----|-----|
| Mind Wandering | Expert | 0.60 | 0.65 | 0.40 | 0.65 |
| Mind Wandering | Novice | 0.75 | 0.40 | 0.35 | 0.40 |

**Key Distinguishing Pattern:**
- **Higher DMN during mind wandering** in both groups, with higher Novice DMN (0.75) reflects robust DMN recruitment during spontaneous thought and reduced DMN activity in experienced meditators even at rest (Brewer et al., 2011; Hasenkamp et al., 2012).
- **Higher Expert VAN/FPN** (0.65 each) during mind wandering models "background monitoring" and stronger coupling between DMN and frontoparietal/salience networks reported in mindfulness interventions and experienced meditators (Kilpatrick et al., 2011; Taren et al., 2015; Lin et al., 2020).

### Meta-Awareness ("Aha" Moment)

| State | Profile | DMN | VAN | DAN | FPN |
|-------|---------|-----|-----|-----|-----|
| Meta-Awareness | Expert | 0.40 | 0.80 | 0.50 | 0.70 |
| Meta-Awareness | Novice | 0.50 | 0.60 | 0.50 | 0.55 |

**Key Distinguishing Pattern:**
- **High Expert VAN** (0.80) and **FPN** (0.70) capture the salience-driven "pop-out" of awareness plus executive control during the awareness interval in FA meditation (Hasenkamp et al., 2012).
- **DMN remains moderate** (0.40-0.50) rather than fully suppressed, reflecting that content is still present but being re-evaluated, consistent with awareness intervals where DMN has not yet dropped to focus levels.
- **Larger VAN jump in experts** vs novices is compatible with faster error/performance monitoring and more efficient salience processing after long-term practice (Brefczynski-Lewis et al., 2007; Kral et al., 2022).

### Redirect (Shifting Back to Breath)

| State | Profile | DMN | VAN | DAN | FPN |
|-------|---------|-----|-----|-----|-----|
| Redirect | Expert | 0.35 | 0.55 | 0.65 | 0.55 |
| Redirect | Novice | 0.45 | 0.50 | 0.65 | 0.70 |

**Key Distinguishing Pattern:**
- **Higher DAN in both profiles** (0.65) reflects the shifting interval's reliance on dorsal attention and control to reorient to the object (Hasenkamp et al., 2012).
- **Novice FPN > Expert FPN** (0.70 vs 0.55) captures more effortful reorienting for novices, consistent with attentional expertise results where experts perform better with less "cost" in control regions in some tasks (Brefczynski-Lewis et al., 2007; Kral et al., 2022).
- **DMN is reduced but not zero** (0.35-0.45) because some residual self-related content is still present during the shift, in line with the temporal progression from mind wandering to focus rather than an instantaneous switch (Hasenkamp et al., 2012).

---

## 2. Functional Coupling & (Anti)Correlations

For algorithm development, these correlations define the temporal dependencies between networks and should be implemented as state-specific lateral coupling mechanisms.

### A. The Focus & Redirect States (Task-Positive Dominance)

**Expert Focus:**
- **Strong DMN–DAN anti-correlation** (≈ -0.70) and **DMN–FPN anti-correlation** (≈ -0.60) reflect deep suppression of self-referential thought (Hasenkamp et al., 2012; Taren et al., 2015).

**Redirect Coupling:**
- The **DAN–FPN coupling peaks** here (≈ 0.70 for Experts, 0.60 for Novices), as these networks coordinate to shift attention back to the anchor (Hasenkamp et al., 2012; Tello et al., 2023).

### B. The Mind Wandering State (DMN Dominance)

**Novice Signature:**
- Strong anti-correlation between DMN and all task-positive networks (DAN/FPN ≈ -0.60), signifying the novice is "lost" in thought with no monitoring (Brewer et al., 2011; Hasenkamp et al., 2012).

**Expert Signature:**
- A unique **positive DMN–FPN/VAN correlation** (≈ 0.20 to 0.30). This models the "background monitoring" that allows experts to recognize wandering more quickly (Kilpatrick et al., 2011; Taren et al., 2015; Lin et al., 2020).

### C. The Meta-Awareness State (The Switch)

**Peak Coupling:**
- The **FPN–VAN coupling is at its highest** in this state (0.70 for Experts, 0.50 for Novices), representing the Salience network (VAN) handing off control to the Executive network (FPN) (Hasenkamp et al., 2012; Hasenkamp & Barsalou, 2012).

---

## 3. Key Patterns for Algorithm Development

### A. DMN Suppression Delta

The most reliable differentiator of expertise is the degree of DMN suppression during Breath Focus (Experts ≈ 0.30 vs. Novices ≈ 0.55). This metric can be used for:
- Expertise classification
- Training progress tracking
- State detection algorithms

**Implementation:** Compare DMN activation levels during `breath_focus` state between expert and novice profiles.

### B. The VAN Spike

An algorithm can detect the transition from Mind Wandering to Redirect by looking for:
- A sharp VAN activation peak
- Combined with an FPN-VAN coupling increase

**Implementation:** Monitor VAN accumulator for spikes above threshold (e.g., 0.7) combined with increased FPN-VAN correlation during state transitions.

### C. Neural Efficiency Ratio

The ratio of DAN/FPN during focus is a metric for expertise:
- **Higher ratio** (stronger DAN, lower FPN) suggests "effortless" concentration in experts
- **Lower ratio** (high FPN) suggests "effortful" focus in novices

**Formula:** `neural_efficiency_ratio = DAN_activation / FPN_activation` during `breath_focus` state.

**Expected Values:**
- Expert: 0.60 / 0.45 ≈ 1.33
- Novice: 0.60 / 0.65 ≈ 0.92

### D. Meta-Cognitive Monitoring

Positive correlation between DMN and FPN during high-DMN periods (Mind Wandering) is a strong indicator of a trained, expert brain state.

**Implementation:** 
- Compute DMN-FPN correlation during `mind_wandering` state
- Expert should show positive correlation (≈ 0.20-0.30)
- Novice should show negative correlation (≈ -0.60)

---

# Part II: Current Implementation (MVOU Architecture)

## 4. Architecture: Current 2-File Structure

### Current Implementation Status

**✅ Implemented:**
- Layer 1: `generative_process.py` - MVOU dynamics with state-specific Θ matrices
- Layers 2 & 3: `meditation_model.py` - Combined Generative Model and Recognition/Inference
- Markov Blanket: Nested inside `ActInfAgent` (agent owns its statistical boundary)
- Bidirectional flow: Perception (bottom-up) + Action (top-down via Markov Blanket)

### Current Structure

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: GENERATIVE PROCESS (generative_process.py)          │
│ - MVOU dynamics: dx = -Θ(x - μ)dt + ΣdW                      │
│ - State-specific Θ matrices (coupling embedded)               │
│ - Produces "observed" network signals x(t)                    │
│ - Accepts active_states from Markov Blanket (downward causation)│
└─────────────────────────────────────────────────────────────┘
                          ↓ (observations)
┌─────────────────────────────────────────────────────────────┐
│ Markov Blanket (markov_blanket.py)                           │
│ - Sensory States: Observations from Layer 1                    │
│ - Active States: Prescriptions from Agent (noise_reduction,   │
│   dwell_modifier, fatigue_buffer)                             │
└─────────────────────────────────────────────────────────────┘
                          ↓ (sensory)        ↑ (active)
┌─────────────────────────────────────────────────────────────┐
│ Layers 2 & 3: ACTINF AGENT (meditation_model.py)              │
│ - Layer 2: Generative Model (W matrix, predictions)          │
│ - Layer 3: Recognition/Inference (VFE, perceptual_inference) │
│ - prescriptive_action: Selects policies, updates blanket     │
└─────────────────────────────────────────────────────────────┘
```

**Key Principle:** Generative Process ≠ Generative Model
- **Process** (Layer 1): "Real world" that produces observations (can be modulated via active_states)
- **Model** (Layer 2): Agent's predictions (can be wrong, can be adjusted)
- **Action**: Agent adjusts internal states and prescribes modulations via Markov Blanket

---

## 5. MVOU Mathematical Formulation

### The Stochastic Differential Equation

The Generative Process evolves according to:

```
d𝐱_t = -Θ(𝐱_t - μ(t))dt + ΣdW_t
```

**Discretized (Euler-Maruyama with sub-stepping):**
```
𝐱_{t+1} = 𝐱_t - Θ(𝐱_t - μ(t))dt + Σ·ε_t·√dt
```

Where:
- **𝐱_t** = [DMN, VAN, DAN, FPN]^T (4×1 state vector)
- **μ(t)** = Mean targets (empirical profiles for current state)
- **Θ** = 4×4 drift matrix (coupling embedded in off-diagonals)
- **Σ** = 4×4 noise covariance matrix (Cholesky decomposition)
- **dW_t** = Wiener process (Gaussian noise)
- **dt** = 0.5 (sub-stepping: 2 substeps per 1.0 second timestep)

### Drift Matrix Θ Structure

For each state and experience level:

```
Θ = [
    [θ_DMN,DMN,  θ_DMN,VAN,  θ_DMN,DAN,  θ_DMN,FPN],
    [θ_VAN,DMN,  θ_VAN,VAN,  θ_VAN,DAN,  θ_VAN,FPN],
    [θ_DAN,DMN,  θ_DAN,VAN,  θ_DAN,DAN,  θ_DAN,FPN],
    [θ_FPN,DMN,  θ_FPN,VAN,  θ_FPN,DAN,  θ_FPN,FPN]
]
```

**Interpretation:**
- **Diagonal elements** (θ_ii): Mean reversion strength for network i
  - Expert: `θ_ii = 0.25` (memory_factor = 0.75)
  - Novice: `θ_ii = 0.15` (memory_factor = 0.85)
- **Off-diagonal elements** (θ_ij): Coupling strength from network j to network i
  - Negative coupling (anti-correlation) → positive Θ off-diagonal
  - Positive coupling → negative Θ off-diagonal

**Stability:** The system is stable if Θ is positive-definite (all eigenvalues positive). Gershgorin Disk Theorem is used to enforce stability with ε = 0.01.

### Mapping Coupling Patterns to Θ

**Convention:**
- **Anti-Correlation (Negative Coupling):**
  - `DMN_DAN: -0.70` → `Θ[0,2] = 0.7` and `Θ[2,0] = 0.7` (symmetric)
- **Positive Coupling:**
  - `DMN_VAN: 0.20` → `Θ[0,1] = -0.2` and `Θ[1,0] = -0.2` (pulls together)

**Example: Expert Breath Focus**

**Coupling Pattern:**
- Strong DMN-DAN anti-correlation (-0.70)
- Strong DMN-FPN anti-correlation (-0.60)
- Moderate DAN-FPN coupling (0.50)

**Resulting Θ Matrix:**
```
Θ = [
    [0.25,  0.00,  0.70,  0.60],  # DMN: pulled back by DAN(0.7), FPN(0.6)
    [0.00,  0.25,  0.00,  0.00],  # VAN: neutral
    [0.70,  0.00,  0.25, -0.50],  # DAN: competes with DMN(0.7), coupled with FPN(-0.5)
    [0.60,  0.00, -0.50,  0.25]   # FPN: competes with DMN(0.6), coupled with DAN(-0.5)
]
```

---

## 6. Mean Targets μ(t) and Noise

### Mean Targets

The mean vector μ(t) comes from empirical network profiles (Section 1):

```python
mu_data = {
    "breath_focus": {
        "expert": {"DMN": 0.30, "VAN": 0.45, "DAN": 0.60, "FPN": 0.45},
        "novice": {"DMN": 0.55, "VAN": 0.50, "DAN": 0.60, "FPN": 0.65}
    },
    "mind_wandering": {
        "expert": {"DMN": 0.60, "VAN": 0.65, "DAN": 0.40, "FPN": 0.65},
        "novice": {"DMN": 0.75, "VAN": 0.40, "DAN": 0.35, "FPN": 0.40}
    },
    # ... other states
}
```

These are the "ground truth" targets that the MVOU process converges toward.

### Noise Covariance Σ

Currently implemented with independent noise per network:

```python
# Expert: lower noise variance (0.001) for reduced jitter
# Novice: higher noise variance (0.002)
K = np.eye(4) * noise_variance
```

**Future Enhancement:** Add correlated noise via full covariance matrix:
```python
K = [
    [σ_DMN²,      ρ_DMN,VAN·σ_DMN·σ_VAN,  ...],
    [ρ_VAN,DMN,   σ_VAN²,                  ...],
    ...
]
```

**Cholesky Decomposition:**
```python
L = np.linalg.cholesky(K)  # K = L @ L^T
noise = L @ ε_t  # where ε_t ~ N(0, I)
```

---

## 7. State Transitions and Dwell Times

State transitions are handled by the Generative Process based on probabilistic Weibull dwell times:

**Canonical Meditative Cycle:** BF → MW → MA → RA → BF

**Dwell Time Ranges (Weibull distribution):**
- Expert: Longer sustained focus (BF: 15-30s), shorter wandering (MW: 10-20s), fast transitions (MA/RA: 1-4s)
- Novice: Shorter focus (BF: 5-15s), longer wandering (MW: 20-40s), slower transitions (MA/RA: 2-6s)

**Implementation:** `generative_process.py` uses Weibull distribution to sample probabilistic dwell times, adding natural jitter to state transitions.

---

## 8. Generative Model: Weight Matrix (W)

The agent's Generative Model uses a fixed 5×4 Weight Matrix (W) for "Scientific Matrix Form":

```python
# Rows: Thoughtseeds [attend_breath, pain_discomfort, pending_tasks, aha_moment, equanimity]
# Cols: Networks [DMN, VAN, DAN, FPN]
W = [
    [0.20, 0.30, 0.85, 0.50],  # attend_breath: Anchor focus (high DAN, moderate FPN)
    [0.40, 0.75, 0.30, 0.40],  # pain_discomfort: Sensory distraction (high VAN)
    [0.85, 0.40, 0.20, 0.30],  # pending_tasks: Internal narrative (high DMN)
    [0.30, 0.80, 0.40, 0.70],  # aha_moment: State recognition (high VAN, high FPN)
    [0.25, 0.30, 0.50, 0.90]   # equanimity: Executive regulation (very high FPN)
]
```

**Prediction:** `μ_pred = thoughtseed_activations @ W`

**Inference:** `inferred_thoughtseeds = network_observations @ W.T` (matrix-transpose projection)

---

## 9. Bidirectional Flow: Perception and Action

### Bottom-Up Flow (Perception)

1. **Layer 1 → Markov Blanket**: Observations `x(t)` from Generative Process
2. **Markov Blanket → Layer 2**: Sensory states updated
3. **Layer 2 → Layer 3**: Prediction errors `δ = x_observed - μ_predicted`
4. **Layer 3**: Computes VFE, performs `perceptual_inference()` (matrix-transpose projection)

### Top-Down Flow (Action)

1. **Layer 3 → Layer 2**: `prescriptive_action()` selects policies based on VFE:
   - **Aha! Short-Circuit**: High `aha_moment` → reduce noise, shorten dwell
   - **Attentional Sharpening**: High meta-awareness → reduce noise
   - **Equanimity Buffer**: High `equanimity` → reduce fatigue
   - **Precision Reset**: Meta-awareness state → aggressive noise reduction

2. **Layer 2 → Markov Blanket**: Prescriptions update `active_states` (noise_reduction, dwell_modifier, fatigue_buffer)

3. **Markov Blanket → Layer 1**: `active_states` modulate Generative Process:
   - `noise_reduction` → reduces effective noise variance
   - `dwell_modifier` → modifies MW dwell times
   - `fatigue_buffer` → modulates thoughtseed dynamics

**Key Principle:** The agent cannot directly control Layer 1, but can modulate it through the Markov Blanket's active states (downward causation).

---

## 10. Advantages of MVOU Approach

### Mathematical Rigor
- ✅ Unified dynamical system (not patchwork)
- ✅ Stability guarantees (Gershgorin Disk Theorem)
- ✅ Natural handling of coupling (no special cases)
- ✅ Correlated noise capability (biologically plausible)

### Solves Previous Problems
- ✅ **Runaway Problem**: Linear system with stability guarantees
- ✅ **Special Cases**: No need for additive vs multiplicative coupling
- ✅ **Coupling Scale**: Embedded in matrix values
- ✅ **Expert MW DMN**: Natural handling via matrix structure

### Code Simplicity
- ✅ Single matrix multiplication replaces 50+ lines of coupling code
- ✅ State-specific behavior = different Θ matrix (clean)
- ✅ No more `if "DMN_DAN" in coupling:` logic

---

# Part III: Future Architecture Evolution

## 11. Russian Doll Nesting Proposal

### Conceptual Framework: "Inside-Out" Nesting

The proposed architecture implements a "Biological-First" Russian Doll structure where each layer physically "envelopes" the next, reflecting Deep Computational Neurophenomenology (DCN):

```
Layer 1 (NeurobiologicalProcess): Outermost shell
  ↓ envelopes
Layer 2 (GNWBottleneck): Intermediate bridge
  ↓ envelopes
Layer 3 (PhenomenologicalMonitor): Central kernel
```

### Refined Layer Names and Roles

| Layer | Refined Name | Doll Position | Identity |
|-------|--------------|---------------|----------|
| Layer 1 | NeurobiologicalProcess | Outermost Shell | The high-dimensional "90% unconscious" biology |
| Layer 2 | GNWBottleneck | Intermediate Bridge | The intentional arena where thoughtseeds compete |
| Layer 3 | PhenomenologicalMonitor | Central Kernel | The meta-cognitive site of formal phenomenology |

### Structural Hierarchy: Inside-Out Nesting

**Proposed Class Structure:**

```python
class NeurobiologicalProcess:
    """Layer 1: The Outer Biological Envelope."""
    def __init__(self, experience_level='expert'):
        # Layer 1 'owns' the interface to the Umwelt
        self.blanket = MarkovBlanket()
        # Layer 1 'envelopes' Layer 2
        self.gnw_bottleneck = GNWBottleneck(experience_level)
        
        # MVOU Dynamics (unconscious 90%)
        self.x = np.array([0.5, 0.5, 0.5, 0.5])

class GNWBottleneck:
    """Layer 2: The Intentional Bridge."""
    def __init__(self, experience_level):
        # Layer 2 'envelopes' Layer 3
        self.monitor = PhenomenologicalMonitor(experience_level)
        # The Dominant Thoughtseed resides here (the 10% conscious)
        self.z = np.zeros(5) 

class PhenomenologicalMonitor:
    """Layer 3: The Central Kernel."""
    def __init__(self, experience_level):
        # This layer monitors L2 to evaluate Phenotypic Integrity
        self.policies = self._initialize_agent_policies()
        self.vfe_history = []
```

### Key Differences from Current Implementation

**Current:**
- `generative_process.py` (L1) is separate
- `meditation_model.py` (L2+L3) with `MarkovBlanket` nested inside `ActInfAgent`
- Agent "owns" its blanket

**Proposed:**
- `NeurobiologicalProcess` (L1) owns `MarkovBlanket` and envelopes `GNWBottleneck` (L2)
- `GNWBottleneck` (L2) envelopes `PhenomenologicalMonitor` (L3)
- L1 is the outermost shell, L3 is the innermost kernel

---

## 12. Standardized Bidirectional Flow (Proposed)

### Bottom-Up Emergence (Perception)

1. **Sensation**: The Umwelt interacts with the MarkovBlanket (Sensory States)
2. **Biological Realization**: NeurobiologicalProcess (L1) updates its network trajectories based on these inputs
3. **Coarse-Graining**: GNWBottleneck (L2) performs `perceptual_inference` to find the Dominant Thoughtseed
4. **Phenomenological Recognition**: The PhenomenologicalMonitor (L3) observes the Dominant Thoughtseed to update the agent's meta-awareness and VFE

### Top-Down Causation (Action)

1. **Policy Monitoring**: The Monitor (L3) evaluates its PolicyObject list (e.g., "Aha! Redirect")
2. **Precision Gating**: If a policy triggers, L3 prescribes a Precision Reset to the GNWBottleneck (L2)
3. **Downward Constraint**: L2 translates this into Active States (Modulations) in the blanket
4. **Biological Stabilization**: These modulations "radiate out" to the NeurobiologicalProcess (L1), physically suppressing noise and mind-wandering attractors

### The Non-Homuncular prescriptive_action

The action selection logic in the Monitor (L3) is a process of **Precision Weighting** rather than direct control:

```python
def prescriptive_action(self, z_dominant, vfe):
    """
    Layer 3: Monitoring the 'gap' between belief and biology.
    Behavior is generated by gating the precision of the GNW Bottleneck.
    """
    for policy in self.policies:
        if policy.evaluate(z_dominant, vfe):
            # The Monitor 'nudges' the system by adjusting precision
            # This radiates outward to stabilize the biological core
            self.parent_model.apply_precision_gating(policy.prescription)
            break
```

**Key Insight:** The Monitor doesn't "act" on the world; it adjusts precision. This "gating" radiates back out through the bottleneck to stabilize the biological core, avoiding the homunculus problem.

---

## 13. Verification of Phenomenological Integrity

The simulation is successful when the PhenomenologicalMonitor achieves **Neurophenomenological Alignment**—where the meta-cognitive "Monitor" correctly identifies a distraction in the GNWBottleneck and successfully stabilizes the NeurobiologicalProcess before the biological mind-wandering state matures.

**Success Criteria:**
- Monitor detects high VFE (mismatch between belief and biology)
- Monitor triggers appropriate policy (e.g., "Aha! Redirect")
- Precision gating radiates outward through L2 → L1
- Biological core stabilizes before MW attractor matures
- VFE decreases, indicating successful alignment

---

# Part IV: References

## 14. Primary Neuroimaging Studies

**Brewer, J. A., Worhunsky, P. D., Gray, J. R., Tang, Y. Y., Weber, J., & Kober, H.** (2011). Meditation experience is associated with differences in default mode network activity and connectivity. *Proceedings of the National Academy of Sciences*, 108(50), 20254-20259. https://doi.org/10.1073/pnas.1112029108

**Hasenkamp, W., Wilson-Mendenhall, C. D., Duncan, E., & Barsalou, L. W.** (2012). Mind wandering and attention during focused meditation: A fine-grained temporal analysis of fluctuating cognitive states. *NeuroImage*, 59(1), 750-760. https://doi.org/10.1016/j.neuroimage.2011.07.008

**Hasenkamp, W., & Barsalou, L. W.** (2012). Effects of meditation experience on functional connectivity of distributed brain networks. *Frontiers in Human Neuroscience*, 6, 38. https://doi.org/10.3389/fnhum.2012.00038

**Taren, A. A., Creswell, J. D., & Gianaros, P. J.** (2015). Mindfulness meditation training alters stress-related amygdala resting state functional connectivity. *Social Cognitive and Affective Neuroscience*, 10(12), 1758-1768. https://doi.org/10.1093/scan/nsv066

**Brefczynski-Lewis, J. A., Lutz, A., Schaefer, H. S., Levinson, D. B., & Davidson, R. J.** (2007). Neural correlates of attentional expertise in long-term meditation practitioners. *Proceedings of the National Academy of Sciences*, 104(27), 11483-11488. https://doi.org/10.1073/pnas.0606552104

**Kral, T. R. A., Davis, K., Korponay, C., Hirshberg, M. J., Hoel, R., Tello, L. Y., ... & Davidson, R. J.** (2022). Long-term meditation training is associated with enhanced attention and emotion regulation. *Journal of Cognitive Neuroscience*, 34(4), 623-641. https://doi.org/10.1162/jocn_a_01818

**Lin, J. F. L., Kuppens, P., & Tsai, C. H.** (2020). Meditation effect in changing functional integrations across large-scale networks. *Journal of Pacific Rim Psychology*, 14, e11. https://doi.org/10.1017/prp.2020.1

**Kilpatrick, L. A., Suyenobu, B. Y., Smith, S. R., Bueller, J. A., Goodman, T., Creswell, J. D., ... & Naliboff, B. D.** (2011). Impact of mindfulness-based stress reduction on intrinsic brain connectivity. *NeuroImage*, 56(1), 290-298. https://doi.org/10.1016/j.neuroimage.2011.02.034

**Tello, L., Kral, T. R. A., Garfinkel, S. N., Hoel, R., Hirshberg, M. J., & Davidson, R. J.** (2023). Effect of meditation on brain activity during an attention task. *Sensors*, 23(3), 1503. https://doi.org/10.3390/s23031503

## 15. Theoretical Framework

**Friston, K.** (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138. https://doi.org/10.1038/nrn2787

**Friston, K., Kilner, J., & Harrison, L.** (2006). A free energy principle for the brain. *Journal of Physiology-Paris*, 100(1-3), 70-87. https://doi.org/10.1016/j.jphysparis.2006.10.001

---

# Summary

This framework provides:

1. **Empirically-grounded network activation profiles** for four meditative states across expertise levels
2. **State-specific functional coupling patterns** based on neuroimaging correlations
3. **Algorithmic metrics** for expertise detection and state classification
4. **Current technical implementation** (MVOU dynamics, 2-file architecture with Markov Blanket)
5. **Future architectural vision** (Russian Doll nesting for Deep Computational Neurophenomenology)

The framework emphasizes:
- **Neural efficiency** in expert meditators (lower FPN during focus)
- **Background monitoring** capabilities in experts (positive DMN-FPN correlation during mind wandering)
- **Salience-driven awareness** (VAN spikes and FPN-VAN coupling during meta-awareness)
- **State-dependent coupling** as a key differentiator between expert and novice brain states
- **Bidirectional flow** (perception + action) through the Markov Blanket
- **Precision gating** as the mechanism for top-down causation (avoiding homunculus)

This document serves as a comprehensive reference for maintaining neuroscientific validity while developing and refining the computational model.

---

*Last Updated: January 2025*  
*Framework Version: 2.0 (Consolidated)*
