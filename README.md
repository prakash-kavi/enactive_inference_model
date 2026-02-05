# Lean Vipassana Model: Russian Doll Architecture

**Hierarchical active inference for meditation attention dynamics.**

A minimal, pedagogical implementation demonstrating the core architectural principles of a three-layer active inference model for meditation practice.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3 (L3): Metacognitive Monitor                         │
│ • Policy evaluation via Expected Free Energy (EFE)          │
│ • Meta-awareness tracking (attentional quality)             │
│ • Precision modulation (top-down control signal)            │
└────────────────┬────────────────────────────────────────────┘
                 │ Markov Blanket L2↔L3
                 │ Sensory: {meta_awareness, dwell_progress}
                 │ Active:  {precision_modulation, transition_drive}
┌────────────────▼────────────────────────────────────────────┐
│ Layer 2 (L2): Attentional Agent                             │
│ • 5 thoughtseeds (attend_breath, pain_discomfort, ...)      │
│ • VAE encoder/decoder + forward dynamics model              │
│ • Action selection via forward-informed EFE minimization    │
└────────────────┬────────────────────────────────────────────┘
                 │ Markov Blanket L1↔L2
                 │ Sensory: {DMN, VAN, DAN, FPN} activations
                 │ Active:  {target_networks, transition_drive}
┌────────────────▼────────────────────────────────────────────┐
│ Layer 1 (L1): Generative Process (MVOU Dynamics)            │
│ • 4 brain networks (DMN, VAN, DAN, FPN)                     │
│ • 4 meditation states (breath_focus, mind_wandering, ...)   │
│ • Multivariate Ornstein-Uhlenbeck dynamics                  │
│ • State-dependent coupling (Theta matrices)                 │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Principles

![Thoughtseeds Framework](Thoughtseeds_Framework.jpg)

### 1. **Thoughtseeds as Tractable Bottleneck**

**Problem**: High-dimensional neural dynamics (L1: 4 networks × continuous states) are intractable for conscious access.

**Solution**: L2 thoughtseeds compress the "global neurospace" into 5 interpretable dimensions:
- `attend_breath`, `pain_discomfort`, `pending_tasks`, `aha_moment`, `equanimity`

This bottleneck makes mental content **tractable**—the mind can "fathom it"—enabling metacognitive monitoring.

### 2. **Bottom-Up Emergence: Neural Dynamics → Thoughtseeds**

**L1 → L2 Message Passing** (via Markov Blanket):

The architecture exhibits **emergent properties** at each level:

```
L1 Brain Networks          L2 Thoughtseeds
─────────────────          ───────────────
DMN (0.8)  ┐               
VAN (0.3)  ├─→ VAE Encoder → [z₁, z₂, z₃, z₄, z₅]
DAN (0.6)  │                 ↓
FPN (0.5)  ┘                 Semantic Labels:
                             • attend_breath
High-dim, distributed        • pain_discomfort
neural activations           • pending_tasks
(continuous, noisy)          • aha_moment
                             • equanimity
                             
                             Low-dim, interpretable
                             mental content
```

**Emergence Mechanism**:
- **Compression**: VAE encoder reduces 4D network space → 5D thoughtseed space
- **Semantic binding**: Thoughtseed activations acquire meaning through state-dependent priors
- **Attractor dynamics**: Thoughtseeds evolve via state-specific differential equations, creating stable mental "objects"

Bottom-up flow creates **irreducible** mental content that cannot be predicted from lower-level dynamics alone.

### 3. **Top-Down Causation: Meta-Awareness → Precision Modulation**

**L3 → L2 → L1 Control Flow** (via Markov Blankets):

The architecture exhibits **downward causation**—higher levels constrain lower levels:

```
L3 Meta-Awareness                L2 Precision (λ)              L1 Neural Dynamics
─────────────────                ────────────────              ──────────────────
Semantic quality                 Prior strength                Process noise
of thoughtseeds                  in VFE                        modulation
                                                              
High meta-awareness    →    High λ (0.8)           →    Low noise variance
(clear, stable focus)       Tight prior adherence        Stable attractors
                           Strong action weighting       Reduced transitions

Low meta-awareness     →    Low λ (0.3)            →    High noise variance  
(diffuse, wandering)        Loose prior adherence        Fluid dynamics
                           Weak action weighting         Frequent transitions
```

**Causation Mechanism**:
- **Precision modulation**: L3 directly sets λ parameter in L2's VFE computation
- **Action weighting**: λ scales forward model loss (0.05 + 0.1 × λ)
- **Noise reduction**: L2 emits `noise_reduction = 1.0 - 0.6 × λ`, modulating L1 process variance
- **Belief about belief**: L3 observes thoughtseed **quality** (not content) and adjusts inference control

Top-down flow enables **metacognitive regulation**—awareness modulates its own substrate.

### 4. **Bidirectional Causation Loop**

The architecture creates **circular causality** between levels:

```
    Bottom-Up Emergence              Top-Down Causation
    ──────────────────              ──────────────────
         ↑                                  ↓
    L1 Networks  →  L2 Thoughtseeds  ←  L3 Meta-Awareness
    (distributed)   (emergent)           (regulatory)
         ↑                                  ↓
         └──────────  Noise Modulation  ────┘
```

This creates a **circular causality**: 
- Neural activity → emergent thoughtseeds → metacognitive evaluation → precision regulation → constrained neural activity

### 5. **L3 Design: Policy Evaluation via EFE **

**L3 uses Expected Free Energy (EFE) for policy evaluation—no VFE minimization.**

**Design Rationale**:
- L2 minimizes **VFE** (perceptual inference): "How well do thoughtseeds explain neural dynamics?"
- L3 evaluates **EFE** (deliberative planning): "Which policy (stay/transition) minimizes expected future surprise?"

**Why EFE-only for metacognition**:
- Meditators **evaluate** current attentional state quality
- They **select actions** (maintain focus vs. redirect attention)
- They don't build predictive models of their own metacognitive observations

**Cognitive Architecture**:
- Reflexive perception → VFE minimization (L2)
- Deliberative planning → EFE policy selection (L3)

**Benefits**:
- Lighter computational footprint
- Clear functional separation (perception at L2, planning at L3)
- Domain-appropriate for metacognitive monitoring

### 6. **Markov Blankets as Message Passing**

Each blanket enforces architectural separation:
- **Sensory states**: Read-only observations from lower level
- **Active states**: Control signals to lower level
- **EMA smoothing**: Prevents sharp discontinuities across hierarchy

---

## Implementation

```
lean_model/
├── config.py                 # Architecture constants and parameters
├── process.py                # Layer 1: MVOU generative process
├── agent.py                  # Layer 2: VAE + thoughtseeds + forward model
├── monitor.py                # Layer 3: Metacognitive policy evaluator
├── blankets.py               # Markov blanket interfaces
├── train.py                  # BPTT training orchestrator
├── analysis.py               # Metrics computation and visualization
├── utils.py                  # Shared utility functions
└── run_meditation.py         # Command-line interface
```

## Usage

### Forward Simulation

```python
from lean_model import Layer1Process, Layer2Agent, Layer3Monitor
from lean_model import MarkovBlanketL1L2, MarkovBlanketL2L3

# Initialize components
blanket_l1l2 = MarkovBlanketL1L2(smoothing=0.7)
blanket_l2l3 = MarkovBlanketL2L3(smoothing=0.7)

process = Layer1Process(experience_level='expert', seed=42)
agent = Layer2Agent(experience_level='expert', 
                    blanket_l1l2=blanket_l1l2,
                    blanket_l2l3=blanket_l2l3)
monitor = Layer3Monitor(experience_level='expert',
                        blanket_l2l3=blanket_l2l3)

# Forward simulation (no training)
for t in range(100):
    # L1: Generative process
    network_acts, state = process.update(blanket_l1l2.active_states)
    blanket_l1l2.update_sensory_states(network_acts)
    
    # L2: Attentional agent
    z_inferred = agent.perceptual_inference()
    
    # L3: Metacognitive monitor  
    meta_awareness = monitor.update_meta_awareness(state, z_inferred)
    policy = monitor.evaluate_policies(state, vfe=0.5)
    
    # L2 → L1 action
    prescription = agent.prescriptive_action(z_inferred, state)
    blanket_l1l2.update_active_states(prescription)
    
    print(f"t={t}: {state}, meta={meta_awareness:.3f}")
```

### Training

```python
from lean_model.train import train_meditation

# Run training
results = train_meditation(
    experience_level='expert',
    timesteps=10000,
    seed=42
)
```

## Mathematical Framework

### L2 Variational Free Energy
```
F[q(z|x)] = E_q[log q(z|x) - log p(x,z)]
          = -E_q[log p(x|z)] + KL[q(z|x) || p(z)]
          = reconstruction_loss + KL_divergence
```


### L3 Expected Free Energy
```
EFE(π) = E_q[log q(s'|π) - log p(s')] + KL[q(o'|s',π) || p(o'|C)]
       = risk + ambiguity
```

## References

- Active Inference: Friston et al. (2015-2022)
- Enactive Cognition: Varela et al. (1991), Thompson (2007)
- Meditation Neuroscience: Hasenkamp et al. (2012), Lutz et al. (2015)
