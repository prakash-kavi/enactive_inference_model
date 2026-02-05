# Vipassana Meditation Model: Hierarchical Active Inference

**A minimal implementation of three-layer active inference for meditation attention dynamics.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Metacognitive Monitor                              │
│ • Evaluates attentional quality (meta-awareness)            │
│ • Computes Expected Free Energy (EFE) for policy selection  │
│ • Modulates L2 precision (top-down control)                 │
└────────────────┬────────────────────────────────────────────┘
                 │ Markov Blanket L2↔L3
                 │ Sensory: meta_awareness, state_progress
                 │ Active:  precision_modulation, exit_drive
┌────────────────▼────────────────────────────────────────────┐
│ Layer 2: Attentional Agent (Thoughtseeds)                   │
│ • Compresses neural dynamics into 5 thoughtseeds            │
│ • VAE encoder/decoder + forward dynamics model              │
│ • Action selection minimizes EFE informed by predictions    │
└────────────────┬────────────────────────────────────────────┘
                 │ Markov Blanket L1↔L2
                 │ Sensory: DMN, VAN, DAN, FPN activations
                 │ Active:  target_networks, state_transition
┌────────────────▼────────────────────────────────────────────┐
│ Layer 1: Neural Generative Process (MVOU)                   │
│ • 4 brain networks (DMN, VAN, DAN, FPN)                     │
│ • 4 meditation states (BF, MW, MA, RA)                      │
│ • Multivariate Ornstein-Uhlenbeck dynamics                  │
│ • State-dependent coupling (Θ matrices)                     │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### States (4)
- **BF** (Breath Focus): Stable attention on breath sensations
- **MW** (Mind Wandering): Spontaneous thought proliferation
- **MA** (Meta-Awareness): Recognition of mind-wandering
- **RA** (Redirect Attention): Volitional return to breath

### Networks (4)
- **DMN** (Default Mode): Self-referential processing
- **VAN** (Ventral Attention): Stimulus-driven detection
- **DAN** (Dorsal Attention): Goal-directed control
- **FPN** (Frontoparietal): Cognitive flexibility

### Thoughtseeds (5)
Compress high-dimensional neural state into interpretable mental content:
- `attend_breath`: Breath sensation focus
- `pain_discomfort`: Physical discomfort awareness
- `pending_tasks`: Task-related rumination
- `aha_moment`: Insight/creative thought
- `equanimity`: Acceptance/non-reactivity

---

## Installation

```bash
pip install torch numpy matplotlib
```

**Requirements:**
- Python 3.8+
- PyTorch 1.10+
- NumPy, Matplotlib

---

## Usage

### Train Both Phenotypes (Expert & Novice)

```bash
python run_meditation.py run
```

This trains both expert and novice models for 10,000 timesteps (default), saves results to `data/`, and generates all plots.

**Options:**
```bash
python run_meditation.py run --timesteps 10000  # Custom training length
```

### Generate Plots from Existing Data

```bash
python run_meditation.py plot
```

Generates 10 publication-quality figures from saved training results.

---

## Output

### Training Results
Saved to `data/`:
- `training_results_expert_seed42.json`
- `training_results_novice_seed42.json`

Each contains:
- Full state/network/thoughtseed trajectories
- Free energy history
- Meta-awareness evolution
- Transition statistics
- Learned model parameters

### Plots
Generated in `plots/`:

**Convergence:**
- `FigS1_Convergence_Expert.png` - Free energy stabilization, state occupancy
- `FigS1_Convergence_Novice.png`

**Comparison:**
- `Fig3A_Network_Radar.png` - Network profiles across states (Expert vs Novice)
- `Fig3B_FE_and_Dwell.png` - Free energy and dwell times per state
- `Fig3C_Transitions.png` - State transition probability matrices
- `Fig3D_Belief_about_Belief.png` - L3 meta-awareness & L2 free energy evolution

**Dynamics:**
- `Fig4A_Hierarchy_Novice.png` - 3-layer hierarchical dynamics over time
- `Fig4B_Hierarchy_Expert.png`

**State Space:**
- `Fig5A_Attractor2D.png` - 2D thoughtseed trajectory (Novice vs Expert)
- `Fig5B_Attractor3D.png` - 3D free energy landscape

---

## Model Features

### 1. Hierarchical Markov Blankets
Each layer interfaces through Markov blankets defining:
- **Sensory states**: What the layer observes from below
- **Active states**: How the layer influences layers below

### 2. Thoughtseeds as Tractable Bottleneck
Layer 2 compresses 4 network activations → 5 thoughtseeds, making neural state "tractable" for conscious access and metacognitive monitoring.

### 3. Forward Dynamics Model
Layer 2 learns to predict future thoughtseed states, enabling:
- Anticipatory action selection
- Policy evaluation beyond immediate outcomes
- Counterfactual reasoning ("what if I stay in MW?")

### 4. BPTT Learning
Backpropagation Through Time optimizes:
- VAE encoder/decoder (representation learning)
- Forward model (dynamics prediction)
- Action policies (minimize expected free energy)

### 5. Expert vs Novice Phenotypes
**Expert:**
- **Unfrozen Encoder:** Learns "Amortized Inference" (fast, intuitive state recognition).
- **Universal Priors:** Accesses the "Universal/Goal" priors effectively.
- **Physiology:** Stronger FPN activation, longer BF dwell times.

**Novice:**
- **Frozen Encoder:** Lacks "Amortized Inference" (relies on slow, effortful VFE minimization).
- **Universal Priors:** Holds the same goal (Focus) but lacks the intuition to recognize it.
- **Physiology:** DMN-dominant profile, shorter BF dwell times.

---

## Key Results

**Behavioral Signatures:**
- **Expert**: 119 timesteps average BF dwell, 59% MW→MA transition rate
- **Novice**: 56 timesteps average BF dwell, 66% MW→MA transition rate (but higher MA threshold)

**Learning Dynamics:**
- Expert converges to lower free energy (better prediction accuracy)
- Novice shows higher meta-awareness variability
- Forward model reduces action prediction errors by ~40%

**Network Profiles:**
- BF state: DAN+FPN dominant
- MW state: DMN dominant
- MA state: VAN+FPN (detection + reorienting)
- RA state: Balanced DAN+FPN (executive control)

---

## File Structure

```
.
├── run_meditation.py          # Main entry point
├── model/                     # Core Logic
│   ├── train.py               # MeditationTrainer class
│   ├── process.py             # Layer1Process (MVOU dynamics)
│   ├── agent.py               # Layer2Agent (VAE + forward model)
│   ├── monitor.py             # Layer3Monitor (EFE policy)
│   └── blankets.py            # Markov blanket interfaces
├── utils/                     # Utilities & Config
│   ├── config.py              # Constants and universal priors
│   ├── math_utils.py          # Tensor/math operations
│   └── analysis_utils.py      # Metrics computation
├── data/                      # Training results (JSON)
├── plots/                     # Generated figures (PNG)
└── viz/                       # Plotting modules
    ├── lean_convergence.py
    ├── lean_comparison.py
    ├── lean_hierarchy.py
    ├── lean_attractors.py
    ├── lean_diagnostics.py
    └── plotting_utils.py
```

**Total:** ~2,000 lines of core implementation + visualization

---

## Technical Details

### Layer 1: Generative Process
Multivariate Ornstein-Uhlenbeck (MVOU) process:
```
dx = Θ(s)[μ(s) - x]dt + σ(s)dW
```
- State-dependent coupling matrices Θ(s)
- State-specific attractors μ(s)
- Gaussian noise σ(s)

### Layer 2: Thoughtseed Dynamics
VAE architecture:
- **Encoder**: Networks → Thoughtseeds (4 → 5 latent dims)
- **Decoder**: Thoughtseeds → Networks (reconstruction)
- **Forward Model**: Predicts thoughtseeds at t+1 given current state and action

Free Energy:
```
F = Reconstruction_Error + KL_Divergence + Forward_Prediction_Error
```

### Layer 3: Policy Selection
Expected Free Energy per policy π:
```
EFE(π) = E_q[log q(o|π) - log p(o|C)] + E_q[KL[q(s|π)||q(s)]]
         \_____________v_____________/   \__________v__________/
              Pragmatic value              Epistemic value
```

---

## Configuration

Edit `config.py` to modify:
- Network/state parameters (Θ matrices, μ attractors)
- Thoughtseed Priors (THOUGHTSEED_STATE_PRIORS)
- Learning rates (0.01 - 0.02)
- Loss weights (forward model, KL divergence, recognition loss)
- Expertise levels (Phenotypes defined by Encoder Plasticity)

---

## Reproducibility

Fixed random seed (42) ensures identical results across runs. Training produces deterministic trajectories given the same initialization.

---

## Citation

If you use this model in your research:

```
@article{enactive_inference_mental_action_via_vipassana_simulation_2026,
  author = {Author, A. and Author, B. and Author, C.},
  title = {Attentional Agents and Enactive Inference: Computational Phenomenology of Mental Action},
  journal = {}
  author={[Authors]},
  year={2026}
}
```
