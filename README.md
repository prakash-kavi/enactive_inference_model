# Vipassana Meditation Model: Hierarchical Active Inference

**A minimal implementation of three-layer active inference for meditation attention dynamics.**

---

## Architecture
![Thoughtseeds Framework](Thoughtseeds_Framework.jpg)
```
+--------------------------------------------------------------+
| Layer 3: Metacognitive Monitor                               |
| - Tracks meta-awareness from L2 thoughtseeds                 |
| - Infers policy precision via Gamma posterior                |
| - Sends precision_sensory + policy_precision to L2           |
+------------------------------+-------------------------------+
               | Markov Blanket L2<->L3
               | Sensory: meta_awareness, opacity
| Active:  precision_sensory, policy_precision
+------------------------------v-------------------------------+
| Layer 2: Attentional Agent (Thoughtseeds)                    |
| - Compresses neural dynamics into 5 thoughtseeds             |
| - VAE encoder/decoder + forward dynamics model               |
| - Policy posterior q(pi) via softmax of G(pi)                |
+------------------------------+-------------------------------+
               | Markov Blanket L1<->L2
               | Sensory: DMN, VAN, DAN, FPN activations
| Active:  mu_x, policy_drive, precision_gain
+------------------------------v-------------------------------+
| Layer 1: Neural Generative Process (MVOU)                    |
| - 4 brain networks (DMN, VAN, DAN, FPN)                      |
| - 4 meditation states (BF, MW, MA, RA)                       |
| - Multivariate Ornstein-Uhlenbeck dynamics                   |
| - State-dependent coupling (Theta matrices)                 |
+--------------------------------------------------------------+
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
- Python 3.9+
- PyTorch 1.13+
- NumPy 1.25+, Matplotlib 3.8+

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
- Action prediction error summaries

### Plots
Generated in `plots/`:

**Convergence:**
- `FigS1_Convergence_Expert.png` - Free energy stabilization, state occupancy
- `FigS1_Convergence_Novice.png`

**Comparison:**
- `Fig3A_Network_Radar.png` - Network profiles across states (Expert vs Novice)
- `Fig3B_FE_and_Dwell.png` - Free energy and dwell times per state
- `Fig3C_Transitions.png` - State transition probability matrices
- `Fig3D.png` - L3 meta-awareness & L2 free energy evolution

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
Layer 2 compresses 4 network activations -> 5 thoughtseeds, making neural state "tractable" for conscious access and metacognitive monitoring.

### 3. Forward Dynamics Model
Layer 2 learns to predict future network activations from (x, z), enabling:
- Anticipatory action selection
- Policy evaluation beyond immediate outcomes
- Counterfactual reasoning ("what if I stay in MW?")

### 4. BPTT Learning
Backpropagation Through Time optimizes:
- VAE encoder/decoder (representation learning)
- Forward model (dynamics prediction)
- Loss = VFE + forward prediction error (+ recognition loss for expert)
- L3 precision is inferred via a Gamma posterior on forward prediction error

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

**Behavioral Signatures (run-dependent):**
- Expert typically shows lower free energy and a more stable breath-focus basin
- Novice shows broader excursions and shallower basins
- See `data/` and `plots/` for the current run's quantitative summaries

---

## File Structure

```
.
+-- run_meditation.py          # Main entry point
+-- model/                     # Core Logic
|   +-- train.py               # MeditationTrainer class
|   +-- process.py             # Layer1Process (MVOU dynamics)
|   +-- agent.py               # Layer2Agent (VAE + forward model)
|   +-- monitor.py             # Layer3Monitor (precision inference)
|   +-- blankets.py            # Markov blanket interfaces
+-- utils/                     # Utilities & Config
|   +-- config.py              # Constants and universal priors
|   +-- math_utils.py          # Tensor/math operations
|   +-- analysis_utils.py      # Metrics computation
+-- data/                      # Training results (JSON)
+-- plots/                     # Generated figures (PNG)
+-- viz/                       # Plotting modules
    +-- analysis.py
    +-- attractors.py
    +-- convergence.py
    +-- diagnostics.py
    +-- hierarchy.py
    +-- radar_plot.py
    +-- plotting_utils.py
```

---

## Technical Details

### Layer 1: Generative Process
Multivariate Ornstein-Uhlenbeck (MVOU) process:
```
dx = Theta(s) * [mu(s) - x] dt + sigma(s) dW
```
- State-dependent coupling matrices Theta(s)
- State-specific attractors mu(s)
- Gaussian noise sigma(s)

### Layer 2: Thoughtseed Dynamics
VAE architecture:
- **Encoder**: Networks -> Thoughtseeds (4 -> 5 latent dims)
- **Decoder**: Thoughtseeds -> Networks (reconstruction)
- **Forward Model**: Predicts next networks given (x, z)

Free Energy (VFE used for reporting):
```
F = Reconstruction_Error + KL_Divergence
```
Training loss additionally includes forward prediction error, and expert-only recognition loss.

### Layer 3: Policy Precision + Policy Posterior
Policy posterior uses softmax over G(pi):
```
q(pi) = softmax( log E(pi) - gamma * G(pi) )
```
First-pass G(pi) uses risk against preferred outcomes C:
```
G(pi) = D_KL( x_pred || C )
```
Policy precision gamma is inferred from a Gamma posterior over prediction error.

---

## Configuration

Edit `config.py` to modify:
- Network/state parameters (Theta matrices, mu attractors)
- Thoughtseed priors (THOUGHTSEED_STATE_PRIORS)
- Learning rates (0.01 - 0.02)
- Noise sigma
- Expertise levels (phenotypes defined by encoder plasticity)

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
