# Vipassana Meditation Model: Hierarchical Active Inference



---

## Architecture
![Thoughtseeds Framework](Thoughtseeds_Framework.jpg)
```
+--------------------------------------------------------------+
| Layer 3: Metacognitive Monitor                               |
| - Tracks meta-awareness from L2 thoughtseeds                 |
| - Sends meta-awareness to L2 (sensory) to modulate precision                       |
+------------------------------+-------------------------------+
               | Markov Blanket L2<->L3
               | Sensory: meta_awareness
               | Active:  precision_sensory, policy_precision
+------------------------------v-------------------------------+
| Layer 2: Attentional Agent (Thoughtseeds)                    |
| - Compresses neural dynamics into 5 thoughtseeds             |
| - VAE encoder/decoder + forward dynamics model               |
| - Policy posterior q(pi) via softmax of G(pi)                |
| - Policy precision from entropy(q_pi)
| - Sensory precision from forward prediction error + meta-awareness            |
+------------------------------+-------------------------------+
               | Markov Blanket L1<->L2
               | Sensory: DMN, VAN, DAN, FPN activations
               | Active:  mu_x, policy_drive, precision_gain, noise_reduction
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
python run_enactive_inference.py run
```

This trains both expert and novice models for 10,000 timesteps (default), saves results to `data/`, and generates all plots.

**Options:**
```bash
python run_enactive_inference.py run --timesteps 10000  # Custom training length
```

### Generate Plots from Existing Data

```bash
python run_enactive_inference.py plot
```

Generates 8 publication-quality figures from saved training results.

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

**Dynamics:**
- `Fig4A_Hierarchy_Novice.png` - 3-layer hierarchical dynamics over time
- `Fig4B_Hierarchy_Expert.png`

**State Space:**
- `Fig5_PCA_Trajectories.png` - PCA trajectories across the hierarchy (L2 thoughtseeds + L1 networks)

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
- Loss = VFE + w_fwd * forward prediction error + alpha_rec * recognition loss
- Recognition loss uses alpha_rec=1 for expert and alpha_rec=lr_novice/lr_expert for novice
- Policy precision is derived from policy posterior entropy (q_pi)
- Sensory precision blends forward prediction error with meta-awareness

### 5. Expert vs Novice Phenotypes
**Expert:**
- **Unfrozen Encoder:** Learns "Amortized Inference" (fast, intuitive state recognition).
- **Universal Priors:** Accesses the "Universal/Goal" priors effectively.
- **Physiology:** Stronger FPN activation, longer BF dwell times.

**Novice:**
- **Weak Amortized Inference:** Learns more slowly due to lower learning rate, so recognition is less reliable.
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
+-- run_enactive_inference.py  # Main entry point
+-- model/                     # Core Logic
|   +-- training_loop.py       # MeditationTrainer class
|   +-- l1_generative_process.py  # Layer1Process (MVOU dynamics)
|   +-- l2_recognition.py         # Layer2Agent (VAE + forward model)
|   +-- l3_metacognition.py       # Layer3Monitor (meta-awareness tracking)
|   +-- markov_blankets.py        # Markov blanket interfaces
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

### Notation
- `x in R^4`: L1 network activations ordered {DMN, VAN, DAN, FPN}
- `z in [0,1]^5`: L2 thoughtseed activations
- `s in {BF, MW, MA, RA}`: meditation state
- `mu_x(s)`, `mu_z(s)`: state-conditioned network and thoughtseed priors
- `Theta(s)`: state-conditioned coupling matrix
- `NOISE_LEVEL`: L1 process noise variance

### Layer 1: Generative Process (MVOU)
Continuous-time dynamics:
```
dx = -Theta(s) (x - mu_x(s)) dt + sigma dW
```
with `sigma^2 = NOISE_LEVEL`. Euler integration is used with state-specific `Theta(s)`.

### Layer 2: Recognition + Variational Inference (VAE)
VAE components:
- **Encoder**: `q(z|x)` (networks -> thoughtseeds)
- **Decoder**: `p(x|z)` (reconstruction)
- **Forward model**: `f(x, z)` predicts next networks

Per-step VFE:
```
F(z) = MSE(decode(z), x) + KL_Bernoulli(z || mu_z(s))
```

Fixed-step VI (2 steps, lr=0.2) optimizes:
```
L(z) = recon_loss + KL
     + precision_w * MSE(z, z_rec)
     + (1 - precision_w) * MSE(z, z_prev)
```
Initialization:
```
z_init = 0.5 * z_prev + 0.5 * z_rec
z_init = precision_w * z_init + (1 - precision_w) * mu_z(s)
```
where `precision_w = clip(precision_sensory)` in [0, 1].

### Sensory Precision (from prediction error)
Forward prediction:
```
x_pred = f(x_{t-1}, a_{t-1})
epsilon_fwd = mean((x_t - x_pred)^2)
```
Precision update:
```
precision_sensory = NOISE_LEVEL / (NOISE_LEVEL + epsilon_fwd + eps)
```

### Layer 3: Meta-Awareness
Meta-awareness is a weighted sum of thoughtseeds:
```
meta = sum_i w_i z_i / sum_i w_i
```
clipped to [0, 1], with state-dependent weights.

### Policy Inference (L2)
Candidate policies: stay in `s` or transition to each other state.
Hazard from dwell:
```
h = clip(dwell_progress^2)
```
Policy prior:
```
E(stay) = 1 - h
E(s') = h * P(s' | s)
```
Expected free energy:
```
C_s = decode(mu_z(s))
G(pi) = KL_Bernoulli(x_pred || C_s) + H(x_pred)
```
Policy posterior:
```
q(pi) = softmax(log E(pi) - gamma * G(pi))
```
Policy precision:
```
gamma = 1 - H(q(pi)) / log(N_pi)
```
Action target:
```
mu = sum_pi q(pi) * mu_z(s_pi)
mu = (1 - h) * mu_current + h * mu
mu_x = decode(mu)
```

### Learning Objective
Auto-balanced forward weight:
```
w_fwd = sum_t F_t / (sum_t epsilon_fwd + eps)
```
Total loss:
```
L_total = F + w_fwd * epsilon_fwd + alpha_rec * L_rec
```
where `L_rec = MSE(encode(x), z*)`. Experts use `alpha_rec = 1`. Novices use a weak
weight tied to learning rate: `alpha_rec = lr_novice / lr_expert`.

### Training Loop
BPTT windows of 50 steps; gradients accumulated per window.

---

## Configuration

Edit `config.py` to modify:
- Network/state parameters (Theta matrices, mu attractors)
- Thoughtseed priors (THOUGHTSEED_STATE_PRIORS)
- Learning rates (0.01 - 0.02)
- Process noise (NOISE_LEVEL)
- Phenotype differences (learning rate + recognition loss scaling)

---

## Reproducibility

Fixed random seed (42) ensures identical results across runs. Training is stochastic (e.g., MW dominance sampling), but fully seed-controlled.

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
This repository is a significant step forward in enhancing the Thoughtseeds Framework fadapting code snippets from below:
  https://github.com/prakash-kavi/thoughtseeds_vipassana 
  https://github.com/prakash-kavi/viapssana_ts2  
  https://github.com/prakash-kavi/aif_iwai2025_thoughtseeds