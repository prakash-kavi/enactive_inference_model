# Enactive Inference Model

---

## Architecture
![Thoughtseeds Framework](figures/fig1.jpg)
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
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- See `requirements.txt`

---

## Usage

### Train Both Phenotypes (Expert & Novice)

```bash
python run_enactive_inference.py run
```

This trains both expert and novice models for 10,000 timesteps (default), saves results to `data/`, and generates all plots.

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
Generated in `figures/`:

**Convergence:**
- `FigS1_Convergence_Expert.pdf` - Loss/free-energy convergence, state occupancy
- `FigS1_Convergence_Novice.pdf`

**Comparison:**
- `fig2a.pdf` - Network profiles across states (Expert vs Novice)
- `fig2b.pdf` - Dwell times per state (timesteps)
- `fig2c.pdf` - State transition probability matrices

**Dynamics:**
- `fig3a.pdf` - 3-layer hierarchical dynamics over time
- `fig3b.pdf`

**State Space:**
- `fig4.pdf` - PCA trajectories across the hierarchy (L2 thoughtseeds + L1 networks)

---

## Model Features

### 1. Hierarchical Markov Blankets
Each layer interfaces through Markov blankets defining:
- **Sensory states**: What the layer observes from below
- **Active states**: How the layer influences layers below

### 2. Thoughtseeds as Tractable Bottleneck
Layer 2 compresses 4 network activations -> 5 thoughtseeds, making neural state "tractable" for conscious access and metacognitive monitoring.

### 3. Forward Dynamics Model
Layer 2 predicts next-step network activations from (x_t, z_t). This provides a prospective signal for policy scoring (stay/switch), and supplies forward surprisal for precision calibration.

### 4. BPTT Learning
Backpropagation Through Time optimizes:
- VAE encoder/decoder (latent structure learning)
- Forward model (dynamics prediction)
- Loss = VFE + w_fwd * forward prediction error + alpha_rec * recognition loss
- Recognition loss uses alpha_rec=1 for expert and alpha_rec=lr_novice/lr_expert for novice
- Policy precision is derived from policy posterior entropy (q_pi)
- Sensory precision blends forward prediction error with meta-awareness

### 5. Expert vs Novice Phenotypes
**Expert:**
- Learns "Amortized Inference" (fast, intuitive state recognition).
- Uses expert priors for dwell ranges and transition probabilities.
- Stronger FPN activation, longer BF dwell times.

**Novice:**
- Weak Amortized Inference: Learns more slowly due to lower learning rate, so recognition is less reliable.
- Uses novice priors for dwell ranges and transition probabilities.
- DMN-dominant profile, shorter BF dwell times.

---

## Key Results

**Behavioral Signatures (run-dependent):**
- Expert typically shows lower free energy and a more stable breath-focus basin
- Novice shows broader excursions and shallower basins
- See `data/` and `figures/` for the current run's quantitative summaries

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
+-- data/                      # Training results (JSON)
+-- figures/                   # Generated figures (PDF)
+-- viz/                       # Plotting modules
    +-- analysis.py
    +-- analysis_utils.py
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
- **Decoder**: `p(x|z)` (likelihood / surprisal)
- **Forward model**: `f(x, z)` predicts next networks

Per-step VFE:
```
F(z) = Surprisal_NLL_Bernoulli(decode(z), x) + KL_Bernoulli(z || mu_z(s))
```

Fixed-step VI (2 steps, lr=0.2) optimizes:
```
L(z) = surprisal + KL
     + pi_w * MSE(z, z_rec)
     + (1 - pi_w) * MSE(z, z_prev)
```
Initialization:
```
z_init = 0.5 * z_prev + 0.5 * z_rec
z_init = pi_w * z_init + (1 - pi_w) * mu_z(s)
```
where `pi_w = clip(precision_sensory)` in [0, 1].

### Sensory Precision (from forward surprisal)
Forward prediction and surprisal (S_forward):
```
x_pred = f(x_{t-1}, a_{t-1})
S_forward = Surprisal_NLL_Bernoulli(x_pred, x_t)
```
We map forward surprisal to sensory precision (`precision_sensory`), so higher surprise implies lower precision.
Precision update (Option A, Act-Inf aligned):
```
precision_sensory = exp(-S_forward)
precision_sensory = clip(precision_sensory, CLIP_MIN, CLIP_MAX)
precision_sensory = fuse_logit(precision_sensory, meta)
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
w_fwd = EMA(F_t) / (EMA(S_forward) + eps)
```
Total loss:
```
L_total = F + w_fwd * S_forward + alpha_rec * L_rec
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
This repository is a significant step forward in enhancing the Thoughtseeds Framework for Enactive Inference. It builds upon the foundational work of the Thoughtseeds Framework, adapting code snippets from below:
  https://github.com/prakash-kavi/thoughtseeds_vipassana 
  https://github.com/prakash-kavi/viapssana_ts2  
  https://github.com/prakash-kavi/aif_iwai2025_thoughtseeds
