# Enactive Inference Model

---

## Architecture
![Meditative Cycle](figures/fig1.jpg) ![Thoughtseeds Framework](figures/fig2.jpg)
```
+--------------------------------------------------------------+
| Layer 3: Metacognitive Monitor                               |
| - Meta-awareness: gated divergence (policy evidence vs habit) |
| - Selects policy posterior q(pi); modulates policy precision  |
+------------------------------+-------------------------------+
               | Markov Blanket L2<->L3
               | Sensory: policy evidence G(pi), state belief
               | Active:  policy posterior q(pi)
+------------------------------v-------------------------------+
| Layer 2: Attentional Agent (Thoughtseeds)                    |
| - Compresses neural dynamics into 5 thoughtseeds             |
| - Encoder/decoder + forward model f(x,z)                     |
| - Evaluates expected free energy G(pi); passes evidence to L3 |
| - Sensory precision from forward surprisal (exp form)         |
+------------------------------+-------------------------------+
               | Markov Blanket L1<->L2
               | Sensory: s_t, x_t, dwell_progress d_t
               | Active:  mu_x, transition_drive u_t, policy_state_probs
+------------------------------v-------------------------------+
| Layer 1: Neural Generative Process (MVOU)                    |
| - 4 brain networks (DMN, VAN, DAN, FPN)                      |
| - 4 meditation states (BF, MW, MA, RA)                       |
| - Multivariate Ornstein-Uhlenbeck dynamics                   |
| - Attractor mixing: mu <- (1-m_t)mu + m_t mu_x               |
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
- Runs on CPU only; no GPU required or used.

---

## Usage

### Run Full Pipeline (Learning + Simulation + Plots)

```bash
python run_enactive_inference.py run
```

This runs a **two-phase protocol** per phenotype: (1) **Learning phase**: variational EM for 10,000 timesteps to fit encoder, decoder, forward model, and habit prior; (2) **Simulation phase**: inference-only for 10,000 timesteps with frozen parameters. Results are saved to `data/` and all plots are generated in `figures/`.

### Generate Plots from Existing Data

```bash
python run_enactive_inference.py plot
```

Generates publication-quality figures from saved results. When available, plots use **simulation results** (inference-only) rather than training results; otherwise falls back to training results.

---

## Output

### Results (saved to `data/`)
- `training_results_expert_seed42.json` / `training_results_novice_seed42.json` — learning-phase trajectories
- `simulation_results_expert_seed42.json` / `simulation_results_novice_seed42.json` — inference-only trajectories (used for figures when available)

Each contains: state/network/thoughtseed histories, free energy, meta-awareness, transition statistics, prediction errors.

### Plots (generated in `figures/`)
**Convergence (FigS1):**
- `FigS1_Convergence_Expert.pdf`, `FigS1_Convergence_Novice.pdf` — learning vs inference-only stability, cumulative state occupancy

**Comparison (Fig 3):**
- `fig3a.pdf` — Network activation profiles across states (Expert vs Novice)
- `fig3b.pdf` — Dwell times per state (timesteps)
- `fig3c.pdf` — State transition probability matrices

**Hierarchy (Fig 4 & 6):**
- `fig4a.pdf`, `fig4b.pdf` — 3-layer hierarchical dynamics (L3 meta-awareness, L2 dominant thoughtseed, L1 networks)
- `fig6a.pdf`, `fig6b.pdf` — Same with continuous thoughtseed traces

**State space (Fig 5):**
- `fig5.pdf` — PCA trajectories (L2 thoughtseeds + L1 networks)

**Tail window:** Plots and transition/dwell statistics use the last 2,000 steps (converged regime).

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
Backpropagation Through Time (window 25 steps) optimizes:
- Encoder/decoder (latent structure learning)
- Forward model (dynamics prediction)
- Loss = VFE + S_forward + alpha_rec * L_rec (L_rec = MSE(encode(x), z*))
- alpha_rec set adaptively per BPTT window as mean(VFE)/mean(L_rec)

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

**Behavioral signatures (run-dependent, from simulation tail window):**
- Expert: longer Breath Focus dwell, shorter MW/MA/RA dwell; lower DMN, stronger DAN/FPN; tighter recovery loop (MW→MA→RA→BF)
- Novice: shorter BF, longer MW; DMN-dominant; more diffuse transitions
- See `data/` and `figures/` for quantitative summaries from the current run

---

## File Structure

```
.
+-- run_enactive_inference.py  # Main entry point (run | plot)
+-- model/                     # Core logic
|   +-- training_loop.py       # MeditationTrainer (EM, BPTT, simulate)
|   +-- phenotype.py           # Expert/novice phenotype definitions
|   +-- l1_generative_process.py  # Layer1Process (MVOU dynamics)
|   +-- l2_recognition.py         # Layer2Agent (encoder/decoder/forward model)
|   +-- l3_metacognition.py       # Layer3Monitor (meta-awareness, policy selection)
|   +-- markov_blankets.py        # Markov blanket interfaces
+-- utils/
|   +-- config.py              # Constants, priors, BPTT_STEPS, TAIL_STEPS, etc.
|   +-- math_utils.py          # Tensor/math operations
+-- data/                      # Training and simulation results (JSON)
+-- figures/                   # Generated figures (PDF)
+-- scripts/                   # Utilities (e.g. extract_results_stats.py)
+-- viz/                       # Plotting
    +-- analysis.py, analysis_utils.py
    +-- attractors.py, convergence.py, diagnostics.py
    +-- hierarchy.py, radar_plot.py, plotting_utils.py
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
with `sigma^2 = NOISE_LEVEL`. Euler-Maruyama integration (2 substeps per step). When L2 provides descending prediction mu_x, attractor is mixed: mu <- (1-m_t)mu + m_t mu_x. State transitions are dwell-timed; after dwell elapses, transition probability is tilde{u}_t = max(u_t, 1/n); exit priors reweighted by policy posterior.

### Layer 2: Recognition + Variational Inference
Components:
- **Encoder**: `q(z|x)` (networks -> thoughtseeds)
- **Decoder**: `p(x|z)` (likelihood / surprisal)
- **Forward model**: `f(x, z)` predicts next networks

Per-step VFE:
```
F(z) = pi_x * ||x - decode(z)||^2 + ||z - mu_z(s)||^2
```

Fixed-step VI (config: VI_STEPS, VI_LR) minimizes F(z). Encoder provides initialization; state-dependent perturbation applied before VI; result clipped to [0.05, 0.9]. VI refinement triggered when latent mismatch exceeds VI_MISMATCH_THRESHOLD.

### Sensory Precision (from forward surprisal)
Forward prediction and surprisal:
```
x_pred = f(x_{t-1}, mu_{t-1})
S_forward = (1/D_x) ||x_t - x_pred||^2
```
Precision (weights reconstruction term in VFE; clipped in code):
```
pi_x = exp(-S_forward / (sigma^2_fwd + epsilon))
```
with sigma^2_fwd an EMA of S_forward.

### Layer 3: Meta-Awareness
Meta-awareness m_t is a gated divergence between policy evidence and habitual priors:
```
q_evid(pi) = softmax(-G_tilde(pi)),  q_habit(pi) = softmax(bar{l_pi})
gate = q_t(MA) + q_t(RA)
m_t = (1 - exp(-KL(q_evid || q_habit))) * gate
```
EMA-smoothed for stability. Higher m_t shifts policy selection from habits toward evidence.

### Policy Evaluation (L2)
Expected free energy (z-scored across candidates):
```
G(pi) = ||x_pred(pi) - C_{s_pi}||^2 - I(pi)
```
Pragmatic: deviation from preferred network target. Epistemic: I(pi) in thoughtseed space.

### Policy Selection (L3)
Dwell-aware prior: rho_t = d_t^2 (dwell progress squared)
```
E(stay) = 1 - rho_t,   E(s') = rho_t * P(s'|s)
```
Policy posterior (L3):
```
q(pi) = softmax(log E(pi) + (1-m_t) bar{l_pi} - m_t r_t G_tilde(pi))
```
bar{l_pi} = belief-weighted habit prior; r_t = evidence reliability (from policy cost dispersion).

### Action (L2)
```
mu = sum_pi q(pi) * mu_z(s_pi)
mu_x = decode(mu)
```
Transition drive: u_t = 1 - q(pi=stay). L1 uses tilde{u}_t = max(u_t, 1/n) as transition probability floor.

### Learning Objective
Total loss:
```
L_t = F_t + S_forward,t + alpha_rec * L_rec,t
```
where `L_rec,t = MSE(encode(x_t), z*_t)`; alpha_rec set per BPTT window as mean(F)/mean(L_rec).

### Training Loop
BPTT windows of 25 steps; gradients accumulated per window. At BPTT boundaries, blanket sensory states reset; L1 state and L2 thoughtseed activations preserved.

---

## Configuration

Edit `utils/config.py` to modify:
- Network/state parameters (Theta matrices, mu attractors)
- Thoughtseed priors (THOUGHTSEED_STATE_PRIORS)
- Dwell ranges (DWELL_TIMES) and transition priors (STATE_TRANSITION_PROBS)
- Learning rates (0.01 novice, 0.02 expert)
- Process noise (NOISE_LEVEL), BPTT_STEPS (25), TAIL_STEPS (2000)

---

## Reproducibility

Fixed random seed (42) ensures identical results across runs. Training is stochastic (e.g., MW dominance sampling), but fully seed-controlled.

---

## Citation

If you use this model in your research:

```
@article{enactive_inference_thoughtseeds_2026,
  author = {Kavi, P. C. and Friedman, D. A. and Patow, G.},
  title = {Thoughtseeds as Latent Causes in Enactive Inference: A Computational Phenomenology of Focused-Attention Meditation},
  journal = {Proc. R. Soc. A},
  year = {2026}
}
```
This repository is a significant step forward in enhancing the Thoughtseeds Framework for Enactive Inference. It builds upon the foundational work of the Thoughtseeds Framework, adapting code snippets from below:
  https://github.com/prakash-kavi/thoughtseeds_vipassana 
  https://github.com/prakash-kavi/viapssana_ts2  
  https://github.com/prakash-kavi/aif_iwai2025_thoughtseeds
