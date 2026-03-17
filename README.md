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
               | Active:  sensory precision pi_x
+------------------------------v-------------------------------+
| Layer 2: Attentional Agent (Thoughtseeds)                    |
| - Compresses neural dynamics into 5 thoughtseeds             |
| - Encoder/decoder + forward model f(x,z)                     |
| - Evaluates expected free energy G(pi); passes evidence to L3 |
| - Sensory precision from forward surprisal (exp form)         |
+------------------------------+-------------------------------+
               | Markov Blanket L1<->L2
               | Sensory: x_t (DMN,VAN,DAN,FPN), dwell_progress d_t
               | Active:  mu_x, policy_state_probs
+------------------------------v-------------------------------+
| Layer 1: Neural Generative Process (MVOU)                    |
| - 4 brain networks (DMN, VAN, DAN, FPN)                      |
| - 4 meditation states (BF, MW, MA, RA)                       |
| - Multivariate Ornstein-Uhlenbeck dynamics                   |
| - Attractor mixing: mu <- (1-m_t)mu + m_t mu_x               |
+--------------------------------------------------------------+
```

## Summary
- States: BF, MW, MA, RA (focused attention cycle)
- Networks: DMN, VAN, DAN, FPN
- Thoughtseeds: attend_breath, pain_discomfort, pending_tasks, aha_moment, equanimity
- One run per phenotype: train (8k) → eval (2k, frozen) → plot (2k, frozen). Fig S1 uses eval window; fig3–fig5 use plot window (final 2k steps).

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

This runs **one contiguous run** per phenotype (12,000 steps total): **Train** (8,000 steps, learning on) → **Eval** (2,000 steps, frozen, used for Fig S1) → **Plot** (2,000 steps, frozen, used for fig3–fig5). Results are saved to `data/` and all plots are generated in `figures/`.

### Generate Plots from Existing Data

```bash
python run_enactive_inference.py plot
```

Generates publication-quality figures from saved results (full run: train+eval+plot). Fig S1 uses the eval segment; fig3–fig5 use the final plot window.

---

## Output

### Results (saved to `data/`)
- `training_results_expert_seed42.json` / `training_results_novice_seed42.json` — full run (train + eval + plot) per phenotype

Each contains: state/network/thoughtseed histories, free energy, meta-awareness, and transition statistics for all 12,000 steps.

### Plots (generated in `figures/`)
**Convergence (FigS1):**
- `FigS1_Convergence_Expert.pdf`, `FigS1_Convergence_Novice.pdf` — full 12,000-step run; eval and plot windows are shaded for reference

**Comparison (Fig 3):**
- `fig3a.pdf` — Network activation profiles across states (Expert vs Novice)
- `fig3b.pdf` — Dwell times per state (timesteps)
- `fig3c.pdf` — State transition probability matrices

**Hierarchy (Fig 4):**
- `fig4a.pdf`, `fig4b.pdf` — 3-layer hierarchical dynamics (L3 meta-awareness, L2 continuous thoughtseed trajectories, L1 networks)

**State space (Fig 5):**
- `fig5.pdf` — PCA trajectories (L2 thoughtseeds + L1 networks)

**Plot window:** fig3–fig5 and dwell/transition statistics use the final 2,000 steps (plot window). Fig S1 spans the full run with eval (steps 8,000–10,000) and plot (steps 10,000–12,000) windows shaded.

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
|   +-- config.py              # Constants, priors, TRAIN_STEPS, EVAL_STEPS, PLOT_STEPS, etc.
|   +-- math_utils.py          # Tensor/math operations
+-- data/                      # Run results (JSON, one file per phenotype)
+-- figures/                   # Generated figures (PDF)
+-- viz/                       # Plotting
    +-- analysis_utils.py      # prepare_tail_data, dwell/transition logic
    +-- attractors.py, convergence.py, analysis_utils.py
    +-- hierarchy.py, radar_plot.py, plotting_utils.py
```

---

## Configuration

Edit `utils/config.py` to modify:
- Network/state parameters (Theta matrices, mu attractors)
- Thoughtseed priors (THOUGHTSEED_STATE_PRIORS)
- Dwell ranges (DWELL_TIMES) and transition priors (STATE_TRANSITION_PROBS)
- Learning rates (0.01 novice, 0.02 expert)
- Process noise (NOISE_LEVEL), BPTT_STEPS (25), PLOT_STEPS (2000)

---

## Reproducibility

Fixed random seed (42) ensures identical results across runs. Training is stochastic (e.g., MW dominance sampling), but fully seed-controlled.

---

## Citation

If you use this model in your research:

```
@article{enactive_inference_thoughtseeds_2026,
  author = {Kavi, P. C. and Friedman, D. A. and Patow, G.},
  title = {Thoughtseeds as Latent Causes: A Computational Phenomenology of Focused-Attention Meditation},
  journal = {Proc. R. Soc. A},
  year = {2026}
}
```
This repository is a significant step forward in enhancing the Thoughtseeds Framework for Enactive Inference. It builds upon the foundational work of the Thoughtseeds Framework, adapting code snippets from below:

  https://github.com/prakash-kavi/thoughtseeds_vipassana 
  
  https://github.com/prakash-kavi/viapssana_ts2  
  
  https://github.com/prakash-kavi/aif_iwai2025_thoughtseeds
