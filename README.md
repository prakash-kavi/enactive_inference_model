# Enactive Inference Model

This repository implements a three-layer active-inference model of focused-attention meditation. It simulates expert and novice phenotypes across the canonical cycle of breath focus, mind wandering, meta-awareness, and redirecting attention, then produces publication figures and saved JSON results for analysis.

The model couples:
- Layer 1 neural-network dynamics over DMN, VAN, DAN, and FPN
- Layer 2 latent thoughtseed inference and policy evaluation
- Layer 3 metacognitive monitoring, habit priors, and meta-awareness

One run per phenotype consists of 12,000 contiguous timesteps:
- Train: 8,000 steps with learning enabled
- Eval: 2,000 frozen steps
- Plot: 2,000 frozen steps used for Fig. 3-Fig. 5 analyses

Fig. S1 is generated from the full run with eval and plot windows shaded.

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python -m run_enactive_inference run
```

Generate plots from existing saved results:

```bash
python -m run_enactive_inference plot
```

This project runs on CPU only; no GPU support is required or used.

---

## Architecture

![Meditative Cycle](figures/fig1.jpg) ![Thoughtseeds Framework](figures/fig2.jpg)

- Layer 1 is a neural generative process over four large-scale networks (DMN, VAN, DAN, FPN) with four meditation regimes (BF, MW, MA, RA) and multivariate Ornstein-Uhlenbeck dynamics.
- Layer 2 compresses network dynamics into five thoughtseeds, performs encoder/decoder inference, and evaluates expected free energy over stay/switch policy candidates.
- Layer 3 computes meta-awareness as gated divergence between policy evidence and habit, then selects the policy posterior that regulates action selection.
- L1 and L2 interact through a Markov blanket carrying network activity and dwell progress upward, and descending network predictions plus policy-state probabilities downward.
- L2 and L3 interact through a Markov blanket carrying state belief, policy evidence, and thoughtseed activations upward, and sensory precision downward.

### Model Summary

- States: BF, MW, MA, RA
- Networks: DMN, VAN, DAN, FPN
- Thoughtseeds: attend_breath, pain_discomfort, pending_tasks, aha_moment, equanimity

---

## Outputs

### Saved Results

Running the full pipeline writes one JSON file per phenotype to `data/`:

- `training_results_expert_seed42.json`
- `training_results_novice_seed42.json`

Each file contains the full 12,000-step run:
- state history
- network activation history
- thoughtseed activation history
- thoughtseed prior activation history
- free-energy history
- loss history
- meta-awareness history
- raw transition events

### Generated Figures

Plots are written to `figures/`.

**Fig. S1: Convergence diagnostics**
- `FigS1_Convergence_Expert.pdf`
- `FigS1_Convergence_Novice.pdf`
- Full-run diagnostics with eval and plot windows shaded

**Fig. 3: Expert vs novice comparison**
- `fig3a.pdf`: state-conditional network activation profiles
- `fig3b.pdf`: dwell-time comparisons by state
- `fig3c.pdf`: transition probability matrices from the plot window

**Fig. 4: Hierarchical dynamics**
- `fig4a.pdf`: novice hierarchical traces
- `fig4b.pdf`: expert hierarchical traces
- Shows L3 meta-awareness, L2 thoughtseed trajectories, and L1 network dynamics

**Fig. 5: State-space geometry**
- `fig5.pdf`: pooled PCA projections of L2 thoughtseed and L1 network trajectories

**Plot window convention**
- Fig. 3-Fig. 5 and dwell/transition summaries use the final 2,000-step plot window.
- Fig. S1 spans the full run with eval (steps 8,000-10,000) and plot (steps 10,000-12,000) windows highlighted.

---

## Repository Layout

```text
.
+-- run_enactive_inference.py      # Main entry point (run | plot)
+-- model/
|   +-- training_loop.py           # Variational EM loop and result packaging
|   +-- phenotype.py               # Expert/novice phenotype definitions
|   +-- l1_generative_process.py   # Layer 1 MVOU dynamics
|   +-- l2_recognition.py          # Layer 2 inference, decoder, forward model, EFE
|   +-- l3_metacognition.py        # Layer 3 meta-awareness and policy selection
|   +-- markov_blankets.py         # Markov blanket interfaces
+-- utils/
|   +-- config.py                  # Core constants and phenotype tables
|   +-- math_utils.py              # Shared tensor/math helpers
|   +-- extract_stats.py           # CLI stats summary from saved JSON
+-- viz/
|   +-- analysis_utils.py          # Tail-window statistics and aggregation
|   +-- convergence.py             # Fig. S1
|   +-- radar_plot.py              # Fig. 3A
|   +-- hierarchy.py               # Fig. 4
|   +-- attractors.py              # Fig. 5
|   +-- plotting_utils.py          # Shared plotting style/helpers
+-- data/                          # Saved run outputs
+-- figures/                       # Generated and manuscript figure assets
+-- tests/
    +-- test_invariants.py         # Core invariants and numerical checks
```

---

## Configuration

Edit `utils/config.py` to modify:

- network/state parameters (`THETA_BASE`, network attractors)
- thoughtseed priors (`THOUGHTSEED_STATE_PRIORS`)
- dwell ranges (`DWELL_TIMES`)
- transition priors (`STATE_TRANSITION_PROBS`)
- learning rates (`LEARNING_RATES`)
- numerical settings such as `NOISE_LEVEL`, `BPTT_STEPS`, `TRAIN_STEPS`, `EVAL_STEPS`, and `PLOT_STEPS`

---

## Reproducibility

The default seed is fixed at 42.

- `torch.manual_seed` and `np.random.seed` are set in the trainer
- Layer 1 uses its own seeded `RandomState`
- stochastic transitions and process noise are therefore reproducible under the same configuration and dependency environment

---

## Citation

If you use this model in your research:

```bibtex
@article{enactive_inference_thoughtseeds_2026,
  author = {Kavi, P. C. and Friedman, D. A. and Patow, G.},
  title = {Thoughtseeds as Latent Causes: A Computational Phenomenology of Focused-Attention Meditation},
  journal = {Proc. R. Soc. A},
  year = {2026}
}
```

This repository builds on earlier Thoughtseeds-related codebases, including:

- https://github.com/prakash-kavi/thoughtseeds_vipassana
- https://github.com/prakash-kavi/viapssana_ts2
- https://github.com/prakash-kavi/aif_iwai2025_thoughtseeds
