# Vipassana-TS2: Thoughtseeds Framework Implementation

This repository contains the reference implementation of the **Thoughtseeds Framework** for modeling Vipassana meditation dynamics.

This is a stochastic simulation using coupled Ornstein–Uhlenbeck dynamics as an initial Active Inference formulation, serving as a scaffold for future full Active Inference implementations. It is a computational simulation and does not do emprical data fitting or neuromimaging data analysis.

## Conceptual Overview

- Level 1: attentional networks (DMN, VAN, DAN, FPN)
- Level 2: thoughtseed dynamics (competing content-level activations)
- Level 3: meta-cognition (precision modulation and policy switching)

The model minimizes Variational Free Energy (VFE) through perception–action–learning cycles and captures qualitative expert–novice differences via parameterized priors and precision settings.

## Project Structure (key files)

- **`meditation_model.py`**: core agent implementation (`AgentConfig`, `ActInfAgent`) — dynamics, inference, and small learning updates.
- **`meditation_trainer.py`**: `Trainer` class that orchestrates experiment runs (extracted from the agent for testability and CI).
- **`meditation_utils.py`**: I/O helpers, `ou_update`, JSON serialization, and aggregate computations.
- **`meditation_config.py`**: parameter profiles, `DEFAULTS` for central numeric constants.
- **`run_simulation.py`**: high-level entrypoint; uses `Trainer.train()` and supports reproducible runs (seed + output_dir).
- **`viz/`**: plotting scripts (`plot_attractors.py`, `plot_diagnostics.py`, `plot_convergence.py`) and plotting utilities.
- **`data/`**, **`plots/`**: generated JSON outputs and figures.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the simulation (reproducible):
```bash
python run_simulation.py
```
`run_simulation.py` calls `Trainer.train(..., seed=42)` by default; to change seed or output path, edit or call the trainer directly from a small script:

```python
from meditation_model import ActInfAgent
from meditation_trainer import Trainer

agent = ActInfAgent(experience_level='novice', timesteps_per_cycle=1000)
Trainer(agent).train(save_outputs=True, output_dir='data/my_run', seed=12345)
```

3. Regenerate publication figures:
```bash
python -m viz.plot_attractors
python -m viz.plot_diagnostics
python -m viz.plot_convergence
```

Generated outputs appear under `data/` (JSON) and `plots/` (PNGs).

## Reproducibility & Outputs

- The `Trainer.train()` method accepts optional `seed` (sets NumPy RNG) and `output_dir` (path for JSON outputs).
- Default numeric constants and thresholds are centralized in `meditation_config.DEFAULTS` for maintainability.
- JSON outputs include `transition_stats_<level>.json`, `thoughtseed_params_<level>.json`, and `active_inference_params_<level>.json`.

## Useful Commands

```bash
# run full simulation
python run_simulation.py

# regenerate plots
python -m viz.plot_attractors
python -m viz.plot_diagnostics
python -m viz.plot_convergence

# run an ad-hoc trainer with custom seed/output
python - <<'PY'
from meditation_model import ActInfAgent
from meditation_trainer import Trainer
Trainer(ActInfAgent('novice', timesteps_per_cycle=500)).train(save_outputs=True, output_dir='data/run_500', seed=2026)