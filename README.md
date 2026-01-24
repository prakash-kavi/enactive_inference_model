# Vipassana-TS2: Thoughtseeds Framework Implementation

This repository contains the reference implementation of the **Thoughtseeds Framework** for modeling Vipassana meditation dynamics.

![Mediative Cycle](Mediative_cycle.jpg)

This is a stochastic simulation using coupled Ornstein–Uhlenbeck dynamics as an initial Active Inference formulation, serving as a scaffold for future full Active Inference implementations. It is a computational simulation and does not do empirical data fitting or neuroimaging data analysis.

## Conceptual Overview

![Thoughtseeds Framework](Thoughtseeds%20Framework.jpg)

- **Level 1**: Attentional networks (DMN, VAN, DAN, FPN)
- **Level 2**: Thoughtseed dynamics (competing markov-blanketed atentional agents activations)
- **Level 3**: Meta-cognition (precision modulation and policy based switching)

The model minimizes Variational Free Energy (VFE) through perception–action–learning cycles and captures qualitative expert–novice differences via parameterized priors and precision settings.

## Project Structure

### Core Implementation Files

- **`meditation_model.py`**: Core agent implementation (`AgentConfig`, `ActInfAgent`) — dynamics, inference, and learning updates.
- **`meditation_trainer.py`**: `Trainer` class that orchestrates experiment runs (extracted from the agent for testability and CI).
- **`meditation_utils.py`**: I/O helpers, `ou_update`, JSON serialization, and aggregate computations.
- **`config/meditation_config.py`**: Parameter profiles, thoughtseeds, states, network profiles, and tunable parameters.

### Pipeline Scripts

- **`run_training.py`**: Multi-seed convergence study for learning stable attractors (state-network expectations). Runs training across multiple seeds (42, 43, 44) and computes mean attractors for use in simulation.
- **`run_simulation.py`**: Runs simulations using recalibrated attractors from training. Loads mean network profiles and runs the model with fixed attractors (learning disabled).
- **`plot_training.py`**: Visualizes training results:
  - `FigS1_Convergence_{Level}.png`: Convergence diagnostics (Free Energy & State Occupancy) for Seed 42
  - `Fig3A_Radar_Comparison.png`: Learned Network Activation Profiles (Novice vs Expert)
- **`plot_simulation.py`**: Generates all diagnostic and attractor plots from simulation data:
  - `FigS1C_Hierarchy_TimeSeries.png`: Cognitive Hierarchy Time Series
  - `Fig3B_FreeEnergy.png`: Free Energy Bar Chart
  - `Fig3C_DwellTime.png`: Dwell Times
  - `Fig4A_Hierarchy_Novice.png` / `Fig4B_Hierarchy_Expert.png`: Individual Hierarchy Plots
  - `Fig5A_Attractor2D.png` / `Fig5B_Attractor3D.png`: Attractor Landscapes

### Visualization Module (`viz/`)

- **`plot_diagnostics.py`**: Diagnostic plotting functions (hierarchy, free energy, dwell times)
- **`plot_attractors.py`**: Attractor landscape visualizations (2D and 3D)
- **`plot_convergence.py`**: Convergence diagnostic utilities
- **`plotting_utils.py`**: Shared utilities (style settings, data loading, constants)

### Data Structure

```
data/
├── training/                          # Training outputs
│   ├── convergence_summary_{level}.json    # Mean attractors across seeds
│   ├── learned_weights_{level}_seed{seed}.json  # Individual seed results
│   └── convergence_plots_data/        # Full outputs for seed 42 (for plotting)
│       ├── thoughtseed_params_{level}.json
│       ├── active_inference_params_{level}.json
│       └── transition_stats_{level}.json
│
└── simulation/                        # Simulation outputs (using fixed attractors)
    ├── thoughtseed_params_{level}.json
    ├── active_inference_params_{level}.json
    └── transition_stats_{level}.json

plots/
├── training/                          # Training visualizations
│   ├── FigS1_Convergence_Novice.png
│   ├── FigS1_Convergence_Expert.png
│   └── Fig3A_Radar_Comparison.png
│
└── simulation/                        # Simulation visualizations
    ├── FigS1C_Hierarchy_TimeSeries.png
    ├── Fig3B_FreeEnergy.png
    ├── Fig3C_DwellTime.png
    ├── Fig4A_Hierarchy_Novice.png
    ├── Fig4B_Hierarchy_Expert.png
    ├── Fig5A_Attractor2D.png
    └── Fig5B_Attractor3D.png
```

## Workflow: Training → Simulation

The project follows a two-stage pipeline:

### Stage 1: Training (Calibration)

**Purpose**: Learn stable state-network attractors through multi-seed convergence study.

```bash
python run_training.py
```

**What it does**:
- Runs training for both novice and expert levels across 3 seeds (42, 43, 44)
- For each seed, trains the agent with learning enabled (`enable_learning=True`)
- Extracts final learned `state_network_expectations` for each seed
- Computes mean and standard deviation across seeds
- Saves convergence summaries to `data/training/convergence_summary_{level}.json`
- Saves full time-series outputs for seed 42 to `data/training/convergence_plots_data/` (for plotting)

**Outputs**:
- `data/training/convergence_summary_{level}.json`: Contains `network_profiles_mean` (the recalibrated attractors)
- `data/training/learned_weights_{level}_seed{seed}.json`: Individual seed results
- `data/training/convergence_plots_data/*`: Full outputs for seed 42

**Visualization**:
```bash
python plot_training.py
```
Generates convergence diagnostics and radar comparison plots in `plots/training/`.

### Stage 2: Simulation (Application)

**Purpose**: Run simulations using the fixed, learned attractors from training.

```bash
python run_simulation.py
```

**What it does**:
- Loads mean network profiles from `data/training/convergence_summary_{level}.json`
- Initializes agents with these trained attractors (overrides default expectations)
- Runs simulations with learning disabled (`enable_learning=False`)
- Saves simulation outputs to `data/simulation/`

**Outputs**:
- `data/simulation/thoughtseed_params_{level}.json`
- `data/simulation/active_inference_params_{level}.json`
- `data/simulation/transition_stats_{level}.json`

**Visualization**:
```bash
python plot_simulation.py
```
Generates all diagnostic and attractor plots in `plots/simulation/`.

## Reproducibility & Outputs

- The `Trainer.train()` method accepts optional `seed` (sets NumPy RNG) and `output_dir` (path for JSON outputs).
- Default numeric constants and thresholds are centralized in `config/meditation_config.py` for maintainability.
- Random number seed is set to 42 by default for simulations.
- Training uses seeds [42, 43, 44] for convergence study.

## Key Concepts

### Attractors (State-Network Expectations)

During training, the model learns **state-network attractors**: expected network activation profiles for each meditative state. These are stored as `state_network_expectations` in the agent's `learned_network_profiles`.

- **Training**: Attractors are learned and updated during training runs
- **Simulation**: Attractors are fixed (loaded from training means) and used to guide dynamics

### Meditative States

Four empirically-based meditative states:
- **Breath Focus** (BF): Focused attention on breath
- **Mind Wandering** (MW): Default mode, distracted state
- **Meta-Awareness** (MA): Awareness of mental states
- **Redirect Breath** (RA): Returning attention to breath

### Thoughtseeds

Five hypothetical constructs representing competing content-level activations:
- `attend_breath`
- `pain_discomfort`
- `pending_tasks`
- `aha_moment`
- `equanimity`

## Useful Commands

```bash
# Run full training pipeline (learn attractors)
python run_training.py

# Generate training visualizations
python plot_training.py

# Run simulation with trained attractors
python run_simulation.py

# Generate simulation visualizations
python plot_simulation.py

# Run individual visualization modules (legacy)
python -m viz.plot_convergence
python -m viz.plot_attractors
python -m viz.plot_diagnostics
```

## Requirements

See `requirements.txt` for Python package dependencies. Key dependencies include:
- `numpy`
- `matplotlib`
- `scipy` (for Ornstein–Uhlenbeck dynamics)

## Notes

- The model is a computational simulation framework, not an empirical data analysis tool.
- Training must be run before simulation to generate the required attractor files.
- Seed 42 outputs are saved during training for convergence visualization purposes.
