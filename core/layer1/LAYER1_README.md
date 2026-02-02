# Layer-1: Generative Process — Mathematical Model

## Overview

Layer-1 implements the **generative process** for brain network dynamics during meditation, modeled as a **Multivariate Ornstein-Uhlenbeck (MVOU) process** with state-dependent dynamics. It consists of two coupled components:

1. **Generative Process** (`generative_process.py`): MVOU dynamics for network activations
2. **State Machine** (`state_machine.py`): Markovian state transitions with hazard-based switching

---

## 1. Network Dynamics (MVOU Process)

### 1.1 Core Equation

Network activations $\mathbf{x}(t) \in \mathbb{R}^4$ (DMN, VAN, DAN, FPN) evolve according to:

$$
d\mathbf{x} = -\boldsymbol{\Theta}(s) (\mathbf{x} - \boldsymbol{\mu}(s)) \, dt + \boldsymbol{\Sigma}(s) \, d\mathbf{W}
$$

where:
- $s \in \{\text{BF}, \text{MW}, \text{MA}, \text{RA}\}$ is the current meditative state
- $\boldsymbol{\Theta}(s) \in \mathbb{R}^{4 \times 4}$ is the **coupling/stiffness matrix** (state-dependent)
- $\boldsymbol{\mu}(s) \in \mathbb{R}^4$ is the **attractor mean** (target activation profile)
- $\boldsymbol{\Sigma}(s) = \sigma(s) \mathbf{I}$ is the **diffusion matrix**
- $d\mathbf{W}$ is Wiener noise (Brownian motion)

### 1.2 Coupling Matrix $\boldsymbol{\Theta}(s)$

The coupling matrix encodes network interactions:

$$
\Theta_{ij}(s) = 
\begin{cases}
\theta_{\text{diag}}(s) & \text{if } i = j \text{ (self-restoring)} \\
\theta_{\text{base}}(s, i, j) + \Delta\theta_{\text{expert}}(s, i, j) & \text{if } i \neq j
\end{cases}
$$

**Interpretation:**
- **Positive coupling** ($\theta > 0$): Inhibition/restoring force (pulls networks back to attractor)
- **Negative coupling** ($\theta < 0$): Synergy/divergence (mutual activation)

**State-specific base coupling** (`THETA_BASE` in config):
- **Breath Focus (BF)**: DMN-DAN inhibition (0.5), weak DAN-FPN coupling (0.15)
- **Mind Wandering (MW)**: DMN-VAN synergy (-0.3), DMN-FPN synergy (-0.15)
- **Meta-Awareness (MA)**: Strong VAN-FPN synergy (-0.75), moderate cross-network coupling
- **Redirect Attention (RA)**: Mixed DMN-DAN/FPN inhibition and DAN-FPN reciprocal coupling

**Expert modulation** (applied only for `level='expert'`):
- **BF**: Increased diagonal stiffness (+0.4) for stability
- **RA**: Amplified DMN-DAN coupling (×1.5) for stronger control
- **MA**: Amplified VAN-FPN synergy (×1.4) for deeper awareness

### 1.3 Stiffness Clamping

To prevent numerical instability, off-diagonal coupling is bounded:

$$
\sum_{j \neq i} |\Theta_{ij}| \leq \theta_{\max} - \epsilon
$$

where $\theta_{\max} = 2.5$, $\epsilon = 0.1$. If violated:

$$
\Theta_{ij} \leftarrow \Theta_{ij} \cdot \frac{\theta_{\max} - \epsilon}{\sum_{k \neq i} |\Theta_{ik}|}
$$

Diagonal terms are adjusted to maintain positive definiteness:

$$
\Theta_{ii} = \max\left(\Theta_{ii}, \sum_{j \neq i} |\Theta_{ij}| + \epsilon \right)
$$

### 1.4 Attractor Means $\boldsymbol{\mu}(s)$

Target network activation profiles (`NETWORK_PROFILES` in config):

| State | DMN | VAN | DAN | FPN |
|-------|-----|-----|-----|-----|
| **BF (novice)** | 0.50 | 0.45 | 0.58 | 0.60 |
| **BF (expert)** | 0.40 | 0.45 | 0.60 | 0.65 |
| **MW (novice)** | 0.82 | 0.35 | 0.30 | 0.33 |
| **MW (expert)** | 0.65 | 0.50 | 0.40 | 0.50 |
| **MA (novice)** | 0.45 | 0.85 | 0.42 | 0.56 |
| **MA (expert)** | 0.40 | 0.78 | 0.42 | 0.55 |
| **RA (novice)** | 0.40 | 0.50 | 0.78 | 0.74 |
| **RA (expert)** | 0.35 | 0.50 | 0.78 | 0.70 |

**Learned attractors**: During training, Layer-2 learns refined attractors via backpropagation. If provided, these override base profiles.

### 1.5 Diffusion $\boldsymbol{\Sigma}(s)$

Noise magnitude controls stochasticity:

$$
\sigma(s) = \sqrt{\sigma_{\text{base}} \cdot g_{\text{noise}}}
$$

where:
- $\sigma_{\text{base}} = 0.002$ (expert), $0.005$ (novice)
- $g_{\text{noise}} \in [0, 2]$ is Layer-2 noise reduction control (Markov blanket input)
- Minimum clamp: $\sigma \geq 0.0005$

### 1.6 Numerical Integration

Euler-Maruyama with 2 substeps per timestep $dt = 0.2$s:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\Delta t}{2} \boldsymbol{\Theta}(s_t) (\mathbf{x}_t - \boldsymbol{\mu}(s_t)) + \sigma(s_t) \sqrt{\frac{\Delta t}{2}} \, \boldsymbol{\epsilon}_t
$$

where $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

Activations clipped: $\mathbf{x} \in [0.05, 0.9]$.

**Exponential smoothing** for observations:

$$
\tilde{\mathbf{x}}_t = \alpha \mathbf{x}_t + (1 - \alpha) \tilde{\mathbf{x}}_{t-1}
$$

with $\alpha = 0.9$ (learned attractors) or $0.7$ (base profiles).

---

## 2. State Machine

### 2.1 State Space

Four meditative states form a cognitive cycle:

$$
\mathcal{S} = \{\text{breath\_focus (BF)}, \text{mind\_wandering (MW)}, \text{meta\_awareness (MA)}, \text{redirect\_attention (RA)}\}
$$

### 2.2 Dwell Times

Each state has a minimum and maximum dwell duration $(d_{\min}, d_{\max})$ (in seconds):

| State | Novice | Expert |
|-------|--------|--------|
| BF | (5, 15) | (15, 30) |
| MW | (20, 40) | (10, 20) |
| MA | (2, 6) | (1, 4) |
| RA | (3, 8) | (1, 4) |

Actual dwell sampled from:

$$
d_{\text{actual}} \sim d_{\min} + \text{Beta}(2.0, 2.0) \cdot (d_{\max} - d_{\min})
$$

Beta(2.0, 2.0) produces a bell-shaped distribution centered around $(d_{\min} + d_{\max})/2$ with moderate spread.

### 2.3 Hazard-Based Transitions

At each timestep, transition probability (hazard rate) is computed:

$$
h(t) = 
\begin{cases}
0 & \text{if } t < t_{\text{refract}} \text{ (refractory period)} \\
0 & \text{if } t < d_{\min} \text{ (min dwell not satisfied)} \\
1 & \text{if } t \geq d_{\text{actual}} \text{ (max dwell reached)} \\
h_{\text{base}}(s, t) & \text{otherwise}
\end{cases}
$$

where $t_{\text{refract}} = 0.4$s.

**Base hazard** depends on state:

$$
h_{\text{base}}(s, t) = 
\begin{cases}
0.015 + 0.20p + 0.18g + \gamma \cdot \max(0, B - B_{\text{thresh}}) & \text{if } s = \text{MW} \\
0.008 + 0.14p + 0.15g & \text{otherwise}
\end{cases}
$$

where:
- $p = t / d_{\text{actual}}$ is dwell progress
- $g \in [0, 1]$ is Layer-2 transition drive (Markov blanket input)
- $B$ is mind-wandering burden (see Section 3)
- $\gamma$ is MW detection gain (1.20 novice, 0.95 expert)
- $B_{\text{thresh}}$ is burden detection threshold (0.11 novice, 0.09 expert)

Hazard clamped: $h \in [0, 0.95]$.

### 2.4 Transition Probabilities

Upon transition, next state sampled from:

$$
P(s' | s) = \text{softmax}\left(\mathbf{w}(s)\right)
$$

with weights $\mathbf{w}$ modulated by three biases:

**1. Cycle-forward bias** (from Layer-2 drive):

$$
\mathbf{w} \leftarrow (1 - 0.35g) \mathbf{w}_{\text{base}} + 0.35g \cdot \mathbf{e}_{s_{\text{next}}}
$$

where $s_{\text{next}}$ is the next state in the cycle (BF → MW → MA → RA → BF).

**2. MW persistence bias** (exit toward meta-awareness):

$$
\text{If } s = \text{MW} \text{ and } p > 0.6: \quad \mathbf{w} \leftarrow (1 - \beta) \mathbf{w} + \beta \cdot \mathbf{e}_{\text{MA}}
$$

where $\beta = (p - 0.6) / 0.4 \in [0, 1]$.

**3. RA reorienting bias** (preference toward BF):

$$
\text{If } s = \text{RA}: \quad \mathbf{w} \leftarrow (1 - b_{\text{RA}}) \mathbf{w} + b_{\text{RA}} \cdot \mathbf{e}_{\text{BF}}
$$

where $b_{\text{RA}} = 0.0$ (novice), $0.22$ (expert).

**Base transition probabilities** (`STATE_TRANSITION_PROBS` in config):

| From → To | BF | MW | MA | RA |
|-----------|----|----|----|----|
| **BF (novice)** | — | 0.90 | 0.08 | 0.02 |
| **BF (expert)** | — | 0.35 | 0.45 | 0.20 |
| **MW (novice)** | 0.15 | — | 0.68 | 0.17 |
| **MW (expert)** | 0.18 | — | 0.56 | 0.26 |
| **MA (novice)** | 0.15 | 0.06 | — | 0.79 |
| **MA (expert)** | 0.14 | 0.01 | — | 0.85 |
| **RA (novice)** | 0.88 | 0.07 | 0.05 | — |
| **RA (expert)** | 0.59 | 0.07 | 0.34 | — |

---

## 3. Mind-Wandering Detection (Burden Accumulation)

### 3.1 Burden Computation

MW detection burden $B_t$ is a weighted sum of two generative costs:

$$
B_t = w_{\text{act}} \cdot C_{\text{act}}(t) + w_{\text{coup}} \cdot C_{\text{coup}}(t)
$$

**Activation cost** (deviation from attractor):

$$
C_{\text{act}}(t) = \min\left(2.5, \frac{\|\mathbf{x}_t - \boldsymbol{\mu}(s_t)\|^2}{s_{\text{act}}}\right)
$$

**Coupling cost** (network interaction load):

$$
C_{\text{coup}}(t) = \min\left(2.5, \frac{\text{mean}_{i \neq j} |\Theta_{ij}(s_t)|}{s_{\text{coup}}}\right)
$$

Weights and scales (novice/expert):

| Parameter | Novice | Expert |
|-----------|--------|--------|
| $w_{\text{act}}$ | 1.25 | 0.80 |
| $w_{\text{coup}}$ | 0.75 | 0.45 |
| $s_{\text{act}}$ | 0.02 | 0.02 |
| $s_{\text{coup}}$ | 0.25 | 0.25 |

Costs capped at 2.5 to prevent scale dominance.

### 3.2 Exponential Moving Average

Burden evolves as:

$$
B_t = 
\begin{cases}
(1 - \alpha) B_{t-1} + \alpha (w_{\text{act}} C_{\text{act}} + w_{\text{coup}} C_{\text{coup}}) & \text{if } s_t = \text{MW} \\
(1 - \delta) B_{t-1} & \text{otherwise}
\end{cases}
$$

where:
- $\alpha$ = accumulation rate (0.02 novice, 0.05 expert)
- $\delta$ = decay rate (0.08 novice, 0.15 expert)

Burden influences MW exit hazard (Section 2.3).

---

## 4. Redirect Attention Geometry Modulation

During $s = \text{RA}$, MVOU dynamics are modified to facilitate reorienting to BF:

### 4.1 Attractor Pull

Attractor mean blended with BF target:

$$
\boldsymbol{\mu}_{\text{RA}}(t) = (1 - \rho) \boldsymbol{\mu}(\text{RA}) + \rho \boldsymbol{\mu}(\text{BF})
$$

where:

$$
\rho = \frac{p_{\text{pull}}}{1 + p_{\text{pull}}}
$$

with $p_{\text{pull}} = 0.30$ (novice), $2.0$ (expert).

### 4.2 Diffusion Scaling

Noise variance amplified (novices) or reduced (experts):

$$
\sigma_{\text{RA}} = \sigma_{\text{base}} \cdot r_{\text{diff}}
$$

where $r_{\text{diff}} = 3.5$ (novice), $0.5$ (expert).

**Result**: Novices experience high-noise exploration (slow convergence), experts have smooth rapid reorienting.

### 4.3 Coupling Augmentation

Diagonal stiffness increased:

$$
\Theta_{ii}(\text{RA}) \leftarrow \Theta_{ii}(\text{RA}) + 0.12 \cdot p_{\text{pull}}
$$

Strengthens self-restoring forces during reorienting.

---

## 5. Markov Blanket Interface (L1 ↔ L2)

Layer-1 **only exposes** through the Markov blanket:

### 5.1 Outputs (L1 → L2)

$$
\mathcal{B}_{\text{L1→L2}} = \{\tilde{\mathbf{x}}, s\}
$$

- $\tilde{\mathbf{x}} \in \mathbb{R}^4$: Smoothed network activations (DMN, VAN, DAN, FPN)
- $s \in \mathcal{S}$: Current meditative state

### 5.2 Inputs (L2 → L1)

$$
\mathcal{B}_{\text{L2→L1}} = \{g, g_{\text{noise}}, \boldsymbol{\mu}_{\text{agent}}, \beta_{\text{enactive}}\}
$$

- $g \in [0, 1]$: Transition drive (modulates hazard and cycle-forward bias)
- $g_{\text{noise}} \in [0, 2]$: Noise reduction control (scales diffusion)
- $\boldsymbol{\mu}_{\text{agent}} \in \mathbb{R}^4$ (optional): Agent-learned attractor bias
- $\beta_{\text{enactive}} \in [0, 1]$ (optional): Enactive bias strength

If agent bias provided:

$$
\boldsymbol{\mu}_{\text{effective}} = (1 - \beta_{\text{enactive}}) \boldsymbol{\mu}(s) + \beta_{\text{enactive}} \boldsymbol{\mu}_{\text{agent}}
$$

---

## 6. Implementation Details

### 6.1 Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_STIFFNESS` | 2.5 | Maximum off-diagonal coupling sum |
| `STIFFNESS_MARGIN` | 0.1 | Safety margin for clamping |
| `N_SUBSTEPS` | 2 | Euler-Maruyama substeps per dt |
| `INIT_ACTIVATION` | 0.5 | Initial network activations |
| `BASE_VARIANCE` (expert) | 0.002 | Expert MVOU noise variance |
| `BASE_VARIANCE` (novice) | 0.005 | Novice MVOU noise variance |
| `MIN_VARIANCE` | 0.0005 | Minimum noise floor |
| `EPS` | 1e-6 | Numerical stability epsilon |

### 6.2 Files

- **`generative_process.py`**: `Layer1Process` class implementing MVOU dynamics
- **`state_machine.py`**: `StateMachine` class implementing hazard-based transitions
- **`layer1_config.py`**: Configuration constants (coupling maps, attractor profiles, dwell times, generative costs)
- **`__init__.py`**: Public API exposing `Layer1Process` and `StateMachine`

### 6.3 External Dependencies

- PyTorch (MVOU tensor operations)
- NumPy (state machine sampling)
- `utils.meditation_config`: Global constants (STATES, NETWORKS, DEFAULTS, EXPERIENCE_LEVELS, EPS)

---

## 7. Key Architectural Principles

1. **Encapsulation**: All MVOU coupling matrices, state transition probabilities, and generative costs are internal to Layer-1. Only smoothed activations and current state cross the Markov blanket.

2. **Expertise Differentiation**: Novice vs. expert behavior encoded in:
   - Dwell time ranges (novices dwell longer in MW, shorter in BF)
   - MVOU variance (novices have 2.5× higher noise)
   - Coupling modulation (experts have stronger BF stabilization and RA reorienting)
   - MW burden sensitivity (novices accumulate faster, decay slower)

3. **State-Aware Costs**: Generative costs (MW burden, RA geometry) are computed from MVOU dynamics, creating feedback loops that bias state transitions toward cognitive regulation.

4. **Hierarchical Control**: Layer-2 can modulate Layer-1 dynamics via transition drive and noise reduction, but cannot directly set states or override MVOU physics.

---

## 8. Mathematical Validation

### 8.1 Stability

MVOU process is **mean-reverting** (stable) if:

$$
\text{Re}(\lambda_i(\boldsymbol{\Theta})) > 0 \quad \forall i
$$

Stiffness clamping (Section 1.3) ensures diagonal dominance, guaranteeing positive eigenvalues.

### 8.2 Ergodicity

For fixed state $s$, stationary distribution is Gaussian:

$$
\mathbf{x}_{\infty} \sim \mathcal{N}\left(\boldsymbol{\mu}(s), \boldsymbol{\Theta}^{-1}(s) \boldsymbol{\Sigma}(s) \boldsymbol{\Sigma}^T(s) \boldsymbol{\Theta}^{-T}(s)\right)
$$

### 8.3 Markov Property

State transitions satisfy:

$$
P(s_{t+1} | s_t, s_{t-1}, \ldots) = P(s_{t+1} | s_t)
$$

Hazard computation depends only on current state, dwell progress, and instantaneous burden/drive.

---

## References

- **Ornstein-Uhlenbeck Process**: Uhlenbeck & Ornstein (1930), "On the Theory of the Brownian Motion"
- **Active Inference**: Friston et al. (2017), "Active Inference: A Process Theory"
- **Meditation Neuroscience**: Hasenkamp et al. (2012), "Mind wandering and attention during focused meditation"

