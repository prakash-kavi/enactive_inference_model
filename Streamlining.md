## L2/L3 Simplification and Markov Blanket Design

This note documents structure-only streamlining of L2/L3 logic and Markov blankets. The goal is to reduce duplicated or tangled implementation while preserving the exact behavior described in the main manuscript and `supplementary.tex`. No algorithmic changes are introduced.

### 1. Baseline (fixed contract)

- **L1 process (frozen)**: L1 is the generative process (switching OU) and is not optimized by EM. No changes to its dynamics, parameters, or role are allowed in this pass.
- **Architecture**:
  - L1: generative process over networks \(\mathbf{x}_t\).
  - L2: thoughtseed inference and precision control (\(\mathbf{z}_t\), \(q(s_t)\), \(\pi_{x,t}\), VFE).
  - L3: meta-awareness and policy selection (\(m_t\), \(q(\pi_t)\), habit learning).
- **Core equations** (from `supplementary.tex`) are the ground truth:
  - VFE \(F(\mathbf{z}_t)\) with accuracy/complexity terms.
  - State belief \(q(s_t)\) from Gaussian evidence in \(\mathbf{z}\)-space.
  - Policy evaluation \(G(\pi)\), policy posterior \(q(\pi_t)\).
  - Meta-awareness dynamics \(m_t\).
  - Habit update \(oldsymbol{lpha}_s\).
  - Markov blanket channels \(\mathbf{y}^{1	o 2}_t\), \(\mathbf{a}^{2	o 1}_t\), \(\mathbf{y}^{2	o 3}_t\), \(\mathbf{a}^{3	o 2}_t\).

All streamlining must preserve these equations and numerical behavior (no plot-level changes).

### 2. L2: Recognition, VFE, and state belief

- **Single VFE implementation**:
  - Keep one implementation of
    \[
    F(\mathbf{z}_t) =
    \pi_{x,t}\,\lVert \mathbf{x}_t - \hat{\mathbf{x}}_t Vert^2
    + \lVert \mathbf{z}_t - oldsymbol{\mu}_z(s_t) Vert^2.
    \]
  - Any loss that refers to VFE should call `compute_vfe` rather than recomputing terms.

- **Precision flow (\(\pi_{x,t}\))**:
  - Precision is computed once in the training loop from forward surprisal \(S_{\mathrm{fwd},t}\) using the SI mapping, then passed as scalar `precision_sensory` through the L2--L3 blanket.
  - L2 exposes a single helper (e.g. `_precision_sensory`) that reads this scalar, clips it to \([	exttt{CLIP_MIN}, 	exttt{CLIP_MAX}]\), and returns one float used for:
    - reconstruction weight in `compute_vfe`;
    - any encoder/prior blend for VI initialization.
  - No alternative precision weights are introduced.

- **VI refinement responsibility**:
  - `update_posterior_z` is responsible only for:
    - initializing from a blend of \(\mathbf{z}^{\mathrm{enc}}_t\) and \(\mathbf{z}^{\mathrm{prior}}_t\);
    - fixed-step gradient updates on \(F(\mathbf{z}_t)\);
    - returning the refined \(\mathbf{z}_t\).
  - Logging/reshaping/bookkeeping should sit outside the VI core.

- **State belief gateway**:
  - `infer_state_belief` is the unique gateway that maps \(\mathbf{z}_t\) to \(q(s_t)\):
    \[
    q(s_t)=\operatorname{softmax}\!\left(-rac{\lVert \mathbf{z}_t - oldsymbol{\mu}_z(s) Vert^2}{	au_z}ight),
    \quad 	au_z = 2D_z\sigma_z^2.
    \]
  - No other code path recomputes state beliefs from \(\mathbf{z}_t\); consumers (including L3) read `state_belief` from the blanket or L2's interface.

### 3. L3: Meta-awareness, policy prior, and habit learning

- **Meta-awareness isolation**:
  - All logic for \(m_t\) lives inside one function (e.g. `update_meta_awareness_from_conflict`), implementing:
    - construction of \(q_{\mathrm{evid}}\) and \(q_{\mathrm{habit}}\);
    - KL-based divergence with GNW-style gate;
    - OU relaxation toward \(m_t^*\);
    - clipping to \([\lambda_{\min}, 1]\).
  - Outside this function, `meta_awareness` is treated as an opaque scalar.

- **Policy selection mirrors SI**:
  - `select_policy` should follow the SI equations directly:
    1. Policy prior: \(\log p_{	ext{prior}}(\pi)=\log p_{	ext{dwell}}(\pi)+\ln p_h(\pi)\).
    2. Policy precision: \(\gamma_t=	ext{clip}(m_t)\).
    3. Policy posterior: \(q(\pi_t)=\operatorname{softmax}(\log p_{	ext{dwell}}+\ln p_h-\gamma_t\,	ilde{G})\).

- **Habit learning as a single update rule**:
  - Keep the EMA/Dirichlet-style update:
    \[
    oldsymbol{lpha}_s \leftarrow (1-\kappa w_s)\,oldsymbol{lpha}_s + \kappa w_s\,\mathbf{q}(\pi_t),
    \quad \ln p_h(\pi\mid s)=\log\left(rac{oldsymbol{lpha}_s}{\mathbf{1}^Toldsymbol{lpha}_s}ight).
    \]
  - No other place alters \(oldsymbol{lpha}_s\) or defines a competing habit prior.

### 4. Markov blanket interfaces

- **L1--L2 blanket**:
  - Sensory: \(\mathbf{y}^{1	o 2}_t=(s_t,\mathbf{x}_t,d_t)\) via keys like `"state"`, `"x"`, `"dwell_progress"`.
  - Active: \(\mathbf{a}^{2	o 1}_t=(oldsymbol{\mu}_{x,t},\mathbf{q}(\pi_t))\) via keys `"mu_x"`, `"policy_state_probs"`.
  - Any blending between state attractors and policy-weighted predictions is computed upstream (training loop or L2), so L1 receives a single effective \(oldsymbol{\mu}_{x,t}\).

- **L2--L3 blanket**:
  - Sensory: \(\mathbf{y}^{2	o 3}_t=(q(s_t),\Pi_t,p_{	ext{dwell}}(\pi_t),G_t(\pi))\) via keys `"state_belief"`, `"policy_candidates"`, `"policy_priors"`, `"policy_costs"`.
  - Active: \(\mathbf{a}^{3	o 2}_t=(\pi_{x,t})\) via key `"precision_sensory"`.
  - The selected policy posterior \(q(\pi_t)\) is returned directly from L3 to L2 as a function result, not encoded as a blanket variable.
  - Adjacent-layer interaction only: L3 never reads L1 directly and L1 never reads L3 directly.

### 5. EM step hygiene (workflow)

- **E-step**: inference only (update \(\mathbf{z}_t\), \(q(s_t)\), \(q(\pi_t)\), \(m_t\)); no parameter updates.
- **M-step**: parameter updates only (encoder/decoder/forward model; habit prior update from E-step statistics).
- **Generative process**: L1 remains external and is never updated by EM.

### 6. Invariance checks after refactors

- Fix random seeds and run short diagnostics before and after refactors.
- Log key trajectories (\(\mathbf{z}_t\), \(q(s_t)\), \(m_t\), \(q(\pi_t)\), \(\pi_{x,t}\)).
- Accept changes only if differences are within numerical tolerance.

This document is a baseline contract: if a proposed change respects the constraints above, it is a simplification of the current L2/L3 and Markov blanket implementation, not a new model.
