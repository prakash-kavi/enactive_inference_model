# Knob Reduction Opportunities

Candidates for simplifying the codebase by removing or consolidating configurable parameters.

---

## 1. **Remove (no-op or always-same)**

| Knob | Location | Current | Recommendation |
|------|----------|---------|----------------|
| **PRECISION_WEIGHT_SCALE** | config.py:153 | 1.0 | Remove; it multiplies precision by 1.0 and has no effect. Inline the multiplication away. |
| **smoothing** (Markov blankets) | markov_blankets.py, l2, l3, training_loop | Always 0.0 | Remove the parameter; pass-through overwrite is the only behavior used. Or make it a fixed constant if you ever want non-zero. |
| **base_precision** init | markov_blankets.py:111, 116 | 0.5 | Could use CLIP_MIN or a single constant; not config-driven. |

---

## 2. **Consolidate (multiple knobs → fewer)**

| Knobs | Location | Current | Recommendation |
|-------|----------|---------|----------------|
| **L3_POLICY_LR_BY_LEVEL + L3_POLICY_STRENGTH_BY_LEVEL** | config.py:168-174 | 4 values (2 per phenotype) | Consider a single **L3_HABIT_STRENGTH** per phenotype; policy_strength controls both the prior weight and effective update rate. Or fix one and vary only the other. |
| **Z_NOISE_STD_BY_STATE** | config.py:178-183 | 4 values (BF 0.02, MW 0.08, MA 0.03, RA 0.03) | Reduce to 2: `high_noise` (MW=0.08), `low_noise` (others=0.03). Or even 1 constant (e.g. 0.03) if MW can tolerate it. |
| **base_diag (Theta)** | l1_generative_process.py:120 | 0.50 (MW) vs 0.15 (else) | Could move to config as 2 constants: `THETA_DIAG_MW`, `THETA_DIAG_DEFAULT`, or fold into THETA_BASE if that structure allows. |
| **theta_boost multipliers** | l1_generative_process.py:137, 141-142, 146-148 | 0.4, 1.5, 1.4 (hardcoded) | Move to config: `THETA_BOOST_BF`, `THETA_BOOST_RA`, `THETA_BOOST_MA` — or use one `THETA_BOOST_SCALE` (e.g. 1.3) applied uniformly if the ratios can be fixed. |

---

## 3. **Fix to constants (config → hardcode)**

| Knob | Location | Recommendation |
|------|----------|----------------|
| **PRIOR_VARIANCE_Z** | config.py:157 | Used in state belief softmax and epistemic proxy. If you never tune it, fix to 1.0 in code and remove from config. |
| **drive_boost coefficient** | l1_generative_process.py:70 | `0.5 * transition_drive` — fix 0.5 as constant or add L1_DRIVE_BOOST_SCALE=0.5 to config only if you plan to vary it. |
| **bias_strength (L2→L1 blend)** | l1_generative_process.py:206 | `mu = 0.5 * mu + 0.5 * mu_x` — currently fixed; either add to config or leave as constant (no new knob). |
| **z_init blend** | l2_recognition.py:138 | `z_init = 0.5 * z_prev + 0.5 * z_recognition` — fix as constant unless you have a reason to vary. |
| **grad clip** | training_loop.py:313 | `max_norm=1.0` — standard; leave as constant. |
| **_clamp_theta constants** | l1_generative_process.py:161-170 | 0.1, MAX_STIFFNESS-0.1 — implementation detail; leave as constants. |

---

## 4. **VI knobs (potential simplification)**

| Knobs | Location | Current | Recommendation |
|-------|----------|---------|----------------|
| **VI_STEPS, VI_LR, VI_MISMATCH_THRESHOLD** | config.py:112-115 | 2, 0.2, 0.02 | Option A: Fix VI_STEPS=2, VI_LR=0.2, remove VI_MISMATCH_THRESHOLD and always run 1–2 steps (simplest). Option B: Keep VI_STEPS and VI_LR; set VI_MISMATCH_THRESHOLD = 0.01 (tighter) or derive from prior variance. |
| **grad clamp in VI** | l2_recognition.py:217 | `clamp(grad, -5.0, 5.0)` | Fix as constant; rarely needs tuning. |

---

## 5. **Precision / EMA knobs**

| Knobs | Location | Current | Recommendation |
|-------|----------|---------|----------------|
| **PRECISION_TAU, L3_META_TAU** | config.py:155, 184 | 4.0, 1.0 | Consider one EMA time constant for "sensory/surprisal" (PRECISION_TAU) and one for "meta-awareness" (L3_META_TAU). If they can share a single value, use **EMA_TAU = 2.0** for both. |
| **CLIP_MIN, CLIP_MAX** | config.py:16-17 | 0.05, 0.9 | Keep; these are model constraints. |

---

## 6. **Architectural (low priority)**

| Knob | Location | Recommendation |
|------|----------|----------------|
| **hidden_dim** | l2_recognition.py:40, 111 | 32 — fix unless you do architecture search. |
| **N_SUBSTEPS** | l1_generative_process.py:28 | 2 — fix for MVOU stability. |
| **INIT_ACTIVATION** | l1_generative_process.py:29 | 0.5 — could use (CLIP_MIN+CLIP_MAX)/2. |

---

## Summary: Highest-impact reductions

1. **Remove PRECISION_WEIGHT_SCALE** — no-op at 1.0.
2. **Remove or hardcode Markov blanket smoothing** — always 0.
3. **Reduce Z_NOISE_STD_BY_STATE** — 4 → 2 values (MW vs others).
4. **Fix PRIOR_VARIANCE_Z** — hardcode 1.0 if not tuned.
5. **Consolidate L3 knobs** — explore single L3_HABIT_STRENGTH per phenotype.
6. **Optionally unify PRECISION_TAU and L3_META_TAU** — if one value works for both EMAs.
