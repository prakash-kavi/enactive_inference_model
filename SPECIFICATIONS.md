# Vipassana TS2 Transition Specification (v1.2)

## 1) Goals and non-goals
- Keep core behavior stable while removing the self_reflection thoughtseed.
- Introduce a derived reflection_score to drive meta-awareness.
- Avoid major architecture changes (global workspace, energy system) in this phase.

## 2) Canonical entities (match current code)
- Networks: DMN, VAN, DAN, FPN
- Thoughtseeds: attend_breath, pain_discomfort, pending_tasks, equanimity
- States: breath_focus, mind_wandering, meta_awareness, redirect_breath
- Experience levels: novice, expert

## 3) Current dynamics to preserve (compatibility contract)
- Networks are driven by thoughtseeds and state expectations, then updated via OU.
- Thoughtseeds evolve toward state-dependent targets with OU smoothing and network modulation.
- Meta-awareness is derived from state base levels and a compact reflection signal.
- State transitions remain VFE-driven with dwell-time constraints.

## 4) Derived reflection_score (replaces self_reflection thoughtseed)
### 4.1 Rationale
Self_reflection is removed as a thoughtseed to reduce redundancy and prevent conflation of
rumination (DMN-heavy) with skillful reflection (FPN-supported). A compact derived signal
captures reflective capacity without expanding the thoughtseed set.

### 4.2 Definition
Let:
- eq = z[equanimity]
- dmn = N[DMN]
- fpn = N[FPN]

Then:

reflection_score = sigmoid(
  w_eq * eq +
  w_fpn * fpn -
  w_dmn * dmn +
  b_reflection
)

Meta-awareness:
- meta_awareness = base_by_state + k_reflection * reflection_score

Notes:
- reflection_score is a derived signal, not a thoughtseed.
- Use fixed constants in code for now (no extra config surface in Phase 1).

### 4.3 VFE precision (unchanged intent)
Sensory precision should scale with salience; use VAN activation directly (not a thoughtseed).
Prior precision remains scaled by meta-awareness.

## 5) Removal scope (Phase 1)
- Remove self_reflection from thoughtseed list, targets, modulators, and network profiles.
- Remove self_reflection-specific modulation parameters (DMN/VAN/FPN reflection knobs).
- Update plots and analysis to expect 4 thoughtseeds.

## 6) Outputs for evaluation
- Keep existing outputs.
- Add reflection_score_history to time-series outputs for analysis.

## 7) Transition plan
Phase 1 (this change): remove self_reflection thoughtseed, add reflection_score, clean config.
Phase 2 (optional): further simplify transition logic or add higher-level control if needed.
