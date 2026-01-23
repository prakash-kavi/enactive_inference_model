# Vipassana TS2 Transition Specification (v1.1)

## 1) Goals and non-goals
- Keep existing code behavior stable; changes must be additive.
- Focus only on improving self_reflection handling.
- Defer major architecture changes (global workspace, energy system) to a later phase.

## 2) Canonical entities (match current code)
- Networks: DMN, VAN, DAN, FPN
- Thoughtseeds: breath_focus, pain_discomfort, pending_tasks, self_reflection, equanimity
- States: breath_control, mind_wandering, meta_awareness, redirect_breath
- Experience levels: novice, expert

## 3) Current dynamics to preserve (compatibility contract)
- Network activations are driven by thoughtseeds and state expectations, then updated via OU.
- Thoughtseeds evolve toward state-dependent targets with OU smoothing and network modulation.
- Meta-awareness is derived from state base levels plus thoughtseed influences.
- State transitions are VFE-driven with dwell-time constraints.

## 4) Self-reflection: new addressing without breaking behavior
### 4.1 Rationale
The current model treats self_reflection as both a thoughtseed and a driver of meta-awareness.
This can blur rumination (DMN-heavy) with skillful reflection (FPN-supported). We want a
bounded reflection signal that distinguishes these modes, without changing the thoughtseed list.

### 4.2 Additive model: reflection_score (derived, optional)
Introduce a derived reflection_score that gates how self_reflection affects meta-awareness and
precision. This is an optional, additive change that defaults to the current behavior when
disabled.

Let:
- sr = z[self_reflection]
- eq = z[equanimity]
- dmn = N[DMN]
- fpn = N[FPN]

Then:

reflection_score = sigmoid(
  w_sr * sr +
  w_eq * eq +
  w_fpn * fpn -
  w_dmn * dmn +
  b_reflection
)

Usage (when enabled):
- meta_awareness = base_awareness + k_reflection * reflection_score + k_eq * eq
- Optionally scale sensory/prior precision (VFE) by reflection_score instead of raw sr

### 4.3 Backward compatibility
If reflection_score parameters are absent or disabled, keep current logic:
- meta_awareness = base + (self_reflection * 0.1) + (equanimity * 0.1)
- self_reflection remains a thoughtseed and stays in all logging and targets

## 5) Transition policy (unchanged)
- Maintain current dwell-time and VFE accumulator policy.
- reflection_score may only modulate meta-awareness or precision; it is not a new gate.

## 6) Optional config additions (non-breaking)
Add a new optional block to JSON profiles or config overrides:

self_reflection_policy:
  enabled: false
  w_sr: 1.0
  w_eq: 0.6
  w_fpn: 0.4
  w_dmn: 0.5
  b_reflection: -0.2
  k_reflection: 0.3

When disabled or missing, behavior is identical to the current code.

## 7) Transition plan
Phase 0 (doc-only): align names and clarify compatibility.
Phase 1 (safe code): add reflection_score computation with defaults that reproduce current behavior.
Phase 2 (optional): explore global workspace and energy dynamics if needed.

## 8) Outputs for evaluation
- Keep existing outputs.
- If reflection_score is enabled, log it as a time series for analysis.
