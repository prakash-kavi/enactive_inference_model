# Phase 4: Enactive Inference - Implementation Complete

## Summary

Implemented **minimal enactive inference** with forward dynamics and metacognitive control:
1. **Forward dynamics model** learns to predict action consequences  
2. **L3 precision-weighted loss** - dynamic accountability based on metacognition
3. **Forward-informed action selection** - evaluates actions via forward prediction
4. **Correct action conditioning** - forward model receives selected action as input
5. **Proper gradient flow** - forward model trainable via prediction error

## Mathematical Formulation

```
# Enactive loop
z_t = encoder(x_t)                           # Current thoughtseed
a_t* = argmin_a ||forward(x_t, a) - goal_a||²  # Select action via forward evaluation
x̂_{t+1} = forward(x_{t}, a_{t}*)             # Predict chosen action's consequence

# Forward model training (gradients flow here)
forward_loss = ||x_{t+1} - forward(x_t, a_t)||²

# L3-modulated loss
λ(t) = 0.05 + 0.1 * precision_L3(t)          # Dynamic weighting
Loss = reconstruction + KL + λ(t) * forward_loss
```

**Key architectural decision**: Action parameters (`mu_params`) are NOT trained via forward loss.
- Forward model learns state-action dynamics (gets gradients)
- Action selection USES forward predictions for evaluation (no_grad)
- Action parameters trained via VAE reconstruction loss only

This separation ensures stable training and avoids coupling that could destabilize the main VAE.

## Implementation Details

### Files Modified

**1. core/layer2/vae.py** (+30 lines)
- `forward_net`: [x_t (4) + z_t (5)] → hidden (32) → x̂_{t+1} (4)
- `predict_next(x_t, z_t)` method

**2. core/train/trainer.py** (+35 lines)
- L3 precision-weighted action loss: `λ = 0.05 + 0.1 * precision`
- **Fixed gradient flow**: Restructured to call `predict_next()` fresh each iteration
- Stores previous actual state+action (detached), computes prediction on current step
- **Fixed timing**: Action selection BEFORE forward prediction
- Action-conditioned: Uses `selected_action_mu` from policy selection

**3. core/layer2/bottleneck.py** (+45 lines)
- **Fixed action conditioning**: Uses `mu_candidate` as forward model input
- Evaluates candidate actions: current state + likely transitions
- **Action selection in no_grad**: Predictions used for evaluation only, no gradient flow to mu_params
- Returns `selected_action_mu` to trainer for action-conditioned forward prediction
- Removed unused `selected_state` variable

**4. core/train/logger.py** (+15 lines)
- Added `action_pred_error_history` tracking
- **Added per-state aggregates**: `average_action_pred_error_by_state`
- Export to `active_inference_params_*.json`

## Results

### Learning Performance

| Level  | Initial Error | Final Error | Reduction |
|--------|---------------|-------------|-----------|
| Expert | 0.00887       | 0.00495     | **44.1%** |
| Novice | 0.03058       | 0.02005     | **34.4%** |

### Per-State Action Prediction Errors (Expert)

- **Breath Focus**: 0.00526 (best - stable attractor)
- **Redirect Attention**: 0.00534 (controlled transition)
- **Meta Awareness**: 0.00876 (medium - detection phase)  
- **Mind Wandering**: 0.01008 (highest - unstable state)

### Key Features

1. **Action-Conditioned Forward Model**: Uses candidate action as input, not current state
2. **Correct Temporal Order**: Action chosen → prediction made → evaluation next step
3. **L3 Metacognitive Gating**: Precision modulates action accountability (0.05-0.15 range)
4. **Per-State Analytics**: Errors aggregated by meditation state for validation

## Fixes Applied (Codex Reviews - Feb 3)

### Round 1: Action Conditioning & Structure ✅

1. **HIGH: Action not truly conditioned** 
   - **Issue**: Forward model used same `z` for all candidates, only target differed
   - **Fix**: Now uses `mu_candidate` as action input → true action-consequence simulation
   - Files: bottleneck.py lines 316-325

2. **HIGH: Timing mismatch**
   - **Issue**: Prediction computed BEFORE action selection
   - **Fix**: Moved `prescriptive_action()` before `predict_next()` - correct causal order
   - Files: trainer.py line 158 before 162

3. **MEDIUM: Missing aggregates**
   - **Issue**: No per-state action error in JSON exports
   - **Fix**: Added `average_action_pred_error_by_state` to `active_inference_params_*.json`
   - Files: logger.py lines 270-280, 434

4. **LOW: Dead code**
   - **Issue**: `_last_x_actual` set but never used
   - **Fix**: Removed unused variable
   - Files: trainer.py line 90 deleted

### Round 2: Gradient Flow & Final Cleanup ✅

5. **HIGH: Forward loss non-trainable**
   - **Issue**: `detach()` on stored prediction broke gradient graph
   - **Fix**: Store previous actual state+action (detached), call `predict_next()` fresh each step
   - Gradients now flow through forward_net to train dynamics model
   - Files: trainer.py lines 146-165

6. **HIGH: Trainer used wrong input**
   - **Issue**: `predict_next(x_current, activations)` instead of selected action
   - **Fix**: Extract `selected_action_mu` from prescription, use in forward prediction
   - Files: trainer.py line 162-164

7. **MEDIUM: Unused selected_state**
   - **Issue**: `selected_state` computed but never used after assignment
   - **Fix**: Replaced with `_` to indicate intentionally unused
   - Files: bottleneck.py line 333

### Round 3: Architectural Clarification ✅

8. **DOCUMENTED: Action params not trained by forward loss**
   - **Observation**: `mu_params` don't receive gradients from forward loss (by design)
   - **Reason**: Action evaluation in `no_grad()`, previous state+action stored detached
   - **Justification**: Clean separation - forward model learns dynamics, VAE learns representations
   - **Status**: Intentional architecture, avoids training instability
   - This is forward-*informed* selection, not forward-*optimized* policy learning

## Architectural Design

### What Gets Trained by Forward Loss
✅ **Forward dynamics model** (`forward_net` in VAE)
- Receives gradients via `predict_next()` call in loss computation
- Learns mapping: (x_t, action) → x_{t+1}
- Trained by prediction error: `||x_{t+1} - forward(x_t, a_t)||²`

### What Doesn't Get Trained by Forward Loss (By Design)
❌ **Action parameters** (`mu_params` in bottleneck)
- Action evaluation happens in `torch.no_grad()` context
- Previous state+action stored with `.detach()`
- Trained only via main VAE reconstruction loss

### Why This Separation?
This is **forward-informed** action selection, not full model-based RL:
1. **Stability**: Avoids coupling that could destabilize VAE training
2. **Simplicity**: Clean separation of concerns (perception vs dynamics)
3. **Sufficiency**: Forward predictions guide action choice without direct optimization
4. **Future work**: Full policy optimization via forward model would be Phase 5

The forward model learns "what happens if I take action X", and action selection uses this knowledge to choose better actions, but the action parameters themselves aren't directly optimized by forward loss.

## Phase 4 Status

### Completed ✅
- ✅ Forward model implemented and trainable (gradient flow verified)
- ✅ L3 precision-weighted loss (dynamic 0.05-0.15 range)
- ✅ Action-conditioned forward prediction (uses selected_action_mu)
- ✅ Correct timing (action selection before prediction)
- ✅ Per-state aggregates exported to JSON
- ✅ All Codex review issues resolved (3 rounds)
- ✅ Visualization created (learning curves)
- ✅ Main training convergence maintained
- ✅ Architecture documented (forward-informed vs forward-optimized)

### Out of Scope (Deferred to Future Work)
- ❌ **Thoughtseed Competition** - Each thoughtseed with own forward model
- ❌ **Full Policy Optimization** - Gradient-based action parameter training via forward loss (would be model-based RL/planning)
- ❌ **Multi-step Planning** - Recursive forward prediction for longer horizons

### Next: Phase 5 (Scientific Validation)
- Ablation studies (disable forward model)
- MW→MA detection latency (novice vs expert)
- Free energy decomposition validation
- Statistical significance tests
- Compare with/without forward-informed actions

## Code Summary

**Total Addition**: ~130 lines
- vae.py: +30 (forward model network and prediction method)
- trainer.py: +35 (restructured forward loss with gradient flow)
- bottleneck.py: +45 (forward-informed action selection)
- logger.py: +15 (per-state aggregates)
- plot_phase4.py: +80 (new visualization script)

**Complexity**: Moderate
- Added action evaluation loop with forward prediction
- Dynamic L3 precision weighting
- Per-state analytics and aggregation
- Restructured loss computation for proper gradient flow

**Maintenance**: Clean
- All Phase 4 code marked with comments
- No dead code remaining after cleanup
- Clear architectural separation documented

---

**Status**: ✅ Phase 4 Complete, Validated & Documented  
**Date**: February 3, 2026  
**Review**: Passed 3 rounds of Codex review with all issues resolved
