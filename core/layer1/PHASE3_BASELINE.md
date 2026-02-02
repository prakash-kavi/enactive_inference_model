# Phase 3 Baseline — Layer-1 Parameters (FROZEN)

**Date**: February 2, 2026  
**Version**: Phase 3.3 (MW burden + RA reorienting complete)  
**Status**: ✅ Frozen for post-Phase-5 tuning

---

## Rationale

This baseline captures Layer-1 parameters after completing Phase 3 deliverables:
- ✅ MW detection burden accumulation
- ✅ RA reorienting geometry modulation  
- ✅ State-dependent policy temperature
- ✅ Architectural refactoring (config separation, streamlining)
- ✅ Bug fixes (hazard clipping, config drift)

**Purpose**: Establish a stable reference point before advancing to Phase 4 (policy evaluation) and Phase 5 (full integration). Post-Phase-5, comprehensive realism tuning can be performed against this baseline.

---

## Key Parameter Values (Frozen)

### MVOU Dynamics
| Parameter | Expert | Novice |
|-----------|--------|--------|
| Base variance | 0.002 | 0.005 |
| Min variance | 0.0005 | 0.0005 |
| Max stiffness | 2.5 | 2.5 |
| Stiffness margin | 0.1 | 0.1 |
| Smoothing alpha (learned) | 0.9 | 0.9 |
| Smoothing alpha (base) | 0.7 | 0.7 |

### State Machine
| Parameter | Expert | Novice |
|-----------|--------|--------|
| BF dwell (s) | (15, 30) | (5, 15) |
| MW dwell (s) | (10, 20) | (20, 40) |
| MA dwell (s) | (1, 4) | (2, 6) |
| RA dwell (s) | (1, 4) | (3, 8) |
| Dwell distribution | Beta(2.0, 2.0) | Beta(2.0, 2.0) |
| Refractory period | 0.4s | 0.4s |

### MW Detection Burden
| Parameter | Expert | Novice |
|-----------|--------|--------|
| Accumulation rate (α) | 0.05 | 0.02 |
| Detection threshold | 0.09 | 0.11 |
| Detection gain (γ) | 0.95 | 1.20 |
| Decay rate (δ) | 0.15 | 0.08 |
| Activation weight | 0.80 | 1.25 |
| Coupling weight | 0.45 | 0.75 |
| Activation scale | 0.02 | 0.02 |
| Coupling scale | 0.25 | 0.25 |

### RA Reorienting
| Parameter | Expert | Novice |
|-----------|--------|--------|
| BF pull strength | 2.0 | 0.30 |
| Diffusion scale | 0.5 | 3.5 |
| Transition bias to BF | 0.22 | 0.0 |

---

## Validation Metrics (Phase 3 End)

### Training Convergence
- **Novice**: 300 episodes, max std = 0.061
- **Expert**: 300 episodes, max std = 0.039
- Both levels converged successfully

### Simulation Behavior
- **Novice MW burden**: 0.174 (mean)
- **Expert MW burden**: 0.097 (mean)
- **Expert RA success rate**: 95.2%
- **Novice RA success rate**: 100% (edge case - acceptable)

### Dwell Time Statistics
- Variability increased with Beta(2.0, 2.0) (was Beta(0.6, 0.6))
- Expert BF actual/limit ratio: ~0.74 ± 0.16
- Novice MW actual/limit ratio: ~0.74 ± 0.17

---

## Files Frozen

1. **`layer1_config_phase3_baseline.py`** - Full parameter snapshot
2. **`generative_process.py`** - MVOU implementation (206 lines → 177 lines after streamlining)
3. **`state_machine.py`** - Hazard-based transitions (155 lines → 127 lines after streamlining)

---

## Post-Phase-5 Tuning Targets

Deferred to comprehensive calibration pass:

### Stochasticity
- [ ] Increase dwell time variability (consider wider ranges or bimodal Beta)
- [ ] Add hazard noise injection for less deterministic transitions
- [ ] Calibrate FE entropy targets

### Realism
- [ ] Tune novice RA success rate to 80-90% (currently 100%)
- [ ] Adjust MVOU variance for more dynamic network fluctuations
- [ ] Refine burden normalization scales

### Expert-Novice Differentiation
- [ ] Validate expertise gaps match empirical meditation literature
- [ ] Ensure smooth learning trajectory from novice → expert

---

## Restoration Instructions

To revert to Phase 3 baseline after tuning experiments:

```bash
# Restore config
Copy-Item core\layer1\layer1_config_phase3_baseline.py core\layer1\layer1_config.py

# Re-run training with baseline
python -m run.run_training

# Verify baseline behavior
python -m run.run_simulation
python -m run.plot_training
```

---

## Change Log

**Phase 3.0**: Initial MW burden prototype  
**Phase 3.1**: Policy diagnostics logging  
**Phase 3.2**: Explicit q(pi) scaffold  
**Phase 3.3**: RA reorienting + burden normalization + state-dependent temperature  
**Phase 3.4**: Architectural refactor (config separation, streamlining, ~18% code reduction)  
**Phase 3.5**: Bug fixes (hazard clipping, config drift, import cleanup)  
**Phase 3.6**: Stochasticity improvements (Beta(2.0, 2.0), novice variance 0.005)  
**✅ FROZEN** (Feb 2, 2026)

---

## Next Phase

**Phase 4**: Policy evaluation and EFE term validation  
**Focus**: Layer-3 inference correctness, horizon mechanics, ambiguity-risk decomposition

Layer-1 remains **read-only** until Phase 5 completion.
