# Phase 4 Ablation Study Results

**Seed**: 42
**Timesteps**: 10000


## Expert Results

| Configuration | MW Dwell | BF Dwell | MW->MA Rate | Action Error (MW) |
|---------------|----------|----------|-------------|------------------|
| **Baseline** (full) | 15.6 | 33.2 | 0.720 | 0.0032089150246090003 |
| No Forward Model | 15.6 (+0.0) | 33.2 | 0.720 (+0.000) | 0.00000 |
| No L3 Weighting | 15.6 (+0.0) | 33.2 | 0.720 (+0.000) | 0.00310 |
| No Forward Actions | 15.6 (+0.0) | 33.2 | 0.720 (+0.000) | 0.00296 |

### Interpretation

- **MW Dwell**: Lower is better (faster escape from mind wandering)
- **MW->MA Rate**: Higher is better (more awareness transitions)
- **Positive deltas** = component removal hurt performance
- **Negative deltas** = component removal helped (rare, suggests overfitting)


## Novice Results

| Configuration | MW Dwell | BF Dwell | MW->MA Rate | Action Error (MW) |
|---------------|----------|----------|-------------|------------------|
| **Baseline** (full) | 14.3 | 92.3 | 0.828 | 0.005154232433027802 |
| No Forward Model | 14.4 (+0.1) | 92.2 | 0.828 (+0.000) | 0.00000 |
| No L3 Weighting | 14.4 (+0.0) | 92.2 | 0.828 (+0.000) | 0.00511 |
| No Forward Actions | 14.4 (+0.0) | 92.2 | 0.828 (+0.000) | 0.00540 |

### Interpretation

- **MW Dwell**: Lower is better (faster escape from mind wandering)
- **MW->MA Rate**: Higher is better (more awareness transitions)
- **Positive deltas** = component removal hurt performance
- **Negative deltas** = component removal helped (rare, suggests overfitting)

