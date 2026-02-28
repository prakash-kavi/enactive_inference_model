"""Analysis and visualization utilities for meditation state dynamics
Provides metrics computation and plotting utilities:
- State dwell times (attractor stability)
- Transition dynamics (state space trajectory)
- Free energy evolution (learning convergence)
- Meta-awareness trajectory (content-based monitoring)
"""

import numpy as np
from typing import Dict

from utils.config import STATES
from viz.analysis_utils import get_tail_window, compute_tail_statistics, compute_residual_scales, TAIL_STEPS

def compute_metrics(results: Dict, use_tail: bool = True) -> Dict:
    """Compute summary metrics from training results (default: uses tail window).
    
    Args:
        results: Training results dict from MeditationTrainer
        use_tail: If True, computes metrics on the last TAIL_STEPS (converged behavior).
                  If False, computes on full lifetime history.
    
    Returns:
        Summary metrics dict
    """
    # 1. Basic counts
    run_len = len(results['state_history'])
    metrics = {
        'experience_level': results['experience_level'],
        'timesteps': run_len,
        'analyzed_steps': TAIL_STEPS if (use_tail and run_len > TAIL_STEPS) else run_len,
        'num_transitions': len(results['transitions']),
    }
    
    # 2. Compute Statistics (Tail or Full)
    steps_to_analyze = TAIL_STEPS if use_tail else run_len
    stats = compute_tail_statistics(results, STATES, tail_steps=steps_to_analyze)
    
    # Average dwell times
    for state in STATES:
        dwell = stats['dwell_times'].get(state, 0.0)
        metrics[f'avg_dwell_{state}'] = dwell
    
    # Transition rates
    trans_matrix = stats['transition_matrix']
    if 'mind_wandering' in trans_matrix and 'meta_awareness' in trans_matrix['mind_wandering']:
        metrics['mw_to_ma_rate'] = trans_matrix['mind_wandering']['meta_awareness']
    else:
        metrics['mw_to_ma_rate'] = 0.0
    
    # 3. Free energy convergence (Always look at history absolute positions)
    fe_history = results['free_energy_history']
    if len(fe_history) > 100:
        metrics['initial_fe'] = float(np.mean(fe_history[:100]))
        metrics['final_fe'] = float(np.mean(fe_history[-100:]))
        metrics['fe_reduction'] = metrics['initial_fe'] - metrics['final_fe']
    else:
        metrics['initial_fe'] = fe_history[0] if fe_history else 0.0
        metrics['final_fe'] = fe_history[-1] if fe_history else 0.0
        metrics['fe_reduction'] = 0.0
    
    # 4. Action prediction errors 
    for state, error in results['avg_action_errors'].items():
        metrics[f'action_error_{state}'] = error

    # 4b. EFE diagnostics (tail window means)
    tail = get_tail_window(results, tail_steps=steps_to_analyze)
    if tail.get('efe_prag_history'):
        metrics['efe_prag_mean'] = float(np.mean(tail['efe_prag_history']))
    else:
        metrics['efe_prag_mean'] = 0.0
    if tail.get('efe_epi_history'):
        metrics['efe_epi_mean'] = float(np.mean(tail['efe_epi_history']))
    else:
        metrics['efe_epi_mean'] = 0.0

    # 5. Residual-based Gaussian scales (tail window)
    residual_scales = compute_residual_scales(results, tail_steps=steps_to_analyze)
    metrics.update(residual_scales)
    
    return metrics



def print_summary(results: Dict) -> None:
    """Print text summary of training results."""
    metrics = compute_metrics(results)
    
    print(f"\n{'='*60}")
    print(f"MEDITATION TRAINING SUMMARY: {results['experience_level'].upper()}")
    print(f"{'='*60}")
    print(f"Timesteps: {metrics['timesteps']}")
    print(f"Total transitions: {metrics['num_transitions']}")
    print(f"\nFree Energy:")
    print(f"  Initial: {metrics['initial_fe']:.4f}")
    print(f"  Final:   {metrics['final_fe']:.4f}")
    print(f"  Reduction: {metrics['fe_reduction']:.4f}")
    print(f"\nAverage Dwell Times (steps):")
    for state in STATES:
        dwell = metrics.get(f'avg_dwell_{state}', 0)
        print(f"  {state.replace('_', ' ').title():25s}: {dwell:6.1f}")
    print(f"\nKey Transition:")
    print(f"  MW -> MA rate: {metrics['mw_to_ma_rate']:.3f}")

    print(f"\nAction Prediction Errors:")
    for state in STATES:
        error = metrics.get(f'action_error_{state}', 0)
        if error > 0:
            print(f"  {state.replace('_', ' ').title():25s}: {error:.5f}")
    if metrics.get('efe_prag_mean', 0) > 0 or metrics.get('efe_epi_mean', 0) > 0:
        print(f"\nEFE Diagnostics (tail mean):")
        print(f"  pragmatic: {metrics.get('efe_prag_mean', 0):.6f}")
        print(f"  epistemic: {metrics.get('efe_epi_mean', 0):.6f}")
    if metrics.get('sigma_fwd2', 0) > 0:
        print(f"\nResidual Scales (tail window):")
        print(f"  sigma_fwd^2:         {metrics.get('sigma_fwd2', 0):.6f}")
    print(f"{'='*60}\n")
