"""Analysis and visualization for lean meditation model.

Provides metrics computation and plotting utilities aligned with paper narrative:
- State dwell times (attractor stability)
- Transition dynamics (state space trajectory)
- Free energy evolution (learning convergence)
- Meta-awareness trajectory (content-based monitoring)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from utils.config import STATES
from utils.analysis_utils import get_tail_window, compute_tail_statistics, TAIL_STEPS

# Color palette for states
STATE_COLORS = {
    'breath_focus': '#2E7D32',      # Green
    'mind_wandering': '#C62828',    # Red
    'meta_awareness': '#F57C00',    # Orange
    'redirect_attention': '#1565C0' # Blue
}

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
    
    return metrics


def plot_belief_about_belief(novice_results: Dict, expert_results: Dict, save_path: str = None) -> None:
    """Create combined L3 Meta-Awareness and L2 Free Energy plot.
    
    Shows L3 on top row, L2 on bottom row; Novice left, Expert right.
    
    Args:
        novice_results: Training results for novice
        expert_results: Training results for expert
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("L3 Meta-Awareness + Opacity and L2 Variational Free Energy Evolution", 
                 fontsize=14, fontweight='bold')
    
    # Extract data first
    meta_novice = np.array(novice_results['meta_awareness_history'])
    meta_expert = np.array(expert_results['meta_awareness_history'])
    opacity_novice = np.array(novice_results.get('opacity_history', []))
    opacity_expert = np.array(expert_results.get('opacity_history', []))
    fe_novice = np.array(novice_results['free_energy_history'])
    fe_expert = np.array(expert_results['free_energy_history'])
    timesteps_novice = np.arange(len(meta_novice))
    timesteps_expert = np.arange(len(meta_expert))
    
    # Calculate Expert MW Baseline (The Standard for "Unconsciousness" / Noise Floor)
    mw_expert_vals = [meta_expert[i] for i, s in enumerate(expert_results.get('state_history', [])) if s == 'mind_wandering']
    
    threshold_val = 0.25 # Fallback
    threshold_label = 'Ref. Baseline'
    
    if mw_expert_vals:
        # Use Expert MW as the "Noise Floor" reference for EVERYONE
        # Rationale: Experts have minimal Meta-Awareness during Mind Wandering. 
        # Deviations from this baseline indicate signal (or noise in Novices).
        baseline = np.mean(mw_expert_vals)
        # We plot the mean + 1 std dev to show the "Top of Noise"
        std_dev = np.std(mw_expert_vals)
        threshold_val = baseline + 2 * std_dev
        threshold_label = f'Expert Baseline+2σ ({threshold_val:.2f})'

    # ===== L3 Meta-Awareness - Novice (Top-Left) =====
    ax = axes[0, 0]
    ax.plot(timesteps_novice, meta_novice, color='#F57C00', linewidth=0.5, alpha=0.6, label='Meta-awareness')
    if opacity_novice.size == meta_novice.size and opacity_novice.size > 0:
        ax.plot(timesteps_novice, opacity_novice, color='#00796B', linewidth=0.6, alpha=0.7, label='Opacity')
    
    ax.axhline(y=threshold_val, color='gray', linestyle='--', alpha=0.7, label=threshold_label)
        
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Meta-Awareness')
    ax.set_title('L3 Meta-Awareness ("Belief about Belief") - Novice')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # ===== L3 Meta-Awareness - Expert (Top-Right) =====
    ax = axes[0, 1]
    
    ax.plot(timesteps_expert, meta_expert, color='#F57C00', linewidth=0.5, alpha=0.6, label='Meta-awareness')
    if opacity_expert.size == meta_expert.size and opacity_expert.size > 0:
        ax.plot(timesteps_expert, opacity_expert, color='#00796B', linewidth=0.6, alpha=0.7, label='Opacity')
    
    ax.axhline(y=threshold_val, color='gray', linestyle='--', alpha=0.7, label=threshold_label)
        
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Meta-Awareness')
    ax.set_title('L3 Meta-Awareness ("Belief about Belief") - Expert')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Determine common Y-axis for Free Energy
    fe_all = np.concatenate([fe_novice, fe_expert])
    fe_min, fe_max = np.min(fe_all), np.max(fe_all)
    fe_buffer = (fe_max - fe_min) * 0.1
    fe_ylim = (fe_min - fe_buffer, fe_max + fe_buffer)

    # ===== L2 Free Energy - Novice (Bottom-Left) =====
    ax = axes[1, 0]
    ax.plot(timesteps_novice, fe_novice, color='#1976D2', linewidth=0.5, alpha=0.6)
    
    # Running average
    window = min(100, len(fe_novice) // 10)
    if window > 1:
        fe_smooth = np.convolve(fe_novice, np.ones(window)/window, mode='valid')
        ax.plot(timesteps_novice[:len(fe_smooth)], fe_smooth, 
                color='#0D47A1', linewidth=2, label='Running avg')
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Variational Free Energy')
    ax.set_title('L2 Belief Divergence (Variational Free Energy) - Novice')
    ax.set_ylim(fe_ylim)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # ===== L2 Free Energy - Expert (Bottom-Right) =====
    ax = axes[1, 1]
    ax.plot(timesteps_expert, fe_expert, color='#1976D2', linewidth=0.5, alpha=0.6)
    
    # Running average
    window = min(100, len(fe_expert) // 10)
    if window > 1:
        fe_smooth = np.convolve(fe_expert, np.ones(window)/window, mode='valid')
        ax.plot(timesteps_expert[:len(fe_smooth)], fe_smooth, 
                color='#0D47A1', linewidth=2, label='Running avg')
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Variational Free Energy')
    ax.set_title('L2 Belief Divergence (Variational Free Energy) - Expert')
    ax.set_ylim(fe_ylim)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


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
    print(f"{'='*60}\n")
