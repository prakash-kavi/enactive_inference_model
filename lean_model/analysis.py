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

from .config import STATES

# Color palette for states
STATE_COLORS = {
    'breath_focus': '#2E7D32',      # Green
    'mind_wandering': '#C62828',    # Red
    'meta_awareness': '#F57C00',    # Orange
    'redirect_attention': '#1565C0' # Blue
}

def compute_metrics(results: Dict) -> Dict:
    """Compute summary metrics from training results.
    
    Args:
        results: Training results dict from MeditationTrainer
    
    Returns:
        Summary metrics dict
    """
    metrics = {
        'experience_level': results['experience_level'],
        'timesteps': results['timesteps'],
        'num_transitions': len(results['transitions']),
    }
    
    # Average dwell times
    for state in STATES:
        dwell = results['avg_dwell_times'].get(state, 0.0)
        metrics[f'avg_dwell_{state}'] = dwell
    
    # Transition rates
    trans_matrix = results['transition_matrix']
    if 'mind_wandering' in trans_matrix and 'meta_awareness' in trans_matrix['mind_wandering']:
        metrics['mw_to_ma_rate'] = trans_matrix['mind_wandering']['meta_awareness']
    else:
        metrics['mw_to_ma_rate'] = 0.0
    
    # Free energy convergence
    fe_history = results['free_energy_history']
    if len(fe_history) > 100:
        metrics['initial_fe'] = float(np.mean(fe_history[:100]))
        metrics['final_fe'] = float(np.mean(fe_history[-100:]))
        metrics['fe_reduction'] = metrics['initial_fe'] - metrics['final_fe']
    else:
        metrics['initial_fe'] = fe_history[0] if fe_history else 0.0
        metrics['final_fe'] = fe_history[-1] if fe_history else 0.0
        metrics['fe_reduction'] = 0.0
    
    # Action prediction errors
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
    fig.suptitle("L3 Meta-Awareness and L2 Variational Free Energy Evolution", 
                 fontsize=14, fontweight='bold')
    
    # ===== L3 Meta-Awareness - Novice (Top-Left) =====
    ax = axes[0, 0]
    meta_novice = np.array(novice_results['meta_awareness_history'])
    timesteps_novice = np.arange(len(meta_novice))
    
    ax.plot(timesteps_novice, meta_novice, color='#F57C00', linewidth=0.5, alpha=0.6)
    
    # Running average
    window = min(100, len(meta_novice) // 10)
    if window > 1:
        meta_smooth = np.convolve(meta_novice, np.ones(window)/window, mode='valid')
        ax.plot(timesteps_novice[:len(meta_smooth)], meta_smooth, 
                color='#E65100', linewidth=2, label='Running avg')
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Meta-Awareness')
    ax.set_title('L3 Meta-Awareness ("Belief about Belief") - Novice')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== L3 Meta-Awareness - Expert (Top-Right) =====
    ax = axes[0, 1]
    meta_expert = np.array(expert_results['meta_awareness_history'])
    timesteps_expert = np.arange(len(meta_expert))
    
    ax.plot(timesteps_expert, meta_expert, color='#F57C00', linewidth=0.5, alpha=0.6)
    
    # Running average
    window = min(100, len(meta_expert) // 10)
    if window > 1:
        meta_smooth = np.convolve(meta_expert, np.ones(window)/window, mode='valid')
        ax.plot(timesteps_expert[:len(meta_smooth)], meta_smooth, 
                color='#E65100', linewidth=2, label='Running avg')
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Meta-Awareness')
    ax.set_title('L3 Meta-Awareness ("Belief about Belief") - Expert')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== L2 Free Energy - Novice (Bottom-Left) =====
    ax = axes[1, 0]
    fe_novice = np.array(novice_results['free_energy_history'])
    timesteps_novice = np.arange(len(fe_novice))
    
    ax.plot(timesteps_novice, fe_novice, color='#1976D2', linewidth=0.5, alpha=0.6)
    
    # Running average
    window = min(100, len(fe_novice) // 10)
    if window > 1:
        fe_smooth = np.convolve(fe_novice, np.ones(window)/window, mode='valid')
        ax.plot(timesteps_novice[:len(fe_smooth)], fe_smooth, 
                color='#0D47A1', linewidth=2, label='Running avg')
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Variational Free Energy')
    ax.set_title('L2 Free Energy Evolution - Novice')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== L2 Free Energy - Expert (Bottom-Right) =====
    ax = axes[1, 1]
    fe_expert = np.array(expert_results['free_energy_history'])
    timesteps_expert = np.arange(len(fe_expert))
    
    ax.plot(timesteps_expert, fe_expert, color='#1976D2', linewidth=0.5, alpha=0.6)
    
    # Running average
    window = min(100, len(fe_expert) // 10)
    if window > 1:
        fe_smooth = np.convolve(fe_expert, np.ones(window)/window, mode='valid')
        ax.plot(timesteps_expert[:len(fe_smooth)], fe_smooth, 
                color='#0D47A1', linewidth=2, label='Running avg')
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Variational Free Energy')
    ax.set_title('L2 Free Energy Evolution - Expert')
    ax.legend()
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
