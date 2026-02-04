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


def plot_training_summary(results: Dict, save_path: str = None) -> None:
    """Create comprehensive training summary plot.
    
    Args:
        results: Training results dict
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Meditation Training Summary: {results['experience_level'].capitalize()} (Seed {results['seed']})", 
                 fontsize=14, fontweight='bold')
    
    # ===== Free Energy Evolution =====
    ax = axes[0, 0]
    fe = np.array(results['free_energy_history'])
    timesteps = np.arange(len(fe))
    
    ax.plot(timesteps, fe, color='#1976D2', linewidth=0.5, alpha=0.6)
    
    # Running average
    window = min(100, len(fe) // 10)
    if window > 1:
        fe_smooth = np.convolve(fe, np.ones(window)/window, mode='valid')
        ax.plot(timesteps[:len(fe_smooth)], fe_smooth, color='#0D47A1', linewidth=2, label='Running avg')
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Variational Free Energy')
    ax.set_title('L2 Free Energy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== Meta-Awareness Trajectory =====
    ax = axes[0, 1]
    meta = np.array(results['meta_awareness_history'])
    
    ax.plot(timesteps, meta, color='#F57C00', linewidth=0.5, alpha=0.6)
    
    # Running average
    if window > 1:
        meta_smooth = np.convolve(meta, np.ones(window)/window, mode='valid')
        ax.plot(timesteps[:len(meta_smooth)], meta_smooth, color='#E65100', linewidth=2, label='Running avg')
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Meta-Awareness')
    ax.set_title('L3 Meta-Awareness ("Belief about Belief")')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== State Dwell Times =====
    ax = axes[1, 0]
    dwell_data = results['avg_dwell_times']
    states = [s for s in STATES if dwell_data.get(s, 0) > 0]
    dwells = [dwell_data[s] for s in states]
    colors = [STATE_COLORS[s] for s in states]
    
    bars = ax.bar(states, dwells, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Average Dwell Time (steps)')
    ax.set_title('Attractor Stability (Dwell Times)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, dwell in zip(bars, dwells):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{dwell:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    # ===== Transition Matrix =====
    ax = axes[1, 1]
    trans_matrix = results['transition_matrix']
    
    # Build matrix array
    matrix_data = np.zeros((len(STATES), len(STATES)))
    for i, from_state in enumerate(STATES):
        for j, to_state in enumerate(STATES):
            matrix_data[i, j] = trans_matrix.get(from_state, {}).get(to_state, 0.0)
    
    im = ax.imshow(matrix_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(range(len(STATES)))
    ax.set_yticks(range(len(STATES)))
    ax.set_xticklabels([s.replace('_', '\n') for s in STATES], fontsize=8)
    ax.set_yticklabels([s.replace('_', ' ').title() for s in STATES], fontsize=8)
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title('Transition Dynamics')
    
    # Add text annotations
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            value = matrix_data[i, j]
            if value > 0.01:  # Only show significant transitions
                ax.text(j, i, f'{value:.2f}',
                       ha='center', va='center',
                       color='white' if value > 0.5 else 'black',
                       fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Probability')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_state_trajectory(results: Dict, max_steps: int = 500, save_path: str = None) -> None:
    """Plot state trajectory over time (first N steps).
    
    Args:
        results: Training results dict
        max_steps: Maximum timesteps to plot
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    
    state_seq = results['state_sequence'][:max_steps]
    timesteps = np.arange(len(state_seq))
    
    # Map states to numeric values for plotting
    state_to_num = {s: i for i, s in enumerate(STATES)}
    state_nums = [state_to_num[s] for s in state_seq]
    
    # Create colored segments
    for state, color in STATE_COLORS.items():
        mask = np.array(state_seq) == state
        ax.scatter(timesteps[mask], np.array(state_nums)[mask], 
                  c=color, label=state.replace('_', ' ').title(),
                  s=10, alpha=0.7)
    
    # Mark transitions
    trans = results['transitions']
    trans_times = [t['timestamp'] for t in trans if t['timestamp'] < max_steps]
    if trans_times:
        for t_time in trans_times:
            ax.axvline(x=t_time, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    ax.set_yticks(range(len(STATES)))
    ax.set_yticklabels([s.replace('_', ' ').title() for s in STATES])
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Meditation State')
    ax.set_title(f'State Space Trajectory: {results["experience_level"].capitalize()} (First {max_steps} steps)')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory plot saved to {save_path}")
    else:
        plt.show()


def compare_expertise(expert_results: Dict, novice_results: Dict, save_path: str = None) -> None:
    """Compare expert vs novice phenotypes side-by-side.
    
    Args:
        expert_results: Expert training results
        novice_results: Novice training results
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Expert vs Novice Meditation Phenotypes", fontsize=14, fontweight='bold')
    
    # ===== Dwell Time Comparison =====
    ax = axes[0, 0]
    x = np.arange(len(STATES))
    width = 0.35
    
    expert_dwells = [expert_results['avg_dwell_times'].get(s, 0) for s in STATES]
    novice_dwells = [novice_results['avg_dwell_times'].get(s, 0) for s in STATES]
    
    ax.bar(x - width/2, expert_dwells, width, label='Expert', color='#1976D2', alpha=0.7)
    ax.bar(x + width/2, novice_dwells, width, label='Novice', color='#D32F2F', alpha=0.7)
    
    ax.set_ylabel('Average Dwell Time (steps)')
    ax.set_title('Attractor Stability by Expertise')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in STATES], fontsize=8)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # ===== Free Energy Convergence =====
    ax = axes[0, 1]
    
    expert_fe = expert_results['free_energy_history']
    novice_fe = novice_results['free_energy_history']
    
    # Downsample for clarity
    downsample = max(1, len(expert_fe) // 1000)
    expert_fe_ds = expert_fe[::downsample]
    novice_fe_ds = novice_fe[::downsample]
    
    ax.plot(expert_fe_ds, color='#1976D2', linewidth=1.5, label='Expert', alpha=0.7)
    ax.plot(novice_fe_ds, color='#D32F2F', linewidth=1.5, label='Novice', alpha=0.7)
    
    ax.set_xlabel('Timestep (downsampled)')
    ax.set_ylabel('Variational Free Energy')
    ax.set_title('Learning Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== Meta-Awareness =====
    ax = axes[1, 0]
    
    expert_meta = expert_results['meta_awareness_history']
    novice_meta = novice_results['meta_awareness_history']
    
    expert_meta_ds = expert_meta[::downsample]
    novice_meta_ds = novice_meta[::downsample]
    
    ax.plot(expert_meta_ds, color='#1976D2', linewidth=1.5, label='Expert', alpha=0.7)
    ax.plot(novice_meta_ds, color='#D32F2F', linewidth=1.5, label='Novice', alpha=0.7)
    
    ax.set_xlabel('Timestep (downsampled)')
    ax.set_ylabel('Meta-Awareness')
    ax.set_title('Content-Based Monitoring Quality')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== Transition Rates =====
    ax = axes[1, 1]
    
    # Key transition: MW → MA (awareness arising)
    expert_mw_ma = expert_results['transition_matrix'].get('mind_wandering', {}).get('meta_awareness', 0)
    novice_mw_ma = novice_results['transition_matrix'].get('mind_wandering', {}).get('meta_awareness', 0)
    
    # Key transition: MA → RA (redirecting)
    expert_ma_ra = expert_results['transition_matrix'].get('meta_awareness', {}).get('redirect_attention', 0)
    novice_ma_ra = novice_results['transition_matrix'].get('meta_awareness', {}).get('redirect_attention', 0)
    
    # Key transition: RA → BF (returning)
    expert_ra_bf = expert_results['transition_matrix'].get('redirect_attention', {}).get('breath_focus', 0)
    novice_ra_bf = novice_results['transition_matrix'].get('redirect_attention', {}).get('breath_focus', 0)
    
    trans_labels = ['MW→MA\n(Catch)', 'MA→RA\n(Redirect)', 'RA→BF\n(Return)']
    expert_trans = [expert_mw_ma, expert_ma_ra, expert_ra_bf]
    novice_trans = [novice_mw_ma, novice_ma_ra, novice_ra_bf]
    
    x = np.arange(len(trans_labels))
    ax.bar(x - width/2, expert_trans, width, label='Expert', color='#1976D2', alpha=0.7)
    ax.bar(x + width/2, novice_trans, width, label='Novice', color='#D32F2F', alpha=0.7)
    
    ax.set_ylabel('Transition Probability')
    ax.set_title('Key Transition Dynamics')
    ax.set_xticks(x)
    ax.set_xticklabels(trans_labels, fontsize=8)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
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
