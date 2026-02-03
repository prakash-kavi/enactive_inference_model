"""
Plot Phase 4 enactive inference metrics: forward prediction error over training.
Shows that the forward model learns to predict action consequences.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from viz.plotting_utils import set_plot_style

def rolling_mean(arr, window=50):
    """Compute rolling mean."""
    if len(arr) < window:
        return np.array(arr)
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window

def plot_action_prediction_error():
    """Plot forward model learning curves."""
    set_plot_style()
    
    data_dir = Path("data/training/convergence_plots_data")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, level in enumerate(['novice', 'expert']):
        file_path = data_dir / f"thoughtseed_params_{level}.json"
        
        if not file_path.exists():
            print(f"No data for {level}")
            continue
            
        with open(file_path) as f:
            data = json.load(f)
        
        ape = data['time_series']['action_pred_error_history']
        steps = np.arange(len(ape))
        
        ax = axes[idx]
        
        # Raw error
        ax.plot(steps, ape, color='#cccccc', linewidth=0.8, alpha=0.6, label='Raw error')
        
        # Rolling mean
        window = 50
        ape_smooth = rolling_mean(ape, window)
        steps_smooth = steps[window-1:]
        ax.plot(steps_smooth, ape_smooth, color='#2ECC71' if level == 'expert' else '#E74C3C', 
                linewidth=2.5, label=f'Rolling mean (w={window})')
        
        # Statistics
        first_100 = np.mean(ape[:100])
        last_100 = np.mean(ape[-100:])
        reduction = (first_100 - last_100) / first_100 * 100
        
        ax.axhline(first_100, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Initial avg')
        ax.axhline(last_100, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Final avg')
        
        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('Action prediction error (MSE)', fontsize=11)
        ax.set_title(f'{level.capitalize()}: {reduction:.1f}% reduction', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', frameon=True, fontsize=9)
        ax.grid(True, alpha=0.2)
        
    plt.tight_layout()
    
    output_path = Path("plots/training/Phase4_Forward_Model_Learning.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved Phase 4 plot: {output_path}")
    plt.close()

if __name__ == "__main__":
    plot_action_prediction_error()
