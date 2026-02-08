"""Visualization module for lean meditation model.

Streamlined plotting modules for Royal Society paper:
- lean_convergence: FigS1 (convergence plots)
- lean_comparison: Fig3A (radar)
- lean_diagnostics: Fig3B (FE+Dwell), Fig3C (Transitions), Fig3D (Belief about Belief)
- lean_hierarchy: Fig4A/B (3-level dynamics)
- lean_attractors: Fig5 (PCA trajectories across hierarchy)
"""

from viz import plotting_utils

# Streamlined plotting modules
from viz.convergence import plot_convergence
from viz.radar_plot import plot_comparison
from viz.hierarchy import plot_hierarchy
from viz.attractors import plot_attractor_pca
from viz.diagnostics import plot_fe_and_dwell, plot_transitions

__all__ = [
    'plotting_utils',
    'plot_convergence',
    'plot_comparison',
    'plot_hierarchy',
    'plot_attractor_pca',
    'plot_fe_and_dwell',
    'plot_transitions',
]
