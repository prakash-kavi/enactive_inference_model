"""Visualization module for the meditation model.

Plotting modules for the manuscript:
- convergence: figs1 (convergence plots)
- radar_plot: fig2a (network radar)
- diagnostics: fig2b (dwell), fig2c (transitions)
- hierarchy: fig3a/fig3b (3-level dynamics)
- attractors: fig4 (PCA trajectories across hierarchy)
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
