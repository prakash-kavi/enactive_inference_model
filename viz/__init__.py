"""Visualization module for the meditation model.

Plotting modules for the manuscript:
- convergence: FigS1 (convergence plots)
- radar_plot: fig3a (network radar)
- diagnostics: fig3b (dwell), fig3c (transitions)
- hierarchy: fig4a/fig4b (3-level dynamics)
- attractors: fig5 (PCA trajectories)
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
