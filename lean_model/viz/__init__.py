"""Visualization module for lean meditation model.

Streamlined plotting modules for Royal Society paper:
- lean_convergence: FigS1 (convergence plots)
- lean_comparison: Fig3A (radar)
- lean_diagnostics: Fig3B (FE+Dwell), Fig3C (Transitions), Fig3D (Belief about Belief)
- lean_hierarchy: Fig4A/B (3-level dynamics)
- lean_attractors: Fig5A/B (2D/3D state space)
"""

from . import plotting_utils

# Streamlined plotting modules
from .lean_convergence import plot_convergence
from .lean_comparison import plot_comparison
from .lean_hierarchy import plot_hierarchy
from .lean_attractors import plot_attractor_2d, plot_attractor_3d
from .lean_diagnostics import plot_fe_and_dwell, plot_transitions

__all__ = [
    'plotting_utils',
    'plot_convergence',
    'plot_comparison',
    'plot_hierarchy',
    'plot_attractor_2d',
    'plot_attractor_3d',
    'plot_fe_and_dwell',
    'plot_transitions',
]
