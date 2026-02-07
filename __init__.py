"""Lean Vipassana Model: Russian Doll Architecture for Meditation Attention.

A streamlined implementation demonstrating hierarchical active inference with:
- Layer 1: MVOU generative process (high-dimensional neural substrate)
- Layer 2: Attentional agents (thoughtseeds as tractable bottleneck)
- Layer 3: Metacognitive monitor (awareness as hyperparameter optimization)

Key Features:
- 4 meditation states (breath_focus, mind_wandering, meta_awareness, redirect_attention)
- 4 brain networks (DMN, VAN, DAN, FPN) 
- 5 thoughtseeds (attend_breath, pain_discomfort, pending_tasks, aha_moment, equanimity)
- Phase 4 enactive inference (forward dynamics model)
- Markov blanket message passing interfaces
"""

from utils.config import STATES, NETWORKS, THOUGHTSEEDS, get_params
from model.l1_generative_process import Layer1Process
from model.l2_recognition import Layer2Agent
from model.l3_metacognition import Layer3Monitor
from model.markov_blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from model.training_loop import MeditationTrainer, train_meditation
from viz.analysis import (
    compute_metrics,
    plot_belief_about_belief,
    print_summary
)

__version__ = '1.0.0'

__all__ = [
    # Core components
    'Layer1Process',
    'Layer2Agent',
    'Layer3Monitor',
    'MarkovBlanketL1L2',
    'MarkovBlanketL2L3',
    
    # Training
    'MeditationTrainer',
    'train_meditation',
    
    # Analysis
    'compute_metrics',
    'plot_belief_about_belief',
    'print_summary',
    
    # Configuration
    'STATES',
    'NETWORKS',
    'THOUGHTSEEDS',
    'get_params',
]
