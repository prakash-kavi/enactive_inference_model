"""Core simulation components: Layers, Markov Blankets, and Trainer."""

# Import using relative imports to avoid circular dependencies
from .layer1_brain_networks import MeditationGenerativeProcess
from .layer2_gnw_bottleneck import GNWBottleneck
from .layer3_phenomenological_monitor import WitnessingLayer
from .markov_blanket_l1_l2 import MarkovBlanketL1L2
from .markov_blanket_l2_l3 import MarkovBlanketL2L3
from .meditation_trainer import Trainer

__all__ = [
    'MeditationGenerativeProcess',
    'GNWBottleneck',
    'WitnessingLayer',
    'MarkovBlanketL1L2',
    'MarkovBlanketL2L3',
    'Trainer',
]
