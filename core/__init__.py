"""Core simulation components: layers, Markov blankets, and trainer."""

from .layer1.process import Layer1Process
from .layer2.bottleneck import Layer2AttentionalModel
from .layer2.vae import MeditationVAE
from .layer3.monitor import Layer3Monitor
from .blankets.l1_l2 import MarkovBlanketL1L2
from .blankets.l2_l3 import MarkovBlanketL2L3
from .train.trainer import PracticeTrainer

__all__ = [
    'Layer1Process',
    'Layer2AttentionalModel',
    'Layer3Monitor',
    'MeditationVAE',
    'MarkovBlanketL1L2',
    'MarkovBlanketL2L3',
    'PracticeTrainer',
]
