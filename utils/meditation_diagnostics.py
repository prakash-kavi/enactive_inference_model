"""Diagnostic functions for meditation state detection and expertise classification.
Implements algorithmic rules from the integrated framework combining dwell times,
network profiles, and coupling patterns.
"""

from typing import Dict, Optional
import numpy as np


def compute_neural_efficiency_ratio(network_acts: Dict[str, float], state: str) -> Optional[float]:
    """Compute neural efficiency ratio: DAN/FPN during breath_focus.
    
    Args:
        network_acts: Current network activations
        state: Current meditative state
        
    Returns:
        Ratio (DAN/FPN) if in breath_focus, None otherwise
        
    Expected Values:
        Expert: High ratio (≈1.33) - stable focus, low effort
        Novice: Low ratio (≈0.92) - unstable focus, high effort
    """
    if state != "breath_focus":
        return None
    
    dan = network_acts.get('DAN', 0.0)
    fpn = network_acts.get('FPN', 0.0)
    
    if fpn == 0:
        return None
    
    return dan / fpn


def detect_expert_mind_wandering(network_acts: Dict[str, float]) -> Optional[bool]:
    """Detect expert meta-cognitive signature during mind wandering.
    
    Expert shows background monitoring: DMN > 0.5 AND VAN > 0.6
    Novice shows total capture: DMN > 0.5 AND VAN < 0.5
    
    Args:
        network_acts: Current network activations
        
    Returns:
        True if expert pattern, False if novice pattern, None if DMN < 0.5
    """
    dmn = network_acts.get('DMN', 0.0)
    van = network_acts.get('VAN', 0.0)
    
    if dmn <= 0.5:
        return None
    
    # Expert: background monitoring (high VAN)
    if van > 0.6:
        return True
    
    # Novice: total capture (low VAN)
    if van < 0.5:
        return False
    
    return None


def compute_dmn_dan_anticorrelation(network_acts: Dict[str, float]) -> bool:
    """Check if DMN-DAN anti-correlation indicates successful distraction suppression.
    
    Args:
        network_acts: Current network activations
        
    Returns:
        True if DMN < 0.5 and DAN > 0.5 (strong suppression), False otherwise
    """
    dmn = network_acts.get("DMN", 0.5)
    dan = network_acts.get("DAN", 0.5)
    return dmn < 0.5 and dan > 0.5

