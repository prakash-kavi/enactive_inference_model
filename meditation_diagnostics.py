"""Diagnostic functions for meditation state detection and expertise classification.
Implements algorithmic rules from the integrated framework combining dwell times,
network profiles, and coupling patterns.
"""

from typing import Dict, List, Optional
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


def detect_meta_awareness_transition(network_acts: Dict[str, float], current_state: str, 
                                    fpn_van_coupling: float, threshold: float = 0.6) -> bool:
    """Detect meta-awareness transition based on FPN-VAN coupling peak.
    
    Args:
        network_acts: Current network activations
        current_state: Current meditative state
        fpn_van_coupling: FPN-VAN coupling strength
        threshold: Minimum coupling for detection (default 0.6)
        
    Returns:
        True if peak FPN-VAN coupling detected during meta_awareness window
    """
    if current_state != "meta_awareness":
        return False
    
    # Peak coupling indicates salience → executive handoff
    return fpn_van_coupling >= threshold


def detect_redirect_transition(network_acts: Dict[str, float], current_state: str,
                              dan_fpn_coupling: float, threshold: float = 0.6) -> bool:
    """Detect redirect transition based on DAN-FPN coupling peak.
    
    Args:
        network_acts: Current network activations
        current_state: Current meditative state
        dan_fpn_coupling: DAN-FPN coupling strength
        threshold: Minimum coupling for detection (default 0.6)
        
    Returns:
        True if peak DAN-FPN coupling detected (coordination for reorientation)
    """
    if current_state != "redirect_breath":
        return False
    
    # Peak coupling indicates coordination for shifting attention
    return dan_fpn_coupling >= threshold


def detect_van_spike_transition(network_acts: Dict[str, float], van_history: List[float],
                               threshold: float = 0.7) -> bool:
    """Detect VAN spike indicating transition from mind wandering to redirect.
    
    Args:
        network_acts: Current network activations
        van_history: Recent VAN activation history (for spike detection)
        threshold: VAN spike threshold (default 0.7)
        
    Returns:
        True if VAN spike detected (sharp increase above threshold)
    """
    current_van = network_acts.get('VAN', 0.0)
    
    if current_van < threshold:
        return False
    
    # Check for spike: current VAN significantly higher than recent history
    if len(van_history) < 3:
        return current_van >= threshold
    
    recent_mean = np.mean(van_history[-3:])
    spike_magnitude = current_van - recent_mean
    
    # Spike: current VAN is at least 0.15 above recent mean
    return spike_magnitude >= 0.15


def compute_network_coupling(network_acts: Dict[str, float], coupling_type: str) -> float:
    """Compute instantaneous coupling between two networks.
    
    Args:
        network_acts: Current network activations
        coupling_type: Type of coupling (e.g., "DMN_DAN", "FPN_VAN")
        
    Returns:
        Coupling value (product of network activations, normalized)
    """
    if "_" not in coupling_type:
        return 0.0
    
    net1, net2 = coupling_type.split("_")
    val1 = network_acts.get(net1, 0.0)
    val2 = network_acts.get(net2, 0.0)
    
    # Instantaneous coupling: product of activations
    return val1 * val2


def validate_state_signature(network_acts: Dict[str, float], state: str, 
                            experience_level: str) -> bool:
    """Validate that network patterns match expected state signatures.
    
    Args:
        network_acts: Current network activations
        state: Current meditative state
        experience_level: 'expert' or 'novice'
        
    Returns:
        True if network patterns match expected signature for state
    """
    dmn = network_acts.get('DMN', 0.0)
    van = network_acts.get('VAN', 0.0)
    dan = network_acts.get('DAN', 0.0)
    fpn = network_acts.get('FPN', 0.0)
    
    if state == "breath_focus":
        # Expert: Low DMN (0.30), Low FPN (0.45)
        # Novice: Higher DMN (0.55), Higher FPN (0.65)
        if experience_level == "expert":
            return dmn < 0.50 and fpn < 0.60
        else:
            return dmn > 0.40 and fpn > 0.50
    
    elif state == "mind_wandering":
        # Expert: DMN moderate (0.60), VAN high (0.65), FPN high (0.65) - background monitoring
        # Novice: DMN high (0.75), VAN low (0.40), FPN low (0.40) - total capture
        # Relaxed validation: Expert should have VAN/FPN high (background monitoring signature)
        # Novice should have VAN/FPN low (total capture signature)
        if experience_level == "expert":
            # Expert: Key signature is high VAN/FPN (background monitoring), DMN can vary
            return van > 0.50 and fpn > 0.50
        else:
            # Novice: Key signature is low VAN/FPN (total capture), high DMN
            return dmn > 0.50 and (van < 0.55 or fpn < 0.55)
    
    elif state == "meta_awareness":
        # High VAN (salience) and FPN (executive)
        return van > 0.55 and fpn > 0.50
    
    elif state == "redirect_breath":
        # High DAN (attention) and FPN (control)
        return dan > 0.55 and fpn > 0.50
    
    return True  # Default: accept if no specific signature
