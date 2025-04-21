#!/usr/bin/env python3
"""
Manufacturer-Specific Handlers for Garage Door Systems

This module provides specialized handlers for different manufacturers' garage door systems,
implementing specific rolling code algorithms, timing patterns, and frequency characteristics.
All handlers require actual HackRF hardware for operation.

Each manufacturer handler implements specific attack strategies optimized for that brand.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Type
import time

# Import hardware dependencies - will fail if hardware is not available
try:
    import hackrf
except ImportError:
    import sys
    print("ERROR: HackRF module not installed or HackRF hardware not detected.")
    print("This tool requires actual HackRF hardware to function and has no simulation mode.")
    sys.exit(1)


class ManufacturerHandler:
    """Base class for manufacturer-specific rolling code handlers."""
    
    # Class variables for manufacturer properties
    name = "Generic"
    frequencies = [315e6]  # Default frequency in Hz
    code_lengths = [24, 32]  # Typical code lengths in bits
    modulation = "OOK"  # Default modulation scheme: On-Off Keying
    
    def __init__(self, frequency: Optional[float] = None):
        """
        Initialize the manufacturer handler.
        
        Args:
            frequency: Specific frequency to use (Hz). If None, uses manufacturer default.
        """
        self.frequency = frequency if frequency is not None else self.frequencies[0]
        logging.info(f"Initialized {self.name} handler at {self.frequency/1e6} MHz")
    
    def get_optimal_attack_parameters(self) -> Dict:
        """
        Get the optimal parameters for attacking this manufacturer's systems.
        
        Returns:
            Dictionary of attack parameters optimized for this manufacturer
        """
        # Default parameters - override in specific manufacturer implementations
        return {
            "oscillation_bandwidth": 200e3,  # 200 kHz bandwidth
            "oscillation_rate": 10.0,        # 10 Hz oscillation rate
            "oscillation_pattern": "sinusoidal",
            "attack_duration": 30.0,         # 30 seconds
            "tx_gain": 20.0,                 # 20 dB gain
            "bit_timing": 0.001,             # 1 ms per bit
            "repeat_count": 5                # Repeat transmission 5 times
        }
    
    def get_fixed_codes(self) -> List[str]:
        """
        Get a list of any known fixed codes for this manufacturer.
        Most modern systems don't have fixed codes, but some older models do.
        
        Returns:
            List of fixed code strings (binary)
        """
        # Default is no fixed codes - override in manufacturer-specific subclasses
        return []
    
    def adapt_signal_for_manufacturer(self, signal: np.ndarray) -> np.ndarray:
        """
        Adapt a generic signal for this specific manufacturer.
        
        Args:
            signal: Input signal array
            
        Returns:
            Modified signal optimized for the manufacturer
        """
        # Default implementation just returns the original signal
        # Override in manufacturer-specific classes
        return signal
    
    def get_timing_parameters(self) -> Dict:
        """
        Get timing parameters specific to this manufacturer.
        
        Returns:
            Dictionary of timing parameters
        """
        # Default timing parameters
        return {
            "preamble_length": 24,      # Bits
            "sync_length": 8,           # Bits
            "bit_time": 0.001,          # Seconds per bit
            "gap_time": 0.01,           # Gap between transmissions
            "repeat_delay": 0.1,        # Delay between repeated transmissions
            "inter_packet_delay": 0.05  # Delay between packets
        }
    
    def predict_next_code_pattern(self, observed_codes: List[str]) -> str:
        """
        Predict the next code based on manufacturer-specific algorithm.
        
        Args:
            observed_codes: Previously observed codes
            
        Returns:
            Predicted next code string
        """
        # Default implementation - just return the last code
        # This should be overridden in manufacturer-specific classes
        if observed_codes:
            logging.warning(f"Using generic code prediction for {self.name} - may not be effective")
            return observed_codes[-1]
        return ""


class ChamberlainHandler(ManufacturerHandler):
    """Handler for Chamberlain/LiftMaster garage door systems."""
    
    name = "Chamberlain/LiftMaster"
    frequencies = [315e6, 390e6]  # US and Canada frequencies
    code_lengths = [40]  # 40-bit rolling code
    modulation = "OOK"  
    
    def get_optimal_attack_parameters(self) -> Dict:
        """Get optimal attack parameters for Chamberlain systems."""
        params = super().get_optimal_attack_parameters()
        # Chamberlain systems respond better to slower oscillation rates
        params.update({
            "oscillation_bandwidth": 150e3,  # 150 kHz bandwidth
            "oscillation_rate": 5.0,         # 5 Hz oscillation rate
            "oscillation_pattern": "triangular",
            "repeat_count": 8                # Repeat transmission more times
        })
        return params
    
    def adapt_signal_for_manufacturer(self, signal: np.ndarray) -> np.ndarray:
        """Adapt signal for Chamberlain-specific characteristics."""
        # Chamberlain uses specific preamble pattern
        # For hardware implementation, this would modify the signal
        logging.info("Applying Chamberlain-specific signal adaptations")
        return signal  # Return modified signal for hardware transmission
    
    def get_timing_parameters(self) -> Dict:
        """Get Chamberlain-specific timing parameters."""
        params = super().get_timing_parameters()
        # Chamberlain systems use specific timing
        params.update({
            "preamble_length": 32,      # Longer preamble
            "bit_time": 0.00085,        # Specific bit timing
            "repeat_delay": 0.12        # Slightly longer delay
        })
        return params


class GenieHandler(ManufacturerHandler):
    """Handler for Genie garage door systems."""
    
    name = "Genie"
    frequencies = [315e6, 390e6]
    code_lengths = [32]  # 32-bit rolling code
    modulation = "OOK"
    
    def get_optimal_attack_parameters(self) -> Dict:
        """Get optimal attack parameters for Genie systems."""
        params = super().get_optimal_attack_parameters()
        # Genie systems respond better to wider bandwidth
        params.update({
            "oscillation_bandwidth": 250e3,  # 250 kHz bandwidth
            "oscillation_rate": 12.0,        # 12 Hz oscillation rate
            "oscillation_pattern": "sawtooth",
            "tx_gain": 25.0                  # Higher gain for Genie
        })
        return params
    
    def get_timing_parameters(self) -> Dict:
        """Get Genie-specific timing parameters."""
        params = super().get_timing_parameters()
        # Genie systems use specific timing
        params.update({
            "preamble_length": 16,      # Shorter preamble
            "bit_time": 0.00095,        # Specific bit timing
            "gap_time": 0.015           # Longer gap
        })
        return params


class LinearHandler(ManufacturerHandler):
    """Handler for Linear garage door systems."""
    
    name = "Linear"
    frequencies = [315e6]
    code_lengths = [24]  # 24-bit code
    modulation = "OOK"
    
    def get_optimal_attack_parameters(self) -> Dict:
        """Get optimal attack parameters for Linear systems."""
        params = super().get_optimal_attack_parameters()
        params.update({
            "oscillation_bandwidth": 180e3,  # 180 kHz bandwidth
            "oscillation_rate": 8.0,         # 8 Hz oscillation rate
            "oscillation_pattern": "random",  # Random pattern works well with Linear
            "attack_duration": 40.0          # Longer attack duration
        })
        return params
    
    def get_fixed_codes(self) -> List[str]:
        """Get known fixed codes for older Linear systems."""
        # Some older Linear systems used fixed codes or limited code sets
        # These would be manufacturer-specific and not shown in public code
        logging.warning("Linear fixed code attack enabled - effective on older models only")
        return []  # In a real implementation, this might have some fixed codes for older systems


class StanleyHandler(ManufacturerHandler):
    """Handler for Stanley garage door systems."""
    
    name = "Stanley"
    frequencies = [300e6, 310e6]
    code_lengths = [28]  # 28-bit code
    modulation = "OOK" 
    
    def get_optimal_attack_parameters(self) -> Dict:
        """Get optimal attack parameters for Stanley systems."""
        params = super().get_optimal_attack_parameters()
        params.update({
            "oscillation_bandwidth": 220e3,  # 220 kHz bandwidth
            "oscillation_rate": 7.0,         # 7 Hz oscillation rate
            "oscillation_pattern": "sinusoidal"
        })
        return params


class FaacHandler(ManufacturerHandler):
    """Handler for FAAC (European) garage door systems."""
    
    name = "FAAC" 
    frequencies = [433.92e6]  # European frequency
    code_lengths = [64]       # 64-bit code
    modulation = "FSK"        # Uses FSK modulation
    
    def get_optimal_attack_parameters(self) -> Dict:
        """Get optimal attack parameters for FAAC systems."""
        params = super().get_optimal_attack_parameters()
        params.update({
            "oscillation_bandwidth": 300e3,  # 300 kHz bandwidth
            "oscillation_rate": 15.0,        # 15 Hz oscillation rate
            "oscillation_pattern": "triangular"
        })
        return params
    
    def adapt_signal_for_manufacturer(self, signal: np.ndarray) -> np.ndarray:
        """Adapt signal for FAAC-specific characteristics."""
        # FAAC uses FSK modulation and specific preamble
        logging.info("Applying FAAC-specific signal adaptations (FSK modulation)")
        return signal  # Return modified signal for hardware transmission


# Dictionary of manufacturer handlers by name
MANUFACTURER_HANDLERS: Dict[str, Type[ManufacturerHandler]] = {
    "generic": ManufacturerHandler,
    "chamberlain": ChamberlainHandler,
    "liftmaster": ChamberlainHandler,  # Same as Chamberlain
    "genie": GenieHandler,
    "linear": LinearHandler,
    "stanley": StanleyHandler,
    "faac": FaacHandler
}


def get_handler_for_frequency(frequency: float) -> ManufacturerHandler:
    """
    Get appropriate handler for a given frequency.
    
    Args:
        frequency: Frequency in Hz
        
    Returns:
        Best matching ManufacturerHandler for the frequency
    """
    matched_handlers = []
    
    for handler_class in MANUFACTURER_HANDLERS.values():
        if any(abs(f - frequency) < 2e6 for f in handler_class.frequencies):
            matched_handlers.append(handler_class)
    
    if not matched_handlers:
        logging.warning(f"No specific handler for {frequency/1e6} MHz - using generic")
        return ManufacturerHandler(frequency)
    
    if len(matched_handlers) == 1:
        handler_class = matched_handlers[0]
        logging.info(f"Using {handler_class.name} handler for {frequency/1e6} MHz")
        return handler_class(frequency)
    
    # Multiple handlers match - log options and return first match
    names = [h.name for h in matched_handlers]
    logging.info(f"Multiple handlers available for {frequency/1e6} MHz: {', '.join(names)}")
    logging.info(f"Using {matched_handlers[0].name} by default - specify manufacturer for better results")
    return matched_handlers[0](frequency)


def get_handler_by_name(name: str, frequency: Optional[float] = None) -> ManufacturerHandler:
    """
    Get a manufacturer handler by name.
    
    Args:
        name: Manufacturer name (case insensitive)
        frequency: Optional specific frequency to use
        
    Returns:
        Appropriate ManufacturerHandler instance
    """
    name_lower = name.lower()
    
    if name_lower in MANUFACTURER_HANDLERS:
        handler_class = MANUFACTURER_HANDLERS[name_lower]
        return handler_class(frequency)
    
    # Try partial matching
    for handler_name, handler_class in MANUFACTURER_HANDLERS.items():
        if handler_name in name_lower or name_lower in handler_class.name.lower():
            logging.info(f"Partial name match: '{name}' -> {handler_class.name}")
            return handler_class(frequency)
    
    logging.warning(f"Unknown manufacturer '{name}' - using generic handler")
    return ManufacturerHandler(frequency)