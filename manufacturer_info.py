#!/usr/bin/env python3
"""
Manufacturer Information Module

This module provides information about garage door system manufacturers without requiring
HackRF hardware. It's used for listing supported manufacturers and their frequencies.
"""

from typing import Dict, List, Type


class ManufacturerInfo:
    """Base class for manufacturer-specific information."""
    
    # Class variables for manufacturer properties
    name = "Generic"
    frequencies = [315e6]  # Default frequency in Hz
    code_lengths = [24, 32]  # Typical code lengths in bits
    modulation = "OOK"  # Default modulation scheme: On-Off Keying
    

class ChamberlainInfo(ManufacturerInfo):
    """Info for Chamberlain/LiftMaster garage door systems."""
    
    name = "Chamberlain/LiftMaster"
    frequencies = [315e6, 390e6]  # US and Canada frequencies
    code_lengths = [40]  # 40-bit rolling code
    modulation = "OOK"  


class GenieInfo(ManufacturerInfo):
    """Info for Genie garage door systems."""
    
    name = "Genie"
    frequencies = [315e6, 390e6]
    code_lengths = [32]  # 32-bit rolling code
    modulation = "OOK"


class LinearInfo(ManufacturerInfo):
    """Info for Linear garage door systems."""
    
    name = "Linear"
    frequencies = [315e6]
    code_lengths = [24]  # 24-bit code
    modulation = "OOK"


class StanleyInfo(ManufacturerInfo):
    """Info for Stanley garage door systems."""
    
    name = "Stanley"
    frequencies = [300e6, 310e6]
    code_lengths = [28]  # 28-bit code
    modulation = "OOK" 


class FaacInfo(ManufacturerInfo):
    """Info for FAAC (European) garage door systems."""
    
    name = "FAAC" 
    frequencies = [433.92e6]  # European frequency
    code_lengths = [64]       # 64-bit code
    modulation = "FSK"        # Uses FSK modulation


# Dictionary of manufacturer handlers by name
MANUFACTURER_INFO: Dict[str, Type[ManufacturerInfo]] = {
    "generic": ManufacturerInfo,
    "chamberlain": ChamberlainInfo,
    "liftmaster": ChamberlainInfo,  # Same as Chamberlain
    "genie": GenieInfo,
    "linear": LinearInfo,
    "stanley": StanleyInfo,
    "faac": FaacInfo
}


def main():
    """Display information about supported manufacturers."""
    print("\nSupported Manufacturers and Frequencies:")
    print("=" * 50)
    for name, info_class in MANUFACTURER_INFO.items():
        freqs = [f"{f/1e6:.2f} MHz" for f in info_class.frequencies]
        mods = info_class.modulation
        print(f"{info_class.name}:")
        print(f"  Frequencies: {', '.join(freqs)}")
        print(f"  Modulation: {mods}")
        print(f"  Code Length: {', '.join(str(cl) for cl in info_class.code_lengths)} bits")
        print("-" * 50)


if __name__ == "__main__":
    main()