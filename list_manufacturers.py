#!/usr/bin/env python3
"""
List supported manufacturers and their frequencies.

This utility script works without HackRF hardware and provides information about
supported garage door manufacturers in the system.
"""

from manufacturer_info import MANUFACTURER_INFO

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