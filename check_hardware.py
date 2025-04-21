#!/usr/bin/env python3
"""
HackRF Hardware Checker

A simple utility to check if HackRF hardware is available and operational.
This can be used independently of the main rolling code analyzer to verify
hardware setup.
"""

import os
import sys
import logging
import subprocess
from datetime import datetime

def setup_logging():
    """Configure basic logging."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = logging.INFO
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure file handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"logs/hardware_check_{timestamp}.log"
    
    # Set up logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_hackrf_package():
    """Check if the hackrf Python package is installed."""
    try:
        import hackrf
        logging.info("HackRF Python package is installed")
        return True
    except ImportError:
        logging.error("HackRF Python package is not installed")
        logging.info("You can install it with: pip install hackrf")
        return False

def check_hackrf_tools():
    """Check if HackRF command-line tools are installed."""
    try:
        result = subprocess.run(['hackrf_info'], 
                               capture_output=True, 
                               text=True, 
                               check=False)
        
        if result.returncode != 0:
            logging.error("HackRF command-line tools are installed but may not be working properly")
            logging.error(f"Error: {result.stderr}")
            return False
        
        logging.info("HackRF command-line tools are installed and working")
        return True
    except FileNotFoundError:
        logging.error("HackRF command-line tools are not installed")
        logging.info("You may need to install the HackRF utilities for your OS")
        return False

def check_hackrf_device():
    """Check if a HackRF device is connected and operational."""
    try:
        # First, try to import the hackrf module
        import hackrf
        
        # Try to open a device
        try:
            device = hackrf.HackRF()
            
            # Get device info
            serial = device.serial_number()
            board_id = device.board_id()
            firmware = device.firmware_version()
            
            logging.info("HackRF device detected and operational")
            logging.info(f"Serial Number: {serial}")
            logging.info(f"Board ID: {board_id}")
            logging.info(f"Firmware Version: {firmware}")
            
            # Clean up
            device.close()
            return True
        except Exception as e:
            logging.error(f"Failed to access HackRF device: {e}")
            logging.info("Check if the device is properly connected and not in use by other applications")
            return False
            
    except ImportError:
        logging.error("Cannot check device: HackRF Python package is not installed")
        return False
    except AttributeError:
        # This may happen in environments where the module exists but hardware support is incomplete
        logging.error("HackRF module exists but appears to be incomplete or incompatible")
        logging.info("This may happen in virtualized or container environments")
        return False

def print_hardware_recommendations():
    """Print recommendations for HackRF hardware setup."""
    print("\n=== Hardware Recommendations ===")
    print("1. Ensure you have a genuine HackRF One or compatible device")
    print("2. Use a high-quality USB cable (USB data cable, not just power)")
    print("3. Connect directly to your computer's USB port (not through a hub)")
    print("4. Ensure proper drivers are installed for your operating system")
    print("5. Make sure no other applications are currently using the HackRF device")
    print("6. Try power cycling the device by disconnecting and reconnecting it")
    print("\nFor more information, visit: https://github.com/greatscottgadgets/hackrf")

def main():
    """Main entry point for hardware checker."""
    setup_logging()
    
    print("\n=== HackRF Hardware Checker ===")
    print("Checking if required HackRF hardware is available...\n")
    
    package_ok = check_hackrf_package()
    tools_ok = check_hackrf_tools()
    device_ok = check_hackrf_device()
    
    print("\n=== Summary ===")
    print(f"HackRF Python Package: {'INSTALLED' if package_ok else 'MISSING'}")
    print(f"HackRF Command Tools: {'INSTALLED' if tools_ok else 'MISSING'}")
    print(f"HackRF Device: {'DETECTED' if device_ok else 'NOT FOUND'}")
    
    if package_ok and tools_ok and device_ok:
        print("\n✅ SUCCESS: HackRF hardware is fully operational")
        print("You can run the main garage door code analyzer with full functionality")
    else:
        print("\n❌ WARNING: HackRF hardware is not fully operational")
        print("You can still use the following limited functionality:")
        print("  - python list_manufacturers.py")
        print("  - python main.py --list-manufacturers")
        print_hardware_recommendations()
    
if __name__ == "__main__":
    main()