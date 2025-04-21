#!/usr/bin/env python3
"""
HackRF Garage Door Signal Analyzer CLI

Dedicated command-line tool for analyzing garage door signals using HackRF.
Provides frequency oscillation, signal capture, and rolling code analysis.
REQUIRES physical HackRF hardware - no simulation mode available.
For educational and security research purposes only.
"""

import argparse
import logging
import sys
import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("garage_code_analyzer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import HackRF module - this is a hard requirement
import hackrf

class HackRFGarageAnalyzer:
    """Main class for HackRF garage door signal analysis."""
    
    def __init__(self, center_freq: float = 315e6, sample_rate: float = 2e6, 
                rx_gain: float = 20.0, tx_gain: float = 20.0):
        """
        Initialize the HackRF analyzer.
        
        Args:
            center_freq: Center frequency in Hz (default: 315 MHz)
            sample_rate: Sample rate in Hz (default: 2 MHz)
            rx_gain: Receiver gain (default: 20.0 dB)
            tx_gain: Transmitter gain (default: 20.0 dB)
        """
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.rx_gain = rx_gain
        self.tx_gain = tx_gain
        self.device = None
        self._buffer_size = 262144  # Default buffer size
        
        # Common garage door frequencies in MHz
        self.common_frequencies = {
            "300 MHz": 300e6,
            "310 MHz": 310e6,
            "315 MHz": 315e6,
            "318 MHz": 318e6,
            "390 MHz": 390e6,
            "433.92 MHz": 433.92e6
        }
    
    def setup_device(self) -> bool:
        """
        Initialize and configure the HackRF device.
        
        Returns:
            True if device was successfully configured, False otherwise
        """
        if self.device is not None:
            return True
            
        try:
            self.device = hackrf.HackRF()
            self.device.sample_rate = self.sample_rate
            self.device.center_freq = self.center_freq
            self.device.rx_gain = self.rx_gain
            self.device.enable_amp = True
            logging.info(f"HackRF initialized (Serial: {self.device.serial_number})")
            logging.info(f"Frequency: {self.center_freq/1e6:.3f} MHz, Sample Rate: {self.sample_rate/1e6:.1f} MHz")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize HackRF: {e}")
            return False
    
    def close_device(self):
        """Close the HackRF device connection."""
        if self.device:
            try:
                self.device.close()
                logging.info("HackRF device closed")
            except Exception as e:
                logging.error(f"Error closing HackRF: {e}")
            finally:
                self.device = None
    
    def scan_frequencies(self) -> Dict[str, float]:
        """
        Scan common garage door frequencies and measure signal strength.
        
        Returns:
            Dictionary mapping frequency names to signal strengths in dB
        """
        if not self.setup_device():
            logging.error("Cannot scan frequencies: HackRF device not available")
            return {}
            
        logging.info("Scanning common garage door frequencies...")
        results = {}
        
        for name, freq in self.common_frequencies.items():
            logging.info(f"Scanning {name}...")
            self.device.center_freq = freq
            
            # Capture short samples to measure strength
            self.device.start_rx_mode()
            time.sleep(0.5)  # Allow AGC to settle
            
            samples = self.device.read_samples(self._buffer_size)
            self.device.stop_rx_mode()
            
            # Calculate power
            power = 10 * np.log10(np.mean(np.abs(samples) ** 2))
            results[name] = power
            logging.info(f"  Signal strength: {power:.2f} dB")
            
            time.sleep(0.2)  # Brief pause between frequencies
        
        # Reset frequency
        self.device.center_freq = self.center_freq
        
        # Find strongest frequency
        strongest = max(results, key=results.get)
        logging.info(f"Strongest signal: {strongest} at {results[strongest]:.2f} dB")
        
        return results
    
    def capture_signal(self, duration: int = 10, filename: Optional[str] = None) -> np.ndarray:
        """
        Capture a signal using HackRF.
        
        Args:
            duration: Duration in seconds to capture
            filename: Optional filename to save the captured data
            
        Returns:
            Numpy array containing the captured signal data
        """
        if not self.setup_device():
            logging.error("Cannot capture signal: HackRF device not available")
            return np.array([])
            
        logging.info(f"Capturing signal at {self.center_freq/1e6:.3f} MHz for {duration} seconds...")
        
        # Calculate number of samples
        num_samples = int(duration * self.sample_rate)
        samples = np.zeros(num_samples, dtype=np.complex64)
        
        try:
            self.device.start_rx_mode()
            
            # Read samples
            samples_read = 0
            with logging.progress_bar(num_samples) as progress:
                while samples_read < num_samples:
                    buffer_size = min(self._buffer_size, num_samples - samples_read)
                    buffer = self.device.read_samples(buffer_size)
                    
                    if len(buffer) == 0:
                        logging.error("HackRF returned empty buffer")
                        break
                        
                    samples[samples_read:samples_read + len(buffer)] = buffer
                    samples_read += len(buffer)
                    progress.update(len(buffer))
            
            self.device.stop_rx_mode()
            
            logging.info(f"Captured {samples_read} samples ({samples_read/self.sample_rate:.2f} seconds)")
            
            # Save to file if requested
            if filename:
                np.save(filename, samples)
                logging.info(f"Saved captured data to {filename}")
            
            return samples
            
        except KeyboardInterrupt:
            logging.info("Capture interrupted by user")
            self.device.stop_rx_mode()
            return samples[:samples_read] if samples_read > 0 else np.array([])
            
        except Exception as e:
            logging.error(f"Error during capture: {e}")
            if self.device:
                self.device.stop_rx_mode()
            return np.array([])
    
    def analyze_signal(self, signal_data: np.ndarray) -> Dict:
        """
        Analyze captured signal to extract key characteristics.
        
        Args:
            signal_data: Captured signal data as numpy array
            
        Returns:
            Dictionary with analysis results
        """
        if len(signal_data) == 0:
            logging.error("Cannot analyze empty signal data")
            return {"error": "Empty signal data"}
            
        logging.info("Analyzing signal characteristics...")
        
        # Calculate signal statistics
        power = np.mean(np.abs(signal_data) ** 2)
        peak = np.max(np.abs(signal_data))
        
        # Calculate frequency spectrum
        spectrum = np.fft.fftshift(np.fft.fft(signal_data))
        freq = np.fft.fftshift(np.fft.fftfreq(len(signal_data), 1/self.sample_rate))
        
        # Find peak frequencies
        peak_idx = np.argsort(np.abs(spectrum))[-5:]
        peak_freqs = freq[peak_idx] + self.center_freq
        
        # Calculate SNR
        signal_power = np.mean(np.sort(np.abs(spectrum) ** 2)[-int(len(spectrum)*0.1):])
        noise_power = np.mean(np.sort(np.abs(spectrum) ** 2)[:int(len(spectrum)*0.8)])
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        
        # Detect possible OOK (On-Off Keying) modulation
        # Look for distinct on/off patterns in signal magnitude
        magnitude = np.abs(signal_data)
        threshold = (np.max(magnitude) + np.mean(magnitude)) / 2
        ook_bits = (magnitude > threshold).astype(int)
        
        # Count transitions as an indicator of potential data
        transitions = np.sum(np.abs(np.diff(ook_bits)))
        ook_detected = transitions > 20  # Arbitrary threshold for "meaningful" transitions
        
        results = {
            "duration": len(signal_data) / self.sample_rate,
            "power_db": 10 * np.log10(power) if power > 0 else -100,
            "peak_amplitude": peak,
            "peak_frequencies_mhz": [f/1e6 for f in peak_freqs],
            "snr_db": snr,
            "ook_modulation_detected": ook_detected,
            "bit_transitions": transitions
        }
        
        for key, value in results.items():
            if isinstance(value, float):
                logging.info(f"  {key}: {value:.2f}")
            else:
                logging.info(f"  {key}: {value}")
                
        return results
    
    def extract_rolling_codes(self, signal_data: np.ndarray) -> List[str]:
        """
        Extract potential rolling codes from the signal.
        
        Args:
            signal_data: Captured signal data
            
        Returns:
            List of extracted code strings
        """
        if len(signal_data) == 0:
            logging.error("Cannot extract codes from empty signal data")
            return []
            
        logging.info("Extracting potential rolling codes...")
        
        # Demodulate OOK signal
        magnitude = np.abs(signal_data)
        threshold = (np.max(magnitude) + np.mean(magnitude)) / 2
        bits = (magnitude > threshold).astype(int)
        
        # Simplify bit sequence (remove repeated bits)
        simplified_bits = []
        for i in range(1, len(bits)):
            if bits[i] != bits[i-1]:
                simplified_bits.append(bits[i])
        
        # Find potential code patterns (looking for fixed-length sequences)
        # Common garage door rolling codes are typically 24-64 bits
        code_candidates = []
        
        # Convert to string for easier processing
        bit_string = ''.join(map(str, simplified_bits))
        
        # Look for repeated patterns that could be codes
        for code_length in [24, 32, 40, 64]:
            logging.info(f"Searching for {code_length}-bit codes...")
            
            # Step through the bit string
            for i in range(0, len(bit_string) - code_length, 8):
                candidate = bit_string[i:i+code_length]
                
                # Simple validation: check for sufficient bit transitions and balance
                ones = candidate.count('1')
                zeros = candidate.count('0')
                transitions = sum(1 for i in range(1, len(candidate)) if candidate[i] != candidate[i-1])
                
                # Heuristic validation: reasonable number of transitions and bit balance
                if transitions >= code_length / 8 and min(ones, zeros) >= code_length / 4:
                    if candidate not in code_candidates:
                        code_candidates.append(candidate)
        
        # Format codes with spaces for readability
        formatted_codes = []
        for code in code_candidates:
            # Insert a space every 8 bits
            formatted = ' '.join(code[i:i+8] for i in range(0, len(code), 8))
            formatted_codes.append(formatted)
        
        logging.info(f"Extracted {len(formatted_codes)} potential rolling codes")
        for i, code in enumerate(formatted_codes):
            logging.info(f"  Code {i+1}: {code}")
            
        return formatted_codes
    
    def oscillate_frequency(self, bandwidth: float = 200e3, rate: float = 10.0, 
                          duration: float = 5.0, pattern: str = 'sinusoidal') -> bool:
        """
        Generate a signal that oscillates around the center frequency.
        
        Args:
            bandwidth: Oscillation bandwidth in Hz
            rate: Oscillation rate in Hz (cycles per second)
            duration: Duration in seconds
            pattern: Oscillation pattern ('sinusoidal', 'linear', 'random')
            
        Returns:
            True if operation was successful, False otherwise
        """
        if not self.setup_device():
            logging.error("Cannot oscillate: HackRF device not available")
            return False
            
        logging.info(f"Oscillating around {self.center_freq/1e6:.3f} MHz, BW: {bandwidth/1e3:.1f} kHz, Rate: {rate} Hz")
        
        try:
            # Calculate number of frequency updates
            # Update 50 times per cycle for smooth oscillation
            updates_per_cycle = 50
            total_updates = int(duration * rate * updates_per_cycle)
            update_interval = duration / total_updates
            
            # Generate frequency pattern
            if pattern == 'sinusoidal':
                # Generate sinusoidal pattern
                t = np.linspace(0, duration, total_updates)
                freqs = self.center_freq + bandwidth/2 * np.sin(2 * np.pi * rate * t)
                
            elif pattern == 'linear':
                # Generate sawtooth pattern
                freqs = []
                for i in range(total_updates):
                    cycle_position = (i % updates_per_cycle) / updates_per_cycle
                    if cycle_position < 0.5:
                        # Rising
                        offset = bandwidth * cycle_position * 2
                    else:
                        # Falling
                        offset = bandwidth * (1 - (cycle_position - 0.5) * 2)
                    freqs.append(self.center_freq - bandwidth/2 + offset)
                    
            elif pattern == 'random':
                # Generate random jumps
                freqs = self.center_freq + np.random.uniform(-bandwidth/2, bandwidth/2, total_updates)
                
            else:
                logging.error(f"Unknown pattern: {pattern}")
                return False
            
            # Configure TX
            self.device.enable_tx()
            self.device.tx_gain = self.tx_gain
            
            # Generate a constant carrier wave
            carrier_wave = np.ones(1024, dtype=np.complex64)
            
            logging.info(f"Starting frequency oscillation for {duration:.1f} seconds...")
            
            # Perform frequency hopping
            start_time = time.time()
            for i, freq in enumerate(freqs):
                if time.time() - start_time > duration:
                    break
                    
                # Update frequency
                self.device.center_freq = int(freq)
                
                # Transmit carrier
                self.device.transmit(carrier_wave)
                
                # Wait for next update
                time.sleep(update_interval)
                
                # Show progress periodically
                if i % 50 == 0:
                    progress = (time.time() - start_time) / duration * 100
                    logging.info(f"  Oscillation progress: {progress:.1f}%")
            
            # Stop transmission
            self.device.disable_tx()
            logging.info("Frequency oscillation completed")
            return True
            
        except KeyboardInterrupt:
            logging.info("Oscillation interrupted by user")
            if self.device:
                self.device.disable_tx()
            return False
            
        except Exception as e:
            logging.error(f"Error during oscillation: {e}")
            if self.device:
                self.device.disable_tx()
            return False
    
    def sweep_frequency(self, start_freq: float, end_freq: float, 
                      sweep_time: float = 5.0, repeats: int = 1) -> bool:
        """
        Perform a linear frequency sweep (chirp).
        
        Args:
            start_freq: Starting frequency in Hz
            end_freq: Ending frequency in Hz
            sweep_time: Time to complete one sweep in seconds
            repeats: Number of sweep repetitions
            
        Returns:
            True if operation was successful, False otherwise
        """
        if not self.setup_device():
            logging.error("Cannot sweep frequency: HackRF device not available")
            return False
            
        logging.info(f"Sweeping from {start_freq/1e6:.3f} MHz to {end_freq/1e6:.3f} MHz")
        logging.info(f"Sweep time: {sweep_time:.1f} seconds, Repeats: {repeats}")
        
        try:
            # Calculate frequency steps
            # Update 100 times per second for smooth sweep
            updates_per_second = 100
            steps_per_sweep = int(sweep_time * updates_per_second)
            step_interval = sweep_time / steps_per_sweep
            
            # Generate frequency steps
            freqs = np.linspace(start_freq, end_freq, steps_per_sweep)
            
            # Configure TX
            self.device.enable_tx()
            self.device.tx_gain = self.tx_gain
            
            # Generate a constant carrier wave
            carrier_wave = np.ones(1024, dtype=np.complex64)
            
            logging.info(f"Starting frequency sweep...")
            
            # Perform sweeps
            for r in range(repeats):
                logging.info(f"Sweep {r+1}/{repeats}")
                
                start_time = time.time()
                for i, freq in enumerate(freqs):
                    if time.time() - start_time > sweep_time:
                        break
                        
                    # Update frequency
                    self.device.center_freq = int(freq)
                    
                    # Transmit carrier
                    self.device.transmit(carrier_wave)
                    
                    # Wait for next update
                    time.sleep(step_interval)
                    
                    # Show progress periodically
                    if i % 50 == 0:
                        progress = (time.time() - start_time) / sweep_time * 100
                        current_freq = self.device.center_freq / 1e6
                        logging.info(f"  Sweep progress: {progress:.1f}% - Current freq: {current_freq:.3f} MHz")
            
            # Stop transmission
            self.device.disable_tx()
            logging.info("Frequency sweep completed")
            return True
            
        except KeyboardInterrupt:
            logging.info("Sweep interrupted by user")
            if self.device:
                self.device.disable_tx()
            return False
            
        except Exception as e:
            logging.error(f"Error during sweep: {e}")
            if self.device:
                self.device.disable_tx()
            return False
    
    def transmit_code(self, code: str, transmit_count: int = 5, 
                     bit_time: float = 0.001) -> bool:
        """
        Transmit a binary code using OOK modulation.
        
        Args:
            code: Binary code string (0s and 1s)
            transmit_count: Number of times to transmit the code
            bit_time: Duration of each bit in seconds
            
        Returns:
            True if operation was successful, False otherwise
        """
        if not self.setup_device():
            logging.error("Cannot transmit code: HackRF device not available")
            return False
            
        # Clean the code
        code = code.replace(' ', '')
        
        if not all(bit in '01' for bit in code):
            logging.error("Invalid code format. Must be binary (0s and 1s)")
            return False
            
        logging.info(f"Transmitting code: {code}")
        logging.info(f"Frequency: {self.center_freq/1e6:.3f} MHz, Bit time: {bit_time*1000:.1f} ms")
        
        try:
            # Configure TX
            self.device.enable_tx()
            self.device.tx_gain = self.tx_gain
            
            # Prepare samples for OOK modulation
            samples_per_bit = int(bit_time * self.sample_rate)
            
            # For bit '1', send carrier wave
            one_bit = np.ones(samples_per_bit, dtype=np.complex64)
            
            # For bit '0', send zeros
            zero_bit = np.zeros(samples_per_bit, dtype=np.complex64)
            
            logging.info(f"Transmitting code {transmit_count} times...")
            
            for i in range(transmit_count):
                logging.info(f"Transmission {i+1}/{transmit_count}")
                
                # Add preamble - a series of alternating 1s and 0s
                preamble = '10' * 8  # 8 pairs of 1-0
                full_code = preamble + code
                
                # Transmit each bit
                for bit in full_code:
                    if bit == '1':
                        self.device.transmit(one_bit)
                    else:
                        self.device.transmit(zero_bit)
                
                # Add a gap between transmissions
                time.sleep(0.05)
            
            # Stop transmission
            self.device.disable_tx()
            logging.info("Code transmission completed")
            return True
            
        except KeyboardInterrupt:
            logging.info("Transmission interrupted by user")
            if self.device:
                self.device.disable_tx()
            return False
            
        except Exception as e:
            logging.error(f"Error during transmission: {e}")
            if self.device:
                self.device.disable_tx()
            return False
            
    def __del__(self):
        """Cleanup HackRF device on object destruction."""
        self.close_device()


def main():
    """Main entry point for the HackRF Garage Door Signal Analyzer."""
    
    # Print disclaimer
    print("\n" + "="*80)
    print(" HackRF Garage Door Signal Analyzer".center(80))
    print(" For educational and security research purposes ONLY".center(80))
    print("="*80)
    print("""
DISCLAIMER: This tool is provided for EDUCATIONAL and SECURITY RESEARCH purposes ONLY.
Use of this software to access garage door systems or other property without explicit
permission from the owner is illegal and unethical. The authors and distributors accept
no liability for misuse of this tool.

By using this software, you agree to:
1. Only use it on systems you own or have explicit permission to test
2. Follow all applicable local, state, and federal laws
3. Use the tool responsibly and ethically
""")
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='HackRF Garage Door Signal Analyzer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan common garage door frequencies')
    
    # Capture command
    capture_parser = subparsers.add_parser('capture', help='Capture and analyze signals')
    capture_parser.add_argument('-f', '--freq', type=float, default=315.0,
                              help='Frequency in MHz')
    capture_parser.add_argument('-d', '--duration', type=int, default=10,
                              help='Duration in seconds')
    capture_parser.add_argument('-o', '--output', type=str,
                              help='Output file for captured data')
    
    # Oscillate command
    oscillate_parser = subparsers.add_parser('oscillate', help='Perform frequency oscillation')
    oscillate_parser.add_argument('-f', '--freq', type=float, default=315.0,
                                help='Center frequency in MHz')
    oscillate_parser.add_argument('-b', '--bandwidth', type=float, default=200.0,
                                help='Bandwidth in kHz')
    oscillate_parser.add_argument('-r', '--rate', type=float, default=10.0,
                                help='Oscillation rate in Hz')
    oscillate_parser.add_argument('-d', '--duration', type=float, default=5.0,
                                help='Duration in seconds')
    oscillate_parser.add_argument('-p', '--pattern', choices=['sinusoidal', 'linear', 'random'],
                                default='sinusoidal', help='Oscillation pattern')
    
    # Sweep command
    sweep_parser = subparsers.add_parser('sweep', help='Perform frequency sweep')
    sweep_parser.add_argument('-s', '--start', type=float, required=True,
                            help='Start frequency in MHz')
    sweep_parser.add_argument('-e', '--end', type=float, required=True,
                            help='End frequency in MHz')
    sweep_parser.add_argument('-d', '--duration', type=float, default=5.0,
                            help='Duration of one sweep in seconds')
    sweep_parser.add_argument('-r', '--repeats', type=int, default=1,
                            help='Number of sweep repetitions')
    
    # Transmit command
    transmit_parser = subparsers.add_parser('transmit', help='Transmit a binary code')
    transmit_parser.add_argument('-f', '--freq', type=float, default=315.0,
                               help='Frequency in MHz')
    transmit_parser.add_argument('-c', '--code', type=str, required=True,
                               help='Binary code to transmit (0s and 1s)')
    transmit_parser.add_argument('-r', '--repeats', type=int, default=5,
                               help='Number of transmission repetitions')
    transmit_parser.add_argument('-b', '--bit-time', type=float, default=1.0,
                               help='Duration of each bit in milliseconds')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display HackRF device information')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create analyzer instance
    analyzer = HackRFGarageAnalyzer()
    
    try:
        if args.command == 'scan':
            analyzer.scan_frequencies()
            
        elif args.command == 'capture':
            # Set frequency
            analyzer.center_freq = args.freq * 1e6
            
            # Capture signal
            signal_data = analyzer.capture_signal(args.duration, args.output)
            
            if len(signal_data) > 0:
                # Analyze signal
                analyzer.analyze_signal(signal_data)
                
                # Extract codes
                analyzer.extract_rolling_codes(signal_data)
            
        elif args.command == 'oscillate':
            # Set frequency
            analyzer.center_freq = args.freq * 1e6
            
            # Convert bandwidth from kHz to Hz
            bandwidth = args.bandwidth * 1e3
            
            # Perform oscillation
            analyzer.oscillate_frequency(bandwidth, args.rate, args.duration, args.pattern)
            
        elif args.command == 'sweep':
            # Convert frequencies from MHz to Hz
            start_freq = args.start * 1e6
            end_freq = args.end * 1e6
            
            # Perform sweep
            analyzer.sweep_frequency(start_freq, end_freq, args.duration, args.repeats)
            
        elif args.command == 'transmit':
            # Set frequency
            analyzer.center_freq = args.freq * 1e6
            
            # Convert bit time from ms to seconds
            bit_time = args.bit_time / 1000.0
            
            # Transmit code
            analyzer.transmit_code(args.code, args.repeats, bit_time)
            
        elif args.command == 'info':
            # Just initialize the device to display info
            if analyzer.setup_device():
                print(f"HackRF device information:")
                print(f"  Serial number: {analyzer.device.serial_number}")
                print(f"  Board ID: {analyzer.device.board_id}")
                print(f"  Firmware version: {analyzer.device.firmware_version}")
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        analyzer.close_device()


if __name__ == "__main__":
    main()