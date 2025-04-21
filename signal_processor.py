#!/usr/bin/env python3
"""
Signal Processor Module

Provides functionality for processing and analyzing radio signals using HackRF hardware.
Direct hardware interaction only, no simulation fallbacks.
"""

import logging
import numpy as np
import pickle
import sys
from typing import Optional, Tuple, List, Dict
import matplotlib.pyplot as plt
from frequency_handler import FrequencyHandler

# Import HackRF - this is required, no fallbacks
import hackrf

class SignalProcessor:
    """Class for processing and analyzing radio frequency signals using HackRF."""

    def __init__(self, sample_rate: float = 2e6, gain: float = 20.0, skip_hardware_check: bool = False):
        """
        Initialize the signal processor with direct HackRF hardware access.
        This class requires actual HackRF hardware to function. No simulation modes available.

        Args:
            sample_rate: Sample rate in Hz (default: 2 MHz)
            gain: Receiver gain (default: 20.0)
            skip_hardware_check: Skip initial hardware verification (default: False)
            
        Raises:
            ValueError: If sample_rate or gain are invalid
            RuntimeError: If HackRF hardware is not available when hardware_check is run
        """
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if gain < 0 or gain > 62:
            raise ValueError("Gain must be between 0 and 62 dB")
            
        self.sample_rate = sample_rate
        self.gain = gain
        self.device = None
        self._buffer_size = 262144  # Optimal buffer size
        logging.debug(f"Initialized SignalProcessor with sample_rate={sample_rate} Hz, gain={gain}")
        
        # Verify HackRF hardware is available immediately (unless skipped)
        if not skip_hardware_check:
            try:
                self.hardware_check()
                logging.info("HackRF hardware verification completed successfully")
            except RuntimeError as e:
                logging.error(f"HackRF hardware verification failed: {e}")
                logging.error("This tool requires actual HackRF hardware. No simulation mode available.")
                raise

    def _setup_hackrf(self):
        """Initialize and configure HackRF device."""
        if self.device is None:
            try:
                self.device = hackrf.HackRF()
                self.device.sample_rate = self.sample_rate
                self.device.rx_gain = self.gain
                self.device.enable_amp = True
                logging.info("HackRF device initialized successfully")
                return True
            except Exception as e:
                logging.error(f"Failed to initialize HackRF: {e}")
                raise RuntimeError(f"HackRF initialization failed: {e}")
        
        return True

    def capture_signal(self, frequency_handler: FrequencyHandler, duration: int = 30) -> np.ndarray:
        """
        Capture a signal at the specified frequency for the given duration using HackRF hardware.
        No simulation fallbacks.

        Args:
            frequency_handler: FrequencyHandler instance
            duration: Duration in seconds

        Returns:
            Numpy array containing captured signal data
            
        Raises:
            RuntimeError: If HackRF hardware is not available or encounters an error
        """
        logging.info(f"Capturing signal for {duration} seconds at {frequency_handler.center_freq/1e6} MHz")

        # Number of samples to capture
        num_samples = int(self.sample_rate * duration)

        try:
            self._setup_hackrf()
            samples = np.zeros(num_samples, dtype=np.complex64)

            self.device.center_freq = frequency_handler.center_freq
            self.device.start_rx_mode()

            # Read samples
            samples_read = 0
            while samples_read < num_samples:
                buffer_size = min(262144, num_samples - samples_read)
                buffer = self.device.read_samples(buffer_size)
                samples[samples_read:samples_read + len(buffer)] = buffer
                samples_read += len(buffer)

            self.device.stop_rx_mode()
            logging.info(f"Captured {len(samples)} samples using HackRF")
            return samples

        except Exception as e:
            logging.error(f"Error during HackRF signal capture: {e}")
            if self.device:
                try:
                    self.device.stop_rx_mode()
                except:
                    pass
            raise RuntimeError(f"HackRF signal capture failed: {e}")

    def save_signal(self, signal_data: np.ndarray, filename: str) -> None:
        """Save captured signal data to a file."""
        logging.info(f"Saving signal data to {filename}")

        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'data': signal_data,
                    'sample_rate': self.sample_rate,
                    'timestamp': np.datetime64('now')
                }, f)
            logging.info(f"Successfully saved {len(signal_data)} samples")
        except Exception as e:
            logging.error(f"Failed to save signal data: {e}")

    def load_signal(self, filename: str) -> np.ndarray:
        """Load signal data from a file."""
        logging.info(f"Loading signal data from {filename}")

        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict) and 'data' in data:
                signal_data = data['data']
                if 'sample_rate' in data:
                    self.sample_rate = data['sample_rate']
                return signal_data
            return data

        except Exception as e:
            logging.error(f"Failed to load signal data: {e}")
            return np.array([])

    def analyze_signal(self, signal_data: np.ndarray) -> Dict:
        """Analyze captured signal characteristics."""
        logging.info("Analyzing signal characteristics")

        if len(signal_data) == 0:
            return {'error': 'Empty signal data'}

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

        return {
            'duration': len(signal_data) / self.sample_rate,
            'sample_rate': self.sample_rate,
            'power': power,
            'peak_amplitude': peak,
            'peak_frequencies': peak_freqs,
            'snr_db': snr
        }

    def analyze_signal_strength(self, signal_data: np.ndarray) -> float:
        """Calculate signal strength in dB."""
        if len(signal_data) == 0:
            return -100

        power = np.mean(np.abs(signal_data) ** 2)
        return 10 * np.log10(power) if power > 0 else -100

    def visualize_signal(self, signal_data: np.ndarray) -> None:
        """Visualize signal in time and frequency domains."""
        if len(signal_data) == 0:
            logging.warning("Cannot visualize empty signal data")
            return

        plt.figure(figsize=(12, 10))

        # Time domain plot
        plt.subplot(3, 1, 1)
        time = np.arange(len(signal_data)) / self.sample_rate
        plt.plot(time, np.abs(signal_data))
        plt.title('Signal Magnitude (Time Domain)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        # Frequency domain plot
        plt.subplot(3, 1, 2)
        spectrum = np.fft.fftshift(np.fft.fft(signal_data))
        freq = np.fft.fftshift(np.fft.fftfreq(len(signal_data), 1/self.sample_rate))
        plt.plot(freq / 1e6, 20 * np.log10(np.abs(spectrum) + 1e-10))
        plt.title('Signal Spectrum (Frequency Domain)')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (dB)')
        plt.grid(True)

        # Spectrogram
        plt.subplot(3, 1, 3)
        segment_size = min(1024, len(signal_data) // 10)
        plt.specgram(signal_data, NFFT=segment_size, Fs=self.sample_rate/1e6,
                    noverlap=segment_size//2, cmap='viridis')
        plt.title('Spectrogram (Time-Frequency)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (MHz)')
        plt.colorbar(label='Power (dB)')

        plt.tight_layout()
        plt.show()

    def hardware_check(self) -> None:
        """
        Verify that HackRF hardware is available and properly configured.
        
        Raises:
            RuntimeError: If HackRF hardware is not properly connected or accessible
        """
        try:
            devices = hackrf.list_devices()
            if not devices:
                raise RuntimeError("No HackRF devices found. Please connect your HackRF device.")
                
            # Retrieve hardware information for logging purposes
            self._setup_hackrf()
            info = {
                "serial": self.device.serial_number,
                "board_id": self.device.board_id,
                "firmware": self.device.firmware_version
            }
            logging.info(f"HackRF hardware detected: {info}")
            
        except Exception as e:
            raise RuntimeError(f"HackRF hardware check failed: {e}")
        
    def __del__(self):
        """Cleanup HackRF device."""
        if self.device:
            try:
                self.device.close()
            except:
                pass