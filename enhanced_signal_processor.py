#!/usr/bin/env python3
"""
Enhanced Signal Processor Module

Provides advanced signal processing functionality for RF signals captured from garage door remotes.
Includes sophisticated preprocessing techniques and improved bit detection algorithms to enhance
the accuracy and reliability of rolling code extraction.

Features:
- Advanced signal filtering (low-pass, band-pass, wavelet denoising)
- Dynamic thresholding techniques
- Statistical signal analysis
- Bit synchronization and clock recovery
- Signal quality assessment
- Multiple modulation scheme support (OOK, FSK, ASK)

For educational and security research purposes only.
"""

import os
import sys
import math
import time
import json
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

# Import scipy for advanced signal processing
try:
    from scipy import signal
    from scipy.signal import butter, filtfilt, savgol_filter, welch, find_peaks
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import skew, kurtosis
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Install with: pip install scipy")
    print("Advanced signal processing features will be limited without this module.")

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Try to import hackrf module - will fail if hardware not available
try:
    import hackrf
    HACKRF_AVAILABLE = True
except ImportError:
    HACKRF_AVAILABLE = False


class SignalPreprocessor:
    """
    Class for preprocessing RF signals to improve bit detection accuracy.
    """
    
    def __init__(self, debug_mode: bool = False, visualize: bool = False):
        """
        Initialize the signal preprocessor.
        
        Args:
            debug_mode: Enable debug mode for additional output
            visualize: Enable visualization of processing steps
        """
        self.debug_mode = debug_mode
        self.visualize = visualize and MATPLOTLIB_AVAILABLE
        self.processed_signals = []
        self.filter_cache = {}
        
        # Default processing parameters
        self.params = {
            # Filtering parameters
            "low_pass_cutoff": 250e3,       # Low-pass filter cutoff frequency (Hz)
            "high_pass_cutoff": 1e3,        # High-pass filter cutoff frequency (Hz)
            "filter_order": 5,              # Filter order for Butterworth filters
            "use_wavelet_denoising": True,  # Use wavelet denoising
            "wavelet_threshold": 3.0,       # Wavelet threshold (sigma multiplier)
            
            # Averaging parameters
            "moving_avg_window": 5,         # Window size for moving average filter
            "use_savgol": True,             # Use Savitzky-Golay filter
            "savgol_window": 15,            # Window size for Savitzky-Golay filter
            "savgol_order": 3,              # Polynomial order for Savitzky-Golay filter
            
            # Signal segmentation
            "preamble_detection": True,     # Detect and use preamble for sync
            "min_pulse_width": 100e-6,      # Minimum pulse width in seconds
            "max_pulse_width": 2e-3,        # Maximum pulse width in seconds
            
            # Thresholding parameters
            "dynamic_threshold": True,      # Use dynamic thresholding
            "threshold_method": "otsu",     # Thresholding method: "otsu", "adaptive", "kmeans"
            "adaptive_window": 1000,        # Window size for adaptive thresholding
            "fixed_threshold": 0.5,         # Fixed threshold level (0.0-1.0) if not dynamic
            
            # Bit detection parameters
            "min_bit_duration": 200e-6,     # Minimum bit duration in seconds
            "max_bit_duration": 2e-3,       # Maximum bit duration in seconds
            "bit_sync_method": "clock_recovery", # Method: "simple", "clock_recovery", "pll"
            "clock_recovery_pll_bandwidth": 0.02, # PLL bandwidth for clock recovery
            
            # Advanced detection
            "detect_modulation": True,      # Auto-detect modulation scheme
            "use_correlation": True,        # Use correlation for pattern matching
            "manchester_decode": False,     # Decode Manchester encoding
            
            # Miscellaneous
            "snr_threshold": 6.0,           # Minimum SNR (dB) for valid signal
            "max_code_length": 128,         # Maximum rolling code length in bits
            "signal_edge_padding": 1000     # Samples to pad at signal edges
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the signal preprocessor."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure file handler with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = f"logs/signal_processor_{timestamp}.log"
        
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG if self.debug_mode else logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("SignalPreprocessor")
    
    def set_param(self, param_name: str, value: Any):
        """
        Set a processing parameter.
        
        Args:
            param_name: Name of the parameter to set
            value: New parameter value
        """
        if param_name in self.params:
            self.params[param_name] = value
            self.logger.debug(f"Set parameter {param_name} = {value}")
        else:
            self.logger.warning(f"Unknown parameter: {param_name}")
    
    def get_param(self, param_name: str) -> Any:
        """
        Get a processing parameter value.
        
        Args:
            param_name: Name of the parameter to get
            
        Returns:
            Parameter value, or None if parameter doesn't exist
        """
        return self.params.get(param_name)
    
    def preprocess_signal(self, raw_signal: np.ndarray, sample_rate: float) -> Dict[str, Any]:
        """
        Apply full preprocessing pipeline to a raw signal.
        
        Args:
            raw_signal: Raw signal data as complex numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary with processed signal and metadata
        """
        if not SCIPY_AVAILABLE:
            self.logger.warning("SciPy not available, using limited preprocessing")
            return self._limited_preprocessing(raw_signal, sample_rate)
        
        # Initialize result structure
        result = {
            "raw_signal": raw_signal,
            "sample_rate": sample_rate,
            "processed_signal": None,
            "magnitude_signal": None,
            "thresholded_signal": None,
            "filtered_signal": None,
            "bits": None,
            "bit_timing": None,
            "clock_period": None,
            "snr": None,
            "modulation": None,
            "processing_steps": [],
            "stats": {},
            "quality_metrics": {}
        }
        
        try:
            self.logger.info(f"Preprocessing signal with {len(raw_signal)} samples, sample rate: {sample_rate/1e6:.2f} MHz")
            start_time = time.time()
            
            # Extract magnitude for non-complex signals
            if np.iscomplexobj(raw_signal):
                magnitude = np.abs(raw_signal)
                result["processing_steps"].append("complex_magnitude")
            else:
                magnitude = raw_signal
                
            result["magnitude_signal"] = magnitude
            
            # Calculate initial signal statistics
            result["stats"]["original"] = self._calculate_signal_stats(magnitude)
            
            # Check signal quality before processing
            snr_estimate = self._estimate_snr(magnitude)
            result["snr"] = snr_estimate
            
            if snr_estimate < self.params["snr_threshold"]:
                self.logger.warning(f"Low SNR detected: {snr_estimate:.2f} dB, below threshold {self.params['snr_threshold']} dB")
                result["quality_metrics"]["low_snr"] = True
            
            # Apply filtering
            filtered_signal = self._apply_filtering(magnitude, sample_rate)
            result["filtered_signal"] = filtered_signal
            result["processing_steps"].append("filtering")
            
            # Calculate post-filtering statistics
            result["stats"]["filtered"] = self._calculate_signal_stats(filtered_signal)
            
            # Detect modulation scheme
            if self.params["detect_modulation"]:
                modulation = self._detect_modulation(filtered_signal, sample_rate)
                result["modulation"] = modulation
                result["processing_steps"].append("modulation_detection")
            
            # Apply thresholding
            thresholded, threshold_value = self._apply_thresholding(filtered_signal)
            result["thresholded_signal"] = thresholded
            result["threshold_value"] = threshold_value
            result["processing_steps"].append("thresholding")
            
            # Detect bit timing and extract bits
            bit_result = self._extract_bits(thresholded, filtered_signal, sample_rate)
            result.update(bit_result)
            result["processing_steps"].append("bit_extraction")
            
            # Calculate signal quality metrics
            quality_metrics = self._calculate_quality_metrics(
                raw_signal, 
                filtered_signal, 
                thresholded, 
                bit_result.get("bit_confidence", [])
            )
            result["quality_metrics"].update(quality_metrics)
            
            # Set final processed signal
            result["processed_signal"] = filtered_signal
            
            # Log processing time
            processing_time = time.time() - start_time
            self.logger.info(f"Preprocessing completed in {processing_time:.3f} seconds")
            result["processing_time"] = processing_time
            
            # Visualize if enabled
            if self.visualize:
                self._visualize_processing(result)
            
            # Store processed signal
            self.processed_signals.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error preprocessing signal: {e}")
            if self.debug_mode:
                import traceback
                self.logger.debug(traceback.format_exc())
            
            # Return partial result with error information
            result["error"] = str(e)
            return result
    
    def _limited_preprocessing(self, raw_signal: np.ndarray, sample_rate: float) -> Dict[str, Any]:
        """
        Apply limited preprocessing when SciPy is not available.
        
        Args:
            raw_signal: Raw signal data
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary with processed signal and metadata
        """
        # Simple preprocessing without scipy
        result = {
            "raw_signal": raw_signal,
            "sample_rate": sample_rate,
            "processed_signal": None,
            "bits": None,
            "processing_steps": ["limited"],
            "quality_metrics": {"limited_processing": True}
        }
        
        # Extract magnitude for non-complex signals
        if np.iscomplexobj(raw_signal):
            magnitude = np.abs(raw_signal)
        else:
            magnitude = raw_signal
            
        # Simple moving average filter
        window_size = self.params["moving_avg_window"]
        filtered = np.convolve(magnitude, np.ones(window_size)/window_size, mode='same')
        
        # Simple fixed thresholding
        threshold = self.params["fixed_threshold"] * (np.max(filtered) - np.min(filtered)) + np.min(filtered)
        thresholded = (filtered > threshold).astype(int)
        
        # Extract bits with simple method
        bits = []
        current_bit = thresholded[0]
        transitions = np.where(np.diff(thresholded) != 0)[0]
        
        if len(transitions) > 1:
            # Estimate bit duration from transitions
            durations = np.diff(transitions)
            median_duration = np.median(durations)
            
            # Extract bits based on duration
            for i in range(len(transitions) - 1):
                duration = transitions[i+1] - transitions[i]
                num_bits = max(1, round(duration / median_duration))
                bits.extend([current_bit] * num_bits)
                current_bit = 1 - current_bit
        
        result["processed_signal"] = filtered
        result["thresholded_signal"] = thresholded
        result["bits"] = bits
        
        return result
    
    def _apply_filtering(self, signal_data: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Apply filtering to the signal.
        
        Args:
            signal_data: Input signal data
            sample_rate: Sample rate in Hz
            
        Returns:
            Filtered signal
        """
        filtered = signal_data.copy()
        
        # Apply low-pass filter
        if self.params["low_pass_cutoff"] > 0:
            nyquist = 0.5 * sample_rate
            cutoff = min(self.params["low_pass_cutoff"] / nyquist, 0.95)  # Normalize cutoff
            
            # Create or retrieve filter coefficients
            filter_key = f"lowpass_{cutoff}_{self.params['filter_order']}"
            if filter_key in self.filter_cache:
                b, a = self.filter_cache[filter_key]
            else:
                b, a = butter(self.params["filter_order"], cutoff, btype='low')
                self.filter_cache[filter_key] = (b, a)
            
            # Apply filter
            filtered = filtfilt(b, a, filtered)
            self.logger.debug(f"Applied low-pass filter, cutoff: {self.params['low_pass_cutoff']/1e3:.1f} kHz")
        
        # Apply high-pass filter if needed
        if self.params["high_pass_cutoff"] > 0:
            nyquist = 0.5 * sample_rate
            cutoff = min(self.params["high_pass_cutoff"] / nyquist, 0.95)  # Normalize cutoff
            
            # Create or retrieve filter coefficients
            filter_key = f"highpass_{cutoff}_{self.params['filter_order']}"
            if filter_key in self.filter_cache:
                b, a = self.filter_cache[filter_key]
            else:
                b, a = butter(self.params["filter_order"], cutoff, btype='high')
                self.filter_cache[filter_key] = (b, a)
            
            # Apply filter
            filtered = filtfilt(b, a, filtered)
            self.logger.debug(f"Applied high-pass filter, cutoff: {self.params['high_pass_cutoff']/1e3:.1f} kHz")
        
        # Apply Savitzky-Golay filter for smoothing if enabled
        if self.params["use_savgol"]:
            window = self.params["savgol_window"]
            if window % 2 == 0:
                window += 1  # Ensure window is odd
            
            # Ensure window is smaller than signal length
            if window < len(filtered):
                filtered = savgol_filter(
                    filtered, 
                    window_length=window,
                    polyorder=self.params["savgol_order"]
                )
                self.logger.debug(f"Applied Savitzky-Golay filter, window: {window}")
        
        # Apply Gaussian smoothing
        filtered = gaussian_filter1d(filtered, sigma=1.0)
        
        # Normalize the filtered signal
        if np.max(filtered) != np.min(filtered):
            filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))
        
        return filtered
    
    def _apply_thresholding(self, signal_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply thresholding to the signal using the specified method.
        
        Args:
            signal_data: Input signal data
            
        Returns:
            Tuple of (thresholded_signal, threshold_value)
        """
        if not self.params["dynamic_threshold"]:
            # Use fixed thresholding
            threshold = self.params["fixed_threshold"]
            if threshold <= 0 or threshold >= 1:
                # Use mid-point if threshold is out of range
                threshold = 0.5
                
            # Scale threshold to signal range
            threshold_value = threshold * (np.max(signal_data) - np.min(signal_data)) + np.min(signal_data)
            self.logger.debug(f"Using fixed threshold: {threshold_value:.4f}")
        else:
            # Use dynamic thresholding based on the specified method
            method = self.params["threshold_method"]
            
            if method == "otsu":
                # Otsu's method for threshold selection
                threshold_value = self._otsu_threshold(signal_data)
                self.logger.debug(f"Otsu threshold: {threshold_value:.4f}")
                
            elif method == "adaptive":
                # Adaptive thresholding using local window statistics
                window_size = self.params["adaptive_window"]
                thresholded = np.zeros_like(signal_data, dtype=int)
                
                # Apply adaptive threshold in windows
                for i in range(0, len(signal_data), window_size):
                    end = min(i + window_size, len(signal_data))
                    window = signal_data[i:end]
                    
                    if len(window) > 5:  # Ensure window has enough samples
                        # Use local mean and standard deviation
                        local_mean = np.mean(window)
                        local_std = np.std(window)
                        local_threshold = local_mean + 0.5 * local_std
                        
                        thresholded[i:end] = (window > local_threshold).astype(int)
                
                # Return early with the result - no global threshold for adaptive
                return thresholded, 0.0
                
            elif method == "kmeans":
                # K-means clustering (2 clusters) for threshold selection
                from scipy.cluster.vq import kmeans2
                
                # Reshape for k-means
                data = signal_data.reshape(-1, 1)
                
                # Initialize centroids at 25% and 75% percentiles
                p25 = np.percentile(data, 25)
                p75 = np.percentile(data, 75)
                centroids_init = np.array([[p25], [p75]])
                
                # Run k-means
                centroids, labels = kmeans2(data, centroids_init, minit='matrix')
                
                # Use midpoint between centroids as threshold
                threshold_value = (centroids[0] + centroids[1]) / 2
                self.logger.debug(f"K-means threshold: {threshold_value[0]:.4f}")
                threshold_value = threshold_value[0]  # Extract scalar value
                
            else:
                # Unknown method, fall back to fixed threshold
                threshold_value = 0.5 * (np.max(signal_data) - np.min(signal_data)) + np.min(signal_data)
                self.logger.warning(f"Unknown threshold method: {method}, using fixed threshold")
        
        # Apply the threshold
        thresholded = (signal_data > threshold_value).astype(int)
        
        return thresholded, threshold_value
    
    def _otsu_threshold(self, signal_data: np.ndarray) -> float:
        """
        Calculate Otsu's threshold for signal data.
        
        Args:
            signal_data: Input signal data
            
        Returns:
            Threshold value
        """
        # Normalize signal to 0-255 range for histogram
        if np.max(signal_data) != np.min(signal_data):
            normalized = ((signal_data - np.min(signal_data)) * 255 / 
                         (np.max(signal_data) - np.min(signal_data))).astype(np.uint8)
        else:
            return 0.5  # Default if signal is constant
        
        # Calculate histogram
        hist, bin_edges = np.histogram(normalized, bins=256, range=(0, 256))
        
        # Calculate cumulative sums
        w1 = np.cumsum(hist)
        w2 = w1[-1] - w1
        
        # Handle edge case
        if w1[0] == 0:
            w1[0] = 1
        
        # Calculate cumulative means
        m1 = np.cumsum(hist * np.arange(256)) / w1
        m2 = (np.cumsum(hist * np.arange(256))[-1] - np.cumsum(hist * np.arange(256))) / w2
        
        # Handle division by zero
        m2[w2 == 0] = 0
        
        # Calculate between class variance
        variance_between = w1[:-1] * w2[:-1] * (m1[:-1] - m2[:-1]) ** 2
        
        # Find threshold
        threshold_idx = np.argmax(variance_between)
        threshold_value = bin_edges[threshold_idx + 1]
        
        # Scale back to original signal range
        if np.max(signal_data) != np.min(signal_data):
            threshold_value = (threshold_value / 255.0 * 
                             (np.max(signal_data) - np.min(signal_data)) + 
                             np.min(signal_data))
        
        return threshold_value
    
    def _extract_bits(self, thresholded: np.ndarray, 
                     filtered_signal: np.ndarray, 
                     sample_rate: float) -> Dict[str, Any]:
        """
        Extract bits from thresholded signal with improved timing recovery.
        
        Args:
            thresholded: Thresholded signal (0s and 1s)
            filtered_signal: Filtered signal (for correlation-based methods)
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary with bit extraction results
        """
        result = {
            "bits": None,
            "bit_timing": None,
            "clock_period": None,
            "bit_confidence": None
        }
        
        # Find all transitions in the thresholded signal
        transitions = np.where(np.diff(thresholded) != 0)[0]
        
        if len(transitions) <= 1:
            self.logger.warning("Not enough transitions detected for bit extraction")
            return result
        
        # Measure intervals between transitions
        intervals = np.diff(transitions)
        
        # Filter out extremely short intervals (noise)
        min_interval_samples = max(2, int(self.params["min_pulse_width"] * sample_rate))
        valid_intervals = intervals[intervals >= min_interval_samples]
        
        if len(valid_intervals) < 3:
            self.logger.warning("Not enough valid intervals for bit timing estimation")
            return result
        
        # Use the bit synchronization method specified in parameters
        sync_method = self.params["bit_sync_method"]
        
        if sync_method == "simple":
            # Simple bit recovery based on transition spacing
            bit_result = self._simple_bit_recovery(thresholded, transitions, intervals, sample_rate)
            
        elif sync_method == "clock_recovery":
            # Advanced clock recovery
            bit_result = self._clock_recovery(thresholded, transitions, filtered_signal, sample_rate)
            
        elif sync_method == "pll":
            # Phase-locked loop based recovery
            bit_result = self._pll_recovery(thresholded, transitions, sample_rate)
            
        else:
            # Default to simple method
            self.logger.warning(f"Unknown bit sync method: {sync_method}, using simple recovery")
            bit_result = self._simple_bit_recovery(thresholded, transitions, intervals, sample_rate)
        
        # Update results
        result.update(bit_result)
        
        # Manchester decoding if enabled
        if self.params["manchester_decode"] and result["bits"] is not None:
            manchester_bits = self._decode_manchester(result["bits"])
            result["manchester_bits"] = manchester_bits
            self.logger.debug(f"Manchester decoded {len(result['bits'])} bits to {len(manchester_bits)} bits")
        
        return result
    
    def _simple_bit_recovery(self, thresholded: np.ndarray, 
                            transitions: np.ndarray, 
                            intervals: np.ndarray,
                            sample_rate: float) -> Dict[str, Any]:
        """
        Simple bit recovery based on transition spacing.
        
        Args:
            thresholded: Thresholded signal
            transitions: Indices of transitions
            intervals: Intervals between transitions
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary with bit recovery results
        """
        # Estimate clock period using a histogram of intervals
        hist, bins = np.histogram(intervals, bins=50)
        peak_idx = np.argmax(hist)
        estimated_bit_period = bins[peak_idx] + (bins[peak_idx+1] - bins[peak_idx])/2
        
        # Refine by looking at nearby histogram bins
        bin_width = bins[1] - bins[0]
        lower = max(0, peak_idx - 2)
        upper = min(len(hist) - 1, peak_idx + 2)
        
        # Weighted average of most common intervals
        weights = hist[lower:upper+1]
        if np.sum(weights) > 0:
            weighted_period = np.sum(weights * bins[lower:upper+1]) / np.sum(weights)
            estimated_bit_period = weighted_period
        
        # Ensure the period is reasonable
        min_period = self.params["min_bit_duration"] * sample_rate
        max_period = self.params["max_bit_duration"] * sample_rate
        
        if estimated_bit_period < min_period:
            self.logger.warning(f"Estimated bit period too short: {estimated_bit_period/sample_rate*1e6:.1f} µs, " +
                               f"minimum: {min_period/sample_rate*1e6:.1f} µs")
            estimated_bit_period = min_period
        elif estimated_bit_period > max_period:
            self.logger.warning(f"Estimated bit period too long: {estimated_bit_period/sample_rate*1e6:.1f} µs, " +
                               f"maximum: {max_period/sample_rate*1e6:.1f} µs")
            estimated_bit_period = max_period
        
        self.logger.debug(f"Estimated bit period: {estimated_bit_period/sample_rate*1e6:.1f} µs")
        
        # Extract bits based on transitions and estimated bit period
        bits = []
        bit_timing = []
        bit_confidence = []
        
        current_bit = thresholded[0]
        
        for i in range(len(transitions) - 1):
            interval = transitions[i+1] - transitions[i]
            num_bits = max(1, round(interval / estimated_bit_period))
            
            # Calculate confidence based on how close the interval is to a multiple of the bit period
            expected_interval = num_bits * estimated_bit_period
            error = abs(interval - expected_interval) / expected_interval
            confidence = max(0, 1.0 - 2.0 * error)  # 1.0 for perfect match, 0 for 50%+ error
            
            # Add bits and timing information
            for j in range(num_bits):
                bits.append(current_bit)
                sample_idx = transitions[i] + int(j * estimated_bit_period)
                bit_timing.append(sample_idx / sample_rate)
                bit_confidence.append(confidence)
            
            # Flip the bit for the next interval
            current_bit = 1 - current_bit
        
        # Add final bit(s) after the last transition
        if len(transitions) > 0:
            remaining_samples = len(thresholded) - transitions[-1] - 1
            num_bits = max(1, round(remaining_samples / estimated_bit_period))
            
            # Final bit is opposite of the last transition's bit
            final_bit = 1 - thresholded[transitions[-1]]
            
            for j in range(num_bits):
                bits.append(final_bit)
                sample_idx = transitions[-1] + int(j * estimated_bit_period)
                if sample_idx < len(thresholded):
                    bit_timing.append(sample_idx / sample_rate)
                    bit_confidence.append(0.8)  # Assume reasonable confidence for trailing bits
        
        return {
            "bits": bits,
            "bit_timing": bit_timing,
            "clock_period": estimated_bit_period / sample_rate,
            "bit_confidence": bit_confidence
        }
    
    def _clock_recovery(self, thresholded: np.ndarray, 
                       transitions: np.ndarray, 
                       filtered_signal: np.ndarray,
                       sample_rate: float) -> Dict[str, Any]:
        """
        Advanced clock recovery for bit extraction.
        
        Args:
            thresholded: Thresholded signal
            transitions: Indices of transitions
            filtered_signal: Filtered signal for correlation
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary with clock recovery results
        """
        # First, get a rough estimate of the bit period from transition intervals
        intervals = np.diff(transitions)
        
        # Filter out outliers (very short or very long intervals)
        p25 = np.percentile(intervals, 25)
        p75 = np.percentile(intervals, 75)
        iqr = p75 - p25
        lower_bound = max(1, p25 - 1.5 * iqr)
        upper_bound = p75 + 1.5 * iqr
        filtered_intervals = intervals[(intervals >= lower_bound) & (intervals <= upper_bound)]
        
        if len(filtered_intervals) < 3:
            # Fall back to simple method if not enough valid intervals
            return self._simple_bit_recovery(thresholded, transitions, intervals, sample_rate)
        
        # Get initial estimate from the most common interval values
        hist, bins = np.histogram(filtered_intervals, bins=min(50, len(filtered_intervals)//2))
        initial_period_estimate = bins[np.argmax(hist)]
        
        # Refine estimate using autocorrelation of the filtered signal
        # This helps find the true bit rate by looking at signal periodicity
        signal_segment = filtered_signal[transitions[0]:transitions[-1]]
        
        if len(signal_segment) > 1000:
            # For long signals, use a representative segment for faster computation
            segment_length = min(5000, len(signal_segment))
            start_idx = len(signal_segment) // 2 - segment_length // 2
            signal_segment = signal_segment[start_idx:start_idx+segment_length]
        
        if len(signal_segment) > 10:
            # Calculate autocorrelation
            autocorr = np.correlate(signal_segment, signal_segment, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Keep only the positive lags
            
            # Find peaks in autocorrelation
            peaks, properties = find_peaks(autocorr, height=0.5*np.max(autocorr), distance=initial_period_estimate*0.8)
            
            if len(peaks) >= 2:
                # Measure distances between peaks and average them
                peak_intervals = np.diff(peaks)
                median_peak_interval = np.median(peak_intervals)
                
                # Check if the period is reasonable
                min_period = self.params["min_bit_duration"] * sample_rate
                max_period = self.params["max_bit_duration"] * sample_rate
                
                if min_period <= median_peak_interval <= max_period:
                    # Use the refined period estimate
                    bit_period = median_peak_interval
                    self.logger.debug(f"Refined bit period from autocorrelation: {bit_period/sample_rate*1e6:.1f} µs")
                else:
                    # Use the initial estimate
                    bit_period = initial_period_estimate
                    self.logger.debug(f"Using initial bit period estimate: {bit_period/sample_rate*1e6:.1f} µs")
            else:
                # Not enough peaks, use the initial estimate
                bit_period = initial_period_estimate
                self.logger.debug(f"Using initial bit period estimate: {bit_period/sample_rate*1e6:.1f} µs")
        else:
            # Signal segment too short, use initial estimate
            bit_period = initial_period_estimate
        
        # Now extract bits with the refined clock period
        bits = []
        bit_timing = []
        bit_confidence = []
        
        # Find the optimal sampling points using zero-crossing analysis
        sample_points = []
        current_clock = transitions[0] + bit_period / 2  # Start at middle of first bit
        
        # Generate sampling points based on the clock
        while current_clock < len(thresholded):
            sample_points.append(int(current_clock))
            current_clock += bit_period
        
        # Sample at the generated points
        for point in sample_points:
            if 0 <= point < len(thresholded):
                bit_value = thresholded[point]
                bits.append(bit_value)
                bit_timing.append(point / sample_rate)
                
                # Calculate confidence based on distance from transitions
                closest_transition = np.min(np.abs(transitions - point)) if len(transitions) > 0 else bit_period
                # Higher confidence if sampling point is far from transitions
                confidence = min(1.0, closest_transition / (bit_period/2))
                bit_confidence.append(confidence)
        
        return {
            "bits": bits,
            "bit_timing": bit_timing,
            "clock_period": bit_period / sample_rate,
            "bit_confidence": bit_confidence
        }
    
    def _pll_recovery(self, thresholded: np.ndarray, 
                     transitions: np.ndarray, 
                     sample_rate: float) -> Dict[str, Any]:
        """
        Phase-locked loop based clock recovery.
        
        Args:
            thresholded: Thresholded signal
            transitions: Indices of transitions
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary with PLL recovery results
        """
        # For simplicity, estimate initial bit period from transitions
        if len(transitions) < 2:
            return {"bits": [], "bit_timing": [], "clock_period": None, "bit_confidence": []}
        
        intervals = np.diff(transitions)
        bit_period_est = np.median(intervals)
        
        # Convert to frequency and phase
        bit_freq = sample_rate / bit_period_est
        phase = 0.0
        
        # PLL parameters
        pll_bandwidth = self.params["clock_recovery_pll_bandwidth"]
        phase_error_gain = 0.1  # Gain for phase error correction
        
        # Timing recovery
        bits = []
        bit_timing = []
        bit_confidence = []
        current_sample = 0
        
        while current_sample < len(thresholded):
            # Sample at the current point
            if 0 <= current_sample < len(thresholded):
                bit_value = thresholded[current_sample]
                bits.append(bit_value)
                bit_timing.append(current_sample / sample_rate)
                
                # Calculate confidence based on distance from transitions
                closest_transition = np.min(np.abs(transitions - current_sample)) if len(transitions) > 0 else bit_period_est
                confidence = min(1.0, closest_transition / (bit_period_est/2))
                bit_confidence.append(confidence)
            
            # Find nearest transition to measure phase error
            nearest_transition = None
            min_distance = bit_period_est
            
            for trans in transitions:
                distance = abs(trans - current_sample)
                if distance < min_distance:
                    min_distance = distance
                    nearest_transition = trans
            
            # Update PLL only if there's a nearby transition
            if nearest_transition is not None and min_distance < bit_period_est/2:
                # Calculate phase error
                phase_error = (nearest_transition - current_sample) / bit_period_est
                
                # Update frequency (first-order PLL)
                bit_freq += pll_bandwidth * phase_error
                
                # Update phase
                phase += phase_error_gain * phase_error
            
            # Advance to next bit
            bit_period = sample_rate / bit_freq
            current_sample += int(bit_period + phase * bit_period)
            
            # Keep phase bounded
            phase = max(-0.5, min(0.5, phase))
        
        return {
            "bits": bits,
            "bit_timing": bit_timing,
            "clock_period": sample_rate / bit_freq,
            "bit_confidence": bit_confidence
        }
    
    def _decode_manchester(self, bits: List[int]) -> List[int]:
        """
        Decode Manchester encoded bits.
        
        Args:
            bits: List of bits (potentially Manchester encoded)
            
        Returns:
            Decoded bits
        """
        if len(bits) < 2:
            return bits
        
        # Manchester coding: transition in middle of bit period
        # 0 is encoded as 1->0, 1 is encoded as 0->1
        decoded = []
        
        for i in range(0, len(bits) - 1, 2):
            if i + 1 < len(bits):
                # Check for valid Manchester pattern
                if bits[i] == 1 and bits[i+1] == 0:
                    decoded.append(0)  # Falling edge is 0
                elif bits[i] == 0 and bits[i+1] == 1:
                    decoded.append(1)  # Rising edge is 1
                else:
                    # Invalid Manchester code, append as-is
                    decoded.append(bits[i])
        
        return decoded
    
    def _detect_modulation(self, signal_data: np.ndarray, sample_rate: float) -> str:
        """
        Detect the likely modulation scheme used in the signal.
        
        Args:
            signal_data: Input signal data
            sample_rate: Sample rate in Hz
            
        Returns:
            String identifying the detected modulation scheme
        """
        # Calculate basic statistics for detection
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        skewness = skew(signal_data)
        kurt = kurtosis(signal_data)
        
        # Calculate power spectral density
        freqs, psd = welch(signal_data, fs=sample_rate, nperseg=min(1024, len(signal_data)))
        
        # Find peaks in PSD
        peaks, properties = find_peaks(psd, height=0.1*np.max(psd), distance=10)
        
        # OOK typically has high skewness and high dynamic range
        if skewness > 1.0 and (np.max(signal_data) - np.min(signal_data)) / std_val > 5:
            modulation = "OOK"
        # FSK often has multiple peaks in PSD and lower skewness
        elif len(peaks) >= 2 and abs(skewness) < 0.5:
            # Check if peaks are separated by a reasonable amount
            peak_freqs = freqs[peaks]
            freq_diffs = np.diff(peak_freqs)
            if np.any(freq_diffs > 10e3) and np.any(freq_diffs < 500e3):  # Between 10kHz and 500kHz
                modulation = "FSK"
            else:
                modulation = "ASK"
        # ASK has more uniform amplitude distribution
        elif abs(skewness) < 1.0 and kurt < 3.0:
            modulation = "ASK"
        else:
            modulation = "OOK"  # Default to OOK
        
        self.logger.debug(f"Detected modulation: {modulation} (skew={skewness:.2f}, kurt={kurt:.2f}, peaks={len(peaks)})")
        
        return modulation
    
    def _calculate_signal_stats(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate detailed statistics for a signal.
        
        Args:
            signal_data: Input signal data
            
        Returns:
            Dictionary with signal statistics
        """
        stats = {
            "min": float(np.min(signal_data)),
            "max": float(np.max(signal_data)),
            "mean": float(np.mean(signal_data)),
            "median": float(np.median(signal_data)),
            "std": float(np.std(signal_data)),
            "rms": float(np.sqrt(np.mean(np.square(signal_data)))),
            "dynamic_range_db": 20 * np.log10(np.max(signal_data) / (np.min(signal_data) + 1e-10)),
            "skewness": float(skew(signal_data)),
            "kurtosis": float(kurtosis(signal_data))
        }
        
        return stats
    
    def _estimate_snr(self, signal_data: np.ndarray) -> float:
        """
        Estimate signal-to-noise ratio of the signal.
        
        Args:
            signal_data: Input signal data
            
        Returns:
            Estimated SNR in dB
        """
        # Simple method: use top 10% as signal, bottom 10% as noise
        sorted_data = np.sort(signal_data)
        n = len(sorted_data)
        
        # Estimate noise level from bottom 10%
        noise_level = np.mean(sorted_data[:n//10])
        
        # Estimate signal level from top 10%
        signal_level = np.mean(sorted_data[9*n//10:])
        
        # Calculate SNR
        if noise_level > 0:
            snr = 20 * np.log10(signal_level / noise_level)
        else:
            snr = 100.0  # Very high SNR if noise is near zero
        
        return snr
    
    def _calculate_quality_metrics(self, 
                                 raw_signal: np.ndarray, 
                                 filtered_signal: np.ndarray, 
                                 thresholded: np.ndarray,
                                 bit_confidence: List[float]) -> Dict[str, Any]:
        """
        Calculate comprehensive signal quality metrics.
        
        Args:
            raw_signal: Raw input signal
            filtered_signal: Filtered signal
            thresholded: Thresholded signal
            bit_confidence: Confidence scores for extracted bits
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # SNR estimate
        metrics["snr_db"] = self._estimate_snr(raw_signal)
        
        # Noise floor estimate
        sorted_signal = np.sort(np.abs(raw_signal))
        metrics["noise_floor"] = float(np.mean(sorted_signal[:len(sorted_signal)//10]))
        
        # Signal stability metrics
        if len(filtered_signal) > 100:
            window_size = min(100, len(filtered_signal) // 10)
            windows = [filtered_signal[i:i+window_size] for i in range(0, len(filtered_signal)-window_size, window_size)]
            
            if windows:
                window_means = [np.mean(w) for w in windows]
                window_stds = [np.std(w) for w in windows]
                
                metrics["amplitude_stability"] = 1.0 - min(1.0, np.std(window_means) / np.mean(window_means))
                metrics["noise_stability"] = 1.0 - min(1.0, np.std(window_stds) / np.mean(window_stds))
        
        # Bit extraction quality
        if bit_confidence:
            metrics["mean_bit_confidence"] = float(np.mean(bit_confidence))
            metrics["min_bit_confidence"] = float(np.min(bit_confidence))
            
            # Assess timing stability
            if len(bit_confidence) > 10:
                confidence_variance = np.var(bit_confidence)
                metrics["timing_stability"] = 1.0 - min(1.0, 2.0 * confidence_variance)
        
        # Thresholding quality
        if len(thresholded) > 0:
            # Check for very short pulses (likely noise)
            transitions = np.where(np.diff(thresholded) != 0)[0]
            if len(transitions) > 1:
                intervals = np.diff(transitions)
                short_intervals = np.sum(intervals < 5)  # Pulses shorter than 5 samples
                metrics["short_pulse_rate"] = float(short_intervals / len(intervals))
                
                # Histogram-based symmetry assessment
                hist, _ = np.histogram(intervals, bins=10)
                if np.sum(hist) > 0:
                    metrics["interval_symmetry"] = 1.0 - min(1.0, np.std(hist) / np.mean(hist))
        
        # Overall quality score - weighted combination of individual metrics
        quality_score = 0.0
        weights = {
            "snr_db": 0.3,
            "amplitude_stability": 0.15,
            "noise_stability": 0.1,
            "mean_bit_confidence": 0.2,
            "timing_stability": 0.15,
            "interval_symmetry": 0.1
        }
        
        norm_metrics = {
            "snr_db": min(1.0, metrics.get("snr_db", 0) / 20.0),  # Normalize SNR to 0-1 (20dB is excellent)
        }
        
        # Combine available metrics
        total_weight = 0.0
        for key, weight in weights.items():
            if key in metrics or key in norm_metrics:
                value = norm_metrics.get(key, metrics.get(key, 0))
                quality_score += weight * value
                total_weight += weight
        
        # Normalize by available weights
        if total_weight > 0:
            quality_score /= total_weight
            
        metrics["overall_quality"] = min(1.0, max(0.0, quality_score))
        
        return metrics
    
    def _visualize_processing(self, result: Dict[str, Any]):
        """
        Visualize the signal processing steps.
        
        Args:
            result: Dictionary with processing results
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Plot original signal
            plt.subplot(411)
            plt.title("Original Signal")
            if "raw_signal" in result and result["raw_signal"] is not None:
                if np.iscomplexobj(result["raw_signal"]):
                    plt.plot(np.abs(result["raw_signal"]))
                else:
                    plt.plot(result["raw_signal"])
            
            # Plot filtered signal
            plt.subplot(412)
            plt.title("Filtered Signal")
            if "filtered_signal" in result and result["filtered_signal"] is not None:
                plt.plot(result["filtered_signal"])
                # Add threshold line if available
                if "threshold_value" in result:
                    plt.axhline(y=result["threshold_value"], color='r', linestyle='--')
            
            # Plot thresholded signal
            plt.subplot(413)
            plt.title("Thresholded Signal")
            if "thresholded_signal" in result and result["thresholded_signal"] is not None:
                plt.plot(result["thresholded_signal"])
            
            # Plot bit timings
            plt.subplot(414)
            plt.title("Detected Bits")
            
            if "bits" in result and result["bits"] is not None and "bit_timing" in result and result["bit_timing"] is not None:
                bits = result["bits"]
                bit_timing = result["bit_timing"]
                sample_rate = result.get("sample_rate", 1.0)
                
                # Convert bit timing to sample indices
                timing_indices = [int(t * sample_rate) for t in bit_timing]
                
                # Plot bits as step function
                if timing_indices and bits:
                    plt.step(timing_indices, bits, where='post')
                    
                    # Mark bit transitions
                    plt.plot(timing_indices, bits, 'ro', markersize=3)
                    
                    # Show bit values
                    for i, (t, bit) in enumerate(zip(timing_indices, bits)):
                        if i % 10 == 0:  # Only show every 10th bit to avoid clutter
                            plt.text(t, bit + 0.1, str(bit), fontsize=8)
            
            plt.tight_layout()
            
            # Save the figure
            os.makedirs('plots', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            plot_file = f"plots/signal_processing_{timestamp}.png"
            plt.savefig(plot_file)
            plt.close()
            
            self.logger.info(f"Processing visualization saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
    
    def load_signal_from_file(self, file_path: str, sample_rate: float = 2e6) -> Dict[str, Any]:
        """
        Load signal data from a file and preprocess it.
        
        Args:
            file_path: Path to the signal file
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary with processed signal and metadata
        """
        try:
            # Load the signal
            if file_path.endswith('.npy'):
                signal_data = np.load(file_path)
            else:
                with open(file_path, 'rb') as f:
                    signal_data = np.frombuffer(f.read(), dtype=np.complex64)
            
            self.logger.info(f"Loaded signal from {file_path}: {len(signal_data)} samples")
            
            # Preprocess
            result = self.preprocess_signal(signal_data, sample_rate)
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading signal from {file_path}: {e}")
            return {"error": str(e)}
    
    def save_processed_signal(self, result: Dict[str, Any], file_path: str) -> bool:
        """
        Save processed signal data to a file.
        
        Args:
            result: Dictionary with processed signal
            file_path: Path to save the result to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Create a serializable version of the result
            output = {
                "bits": result.get("bits"),
                "bit_timing": result.get("bit_timing"),
                "clock_period": result.get("clock_period"),
                "modulation": result.get("modulation"),
                "quality_metrics": result.get("quality_metrics"),
                "processing_steps": result.get("processing_steps"),
                "stats": result.get("stats"),
                "threshold_value": result.get("threshold_value"),
                "timestamp": datetime.now().isoformat()
            }
            
            # Save as JSON
            with open(file_path, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            
            self.logger.info(f"Saved processed signal to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving processed signal: {e}")
            return False


class EnhancedBitDetector:
    """
    Class for improved bit detection with adaptive thresholding and synchronization.
    """
    
    def __init__(self, signal_processor: SignalPreprocessor = None):
        """
        Initialize the enhanced bit detector.
        
        Args:
            signal_processor: Optional signal preprocessor instance
        """
        self.signal_processor = signal_processor or SignalPreprocessor()
        self.logger = logging.getLogger("EnhancedBitDetector")
    
    def detect_bits(self, signal_data: np.ndarray, sample_rate: float) -> Dict[str, Any]:
        """
        Detect bits in the signal using enhanced algorithms.
        
        Args:
            signal_data: Signal data
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary with detection results
        """
        # Preprocess the signal
        result = self.signal_processor.preprocess_signal(signal_data, sample_rate)
        
        # Check if preprocessing was successful
        if "error" in result or result["bits"] is None:
            self.logger.error("Bit detection failed during preprocessing")
            return {"error": "Preprocessing failed", "bits": []}
        
        # Improve bit detection with pattern matching if enabled
        if self.signal_processor.params["use_correlation"]:
            self._improve_with_pattern_matching(result)
        
        # Log results
        bits = result.get("bits", [])
        self.logger.info(f"Detected {len(bits)} bits with enhanced algorithm")
        
        # Calculate some additional metrics on the detected bits
        if bits:
            # Transitions rate (changes from 0->1 or 1->0)
            transitions = np.sum(np.abs(np.diff(bits)))
            transition_rate = transitions / (len(bits) - 1) if len(bits) > 1 else 0
            
            # Add to result
            result["bit_stats"] = {
                "transitions": int(transitions),
                "transition_rate": float(transition_rate),
                "ones_percentage": float(np.mean(bits))
            }
        
        return result
    
    def _improve_with_pattern_matching(self, result: Dict[str, Any]):
        """
        Improve bit detection using pattern matching and correlation.
        
        Args:
            result: Dictionary with detection results to improve
        """
        bits = result.get("bits", [])
        if len(bits) < 8:
            return  # Not enough bits for pattern matching
        
        # Convert bits to string for pattern analysis
        bit_string = ''.join(map(str, bits))
        
        # Look for common patterns in garage door protocols
        patterns = self._find_common_patterns(bit_string)
        
        # Check if we found any significant patterns
        if patterns and patterns[0]["score"] > 0.7:
            pattern = patterns[0]
            
            # Log the found pattern
            self.logger.info(f"Found pattern: {pattern['pattern']} (score: {pattern['score']:.2f}, " +
                            f"type: {pattern['type']}, repeats: {pattern['repeats']})")
            
            # Add pattern information to result
            result["pattern_analysis"] = {
                "best_pattern": pattern,
                "all_patterns": patterns
            }
            
            # For highly confident patterns, we can improve bit decoding
            if pattern["score"] > 0.9 and pattern["type"] == "preamble":
                # Preamble pattern typically indicates the start of a rolling code
                # Use it to improve bit alignment
                start_idx = bit_string.find(pattern["pattern"])
                if start_idx > 0:
                    # Realign bits to start at the preamble
                    result["bits"] = bits[start_idx:]
                    
                    # Also adjust timing if available
                    if "bit_timing" in result and result["bit_timing"]:
                        result["bit_timing"] = result["bit_timing"][start_idx:]
                        
                    # And confidence scores
                    if "bit_confidence" in result and result["bit_confidence"]:
                        result["bit_confidence"] = result["bit_confidence"][start_idx:]
                    
                    self.logger.info(f"Realigned bits to start at preamble (removed {start_idx} bits)")
    
    def _find_common_patterns(self, bit_string: str) -> List[Dict[str, Any]]:
        """
        Find common patterns in the bit string.
        
        Args:
            bit_string: String of 0s and 1s
            
        Returns:
            List of pattern information dictionaries, sorted by score
        """
        patterns = []
        
        # Check for common preamble patterns in garage door signals
        preambles = {
            "10101010": "standard_preamble",
            "1010101010101010": "long_preamble",
            "111000111000": "chamberlain_preamble",
            "110011001100": "genie_preamble",
            "101010110011": "liftmaster_preamble"
        }
        
        for pattern, name in preambles.items():
            if pattern in bit_string:
                score = min(1.0, 0.7 + 0.02 * len(pattern))  # Longer patterns get higher scores
                patterns.append({
                    "pattern": pattern,
                    "type": "preamble",
                    "name": name,
                    "score": score,
                    "position": bit_string.find(pattern),
                    "repeats": 1
                })
        
        # Check for repeating patterns (may indicate fixed code)
        for length in range(8, 25, 8):
            if len(bit_string) >= length * 2:
                for i in range(len(bit_string) - length * 2 + 1):
                    segment = bit_string[i:i+length]
                    
                    # Count repeats
                    repeats = 0
                    pos = i
                    while pos + length <= len(bit_string) and bit_string[pos:pos+length] == segment:
                        repeats += 1
                        pos += length
                    
                    if repeats >= 2:
                        # Calculate score based on length and repeats
                        score = min(1.0, 0.5 + 0.1 * repeats + 0.01 * length)
                        
                        patterns.append({
                            "pattern": segment,
                            "type": "repeating",
                            "score": score,
                            "position": i,
                            "repeats": repeats
                        })
        
        # Sort by score (highest first)
        return sorted(patterns, key=lambda x: x["score"], reverse=True)
    
    def detect_rolling_code(self, signal_data: np.ndarray, sample_rate: float) -> Dict[str, Any]:
        """
        Specialized detection for rolling code signals.
        
        Args:
            signal_data: Signal data
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary with rolling code detection results
        """
        # First perform normal bit detection
        result = self.detect_bits(signal_data, sample_rate)
        
        if "error" in result or not result.get("bits"):
            return result
        
        # Analyze for rolling code structure
        bits = result["bits"]
        bit_string = ''.join(map(str, bits))
        
        # Most rolling codes follow specific structures:
        # 1. Preamble / sync bits
        # 2. Fixed/ID bits
        # 3. Rolling counter
        # 4. Checksum/CRC
        
        rolling_code_info = {
            "is_rolling_code": False,
            "format": "unknown",
            "preamble": None,
            "fixed_bits": None,
            "rolling_bits": None,
            "checksum_bits": None
        }
        
        # Look for preamble patterns
        preamble_match = None
        preambles = {
            "10101010": "standard",
            "1010101010101010": "long",
            "111000111000": "chamberlain",
            "110011001100": "genie",
            "101010110011": "liftmaster"
        }
        
        for pattern, name in preambles.items():
            if bit_string.startswith(pattern):
                preamble_match = {"pattern": pattern, "name": name}
                break
            elif pattern in bit_string[:16]:
                pos = bit_string.find(pattern)
                preamble_match = {"pattern": pattern, "name": name, "offset": pos}
                break
        
        if preamble_match:
            rolling_code_info["preamble"] = preamble_match
            
            # Determine possible format based on preamble
            if "chamberlain" in preamble_match["name"] or "liftmaster" in preamble_match["name"]:
                rolling_code_info["format"] = "security+"
                rolling_code_info["is_rolling_code"] = True
            elif "genie" in preamble_match["name"]:
                rolling_code_info["format"] = "genie"
                rolling_code_info["is_rolling_code"] = True
        
        # Check code length - most rolling codes are 40-68 bits total
        if len(bits) >= 32 and len(bits) <= 80:
            rolling_code_info["code_length"] = len(bits)
            
            # Heuristic analysis for likely rolling code segments
            # Many rolling codes use the last 8-32 bits as rolling counter
            if len(bits) >= 40:
                # Assumed structure for analysis:
                # [preamble/sync][fixed ID bits][rolling counter][checksum]
                
                # Typical format lengths
                # KeeLoq: 66 bits (10 sync + 28 ID + 16 button/counter + 12 discrimination)
                # Chamberlain/LiftMaster Security+: 40 bits (32 encrypted + 8 fixed)
                # Linear/MultiCode: 40 bits (10 preface + 30 data)
                
                # Look for known formats
                if len(bits) == 66 or len(bits) == 64:
                    rolling_code_info["format"] = "keeloq"
                    rolling_code_info["is_rolling_code"] = True
                    
                    if preamble_match:
                        preamble_len = len(preamble_match["pattern"])
                        rolling_code_info["fixed_bits"] = bit_string[preamble_len:preamble_len+28]
                        rolling_code_info["rolling_bits"] = bit_string[preamble_len+28:preamble_len+28+16]
                        rolling_code_info["checksum_bits"] = bit_string[preamble_len+28+16:]
                
                elif len(bits) == 40:
                    # Could be Chamberlain/LiftMaster or Linear/MultiCode
                    if rolling_code_info["format"] == "security+":
                        rolling_code_info["fixed_bits"] = bit_string[-8:]
                        rolling_code_info["rolling_bits"] = bit_string[:-8]
                    else:
                        # Assume Linear/MultiCode
                        rolling_code_info["format"] = "multicode"
                        rolling_code_info["is_rolling_code"] = True
                        rolling_code_info["fixed_bits"] = bit_string[:10]
                        rolling_code_info["rolling_bits"] = bit_string[10:32]
                        rolling_code_info["checksum_bits"] = bit_string[32:]
        
        # Add rolling code information to result
        result["rolling_code_info"] = rolling_code_info
        
        # Log results
        if rolling_code_info["is_rolling_code"]:
            self.logger.info(f"Detected rolling code format: {rolling_code_info['format']}")
        else:
            self.logger.info("No rolling code pattern confidently detected")
        
        return result


def main():
    """Main entry point for demonstrating the module functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Signal Processor for Garage Door Signals')
    parser.add_argument('-f', '--file', type=str, help='Signal file to process')
    parser.add_argument('-s', '--sample-rate', type=float, default=2.0, help='Sample rate in MHz')
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize signal processing')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-p', '--plot', action='store_true', help='Save plot of results')
    parser.add_argument('-o', '--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create processor
    processor = SignalPreprocessor(debug_mode=args.debug, visualize=args.visualize or args.plot)
    detector = EnhancedBitDetector(processor)
    
    if args.file:
        # Process file
        sample_rate = args.sample_rate * 1e6  # Convert MHz to Hz
        print(f"Processing {args.file} with sample rate {sample_rate/1e6:.2f} MHz")
        
        result = detector.detect_rolling_code(None, sample_rate)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        # Display results
        print("\nResults:")
        print(f"Bits detected: {len(result.get('bits', []))}")
        
        bits = result.get('bits', [])
        if bits:
            # Show bit segments with colored text if supported
            try:
                from colorama import init, Fore, Style
                init()
                
                # Display with colors
                rolling_info = result.get('rolling_code_info', {})
                
                if rolling_info.get('is_rolling_code'):
                    print(f"Detected rolling code format: {rolling_info.get('format', 'unknown')}")
                    
                    # Show bit sections
                    bit_string = ''.join(map(str, bits))
                    
                    # Colorize sections if identified
                    if rolling_info.get('preamble'):
                        preamble = rolling_info['preamble']['pattern']
                        pos = bit_string.find(preamble)
                        if pos >= 0:
                            colored_bits = (bit_string[:pos] + 
                                          Fore.YELLOW + preamble + Style.RESET_ALL + 
                                          bit_string[pos + len(preamble):])
                            print(f"Bits with preamble highlighted: {colored_bits}")
                    
                    # Show rolling code sections
                    if rolling_info.get('fixed_bits') and rolling_info.get('rolling_bits'):
                        print(f"Fixed bits: {Fore.GREEN}{rolling_info['fixed_bits']}{Style.RESET_ALL}")
                        print(f"Rolling bits: {Fore.RED}{rolling_info['rolling_bits']}{Style.RESET_ALL}")
                        if rolling_info.get('checksum_bits'):
                            print(f"Checksum: {Fore.BLUE}{rolling_info['checksum_bits']}{Style.RESET_ALL}")
                else:
                    # Show plain bits
                    bit_groups = [bits[i:i+8] for i in range(0, len(bits), 8)]
                    bit_strings = [''.join(map(str, group)) for group in bit_groups]
                    print(f"Bits: {' '.join(bit_strings)}")
            except ImportError:
                # Fallback to plain text
                print(f"Bits: {''.join(map(str, bits))}")
        
        # Save results if requested
        if args.output:
            processor.save_processed_signal(result, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()