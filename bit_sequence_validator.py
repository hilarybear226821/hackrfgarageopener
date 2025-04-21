#!/usr/bin/env python3
"""
Bit Sequence Validator Module

Provides advanced functionality for validating, simplifying, and analyzing bit sequences
extracted from garage door remote signals. Implements sophisticated techniques for:

1. Debouncing and noise filtering of bit sequences
2. Advanced pattern validation through statistical analysis
3. Transition density evaluation for rolling code validation
4. Bit balance analysis (distribution of 1s and 0s)
5. Detection of manufacturer-specific patterns and headers
6. Sequence segmentation and component identification

These techniques significantly improve the accuracy of rolling code identification
by filtering out invalid sequences and confirming likely candidates.

For educational and security research purposes only.
"""

import os
import sys
import math
import json
import logging
import itertools
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from collections import defaultdict, Counter

# Import from other modules if available
try:
    from enhanced_signal_processor import SignalPreprocessor
    SIGNAL_PROCESSOR_AVAILABLE = True
except ImportError:
    SIGNAL_PROCESSOR_AVAILABLE = False


class BitSequenceValidator:
    """
    Class for validating and analyzing bit sequences to identify valid rolling codes.
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the bit sequence validator.
        
        Args:
            debug_mode: Enable debug mode for additional output
        """
        self.debug_mode = debug_mode
        self.validated_sequences = []
        self.known_patterns = self._load_known_patterns()
        
        # Default validation parameters
        self.params = {
            # Debouncing parameters
            "debounce_window": 3,          # Window size for bit debouncing
            "debounce_threshold": 0.5,      # Majority threshold for debouncing
            
            # Balance parameters (1s vs 0s)
            "min_balance_ratio": 0.15,      # Minimum acceptable ratio (e.g., 15% ones)
            "max_balance_ratio": 0.85,      # Maximum acceptable ratio (e.g., 85% ones)
            "check_balance": True,          # Enable balance checking
            
            # Transition parameters (bit flips)
            "min_transition_density": 0.1,  # Minimum transitions per bit
            "max_transition_density": 0.8,  # Maximum transitions per bit
            "check_transitions": True,      # Enable transition density checking
            
            # Length validation
            "valid_lengths": [12, 16, 18, 24, 32, 40, 66, 67, 68], # Common code lengths
            "check_length": True,           # Enable length checking
            
            # Pattern validation
            "check_patterns": True,         # Enable pattern checking
            "min_pattern_score": 0.6,       # Minimum score for pattern match
            "header_weight": 2.0,           # Weight for header patterns in scoring
            
            # Entropy validation
            "min_entropy": 0.5,             # Minimum acceptable entropy (0-1)
            "max_entropy": 0.95,            # Maximum acceptable entropy (0-1)
            "check_entropy": True,          # Enable entropy checking
            
            # Miscellaneous
            "min_sequence_length": 8,       # Minimum sequence length to consider
            "max_sequence_length": 128,     # Maximum sequence length to consider
            "confidence_threshold": 0.65    # Minimum confidence score to accept
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the bit sequence validator."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure file handler with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = f"logs/bit_validator_{timestamp}.log"
        
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG if self.debug_mode else logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("BitSequenceValidator")
    
    def _load_known_patterns(self) -> Dict[str, Any]:
        """
        Load known manufacturer-specific bit patterns and headers.
        
        Returns:
            Dictionary of patterns by manufacturer
        """
        return {
            "general": {
                "preambles": [
                    {"pattern": "10101010", "name": "standard_sync", "score": 0.7},
                    {"pattern": "1010101010101010", "name": "extended_sync", "score": 0.8},
                    {"pattern": "10101010101010101010", "name": "long_sync", "score": 0.85}
                ],
                "postambles": [
                    {"pattern": "10101010", "name": "standard_postamble", "score": 0.6},
                    {"pattern": "00000000", "name": "zero_padding", "score": 0.5}
                ]
            },
            "chamberlain": {
                "preambles": [
                    {"pattern": "111000111000", "name": "security+_preamble", "score": 0.9},
                    {"pattern": "111000111000111000", "name": "security+2.0_preamble", "score": 0.95}
                ],
                "formats": [
                    {"length": 40, "name": "security+", "score": 0.85},
                    {"length": 68, "name": "security+2.0", "score": 0.9}
                ]
            },
            "genie": {
                "preambles": [
                    {"pattern": "110011001100", "name": "genie_preamble", "score": 0.85},
                    {"pattern": "110011001100110011", "name": "genie_long_preamble", "score": 0.9}
                ],
                "formats": [
                    {"length": 24, "name": "genie_old", "score": 0.7},
                    {"length": 40, "name": "genie_intellicode", "score": 0.85}
                ]
            },
            "linear": {
                "preambles": [
                    {"pattern": "101010110011", "name": "linear_preamble", "score": 0.85}
                ],
                "formats": [
                    {"length": 24, "name": "linear_standard", "score": 0.8},
                    {"length": 40, "name": "linear_extended", "score": 0.85}
                ]
            },
            "keeloq": {
                "formats": [
                    {"length": 66, "name": "keeloq_standard", "score": 0.9},
                    {"length": 67, "name": "keeloq_extended", "score": 0.9}
                ],
                "structure": {
                    "header_bits": 10,
                    "id_bits": 28,
                    "counter_bits": 16,
                    "discrimination_bits": 12
                }
            }
        }
    
    def set_param(self, param_name: str, value: Any):
        """
        Set a validation parameter.
        
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
        Get a validation parameter value.
        
        Args:
            param_name: Name of the parameter to get
            
        Returns:
            Parameter value, or None if parameter doesn't exist
        """
        return self.params.get(param_name)
    
    def validate_bit_sequence(self, bit_sequence: Union[List[int], str], 
                            source: str = "unknown",
                            confidence_scores: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Validate a bit sequence and determine if it's likely a valid rolling code.
        
        Args:
            bit_sequence: Sequence of bits as list of integers or string
            source: Source of the bit sequence (for tracking)
            confidence_scores: Optional list of confidence scores for each bit
            
        Returns:
            Dictionary with validation results
        """
        # Convert string to list if needed
        if isinstance(bit_sequence, str):
            bits = [int(b) for b in bit_sequence if b in '01']
        else:
            bits = list(bit_sequence)
        
        # Initialize result structure
        result = {
            "original_sequence": ''.join(map(str, bits)),
            "length": len(bits),
            "source": source,
            "is_valid": False,
            "confidence": 0.0,
            "preprocessed_sequence": None,
            "validation_steps": [],
            "manufacturer": None,
            "format": None,
            "components": {},
            "issues": [],
            "stats": {}
        }
        
        # Early validation - check if sequence is long enough
        if len(bits) < self.params["min_sequence_length"]:
            result["issues"].append(f"Sequence too short: {len(bits)} bits, minimum: {self.params['min_sequence_length']}")
            self.logger.debug(f"Rejected sequence: too short ({len(bits)} bits)")
            return result
        
        if len(bits) > self.params["max_sequence_length"]:
            result["issues"].append(f"Sequence too long: {len(bits)} bits, maximum: {self.params['max_sequence_length']}")
            self.logger.debug(f"Rejected sequence: too long ({len(bits)} bits)")
            return result
        
        # Calculate basic statistics
        stats = self._calculate_sequence_stats(bits)
        result["stats"] = stats
        
        # Preprocess the bit sequence (debouncing)
        processed_bits = self._debounce_bits(bits, confidence_scores)
        result["preprocessed_sequence"] = ''.join(map(str, processed_bits))
        result["validation_steps"].append("debouncing")
        
        # If debouncing significantly changed the sequence, update stats
        if processed_bits != bits:
            debounced_stats = self._calculate_sequence_stats(processed_bits)
            result["stats"]["debounced"] = debounced_stats
            
            # Use the processed bits for further validation
            bits = processed_bits
            
            self.logger.debug(f"Debouncing changed {result['length'] - len(bits)} bits")
        
        # Check sequence length
        length_valid, length_score, length_info = self._validate_length(bits)
        if self.params["check_length"] and not length_valid:
            result["issues"].append(f"Invalid length: {len(bits)} bits. {length_info}")
        else:
            result["validation_steps"].append("length_check")
        
        # Check bit balance (distribution of 1s and 0s)
        balance_valid, balance_score, balance_info = self._validate_balance(bits)
        if self.params["check_balance"] and not balance_valid:
            result["issues"].append(f"Poor bit balance: {balance_info}")
        else:
            result["validation_steps"].append("balance_check")
        
        # Check transition density
        transitions_valid, transitions_score, transitions_info = self._validate_transitions(bits)
        if self.params["check_transitions"] and not transitions_valid:
            result["issues"].append(f"Invalid transition density: {transitions_info}")
        else:
            result["validation_steps"].append("transition_check")
        
        # Check entropy
        entropy_valid, entropy_score, entropy_info = self._validate_entropy(bits)
        if self.params["check_entropy"] and not entropy_valid:
            result["issues"].append(f"Invalid entropy: {entropy_info}")
        else:
            result["validation_steps"].append("entropy_check")
        
        # Check for known patterns
        pattern_valid, pattern_score, pattern_info = self._validate_patterns(bits)
        if self.params["check_patterns"] and not pattern_valid:
            result["issues"].append(f"No recognized patterns: {pattern_info}")
        else:
            result["validation_steps"].append("pattern_check")
            
            # Extract manufacturer and format if available
            if "manufacturer" in pattern_info:
                result["manufacturer"] = pattern_info["manufacturer"]
            if "format" in pattern_info:
                result["format"] = pattern_info["format"]
        
        # Calculate overall confidence score
        confidence = self._calculate_confidence(
            length_score,
            balance_score,
            transitions_score,
            entropy_score,
            pattern_score
        )
        result["confidence"] = confidence
        
        # Determine if sequence is valid based on confidence
        result["is_valid"] = confidence >= self.params["confidence_threshold"]
        
        # Identify sequence components if valid or close to valid
        if result["is_valid"] or confidence >= self.params["confidence_threshold"] * 0.8:
            components = self._identify_sequence_components(bits, result["manufacturer"])
            result["components"] = components
        
        # Store validated sequence if valid
        if result["is_valid"]:
            self.validated_sequences.append(result)
            self.logger.info(f"Valid sequence detected: {len(bits)} bits, confidence: {confidence:.2f}, " +
                           f"manufacturer: {result['manufacturer']}")
        else:
            self.logger.debug(f"Invalid sequence: {len(bits)} bits, confidence: {confidence:.2f}, " +
                             f"issues: {len(result['issues'])}")
        
        return result
    
    def _debounce_bits(self, bits: List[int], 
                     confidence_scores: Optional[List[float]] = None) -> List[int]:
        """
        Apply debouncing to remove noise-induced bit flips.
        
        Args:
            bits: Original bit sequence
            confidence_scores: Optional bit confidence scores
            
        Returns:
            Debounced bit sequence
        """
        if len(bits) <= 1:
            return bits.copy()
        
        window_size = self.params["debounce_window"]
        threshold = self.params["debounce_threshold"]
        
        # If window size is even, add 1 to make it odd
        if window_size % 2 == 0:
            window_size += 1
        
        # If window is too large for the sequence, reduce it
        window_size = min(window_size, len(bits) - 1)
        
        # If window is too small, use minimal window
        window_size = max(window_size, 3)
        
        # Apply debouncing using a sliding window with majority vote
        debounced = bits.copy()
        half_window = window_size // 2
        
        for i in range(len(bits)):
            # Calculate window bounds
            start = max(0, i - half_window)
            end = min(len(bits), i + half_window + 1)
            
            # Get window bits
            window_bits = bits[start:end]
            
            # If confidence scores are provided, use them for weighted voting
            if confidence_scores is not None and len(confidence_scores) == len(bits):
                window_confidence = confidence_scores[start:end]
                
                # Weighted vote
                vote_0 = sum((1 - b) * c for b, c in zip(window_bits, window_confidence))
                vote_1 = sum(b * c for b, c in zip(window_bits, window_confidence))
                
                # Select bit with highest weighted vote
                debounced[i] = 1 if vote_1 > vote_0 else 0
            else:
                # Simple majority vote
                count_1 = sum(window_bits)
                count_0 = len(window_bits) - count_1
                
                # Determine majority (with tie-breaking using threshold)
                if count_1 / len(window_bits) > threshold:
                    debounced[i] = 1
                elif count_0 / len(window_bits) > threshold:
                    debounced[i] = 0
                # else keep original bit
        
        return debounced
    
    def _calculate_sequence_stats(self, bits: List[int]) -> Dict[str, Any]:
        """
        Calculate statistics for a bit sequence.
        
        Args:
            bits: Bit sequence
            
        Returns:
            Dictionary with sequence statistics
        """
        if not bits:
            return {}
        
        # Count ones and zeros
        count_1 = sum(bits)
        count_0 = len(bits) - count_1
        
        # Calculate transitions (bit flips)
        transitions = sum(1 for i in range(1, len(bits)) if bits[i] != bits[i-1])
        
        # Calculate entropy
        p_1 = count_1 / len(bits)
        p_0 = count_0 / len(bits)
        entropy = 0.0
        if p_1 > 0:
            entropy -= p_1 * math.log2(p_1)
        if p_0 > 0:
            entropy -= p_0 * math.log2(p_0)
        # Normalize entropy to 0-1 range (max entropy for binary is 1.0)
        entropy = min(1.0, entropy)
        
        # Calculate run lengths (sequences of consecutive identical bits)
        runs = []
        current_run = 1
        for i in range(1, len(bits)):
            if bits[i] == bits[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)  # Add the last run
        
        # Calculate run statistics
        max_run = max(runs) if runs else 0
        min_run = min(runs) if runs else 0
        avg_run = sum(runs) / len(runs) if runs else 0
        
        return {
            "length": len(bits),
            "count_1": count_1,
            "count_0": count_0,
            "balance_ratio": count_1 / len(bits),
            "transitions": transitions,
            "transition_density": transitions / (len(bits) - 1) if len(bits) > 1 else 0,
            "entropy": entropy,
            "max_run": max_run,
            "min_run": min_run,
            "avg_run": avg_run,
            "run_count": len(runs)
        }
    
    def _validate_length(self, bits: List[int]) -> Tuple[bool, float, str]:
        """
        Validate the length of a bit sequence.
        
        Args:
            bits: Bit sequence
            
        Returns:
            Tuple of (is_valid, score, info_message)
        """
        length = len(bits)
        
        # Check if length matches any of the common rolling code lengths
        valid_lengths = self.params["valid_lengths"]
        
        if length in valid_lengths:
            # Exact match
            score = 1.0
            return True, score, f"Length {length} bits matches common rolling code length"
        
        # Find closest valid length
        closest = min(valid_lengths, key=lambda x: abs(x - length))
        difference = abs(closest - length)
        
        # Calculate score based on how close it is to a valid length
        max_diff = max(10, closest * 0.2)  # Allow up to 20% difference or 10 bits
        score = max(0, 1.0 - (difference / max_diff))
        
        # Check if it's close enough to be considered valid
        if score >= 0.8:
            return True, score, f"Length {length} bits is close to valid length {closest}"
        else:
            # For very long sequences, check if they might contain multiple codes
            if length > 80:
                for valid_len in valid_lengths:
                    if length % valid_len < valid_len * 0.1:  # Less than 10% remainder
                        return True, 0.7, f"Length {length} bits might contain multiple {valid_len}-bit codes"
            
            return False, score, f"Length {length} bits doesn't match any common rolling code length, closest: {closest}"
    
    def _validate_balance(self, bits: List[int]) -> Tuple[bool, float, str]:
        """
        Validate the balance of 1s and 0s in a bit sequence.
        
        Args:
            bits: Bit sequence
            
        Returns:
            Tuple of (is_valid, score, info_message)
        """
        if not bits:
            return False, 0.0, "Empty sequence"
        
        # Calculate balance ratio (proportion of 1s)
        count_1 = sum(bits)
        ratio = count_1 / len(bits)
        
        min_ratio = self.params["min_balance_ratio"]
        max_ratio = self.params["max_balance_ratio"]
        
        # Perfect balance is 0.5
        balance_quality = 1.0 - abs(ratio - 0.5) * 2  # Scale to 0-1
        
        # Check if within acceptable range
        if min_ratio <= ratio <= max_ratio:
            return True, balance_quality, f"Good balance, {ratio:.2f} proportion of 1s"
        else:
            # Extreme imbalance
            if ratio < min_ratio:
                return False, balance_quality, f"Too few 1s: {ratio:.2f}, minimum: {min_ratio}"
            else:
                return False, balance_quality, f"Too many 1s: {ratio:.2f}, maximum: {max_ratio}"
    
    def _validate_transitions(self, bits: List[int]) -> Tuple[bool, float, str]:
        """
        Validate the transition density of a bit sequence.
        
        Args:
            bits: Bit sequence
            
        Returns:
            Tuple of (is_valid, score, info_message)
        """
        if len(bits) <= 1:
            return False, 0.0, "Sequence too short for transition analysis"
        
        # Count transitions (bit flips)
        transitions = sum(1 for i in range(1, len(bits)) if bits[i] != bits[i-1])
        
        # Calculate transition density (transitions per bit)
        density = transitions / (len(bits) - 1)
        
        min_density = self.params["min_transition_density"]
        max_density = self.params["max_transition_density"]
        
        # Ideal density is around 0.5 for a random sequence
        density_quality = 1.0 - abs(density - 0.5) * 2  # Scale to 0-1
        
        # Check if within acceptable range
        if min_density <= density <= max_density:
            return True, density_quality, f"Good transition density: {density:.2f}"
        else:
            # Extreme density
            if density < min_density:
                return False, density_quality, f"Too few transitions: {density:.2f}, minimum: {min_density}"
            else:
                return False, density_quality, f"Too many transitions: {density:.2f}, maximum: {max_density}"
    
    def _validate_entropy(self, bits: List[int]) -> Tuple[bool, float, str]:
        """
        Validate the entropy of a bit sequence.
        
        Args:
            bits: Bit sequence
            
        Returns:
            Tuple of (is_valid, score, info_message)
        """
        if len(bits) <= 1:
            return False, 0.0, "Sequence too short for entropy analysis"
        
        # Count ones and zeros
        count_1 = sum(bits)
        count_0 = len(bits) - count_1
        
        # Calculate entropy
        p_1 = count_1 / len(bits)
        p_0 = count_0 / len(bits)
        entropy = 0.0
        if p_1 > 0:
            entropy -= p_1 * math.log2(p_1)
        if p_0 > 0:
            entropy -= p_0 * math.log2(p_0)
        
        # Normalize entropy to 0-1 range (max entropy for binary is 1.0)
        entropy = min(1.0, entropy)
        
        min_entropy = self.params["min_entropy"]
        max_entropy = self.params["max_entropy"]
        
        # Best entropy is high but not maximum (which might indicate random noise)
        entropy_quality = entropy if entropy <= 0.9 else (1.0 - (entropy - 0.9) * 10)
        
        # Check if within acceptable range
        if min_entropy <= entropy <= max_entropy:
            return True, entropy_quality, f"Good entropy: {entropy:.2f}"
        else:
            # Extreme entropy
            if entropy < min_entropy:
                return False, entropy_quality, f"Too low entropy: {entropy:.2f}, minimum: {min_entropy}"
            else:
                return False, entropy_quality, f"Too high entropy: {entropy:.2f}, maximum: {max_entropy}"
    
    def _validate_patterns(self, bits: List[int]) -> Tuple[bool, float, str]:
        """
        Validate bit patterns against known manufacturer patterns.
        
        Args:
            bits: Bit sequence
            
        Returns:
            Tuple of (is_valid, score, info_message)
        """
        bit_string = ''.join(map(str, bits))
        length = len(bits)
        
        best_pattern = {
            "manufacturer": None,
            "format": None,
            "pattern_type": None,
            "score": 0.0,
            "position": -1
        }
        
        # Check for known preamble patterns
        for manufacturer, patterns in self.known_patterns.items():
            # Skip the general patterns for now
            if manufacturer == "general":
                continue
                
            # Check format lengths
            if "formats" in patterns:
                for format_info in patterns["formats"]:
                    if length == format_info["length"]:
                        current_score = format_info["score"]
                        if current_score > best_pattern["score"]:
                            best_pattern = {
                                "manufacturer": manufacturer,
                                "format": format_info["name"],
                                "pattern_type": "length_match",
                                "score": current_score,
                                "position": -1
                            }
            
            # Check preamble patterns
            if "preambles" in patterns:
                for preamble in patterns["preambles"]:
                    preamble_str = preamble["pattern"]
                    if preamble_str in bit_string:
                        # Check if it's at the beginning (higher score) or elsewhere
                        pos = bit_string.find(preamble_str)
                        
                        # Calculate pattern score, weighted by position
                        position_factor = 1.0 if pos == 0 else (0.8 if pos < 8 else 0.6)
                        current_score = preamble["score"] * position_factor
                        
                        if current_score > best_pattern["score"]:
                            best_pattern = {
                                "manufacturer": manufacturer,
                                "format": None,  # Will be set based on length if possible
                                "pattern_type": "preamble",
                                "pattern_name": preamble["name"],
                                "score": current_score,
                                "position": pos
                            }
                            
                            # Try to determine format from length if we matched a preamble
                            if "formats" in patterns:
                                for format_info in patterns["formats"]:
                                    if length == format_info["length"]:
                                        best_pattern["format"] = format_info["name"]
                                        # Boost score for matching both preamble and length
                                        best_pattern["score"] = min(0.99, best_pattern["score"] + 0.1)
        
        # If no manufacturer-specific pattern was found, check general patterns
        if best_pattern["score"] == 0 and "general" in self.known_patterns:
            general_patterns = self.known_patterns["general"]
            
            # Check general preambles
            if "preambles" in general_patterns:
                for preamble in general_patterns["preambles"]:
                    preamble_str = preamble["pattern"]
                    if preamble_str in bit_string:
                        pos = bit_string.find(preamble_str)
                        position_factor = 1.0 if pos == 0 else (0.7 if pos < 8 else 0.5)
                        current_score = preamble["score"] * position_factor
                        
                        if current_score > best_pattern["score"]:
                            best_pattern = {
                                "manufacturer": "unknown",
                                "format": "general",
                                "pattern_type": "general_preamble",
                                "pattern_name": preamble["name"],
                                "score": current_score,
                                "position": pos
                            }
        
        # Determine if a valid pattern was found
        min_score = self.params["min_pattern_score"]
        pattern_valid = best_pattern["score"] >= min_score
        
        # Prepare info message
        if pattern_valid:
            info = {
                "manufacturer": best_pattern["manufacturer"],
                "format": best_pattern["format"],
                "score": best_pattern["score"]
            }
            info_msg = f"Matched {best_pattern['pattern_type']} pattern for {best_pattern['manufacturer']}"
            if best_pattern["format"]:
                info_msg += f", format: {best_pattern['format']}"
        else:
            info = {}
            info_msg = "No known patterns matched with sufficient confidence"
        
        return pattern_valid, best_pattern["score"], info
    
    def _calculate_confidence(self, length_score: float, balance_score: float, 
                           transitions_score: float, entropy_score: float, 
                           pattern_score: float) -> float:
        """
        Calculate overall confidence score from individual validation scores.
        
        Args:
            length_score: Score from length validation
            balance_score: Score from balance validation
            transitions_score: Score from transition density validation
            entropy_score: Score from entropy validation
            pattern_score: Score from pattern validation
            
        Returns:
            Overall confidence score (0-1)
        """
        # Weights for different aspects of validation
        weights = {
            "length": 0.15,
            "balance": 0.15,
            "transitions": 0.20,
            "entropy": 0.20,
            "pattern": 0.30
        }
        
        # Weighted sum
        confidence = (
            weights["length"] * length_score +
            weights["balance"] * balance_score +
            weights["transitions"] * transitions_score +
            weights["entropy"] * entropy_score +
            weights["pattern"] * pattern_score
        )
        
        # Ensure confidence is in 0-1 range
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _identify_sequence_components(self, bits: List[int], manufacturer: Optional[str]) -> Dict[str, Any]:
        """
        Identify and segment components of a bit sequence.
        
        Args:
            bits: Bit sequence
            manufacturer: Identified manufacturer (if known)
            
        Returns:
            Dictionary with identified components
        """
        bit_string = ''.join(map(str, bits))
        length = len(bits)
        components = {}
        
        # If manufacturer is known, use specific structure information
        if manufacturer and manufacturer in self.known_patterns:
            pattern_info = self.known_patterns[manufacturer]
            
            # If this manufacturer has defined structure
            if "structure" in pattern_info:
                structure = pattern_info["structure"]
                
                # Extract components based on structure
                current_pos = 0
                for component, size in structure.items():
                    if current_pos + size <= length:
                        components[component] = bit_string[current_pos:current_pos+size]
                        current_pos += size
            else:
                # Try to identify components based on common patterns
                # Look for preamble
                if "preambles" in pattern_info:
                    for preamble in pattern_info["preambles"]:
                        preamble_str = preamble["pattern"]
                        if bit_string.startswith(preamble_str):
                            components["preamble"] = preamble_str
                            
                            # Remainder is likely the payload
                            components["payload"] = bit_string[len(preamble_str):]
                            break
                        elif preamble_str in bit_string:
                            pos = bit_string.find(preamble_str)
                            components["pre_preamble"] = bit_string[:pos]
                            components["preamble"] = preamble_str
                            components["payload"] = bit_string[pos+len(preamble_str):]
                            break
        
        # If no components identified yet, use generic structure analysis
        if not components:
            # Look for standard sync/preamble patterns
            preamble_patterns = ["10101010", "1010101010101010", "111000", "110011"]
            for pattern in preamble_patterns:
                if bit_string.startswith(pattern):
                    components["preamble"] = pattern
                    remainder = bit_string[len(pattern):]
                    
                    # For longer sequences, try to identify sections
                    if len(remainder) > 16:
                        # Estimate fixed ID portion and rolling counter
                        id_size = min(len(remainder) // 2, 24)
                        components["fixed_id"] = remainder[:id_size]
                        components["rolling_counter"] = remainder[id_size:]
                    else:
                        components["payload"] = remainder
                    
                    break
                elif pattern in bit_string[:16]:
                    pos = bit_string.find(pattern)
                    components["prefix"] = bit_string[:pos]
                    components["preamble"] = pattern
                    components["payload"] = bit_string[pos+len(pattern):]
                    break
        
        # If still no components identified, use statistical analysis
        if not components and length >= 16:
            # Analyze transition density in sliding windows
            window_size = min(8, length // 3)
            transition_counts = []
            
            for i in range(length - window_size + 1):
                window = bits[i:i+window_size]
                transitions = sum(1 for j in range(1, len(window)) if window[j] != window[j-1])
                transition_counts.append(transitions)
            
            # Find segments with different transition characteristics
            boundaries = []
            avg_transitions = sum(transition_counts) / len(transition_counts)
            
            for i in range(1, len(transition_counts)):
                prev_avg = sum(transition_counts[max(0, i-5):i]) / min(5, i)
                next_avg = sum(transition_counts[i:min(len(transition_counts), i+5)]) / min(5, len(transition_counts)-i)
                
                # Detect significant changes in transition density
                if abs(prev_avg - next_avg) > max(1, avg_transitions * 0.3):
                    boundaries.append(i)
            
            # Use detected boundaries to segment the sequence
            if boundaries:
                current_pos = 0
                segment_num = 1
                
                for boundary in sorted(boundaries):
                    if boundary - current_pos >= 4:  # Only consider segments of sufficient length
                        components[f"segment_{segment_num}"] = bit_string[current_pos:boundary]
                        segment_num += 1
                        current_pos = boundary
                
                # Add final segment
                if length - current_pos >= 4:
                    components[f"segment_{segment_num}"] = bit_string[current_pos:]
            else:
                # If no clear boundaries, split by proportions
                if length >= 40:
                    # For longer codes, common structure: preamble + fixed ID + rolling counter
                    preamble_size = min(16, length // 5)
                    id_size = min(24, length // 3)
                    components["preamble"] = bit_string[:preamble_size]
                    components["fixed_id"] = bit_string[preamble_size:preamble_size+id_size]
                    components["rolling_counter"] = bit_string[preamble_size+id_size:]
                else:
                    # For shorter codes, simpler structure
                    preamble_size = min(8, length // 4)
                    components["preamble"] = bit_string[:preamble_size]
                    components["payload"] = bit_string[preamble_size:]
        
        return components
    
    def batch_validate(self, bit_sequences: List[Union[List[int], str]]) -> List[Dict[str, Any]]:
        """
        Validate multiple bit sequences and sort by confidence.
        
        Args:
            bit_sequences: List of bit sequences to validate
            
        Returns:
            List of validation results, sorted by confidence (high to low)
        """
        results = []
        
        for i, sequence in enumerate(bit_sequences):
            result = self.validate_bit_sequence(sequence, source=f"batch_{i}")
            results.append(result)
        
        # Sort by confidence (descending)
        sorted_results = sorted(results, key=lambda x: x["confidence"], reverse=True)
        
        return sorted_results
    
    def find_best_candidate(self, bit_sequences: List[Union[List[int], str]]) -> Optional[Dict[str, Any]]:
        """
        Find the best candidate rolling code from a list of sequences.
        
        Args:
            bit_sequences: List of bit sequences to evaluate
            
        Returns:
            Best candidate result, or None if no valid candidates
        """
        results = self.batch_validate(bit_sequences)
        
        # Get the best candidate
        if results and results[0]["is_valid"]:
            return results[0]
        elif results and results[0]["confidence"] >= self.params["confidence_threshold"] * 0.8:
            # Return best candidate even if it's slightly below threshold
            return results[0]
        else:
            return None
    
    def extract_valid_segments(self, long_sequence: Union[List[int], str]) -> List[Dict[str, Any]]:
        """
        Extract valid code segments from a longer bit sequence.
        Useful when multiple codes might be present in a single capture.
        
        Args:
            long_sequence: Long bit sequence potentially containing multiple codes
            
        Returns:
            List of validation results for extracted segments
        """
        # Convert to string if needed
        if isinstance(long_sequence, list):
            bit_string = ''.join(map(str, long_sequence))
        else:
            bit_string = long_sequence
        
        valid_segments = []
        min_length = min(self.params["valid_lengths"])
        max_length = max(self.params["valid_lengths"])
        
        # Try different sliding window sizes
        for window_size in self.params["valid_lengths"]:
            # Slide the window with 50% overlap
            step_size = window_size // 2
            
            for start in range(0, len(bit_string) - window_size + 1, step_size):
                segment = bit_string[start:start+window_size]
                result = self.validate_bit_sequence(segment, source=f"segment_{start}")
                
                if result["is_valid"]:
                    # Check if this segment overlaps with previously found segments
                    overlapping = False
                    for existing in valid_segments:
                        existing_start = int(existing["source"].split("_")[1])
                        existing_len = len(existing["original_sequence"])
                        
                        # Check for overlap
                        if (start < existing_start + existing_len and 
                            start + window_size > existing_start):
                            overlapping = True
                            
                            # Keep the one with higher confidence
                            if result["confidence"] > existing["confidence"]:
                                valid_segments.remove(existing)
                                valid_segments.append(result)
                            
                            break
                    
                    if not overlapping:
                        valid_segments.append(result)
        
        # Also try with a flexible window approach for patterns not matching standard lengths
        self._extract_with_pattern_anchors(bit_string, valid_segments)
        
        # Sort by confidence
        valid_segments.sort(key=lambda x: x["confidence"], reverse=True)
        
        return valid_segments
    
    def _extract_with_pattern_anchors(self, bit_string: str, valid_segments: List[Dict[str, Any]]):
        """
        Extract valid segments using known patterns as anchors.
        
        Args:
            bit_string: Complete bit string
            valid_segments: List to append valid segments to
        """
        # Collect all preamble patterns
        preamble_patterns = []
        
        for manufacturer, patterns in self.known_patterns.items():
            if "preambles" in patterns:
                for preamble in patterns["preambles"]:
                    preamble_patterns.append(preamble["pattern"])
        
        # Add general patterns
        if "general" in self.known_patterns and "preambles" in self.known_patterns["general"]:
            for preamble in self.known_patterns["general"]["preambles"]:
                preamble_patterns.append(preamble["pattern"])
        
        # Find all pattern matches
        pattern_positions = []
        
        for pattern in preamble_patterns:
            pos = 0
            while True:
                pos = bit_string.find(pattern, pos)
                if pos == -1:
                    break
                
                pattern_positions.append((pos, pattern))
                pos += 1
        
        # Sort by position
        pattern_positions.sort()
        
        # Extract segments using patterns as anchors
        for i, (pos, pattern) in enumerate(pattern_positions):
            # Determine end position (next pattern or end of string)
            if i < len(pattern_positions) - 1:
                end_pos = pattern_positions[i+1][0]
            else:
                end_pos = len(bit_string)
            
            # Check various potential segment lengths
            for length in self.params["valid_lengths"]:
                if pos + length <= len(bit_string):
                    segment = bit_string[pos:pos+length]
                    result = self.validate_bit_sequence(segment, source=f"anchor_{pos}")
                    
                    if result["is_valid"]:
                        # Check for overlap with existing segments
                        overlapping = False
                        for existing in valid_segments:
                            existing_source = existing["source"]
                            if existing_source.startswith("segment_"):
                                existing_start = int(existing_source.split("_")[1])
                            elif existing_source.startswith("anchor_"):
                                existing_start = int(existing_source.split("_")[1])
                            else:
                                continue
                                
                            existing_len = len(existing["original_sequence"])
                            
                            # Check for overlap
                            if (pos < existing_start + existing_len and 
                                pos + length > existing_start):
                                overlapping = True
                                
                                # Keep the one with higher confidence
                                if result["confidence"] > existing["confidence"]:
                                    valid_segments.remove(existing)
                                    valid_segments.append(result)
                                
                                break
                        
                        if not overlapping:
                            valid_segments.append(result)


class SequenceLibraryManager:
    """
    Class for managing a library of validated rolling code sequences.
    """
    
    def __init__(self, library_file: str = "rolling_code_library.json"):
        """
        Initialize the sequence library manager.
        
        Args:
            library_file: Path to the library file
        """
        self.library_file = library_file
        self.sequences = []
        self.load_library()
        
        self.logger = logging.getLogger("SequenceLibraryManager")
    
    def load_library(self) -> bool:
        """
        Load the sequence library from file.
        
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            if os.path.exists(self.library_file):
                with open(self.library_file, 'r') as f:
                    self.sequences = json.load(f)
                
                self.logger.info(f"Loaded {len(self.sequences)} sequences from library")
                return True
            else:
                self.logger.info("Library file does not exist, creating new library")
                self.sequences = []
                return True
                
        except Exception as e:
            self.logger.error(f"Error loading library: {e}")
            self.sequences = []
            return False
    
    def save_library(self) -> bool:
        """
        Save the sequence library to file.
        
        Returns:
            True if saving was successful, False otherwise
        """
        try:
            with open(self.library_file, 'w') as f:
                json.dump(self.sequences, f, indent=2)
            
            self.logger.info(f"Saved {len(self.sequences)} sequences to library")
            return True
                
        except Exception as e:
            self.logger.error(f"Error saving library: {e}")
            return False
    
    def add_sequence(self, sequence: Dict[str, Any]) -> bool:
        """
        Add a validated sequence to the library.
        
        Args:
            sequence: Validated sequence information
            
        Returns:
            True if the sequence was added, False if it already exists
        """
        # Check if sequence already exists
        for seq in self.sequences:
            if seq["original_sequence"] == sequence["original_sequence"]:
                return False
        
        # Add timestamp
        sequence["timestamp"] = datetime.now().isoformat()
        
        # Add to library
        self.sequences.append(sequence)
        
        # Save library
        self.save_library()
        
        return True
    
    def search_similar_sequences(self, sequence: Union[List[int], str], 
                              threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Search for similar sequences in the library.
        
        Args:
            sequence: Query sequence to match
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar sequences, sorted by similarity
        """
        # Convert query to string if needed
        if isinstance(sequence, list):
            query = ''.join(map(str, sequence))
        else:
            query = sequence
        
        matches = []
        
        for seq in self.sequences:
            original = seq["original_sequence"]
            similarity = self._calculate_similarity(query, original)
            
            if similarity >= threshold:
                matches.append({
                    "sequence": seq,
                    "similarity": similarity
                })
        
        # Sort by similarity (descending)
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        return matches
    
    def _calculate_similarity(self, seq1: str, seq2: str) -> float:
        """
        Calculate the similarity between two bit sequences.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Similarity score (0-1)
        """
        # Handle different lengths
        if len(seq1) != len(seq2):
            # Try to align sequences
            min_len = min(len(seq1), len(seq2))
            max_len = max(len(seq1), len(seq2))
            
            # Only consider sequences of similar length
            if min_len < max_len * 0.8:
                return 0.0
            
            # Find best alignment
            if len(seq1) > len(seq2):
                longer, shorter = seq1, seq2
            else:
                longer, shorter = seq2, seq1
                
            best_score = 0.0
            
            for i in range(len(longer) - len(shorter) + 1):
                window = longer[i:i+len(shorter)]
                match_count = sum(1 for a, b in zip(window, shorter) if a == b)
                score = match_count / len(shorter)
                best_score = max(best_score, score)
                
            return best_score
        else:
            # Same length, direct comparison
            match_count = sum(1 for a, b in zip(seq1, seq2) if a == b)
            return match_count / len(seq1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the sequence library.
        
        Returns:
            Dictionary with library statistics
        """
        stats = {
            "total_sequences": len(self.sequences),
            "manufacturers": defaultdict(int),
            "formats": defaultdict(int),
            "lengths": defaultdict(int)
        }
        
        for seq in self.sequences:
            stats["manufacturers"][seq.get("manufacturer", "unknown")] += 1
            stats["formats"][seq.get("format", "unknown")] += 1
            stats["lengths"][len(seq["original_sequence"])] += 1
        
        return stats


def main():
    """Main entry point for demonstrating module functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bit Sequence Validator for Rolling Codes')
    parser.add_argument('-f', '--file', type=str, help='File containing bit sequences (one per line)')
    parser.add_argument('-s', '--sequence', type=str, help='Bit sequence to validate')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-l', '--library', action='store_true', help='Use sequence library')
    parser.add_argument('--extract', action='store_true', help='Extract valid segments from longer sequence')
    parser.add_argument('-o', '--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    # Create validator
    validator = BitSequenceValidator(debug_mode=args.debug)
    
    # Create library manager if requested
    library = None
    if args.library:
        library = SequenceLibraryManager()
    
    if args.file:
        # Read sequences from file
        try:
            with open(args.file, 'r') as f:
                sequences = [line.strip() for line in f.readlines() if line.strip()]
            
            if args.extract:
                # Treat the file as a single long sequence
                long_sequence = ''.join(sequences)
                print(f"Extracting valid segments from sequence of length {len(long_sequence)}")
                
                valid_segments = validator.extract_valid_segments(long_sequence)
                
                print(f"\nFound {len(valid_segments)} valid segments:")
                for i, result in enumerate(valid_segments):
                    print(f"\nSegment {i+1}:")
                    print(f"  Sequence: {result['preprocessed_sequence']}")
                    print(f"  Confidence: {result['confidence']:.2f}")
                    print(f"  Manufacturer: {result['manufacturer']}")
                    print(f"  Format: {result['format']}")
                    
                    if result['components']:
                        print("  Components:")
                        for name, component in result['components'].items():
                            print(f"    {name}: {component}")
                    
                    # Add to library if requested
                    if library and result["is_valid"]:
                        added = library.add_sequence(result)
                        print(f"  Added to library: {'Yes' if added else 'Already exists'}")
                
                # Save to output file if requested
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(valid_segments, f, indent=2)
                    print(f"Results saved to {args.output}")
            else:
                # Process each sequence separately
                print(f"Validating {len(sequences)} sequences")
                
                valid_count = 0
                for i, sequence in enumerate(sequences):
                    print(f"\nSequence {i+1}: {sequence}")
                    result = validator.validate_bit_sequence(sequence)
                    
                    print(f"  Valid: {result['is_valid']}")
                    print(f"  Confidence: {result['confidence']:.2f}")
                    print(f"  Manufacturer: {result['manufacturer']}")
                    print(f"  Format: {result['format']}")
                    
                    if result["issues"]:
                        print(f"  Issues ({len(result['issues'])}):")
                        for issue in result["issues"]:
                            print(f"    - {issue}")
                    
                    # Count valid sequences
                    if result["is_valid"]:
                        valid_count += 1
                        
                        # Add to library if requested
                        if library:
                            added = library.add_sequence(result)
                            print(f"  Added to library: {'Yes' if added else 'Already exists'}")
                
                print(f"\nFound {valid_count} valid sequences out of {len(sequences)}")
        
        except Exception as e:
            print(f"Error processing file: {e}")
    
    elif args.sequence:
        # Validate single sequence
        sequence = args.sequence.strip()
        print(f"Validating sequence: {sequence}")
        
        if args.extract:
            # Extract valid segments from the sequence
            valid_segments = validator.extract_valid_segments(sequence)
            
            print(f"\nFound {len(valid_segments)} valid segments:")
            for i, result in enumerate(valid_segments):
                print(f"\nSegment {i+1}:")
                print(f"  Sequence: {result['preprocessed_sequence']}")
                print(f"  Confidence: {result['confidence']:.2f}")
                print(f"  Manufacturer: {result['manufacturer']}")
                print(f"  Format: {result['format']}")
                
                if result['components']:
                    print("  Components:")
                    for name, component in result['components'].items():
                        print(f"    {name}: {component}")
                
                # Add to library if requested
                if library and result["is_valid"]:
                    added = library.add_sequence(result)
                    print(f"  Added to library: {'Yes' if added else 'Already exists'}")
        else:
            # Validate the sequence
            result = validator.validate_bit_sequence(sequence)
            
            print(f"Valid: {result['is_valid']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Manufacturer: {result['manufacturer']}")
            print(f"Format: {result['format']}")
            
            if result["issues"]:
                print(f"Issues ({len(result['issues'])}):")
                for issue in result["issues"]:
                    print(f"  - {issue}")
            
            if result["components"]:
                print("Components:")
                for name, component in result["components"].items():
                    print(f"  {name}: {component}")
            
            # Check library for similar sequences
            if library:
                similar = library.search_similar_sequences(sequence)
                
                if similar:
                    print(f"\nFound {len(similar)} similar sequences in library:")
                    for i, match in enumerate(similar[:3]):  # Show top 3
                        print(f"  Match {i+1} (similarity: {match['similarity']:.2f}):")
                        print(f"    Sequence: {match['sequence']['original_sequence']}")
                        print(f"    Manufacturer: {match['sequence']['manufacturer']}")
                        print(f"    Format: {match['sequence']['format']}")
                
                # Add to library if valid
                if result["is_valid"]:
                    added = library.add_sequence(result)
                    print(f"Added to library: {'Yes' if added else 'Already exists'}")
    
    elif library:
        # Show library statistics
        stats = library.get_statistics()
        
        print("Rolling Code Library Statistics")
        print(f"Total sequences: {stats['total_sequences']}")
        
        if stats['manufacturers']:
            print("\nManufacturers:")
            for manufacturer, count in sorted(stats['manufacturers'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {manufacturer}: {count}")
        
        if stats['formats']:
            print("\nFormats:")
            for format_name, count in sorted(stats['formats'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {format_name}: {count}")
        
        if stats['lengths']:
            print("\nSequence lengths:")
            for length, count in sorted(stats['lengths'].items()):
                print(f"  {length} bits: {count}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()