#!/usr/bin/env python3
"""
Code Predictor Module

Provides functionality for predicting potential next rolling codes based on
observed code patterns.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
import hashlib

class CodePredictor:
    """Class for predicting potential next rolling codes."""
    
    def __init__(self, attempts: int = 10):
        """
        Initialize the code predictor.
        
        Args:
            attempts: Number of prediction attempts to generate (default: 10)
        """
        self.attempts = attempts
        self.algorithms = ['linear', 'keeloq', 'xor', 'increment', 'hash']
        logging.debug(f"Initialized CodePredictor with {attempts} attempts")
        
    def predict_next_codes(self, observed_codes: List[str]) -> List[str]:
        """
        Predict potential next rolling codes based on observed codes.
        
        Args:
            observed_codes: List of previously observed codes
            
        Returns:
            List of predicted next code strings
        """
        if len(observed_codes) < 2:
            logging.warning("Need at least 2 observed codes to make predictions")
            return []
            
        logging.info(f"Predicting next codes based on {len(observed_codes)} observed codes")
        
        # Convert binary strings to integers for easier manipulation
        code_values = [int(code, 2) for code in observed_codes]
        
        # Sort codes by value (assuming they were generated in sequence)
        code_values.sort()
        
        # Try different prediction algorithms
        predictions = []
        
        # Generate predictions using different algorithms
        for algorithm in self.algorithms:
            logging.debug(f"Trying prediction algorithm: {algorithm}")
            algorithm_predictions = self._predict_with_algorithm(code_values, algorithm)
            
            if algorithm_predictions:
                # Convert predictions back to binary strings
                for pred_value in algorithm_predictions:
                    # Ensure same bit length as original codes
                    bit_length = len(observed_codes[0])
                    pred_binary = bin(pred_value)[2:].zfill(bit_length)
                    
                    # Only add if not already in predictions and not in observed codes
                    if (pred_binary not in predictions and 
                        pred_binary not in observed_codes):
                        predictions.append(pred_binary)
                        logging.debug(f"Generated prediction: {pred_binary}")
        
        # Limit to the requested number of attempts
        predictions = predictions[:self.attempts]
        
        logging.info(f"Generated {len(predictions)} unique predictions")
        return predictions
    
    def _predict_with_algorithm(self, code_values: List[int], algorithm: str) -> List[int]:
        """
        Predict next codes using the specified algorithm.
        
        Args:
            code_values: List of observed code values as integers
            algorithm: Algorithm to use for prediction
            
        Returns:
            List of predicted code values as integers
        """
        predictions = []
        
        if algorithm == 'linear':
            # Linear extrapolation
            if len(code_values) >= 2:
                last_code = code_values[-1]
                second_last_code = code_values[-2]
                
                # Calculate difference
                diff = last_code - second_last_code
                
                # Generate predictions by adding difference
                for i in range(1, self.attempts + 1):
                    predictions.append(last_code + diff * i)
        
        elif algorithm == 'increment':
            # Simple increment (common in basic systems)
            last_code = code_values[-1]
            
            # Generate predictions by incrementing by 1
            for i in range(1, self.attempts + 1):
                predictions.append(last_code + i)
                
        elif algorithm == 'keeloq':
            # Simulate KeeLoq-like algorithm
            # This is a simplified simulation of how KeeLoq works
            if len(code_values) >= 2:
                last_code = code_values[-1]
                
                # Simulate KeeLoq NLFSR (Non-Linear Feedback Shift Register)
                for i in range(1, self.attempts + 1):
                    # Extract components (simulated)
                    discriminant = last_code & 0xF  # Last 4 bits
                    counter = (last_code >> 4) & 0xFFFF  # 16-bit counter
                    fixed = last_code >> 20  # Fixed portion
                    
                    # Increment counter
                    counter = (counter + 1) & 0xFFFF
                    
                    # Simulate non-linear transformation
                    if counter % 2 == 0:
                        discriminant = (discriminant + 7) & 0xF
                    else:
                        discriminant = (discriminant + 13) & 0xF
                    
                    # Combine back
                    next_code = (fixed << 20) | (counter << 4) | discriminant
                    predictions.append(next_code)
                    last_code = next_code
        
        elif algorithm == 'xor':
            # XOR-based prediction
            if len(code_values) >= 3:
                # Calculate XOR pattern between successive codes
                patterns = []
                for i in range(1, len(code_values)):
                    patterns.append(code_values[i] ^ code_values[i-1])
                
                # Check if pattern is consistent
                if len(set(patterns)) <= 2:  # Allow small variations
                    # Use most common pattern
                    from collections import Counter
                    pattern_count = Counter(patterns)
                    common_pattern = pattern_count.most_common(1)[0][0]
                    
                    # Generate predictions
                    last_code = code_values[-1]
                    for i in range(1, self.attempts + 1):
                        predictions.append(last_code ^ common_pattern)
                        last_code = last_code ^ common_pattern
        
        elif algorithm == 'hash':
            # Hash-based prediction (simplified HOTP-like)
            last_code = code_values[-1]
            
            # Generate predictions using hash function
            for i in range(1, self.attempts + 1):
                # Create hash input
                hash_input = f"{last_code}:{i}".encode()
                
                # Generate hash
                hash_value = hashlib.md5(hash_input).digest()
                
                # Extract 32 bits
                extracted_value = int.from_bytes(hash_value[:4], byteorder='big')
                
                # Mask to the same bit length as original codes
                # Determine bit length from largest observed code
                bit_length = max(code_values).bit_length()
                mask = (1 << bit_length) - 1
                
                next_code = extracted_value & mask
                predictions.append(next_code)
                
        return predictions
    
    def analyze_code_sequence(self, observed_codes: List[str]) -> Dict[str, Any]:
        """
        Analyze observed codes to identify patterns.
        
        Args:
            observed_codes: List of previously observed codes
            
        Returns:
            Dictionary with analysis results
        """
        if len(observed_codes) < 2:
            return {'error': 'Need at least 2 codes for analysis'}
            
        logging.info(f"Analyzing sequence of {len(observed_codes)} observed codes")
        
        # Convert to integers
        values = [int(code, 2) for code in observed_codes]
        values.sort()  # Sort by value
        
        # Calculate differences
        diffs = [values[i] - values[i-1] for i in range(1, len(values))]
        
        # Check for consistent difference (linear progression)
        is_linear = len(set(diffs)) == 1
        
        # Check for xor pattern
        xor_diffs = [values[i] ^ values[i-1] for i in range(1, len(values))]
        is_xor_pattern = len(set(xor_diffs)) == 1
        
        # Check for increment pattern
        is_increment = set(diffs) == {1}
        
        # Determine most likely algorithm
        likely_algorithms = []
        
        if is_increment:
            likely_algorithms.append('increment')
        if is_linear and not is_increment:
            likely_algorithms.append('linear')
        if is_xor_pattern:
            likely_algorithms.append('xor')
            
        # If no obvious pattern, try more complex analysis
        if not likely_algorithms:
            # Check for KeeLoq-like pattern
            # This is a simplified check
            fixed_bits_match = True
            for i in range(1, len(values)):
                # Check if high bits remain constant
                if (values[i] >> 20) != (values[i-1] >> 20):
                    fixed_bits_match = False
                    break
                    
            if fixed_bits_match:
                likely_algorithms.append('keeloq')
            else:
                likely_algorithms.append('unknown')
        
        return {
            'num_codes': len(observed_codes),
            'is_linear': is_linear,
            'is_xor_pattern': is_xor_pattern,
            'is_increment': is_increment,
            'likely_algorithms': likely_algorithms,
            'average_diff': np.mean(diffs) if diffs else 0,
            'prediction_confidence': self._calculate_confidence(likely_algorithms)
        }
    
    def _calculate_confidence(self, likely_algorithms: List[str]) -> float:
        """
        Calculate confidence level for predictions.
        
        Args:
            likely_algorithms: List of likely algorithms
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not likely_algorithms or 'unknown' in likely_algorithms:
            return 0.1
            
        # Assign confidence scores based on algorithm certainty
        if 'increment' in likely_algorithms:
            return 0.9  # High confidence for simple increment
        elif 'linear' in likely_algorithms:
            return 0.8  # Good confidence for linear progression
        elif 'xor' in likely_algorithms:
            return 0.7  # Decent confidence for XOR pattern
        elif 'keeloq' in likely_algorithms:
            return 0.5  # Moderate confidence for KeeLoq-like
        else:
            return 0.3  # Low confidence for other algorithms
