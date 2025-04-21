#!/usr/bin/env python3
"""
Cryptographic Analysis Module for Garage Door Security

This module provides advanced functionality for analyzing and exploiting cryptographic
weaknesses in garage door remote systems, including:

1. Replay Protection Testing - Detects vulnerable systems that don't use proper replay protection
2. Entropy Analysis - Identifies weak encryption by analyzing payload entropy
3. Block Pattern Analysis - Detects ECB mode or static IV usage through repeating patterns
4. Nonce/IV Reuse Detection - Identifies critical vulnerabilities in AES-based remotes

For educational and security research purposes only.
"""

import os
import sys
import math
import time
import json
import binascii
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from collections import defaultdict, Counter

# Import for cryptographic functions
try:
    from Crypto.Cipher import AES, DES, DES3
    from Crypto.Util.Padding import pad, unpad
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Warning: pycryptodome not available. Install with: pip install pycryptodome")
    print("Cryptographic analysis features will be limited without this module.")

# Common fixed key lengths for different block ciphers
COMMON_KEY_SIZES = {
    "AES": [16, 24, 32],    # AES-128, AES-192, AES-256
    "DES": [8],             # DES (64-bit key with parity)
    "3DES": [16, 24],       # 3DES with 2 or 3 keys
    "RC5": [8, 16, 32],     # RC5 with various key sizes
    "KEELOQ": [8]           # KeeLoq (64-bit key)
}

# Common IV/Nonce sizes
COMMON_IV_SIZES = {
    "AES-CBC": 16,          # 128-bit IV
    "AES-CTR": 16,          # 128-bit nonce+counter
    "AES-GCM": 12,          # 96-bit nonce is recommended
    "AES-CCM": 12,          # 96-bit nonce is recommended
    "DES-CBC": 8,           # 64-bit IV
    "3DES-CBC": 8           # 64-bit IV
}

# Known start patterns for major manufacturers
KNOWN_PAYLOAD_PATTERNS = {
    "Chamberlain": {
        "header_bytes": [0x55, 0x01, 0x00],  # Example header
        "iv_position": [3, 19],              # IV from byte 3 to 19 (16 bytes)
        "payload_position": [19, -1],        # Encrypted payload starts at byte 19
        "cipher_type": "AES-128-CBC"         # Using AES-128 in CBC mode
    },
    "Genie": {
        "header_bytes": [0xAA, 0xBB],
        "iv_position": [4, 16],
        "payload_position": [16, -1],
        "cipher_type": "AES-128-CTR"
    },
    "LiftMaster": {
        "header_bytes": [0x55, 0x01, 0x0A],
        "iv_position": [3, 19],
        "payload_position": [19, -1],
        "cipher_type": "AES-128-CBC"
    },
    "Stanley": {
        "header_bytes": [0xCC, 0xDD],
        "iv_position": [2, 10],
        "payload_position": [10, -1],
        "cipher_type": "DES-CBC"
    }
}


class CryptoAnalyzer:
    """
    Class for analyzing cryptographic vulnerabilities in garage door remotes.
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the crypto analyzer.
        
        Args:
            debug_mode: Enable debug mode for additional output
        """
        self.debug_mode = debug_mode
        self.captured_payloads = []
        self.iv_cache = {}
        self.nonce_reuse_detected = False
        self.known_keys = {}  # For testing with known keys
        
        # Setup logging
        self._setup_logging()
        
        # Check crypto library
        if not CRYPTO_AVAILABLE:
            self.logger.warning("Crypto library not available. Some features will be limited.")
    
    def _setup_logging(self):
        """Configure logging for the crypto analyzer."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure file handler with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = f"logs/crypto_analyzer_{timestamp}.log"
        
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG if self.debug_mode else logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("CryptoAnalyzer")
    
    def add_payload(self, payload: bytes, source: str = "capture") -> int:
        """
        Add a payload to the analyzer database.
        
        Args:
            payload: The binary payload data
            source: Source of the payload (capture, file, etc.)
            
        Returns:
            Index of the added payload
        """
        self.captured_payloads.append({
            "data": payload,
            "source": source,
            "timestamp": datetime.now(),
            "analyzed": False,
            "entropy": None,
            "repeating_blocks": None,
            "manufacturer": None,
            "iv": None,
            "vulnerabilities": []
        })
        
        payload_index = len(self.captured_payloads) - 1
        self.logger.info(f"Added payload #{payload_index} from {source} ({len(payload)} bytes)")
        
        return payload_index
    
    def load_payload_from_file(self, file_path: str) -> int:
        """
        Load payload data from a file.
        
        Args:
            file_path: Path to the file containing binary payload data
            
        Returns:
            Index of the loaded payload
        """
        try:
            with open(file_path, 'rb') as f:
                payload = f.read()
            
            return self.add_payload(payload, source=f"file:{file_path}")
        except Exception as e:
            self.logger.error(f"Error loading payload from {file_path}: {e}")
            raise
    
    def analyze_payload(self, payload_index: int, identify_manufacturer: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a specific payload.
        
        Args:
            payload_index: Index of the payload to analyze
            identify_manufacturer: Try to identify the manufacturer based on patterns
            
        Returns:
            Dictionary with analysis results
        """
        if payload_index < 0 or payload_index >= len(self.captured_payloads):
            raise ValueError(f"Invalid payload index: {payload_index}")
        
        payload_entry = self.captured_payloads[payload_index]
        payload = payload_entry["data"]
        
        self.logger.info(f"Analyzing payload #{payload_index} ({len(payload)} bytes)")
        
        # Initialize results
        analysis_results = {
            "payload_size": len(payload),
            "entropy": None,
            "manufacturer": None,
            "iv_detected": None,
            "repeating_blocks": None,
            "vulnerabilities": [],
            "encryption_type": "unknown"
        }
        
        # Calculate entropy
        entropy = self.calculate_entropy(payload)
        analysis_results["entropy"] = entropy
        payload_entry["entropy"] = entropy
        
        # Check for extremely low entropy (possible fixed-pattern or unencrypted)
        if entropy < 3.0:
            analysis_results["vulnerabilities"].append({
                "type": "low_entropy",
                "severity": "high",
                "description": "Extremely low entropy suggests unencrypted or fixed pattern data"
            })
        elif entropy < 5.0:
            analysis_results["vulnerabilities"].append({
                "type": "suspicious_entropy",
                "severity": "medium",
                "description": "Suspiciously low entropy may indicate weak encryption or encoding"
            })
        
        # Detect repeating blocks
        repeating_blocks = self.detect_repeating_blocks(payload)
        analysis_results["repeating_blocks"] = repeating_blocks
        payload_entry["repeating_blocks"] = repeating_blocks
        
        if repeating_blocks["count"] > 0:
            analysis_results["vulnerabilities"].append({
                "type": "repeating_blocks",
                "severity": "high",
                "description": "Repeating blocks detected; likely ECB mode or static IV in use",
                "details": repeating_blocks
            })
        
        # Try to identify manufacturer
        if identify_manufacturer:
            manufacturer = self.identify_manufacturer(payload)
            analysis_results["manufacturer"] = manufacturer
            payload_entry["manufacturer"] = manufacturer
            
            # If manufacturer identified, try more specific tests
            if manufacturer:
                # Extract IV if possible
                iv = self.extract_iv(payload, manufacturer)
                if iv:
                    analysis_results["iv_detected"] = binascii.hexlify(iv).decode()
                    payload_entry["iv"] = iv
                    
                    # Check for IV reuse
                    iv_reuse = self.check_iv_reuse(iv, payload_index)
                    if iv_reuse:
                        analysis_results["vulnerabilities"].append({
                            "type": "iv_reuse",
                            "severity": "critical",
                            "description": "Nonce/IV reuse detected across multiple payloads",
                            "details": {
                                "reused_with": iv_reuse,
                                "iv": binascii.hexlify(iv).decode()
                            }
                        })
                
                # Determine encryption type
                if manufacturer in KNOWN_PAYLOAD_PATTERNS:
                    analysis_results["encryption_type"] = KNOWN_PAYLOAD_PATTERNS[manufacturer]["cipher_type"]
        
        # Check for vulnerable key management
        key_analysis = self.analyze_key_management(payload)
        if key_analysis["vulnerable"]:
            analysis_results["vulnerabilities"].append({
                "type": "weak_key_management",
                "severity": "high" if key_analysis["severity"] == "high" else "medium",
                "description": "Possible weak key derivation or management detected",
                "details": key_analysis["details"]
            })
        
        # Mark payload as analyzed
        payload_entry["analyzed"] = True
        payload_entry["vulnerabilities"] = analysis_results["vulnerabilities"]
        
        self.logger.info(f"Analysis complete for payload #{payload_index}")
        if analysis_results["vulnerabilities"]:
            self.logger.warning(f"Found {len(analysis_results['vulnerabilities'])} vulnerabilities")
        
        return analysis_results
    
    def calculate_entropy(self, data: bytes) -> float:
        """
        Calculate Shannon entropy of the data.
        
        Args:
            data: Binary data to analyze
            
        Returns:
            Entropy value (0.0 to 8.0 for byte data)
        """
        if not data:
            return 0.0
        
        # Count occurrences of each byte value
        counter = Counter(data)
        
        # Calculate entropy
        entropy = 0.0
        for count in counter.values():
            probability = count / len(data)
            entropy -= probability * math.log2(probability)
        
        self.logger.debug(f"Entropy: {entropy:.4f} bits per byte")
        
        return entropy
    
    def detect_repeating_blocks(self, data: bytes, block_size: int = 16) -> Dict[str, Any]:
        """
        Detect repeating blocks in the data, which may indicate ECB mode or static IV.
        
        Args:
            data: Binary data to analyze
            block_size: Size of blocks to check (default: 16 bytes for AES)
            
        Returns:
            Dictionary with detection results
        """
        # Ensure data length is multiple of block_size by padding
        if len(data) < block_size:
            return {"count": 0, "blocks": {}, "positions": {}}
        
        # Check multiple standard block sizes if not specified
        if block_size <= 0:
            results = {}
            for size in [8, 16]:  # Common block sizes (DES, AES)
                results[size] = self.detect_repeating_blocks(data, size)
            return results
        
        # Divide data into blocks
        blocks = {}
        positions = defaultdict(list)
        repeating = 0
        
        # Process each block
        for i in range(0, len(data) - block_size + 1, block_size):
            block = data[i:i+block_size]
            block_hex = binascii.hexlify(block).decode()
            
            positions[block_hex].append(i)
            
            if len(positions[block_hex]) > 1:
                # This is a repeating block
                if block_hex not in blocks:
                    blocks[block_hex] = 1
                    repeating += 1
                blocks[block_hex] += 1
        
        # Filter out non-repeating blocks
        positions = {k: v for k, v in positions.items() if len(v) > 1}
        
        results = {
            "count": repeating,
            "blocks": blocks,
            "positions": positions
        }
        
        if repeating > 0:
            self.logger.info(f"Found {repeating} repeating {block_size}-byte blocks")
            if self.debug_mode:
                for block, positions_list in positions.items():
                    self.logger.debug(f"Block {block[:16]}... repeats at positions: {positions_list}")
        
        return results
    
    def identify_manufacturer(self, data: bytes) -> Optional[str]:
        """
        Attempt to identify the manufacturer based on known patterns.
        
        Args:
            data: Binary payload data
            
        Returns:
            Manufacturer name if identified, None otherwise
        """
        if len(data) < 4:  # Minimum length for identification
            return None
        
        # Check header patterns
        for manufacturer, pattern in KNOWN_PAYLOAD_PATTERNS.items():
            header = pattern.get("header_bytes", [])
            if len(header) > 0 and len(data) >= len(header):
                if list(data[:len(header)]) == header:
                    self.logger.info(f"Identified manufacturer: {manufacturer}")
                    return manufacturer
        
        # Advanced pattern matching could be added here
        
        return None
    
    def extract_iv(self, data: bytes, manufacturer: str) -> Optional[bytes]:
        """
        Extract the IV/nonce from the payload based on manufacturer patterns.
        
        Args:
            data: Binary payload data
            manufacturer: Identified manufacturer
            
        Returns:
            Extracted IV/nonce as bytes if found, None otherwise
        """
        if manufacturer not in KNOWN_PAYLOAD_PATTERNS:
            return None
        
        pattern = KNOWN_PAYLOAD_PATTERNS[manufacturer]
        iv_position = pattern.get("iv_position")
        
        if not iv_position or len(iv_position) != 2:
            return None
        
        start, end = iv_position
        if end < 0:
            end = len(data) + end  # Handle negative indexing
        
        if start >= len(data) or end > len(data) or start >= end:
            return None
        
        iv = data[start:end]
        self.logger.info(f"Extracted IV: {binascii.hexlify(iv).decode()}")
        
        return iv
    
    def check_iv_reuse(self, iv: bytes, current_payload_index: int) -> List[int]:
        """
        Check if this IV has been seen before (critical vulnerability).
        
        Args:
            iv: The extracted IV/nonce
            current_payload_index: Index of the current payload
            
        Returns:
            List of payload indices that used the same IV
        """
        iv_hex = binascii.hexlify(iv).decode()
        
        # Check if IV exists in our cache
        if iv_hex in self.iv_cache:
            reused_with = self.iv_cache[iv_hex]
            self.logger.warning(f"IV REUSE DETECTED! IV {iv_hex} was used in payload(s): {reused_with}")
            
            # Add current payload to the list
            self.iv_cache[iv_hex].append(current_payload_index)
            self.nonce_reuse_detected = True
            
            return reused_with
        else:
            # First time seeing this IV
            self.iv_cache[iv_hex] = [current_payload_index]
            return []
    
    def analyze_key_management(self, data: bytes) -> Dict[str, Any]:
        """
        Analyze potential vulnerabilities in key management.
        
        Args:
            data: Binary payload data
            
        Returns:
            Dictionary with analysis results
        """
        # Default result
        result = {
            "vulnerable": False,
            "severity": "low",
            "details": {}
        }
        
        # Check for common weak key patterns
        weak_patterns = self.check_weak_key_patterns(data)
        if weak_patterns:
            result["vulnerable"] = True
            result["severity"] = "high"
            result["details"]["weak_patterns"] = weak_patterns
        
        # Additional key management checks could be added here
        
        return result
    
    def check_weak_key_patterns(self, data: bytes) -> List[Dict[str, Any]]:
        """
        Check for common weak key derivation patterns.
        
        Args:
            data: Binary payload data
            
        Returns:
            List of detected weak patterns
        """
        weak_patterns = []
        
        # Check for repeating key bytes pattern
        repeating_bytes = self.detect_repeating_bytes(data)
        if repeating_bytes["found"]:
            weak_patterns.append({
                "type": "repeating_key_bytes",
                "description": "Detected repeating byte pattern, possible weak key derivation",
                "details": repeating_bytes
            })
        
        # Check for sequential/incremental pattern
        sequential = self.detect_sequential_pattern(data)
        if sequential["found"]:
            weak_patterns.append({
                "type": "sequential_pattern",
                "description": "Detected sequential byte pattern, possible weak key generation",
                "details": sequential
            })
        
        return weak_patterns
    
    def detect_repeating_bytes(self, data: bytes) -> Dict[str, Any]:
        """
        Detect simple repeating byte patterns that might indicate weak keys.
        
        Args:
            data: Binary data to analyze
            
        Returns:
            Dictionary with detection results
        """
        result = {"found": False, "pattern": None, "length": 0}
        
        # Check common repeating lengths
        for pattern_length in range(1, min(8, len(data) // 2)):
            is_repeating = True
            pattern = data[:pattern_length]
            
            for i in range(pattern_length, len(data), pattern_length):
                chunk = data[i:i+pattern_length]
                if len(chunk) < pattern_length:  # Handle incomplete chunk at end
                    break
                
                if chunk != pattern:
                    is_repeating = False
                    break
            
            if is_repeating:
                result["found"] = True
                result["pattern"] = binascii.hexlify(pattern).decode()
                result["length"] = pattern_length
                self.logger.warning(f"Detected repeating byte pattern: {result['pattern']} (length {pattern_length})")
                break
        
        return result
    
    def detect_sequential_pattern(self, data: bytes) -> Dict[str, Any]:
        """
        Detect sequential/incremental byte patterns that might indicate weak keys.
        
        Args:
            data: Binary data to analyze
            
        Returns:
            Dictionary with detection results
        """
        result = {"found": False, "type": None, "start_value": None}
        
        if len(data) < 4:  # Minimum length to detect pattern
            return result
        
        # Check for incrementing sequence
        incrementing = True
        for i in range(1, min(16, len(data))):
            if (data[i] - data[i-1]) % 256 != 1:
                incrementing = False
                break
        
        if incrementing:
            result["found"] = True
            result["type"] = "incrementing"
            result["start_value"] = data[0]
            self.logger.warning(f"Detected incrementing byte pattern starting with: {data[0]}")
            return result
        
        # Check for decrementing sequence
        decrementing = True
        for i in range(1, min(16, len(data))):
            if (data[i-1] - data[i]) % 256 != 1:
                decrementing = False
                break
        
        if decrementing:
            result["found"] = True
            result["type"] = "decrementing"
            result["start_value"] = data[0]
            self.logger.warning(f"Detected decrementing byte pattern starting with: {data[0]}")
        
        return result
    
    def test_replay_protection(self, device_address: str, replay_payload: bytes, 
                               delay_between_replays: float = 1.0,
                               num_replays: int = 3) -> Dict[str, Any]:
        """
        Test if a device is vulnerable to replay attacks by sending the same payload multiple times.
        
        Args:
            device_address: Address or identifier of target device
            replay_payload: The payload to replay
            delay_between_replays: Delay in seconds between replay attempts
            num_replays: Number of times to replay the payload
            
        Returns:
            Dictionary with test results
        """
        self.logger.info(f"Testing replay protection against {device_address}")
        self.logger.info(f"Payload length: {len(replay_payload)} bytes, " 
                         f"replaying {num_replays} times with {delay_between_replays}s delay")
        
        # This function is normally used with actual hardware
        # Here we'll simulate the process for demonstration
        
        # Record results
        results = {
            "device": device_address,
            "payload_size": len(replay_payload),
            "replays_sent": num_replays,
            "responses": [],
            "vulnerable": False,
            "conclusion": ""
        }
        
        # Send initial payload
        response = self._simulate_send_payload(device_address, replay_payload)
        results["responses"].append({
            "attempt": 0,
            "success": response["success"],
            "response_code": response["code"],
            "response_time": response["response_time"]
        })
        
        initial_success = response["success"]
        self.logger.info(f"Initial send: {'Success' if initial_success else 'Failed'}")
        
        # If initial send fails, device might be unresponsive
        if not initial_success:
            results["conclusion"] = "Device unresponsive to initial payload"
            return results
        
        # Perform replay attempts
        successful_replays = 0
        
        for i in range(1, num_replays + 1):
            time.sleep(delay_between_replays)
            self.logger.info(f"Sending replay {i}/{num_replays}")
            
            response = self._simulate_send_payload(device_address, replay_payload)
            results["responses"].append({
                "attempt": i,
                "success": response["success"],
                "response_code": response["code"],
                "response_time": response["response_time"]
            })
            
            if response["success"]:
                successful_replays += 1
                self.logger.warning(f"Replay {i} SUCCEEDED - device accepted replayed message!")
            else:
                self.logger.info(f"Replay {i} failed - device rejected replayed message")
        
        # Analyze results
        if successful_replays > 0:
            results["vulnerable"] = True
            results["conclusion"] = f"Device is VULNERABLE to replay attacks! {successful_replays}/{num_replays} replays succeeded."
            
            if successful_replays < num_replays:
                results["conclusion"] += " Device may have time-based replay protection."
        else:
            results["conclusion"] = "Device appears to have proper replay protection - all replays rejected."
        
        self.logger.info(results["conclusion"])
        return results
    
    def _simulate_send_payload(self, device_address: str, payload: bytes) -> Dict[str, Any]:
        """
        Simulate sending a payload to a device.
        In a real implementation, this would use the actual hardware.
        
        Args:
            device_address: Address or identifier of target device
            payload: The payload to send
            
        Returns:
            Dictionary with response information
        """
        # This is a stub for demonstration - in real usage, this would
        # interface with HackRF or other hardware to actually send the payload
        
        # Simulate different responses based on payload characteristics to demonstrate
        # different replay protection implementations
        
        # Calculate a hash of the payload to simulate different devices
        import hashlib
        device_hash = int(hashlib.md5(device_address.encode()).hexdigest(), 16) % 4
        
        # Simulate response time
        import random
        response_time = 0.05 + random.random() * 0.1
        time.sleep(response_time)
        
        # Decide if the replay should succeed based on device type
        success = False
        code = "REJECTED"
        
        # Device types:
        # 0: No replay protection (always accepts)
        # 1: Simple counter (accepts first send, rejects all replays)
        # 2: Timed window (accepts first send, rejects replays within 5 seconds, accepts after)
        # 3: Challenge-response (always rejects replays)
        
        # Check if this is a first send or replay
        is_replay = payload in [p["data"] for p in self.captured_payloads]
        
        if device_hash == 0:
            # No replay protection
            success = True
            code = "ACCEPTED"
        elif device_hash == 1:
            # Simple counter
            if not is_replay:
                success = True
                code = "ACCEPTED"
        elif device_hash == 2:
            # Timed window - not implemented in this simulation
            if not is_replay:
                success = True
                code = "ACCEPTED"
        elif device_hash == 3:
            # Challenge-response
            if not is_replay:
                success = True
                code = "ACCEPTED"
        
        return {
            "success": success,
            "code": code,
            "response_time": response_time
        }
    
    def analyze_entropy_across_payloads(self, payload_indices: List[int] = None) -> Dict[str, Any]:
        """
        Analyze entropy changes across multiple payloads to identify potential vulnerabilities.
        
        Args:
            payload_indices: List of payload indices to analyze (defaults to all)
            
        Returns:
            Dictionary with analysis results
        """
        if payload_indices is None:
            payload_indices = list(range(len(self.captured_payloads)))
        
        if not payload_indices:
            return {"error": "No payloads to analyze"}
        
        payloads = [self.captured_payloads[i] for i in payload_indices if i < len(self.captured_payloads)]
        
        # Calculate entropy for each payload if not already done
        for payload in payloads:
            if payload["entropy"] is None:
                payload["entropy"] = self.calculate_entropy(payload["data"])
        
        # Analyze entropy distribution
        entropies = [payload["entropy"] for payload in payloads]
        mean_entropy = sum(entropies) / len(entropies)
        min_entropy = min(entropies)
        max_entropy = max(entropies)
        range_entropy = max_entropy - min_entropy
        
        # Check for consistency
        entropy_std = np.std(entropies) if len(entropies) > 1 else 0
        
        # Calculate entropy by section
        section_analysis = self.analyze_payload_sections(payloads)
        
        # Prepare results
        results = {
            "num_payloads": len(payloads),
            "mean_entropy": mean_entropy,
            "min_entropy": min_entropy,
            "max_entropy": max_entropy,
            "entropy_range": range_entropy,
            "entropy_std": entropy_std,
            "section_analysis": section_analysis,
            "vulnerabilities": []
        }
        
        # Detect potential vulnerabilities
        if mean_entropy < 6.0:
            results["vulnerabilities"].append({
                "type": "low_overall_entropy",
                "severity": "medium" if mean_entropy < 5.0 else "low",
                "description": f"Overall low entropy ({mean_entropy:.2f}) may indicate weak encryption or encoding"
            })
        
        if entropy_std > 1.0:
            results["vulnerabilities"].append({
                "type": "inconsistent_entropy",
                "severity": "high",
                "description": f"High variation in entropy ({entropy_std:.2f}) suggests inconsistent encryption or partial encryption"
            })
        
        # Analyze sections with suspicious entropy
        suspicious_sections = []
        for section, section_data in section_analysis["sections"].items():
            if section_data["entropy"] < 5.0 and section_data["size"] >= 8:
                suspicious_sections.append({
                    "section": section,
                    "entropy": section_data["entropy"],
                    "size": section_data["size"]
                })
        
        if suspicious_sections:
            results["vulnerabilities"].append({
                "type": "suspicious_sections",
                "severity": "medium",
                "description": "Some payload sections have suspiciously low entropy",
                "details": {
                    "sections": suspicious_sections
                }
            })
        
        # Log results
        self.logger.info(f"Analyzed entropy across {len(payloads)} payloads")
        self.logger.info(f"Mean entropy: {mean_entropy:.2f}, Range: {min_entropy:.2f}-{max_entropy:.2f}, Std: {entropy_std:.2f}")
        
        if results["vulnerabilities"]:
            self.logger.warning(f"Found {len(results['vulnerabilities'])} potential entropy-related vulnerabilities")
        
        return results
    
    def analyze_payload_sections(self, payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze different sections of payloads to identify patterns and vulnerabilities.
        
        Args:
            payloads: List of payload entries to analyze
            
        Returns:
            Dictionary with section analysis results
        """
        # Define standard sections to analyze
        sections = {
            "header": {"start": 0, "length": 4, "description": "Initial payload bytes"},
            "middle": {"start": 4, "length": 16, "description": "Middle section (potential IV/nonce)"},
            "body": {"start": 20, "length": -1, "description": "Main payload body"}
        }
        
        section_data = {"sections": {}}
        
        for section_name, section_info in sections.items():
            start = section_info["start"]
            length = section_info["length"]
            
            # Collect data from this section across all payloads
            section_bytes = []
            
            for payload in payloads:
                data = payload["data"]
                if start >= len(data):
                    continue
                
                if length < 0:
                    # Negative length means "until the end"
                    section = data[start:]
                else:
                    end = min(start + length, len(data))
                    section = data[start:end]
                
                if section:
                    section_bytes.append(section)
            
            if not section_bytes:
                continue
            
            # Analyze this section
            section_entropy = sum(self.calculate_entropy(s) for s in section_bytes) / len(section_bytes)
            
            # Check for consistency in this section
            identical = all(s == section_bytes[0] for s in section_bytes[1:]) if len(section_bytes) > 1 else False
            
            section_data["sections"][section_name] = {
                "size": sum(len(s) for s in section_bytes) // len(section_bytes),
                "entropy": section_entropy,
                "identical_across_payloads": identical,
                "description": section_info["description"]
            }
            
            # For headers and potential IVs, check if they're unique across payloads
            if section_name in ["header", "middle"] and len(section_bytes) > 1:
                unique_values = set(bytes(s) for s in section_bytes)
                section_data["sections"][section_name]["unique_values"] = len(unique_values)
                section_data["sections"][section_name]["total_values"] = len(section_bytes)
                
                # If middle section always changes, it might be an IV/nonce
                if section_name == "middle" and len(unique_values) == len(section_bytes):
                    section_data["sections"][section_name]["possible_iv"] = True
                    
                    # Check for sequential patterns in possible IV
                    if len(section_bytes) >= 3:
                        sequential = self.check_sequential_iv_pattern(section_bytes)
                        section_data["sections"][section_name]["sequential_pattern"] = sequential
                
                # If header never changes, it might be a fixed identifier
                if section_name == "header" and len(unique_values) == 1:
                    section_data["sections"][section_name]["fixed_identifier"] = True
        
        return section_data
    
    def check_sequential_iv_pattern(self, iv_sections: List[bytes]) -> Dict[str, Any]:
        """
        Check if IV/nonce sections follow a sequential pattern (counter).
        
        Args:
            iv_sections: List of potential IV/nonce sections
            
        Returns:
            Dictionary with pattern analysis results
        """
        result = {"found": False, "type": None, "consistency": 0.0}
        
        if len(iv_sections) < 3:
            return result
        
        # Check for incrementing counter at different positions
        for pos in range(min(len(s) for s in iv_sections)):
            values = [s[pos] for s in iv_sections]
            
            # Check if values are incrementing
            increments = [(values[i] - values[i-1]) % 256 for i in range(1, len(values))]
            
            if all(inc == increments[0] for inc in increments) and increments[0] > 0:
                result["found"] = True
                result["type"] = "incrementing_counter"
                result["position"] = pos
                result["increment"] = increments[0]
                result["consistency"] = 1.0
                
                self.logger.info(f"Detected sequential counter pattern at position {pos}, increment {increments[0]}")
                return result
        
        # Check for more complex patterns
        # (Could add more sophisticated pattern detection here)
        
        return result
    
    def attempt_known_cipher_attack(self, payload_index: int, cipher_type: str = None) -> Dict[str, Any]:
        """
        Attempt to attack the payload using known vulnerabilities in common cipher implementations.
        
        Args:
            payload_index: Index of the payload to attack
            cipher_type: Known cipher type if available
            
        Returns:
            Dictionary with attack results
        """
        if payload_index < 0 or payload_index >= len(self.captured_payloads):
            raise ValueError(f"Invalid payload index: {payload_index}")
        
        if not CRYPTO_AVAILABLE:
            return {"success": False, "error": "Crypto library not available"}
        
        payload_entry = self.captured_payloads[payload_index]
        payload = payload_entry["data"]
        manufacturer = payload_entry["manufacturer"]
        
        # Determine cipher type
        if cipher_type is None:
            if manufacturer and manufacturer in KNOWN_PAYLOAD_PATTERNS:
                cipher_type = KNOWN_PAYLOAD_PATTERNS[manufacturer]["cipher_type"]
            else:
                cipher_type = "AES-128-CBC"  # Default guess
        
        self.logger.info(f"Attempting {cipher_type} attack on payload #{payload_index}")
        
        # Initialize results
        results = {
            "success": False,
            "cipher": cipher_type,
            "manufacturer": manufacturer,
            "payload_size": len(payload),
            "attempts": 0,
            "vulnerabilities": []
        }
        
        # Extract IV if available
        iv = payload_entry.get("iv")
        if iv is None and manufacturer:
            iv = self.extract_iv(payload, manufacturer)
        
        if iv:
            results["iv"] = binascii.hexlify(iv).decode()
            
            # Check for reused IV
            iv_reuse = self.check_iv_reuse(iv, payload_index)
            if iv_reuse:
                results["vulnerabilities"].append({
                    "type": "iv_reuse",
                    "severity": "critical",
                    "description": "Nonce/IV reuse detected across multiple payloads",
                    "details": {
                        "reused_with": iv_reuse,
                        "iv": binascii.hexlify(iv).decode()
                    }
                })
                
                # If we have IV reuse, we can perform known plaintext attacks
                # if we can guess the plaintext of one message
                if self.known_keys.get(manufacturer):
                    # If we have a known key, try to decrypt and verify 
                    # (this would be used in a real attack scenario)
                    key = self.known_keys[manufacturer]
                    
                    try:
                        if "AES" in cipher_type and "CBC" in cipher_type:
                            cipher = AES.new(key, AES.MODE_CBC, iv)
                        elif "AES" in cipher_type and "ECB" in cipher_type:
                            cipher = AES.new(key, AES.MODE_ECB)
                        
                        # Get the encrypted part
                        if manufacturer in KNOWN_PAYLOAD_PATTERNS:
                            pattern = KNOWN_PAYLOAD_PATTERNS[manufacturer]
                            start, end = pattern["payload_position"]
                            if end < 0:
                                end = len(payload) + end
                            
                            encrypted_part = payload[start:end]
                        else:
                            # Guess the encrypted part is everything after the IV
                            encrypted_part = payload[len(iv):]
                        
                        # Try to decrypt
                        decrypted = cipher.decrypt(encrypted_part)
                        
                        # Check if decryption looks valid (e.g., has valid padding)
                        try:
                            unpadded = unpad(decrypted, AES.block_size)
                            results["success"] = True
                            results["decrypted"] = binascii.hexlify(unpadded).decode()
                            self.logger.info(f"Successfully decrypted payload with known key")
                        except:
                            self.logger.debug("Decryption produced invalid padding")
                    except Exception as e:
                        self.logger.error(f"Error during decryption attempt: {e}")
        
        # Check for other common vulnerabilities
        if "ECB" in cipher_type:
            results["vulnerabilities"].append({
                "type": "ecb_mode",
                "severity": "high",
                "description": "ECB mode does not provide semantic security and is vulnerable to pattern analysis"
            })
        
        # If we found vulnerabilities but couldn't successfully attack
        if results["vulnerabilities"] and not results["success"]:
            self.logger.warning(f"Found {len(results['vulnerabilities'])} vulnerabilities but could not complete attack")
        
        return results
    
    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of all analyzed payloads.
        
        Returns:
            Dictionary with summary information
        """
        analyzed_count = sum(1 for p in self.captured_payloads if p["analyzed"])
        
        # Count vulnerabilities by type and severity
        vulnerabilities = defaultdict(int)
        severity_counts = defaultdict(int)
        manufacturers = defaultdict(int)
        
        for payload in self.captured_payloads:
            if not payload["analyzed"]:
                continue
                
            if payload["manufacturer"]:
                manufacturers[payload["manufacturer"]] += 1
                
            for vuln in payload.get("vulnerabilities", []):
                vuln_type = vuln.get("type", "unknown")
                vulnerabilities[vuln_type] += 1
                
                severity = vuln.get("severity", "low")
                severity_counts[severity] += 1
        
        # Overall security assessment
        critical_vulns = severity_counts.get("critical", 0)
        high_vulns = severity_counts.get("high", 0)
        medium_vulns = severity_counts.get("medium", 0)
        
        if critical_vulns > 0:
            security_rating = 1  # Critical security issues
            assessment = "Critical security vulnerabilities detected! Immediate action recommended."
        elif high_vulns > 0:
            security_rating = min(3, 4 - high_vulns)  # 1-3 depending on count
            assessment = "High security vulnerabilities detected. System is likely vulnerable."
        elif medium_vulns > 0:
            security_rating = min(5, 6 - medium_vulns)  # 4-5 depending on count
            assessment = "Medium security vulnerabilities detected. Security improvements recommended."
        elif severity_counts.get("low", 0) > 0:
            security_rating = 7  # Minor issues
            assessment = "Minor security issues detected. System is relatively secure."
        else:
            security_rating = 9  # No issues found
            assessment = "No significant vulnerabilities detected. System appears secure."
        
        # IV reuse is especially severe
        if self.nonce_reuse_detected:
            security_rating = 1
            assessment = "CRITICAL: IV/nonce reuse detected! System is fundamentally insecure."
        
        # Prepare report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_payloads": len(self.captured_payloads),
            "analyzed_payloads": analyzed_count,
            "manufacturers_detected": dict(manufacturers),
            "vulnerability_summary": {
                "by_type": dict(vulnerabilities),
                "by_severity": dict(severity_counts)
            },
            "security_rating": security_rating,  # 1-10 scale (10 being most secure)
            "assessment": assessment,
            "recommendations": self.generate_recommendations(vulnerabilities, severity_counts)
        }
        
        return report
    
    def generate_recommendations(self, vulnerabilities: Dict[str, int], 
                                severity_counts: Dict[str, int]) -> List[str]:
        """
        Generate specific security recommendations based on discovered vulnerabilities.
        
        Args:
            vulnerabilities: Count of vulnerabilities by type
            severity_counts: Count of vulnerabilities by severity
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # IV/nonce reuse recommendations
        if vulnerabilities.get("iv_reuse", 0) > 0:
            recommendations.append(
                "CRITICAL: Implement proper IV/nonce management. Each transmission must use a unique IV."
            )
            recommendations.append(
                "Consider using a secure counter or random number generator for IVs."
            )
        
        # ECB mode recommendations
        if vulnerabilities.get("ecb_mode", 0) > 0 or vulnerabilities.get("repeating_blocks", 0) > 0:
            recommendations.append(
                "Switch from ECB mode to a secure authenticated encryption mode like GCM or CCM."
            )
            recommendations.append(
                "If CBC mode is required, ensure IVs are random and unique for each message."
            )
        
        # Low entropy recommendations
        if vulnerabilities.get("low_entropy", 0) > 0 or vulnerabilities.get("suspicious_entropy", 0) > 0:
            recommendations.append(
                "Implement stronger encryption with proper key management."
            )
            recommendations.append(
                "Ensure the entire payload is properly encrypted, not just parts of it."
            )
        
        # Weak key management recommendations
        if vulnerabilities.get("weak_key_management", 0) > 0:
            recommendations.append(
                "Implement secure key derivation using industry standard methods like PBKDF2 or Argon2."
            )
            recommendations.append(
                "Avoid predictable or sequential key/IV generation patterns."
            )
        
        # Replay protection recommendations
        if vulnerabilities.get("replay_attack", 0) > 0:
            recommendations.append(
                "Implement proper replay protection using sequence counters or timestamps."
            )
            recommendations.append(
                "Consider adding a challenge-response protocol for critical operations."
            )
        
        # General recommendations
        if severity_counts.get("high", 0) > 0 or severity_counts.get("critical", 0) > 0:
            recommendations.append(
                "Conduct a complete security review and penetration testing of the system."
            )
            recommendations.append(
                "Consider upgrading to newer, more secure models with improved encryption."
            )
        
        return recommendations


def main():
    """Main function for testing the module functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cryptographic Analysis Tool for Garage Door Security')
    parser.add_argument('-f', '--file', type=str, help='Binary payload file to analyze')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-t', '--test', action='store_true', help='Run test suite')
    parser.add_argument('-r', '--replay', type=str, help='Test replay protection against device address')
    parser.add_argument('-o', '--output', type=str, help='Output file for analysis results')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CryptoAnalyzer(debug_mode=args.debug)
    
    if args.test:
        # Run built-in tests
        print("Running test suite...")
        
        # Test with simulated payloads
        for i in range(3):
            # Create test payloads with different characteristics
            if i == 0:
                # Good encryption example (high entropy, no patterns)
                import os
                random_payload = os.urandom(64)
                analyzer.add_payload(random_payload, "test_good")
            elif i == 1:
                # Bad encryption example (ECB mode with repeating blocks)
                bad_payload = b'\xAA\xBB' + (b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F\x10' * 3)
                analyzer.add_payload(bad_payload, "test_bad_ecb")
            elif i == 2:
                # IV reuse example
                header = b'\x55\x01\x00'
                iv = b'\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xAA\xBB\xCC\xDD\xEE\xFF'
                payload = b'\xAA\xBB\xCC\xDD\xEE\xFF\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99'
                reused_iv_payload = header + iv + payload
                analyzer.add_payload(reused_iv_payload, "test_iv_reuse")
        
        # Analyze all payloads
        for i in range(len(analyzer.captured_payloads)):
            analyzer.analyze_payload(i)
        
        # Generate summary report
        report = analyzer.get_summary_report()
        
        print("\nAnalysis completed. Summary:")
        print(f"Total payloads analyzed: {report['analyzed_payloads']}")
        print(f"Security rating: {report['security_rating']}/10")
        print(f"Assessment: {report['assessment']}")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"- {rec}")
    
    elif args.file:
        # Analyze file
        try:
            index = analyzer.load_payload_from_file(args.file)
            results = analyzer.analyze_payload(index)
            
            print(f"\nAnalysis of {args.file}:")
            print(f"Payload size: {results['payload_size']} bytes")
            print(f"Entropy: {results['entropy']:.2f}")
            
            if results['manufacturer']:
                print(f"Identified manufacturer: {results['manufacturer']}")
            
            if results['repeating_blocks']['count'] > 0:
                print(f"Warning: {results['repeating_blocks']['count']} repeating blocks detected!")
            
            if results['vulnerabilities']:
                print("\nVulnerabilities detected:")
                for vuln in results['vulnerabilities']:
                    print(f"- [{vuln['severity'].upper()}] {vuln['description']}")
            else:
                print("\nNo obvious vulnerabilities detected.")
            
            # Save results if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nDetailed results saved to {args.output}")
                
        except Exception as e:
            print(f"Error analyzing file: {e}")
    
    elif args.replay:
        # Test replay protection
        if not args.file:
            print("Error: Replay test requires a payload file (-f)")
            return
        
        try:
            # Load payload
            index = analyzer.load_payload_from_file(args.file)
            payload = analyzer.captured_payloads[index]["data"]
            
            # Run replay test
            results = analyzer.test_replay_protection(args.replay, payload)
            
            print("\nReplay protection test results:")
            print(f"Target device: {results['device']}")
            print(f"Conclusion: {results['conclusion']}")
            
            # Save results if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nDetailed results saved to {args.output}")
                
        except Exception as e:
            print(f"Error in replay test: {e}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()