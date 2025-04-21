#!/usr/bin/env python3
"""
Utility functions for the Garage Door Rolling Code Analysis Tool.
"""

import logging
import coloredlogs
import sys
from typing import Optional, Dict, Any, List

def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
    """
    # Configure coloredlogs for prettier output
    field_styles = {
        'asctime': {'color': 'green'},
        'hostname': {'color': 'magenta'},
        'levelname': {'bold': True, 'color': 'black'},
        'name': {'color': 'blue'},
        'programname': {'color': 'cyan'}
    }
    
    level_styles = {
        'debug': {'color': 'green'},
        'info': {'color': 'blue'},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red'},
        'critical': {'bold': True, 'color': 'red'}
    }
    
    coloredlogs.install(
        level=level,
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        field_styles=field_styles,
        level_styles=level_styles,
        stream=sys.stdout
    )
    
    # Add file handler for logging to file
    file_handler = logging.FileHandler('garage_code_analyzer.log')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)
    
    logging.debug("Logging initialized")

def validate_frequency(frequency: float) -> bool:
    """
    Validate if the given frequency is within expected range for garage door openers.
    
    Args:
        frequency: Frequency in MHz
        
    Returns:
        True if valid, False otherwise
    """
    # Most garage door openers operate between 300-400 MHz
    if 300 <= frequency <= 400:
        return True
    
    logging.warning(f"Frequency {frequency} MHz is outside the typical garage door range (300-400 MHz)")
    return False

def format_binary(binary_str: str, chunk_size: int = 8) -> str:
    """
    Format a binary string into chunks for easier reading.
    
    Args:
        binary_str: Binary string
        chunk_size: Size of each chunk
        
    Returns:
        Formatted binary string
    """
    return ' '.join(binary_str[i:i+chunk_size] for i in range(0, len(binary_str), chunk_size))

def hex_to_binary(hex_str: str) -> str:
    """
    Convert hexadecimal string to binary.
    
    Args:
        hex_str: Hexadecimal string
        
    Returns:
        Binary string
    """
    try:
        value = int(hex_str, 16)
        binary = bin(value)[2:]  # Remove '0b' prefix
        return binary
    except ValueError:
        logging.error(f"Invalid hexadecimal string: {hex_str}")
        return ""

def binary_to_hex(binary_str: str) -> str:
    """
    Convert binary string to hexadecimal.
    
    Args:
        binary_str: Binary string
        
    Returns:
        Hexadecimal string
    """
    try:
        value = int(binary_str, 2)
        return hex(value)[2:]  # Remove '0x' prefix
    except ValueError:
        logging.error(f"Invalid binary string: {binary_str}")
        return ""

def print_results_table(data: List[Dict[str, Any]], title: Optional[str] = None) -> None:
    """
    Print a nicely formatted table of results.
    
    Args:
        data: List of dictionaries with data to display
        title: Optional table title
    """
    if not data:
        logging.warning("No data to display in table")
        return
        
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    
    # Get all column headers
    headers = set()
    for item in data:
        headers.update(item.keys())
    headers = sorted(headers)
    
    # Calculate column widths
    col_widths = {header: len(header) for header in headers}
    for item in data:
        for header in headers:
            if header in item:
                col_widths[header] = max(
                    col_widths[header], 
                    len(str(item[header]))
                )
    
    # Print header row
    header_row = " | ".join(
        header.ljust(col_widths[header]) for header in headers
    )
    print(header_row)
    print("-" * len(header_row))
    
    # Print data rows
    for item in data:
        row = " | ".join(
            str(item.get(header, "")).ljust(col_widths[header]) 
            for header in headers
        )
        print(row)
    
    print()

def print_legal_disclaimer() -> None:
    """Print legal disclaimer about tool usage."""
    disclaimer = """
    LEGAL DISCLAIMER
    ================
    This tool is provided for EDUCATIONAL and SECURITY RESEARCH purposes ONLY.
    
    Use of this software to access garage door systems or other property without
    explicit permission from the owner is illegal and unethical. The authors and
    distributors of this software accept no liability for misuse of this tool.
    
    By using this software, you agree to:
    1. Only use it on systems you own or have explicit permission to test
    2. Follow all applicable local, state, and federal laws
    3. Use the tool responsibly and ethically
    
    The primary purpose of this tool is to help security researchers and
    property owners understand the security implications of rolling code systems.
    """
    
    print(disclaimer)
    logging.info("Legal disclaimer displayed")
    
def print_banner() -> None:
    """Print a banner for the tool."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ██████╗  █████╗ ██████╗  █████╗  ██████╗ ███████╗      ║
    ║  ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██╔════╝ ██╔════╝      ║
    ║  ██║  ███╗███████║██████╔╝███████║██║  ███╗█████╗        ║
    ║  ██║   ██║██╔══██║██╔══██╗██╔══██║██║   ██║██╔══╝        ║
    ║  ╚██████╔╝██║  ██║██║  ██║██║  ██║╚██████╔╝███████╗      ║
    ║   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝      ║
    ║                                                           ║
    ║   ██████╗ ██████╗ ██████╗ ███████╗                       ║
    ║  ██╔════╝██╔═══██╗██╔══██╗██╔════╝                       ║
    ║  ██║     ██║   ██║██║  ██║█████╗                         ║
    ║  ██║     ██║   ██║██║  ██║██╔══╝                         ║
    ║  ╚██████╗╚██████╔╝██████╔╝███████╗                       ║
    ║   ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝                       ║
    ║                                                           ║
    ║   █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗███████╗║
    ║  ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝██╔════╝██╔════╝║
    ║  ███████║██╔██╗ ██║███████║██║   ╚████╔╝ ███████╗█████╗  ║
    ║  ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝  ╚════██║██╔══╝  ║
    ║  ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████║███████╗║
    ║  ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝║
    ║                                                           ║
    ║             GARAGE ROLLING CODE ANALYZER                  ║
    ║             For Educational Purposes Only                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)
    logging.debug("Banner displayed")
