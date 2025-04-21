# Garage Door Rolling Code Analyzer with HackRF

A Python tool for analyzing and bypassing garage door rolling code systems through frequency oscillation and signal manipulation. **For educational and security research purposes only.**

## IMPORTANT: HackRF Hardware Required

This tool **REQUIRES** physical HackRF hardware for most functionality:
- There are NO simulation modes or fallbacks available for RF operations
- Full functionality will not work without a connected HackRF One or compatible device
- All RF signal analysis and transmission depends on direct hardware interaction

### Limited Non-Hardware Functionality

The following limited functionality is available without HackRF hardware:
- Listing supported manufacturer information with `python list_manufacturers.py`
- Viewing supported frequencies and modulation types with `python main.py --list-manufacturers`
- All other operations require the physical HackRF device

## Disclaimer

This tool is provided for **EDUCATIONAL and SECURITY RESEARCH purposes ONLY**.

Use of this software to access garage door systems or other property without explicit permission from the owner is illegal and unethical. The authors and distributors of this software accept no liability for misuse of this tool.

By using this software, you agree to:
1. Only use it on systems you own or have explicit permission to test
2. Follow all applicable local, state, and federal laws
3. Use the tool responsibly and ethically

## Overview

This tool performs hardware-based analysis and manipulation of rolling code systems commonly used in garage door openers. It provides functionality for:

- Analyzing rolling code patterns from garage door systems
- Implementing algorithms to predict next codes
- Frequency oscillation for radio signals (with multiple patterns)
- Advanced frequency sweeping and hopping techniques
- Supporting common garage door frequencies (typically 300-400 MHz)
- Multiple signal modulation capabilities (OOK, FSK, PSK)
- Radio signal generation and analysis
- Rolling code prediction using multiple algorithms
- Command line interface for various operations

## Key Features

### Frequency Oscillation

The tool includes advanced frequency oscillation capabilities that can:
- Generate signals that oscillate around a target frequency
- Perform frequency sweeping across a bandwidth
- Create frequency hopping patterns
- Support various oscillation patterns (linear, sinusoidal, random)

### Signal Generation and Analysis

- Capture and analyze radio signals
- Extract potential rolling codes from signals
- Modulate signals with various encoding schemes
- Visualize signals in time and frequency domains

### Rolling Code Analysis and Prediction

- Extract rolling codes from captured signals
- Analyze code structure and patterns
- Predict potential next codes using multiple algorithms
- Attack various rolling code systems directly via hardware

## Requirements

- Python 3.x
- **HackRF One or compatible hardware device**
- Required packages:
  - hackrf (Python library for HackRF hardware access)
  - numpy
  - matplotlib
  - coloredlogs
  - requests
  - scipy
- C compiler (for the transmitter component)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/garage-code-analyzer.git
cd garage-code-analyzer

# Install HackRF drivers and libraries for your operating system
# (Follow HackRF documentation for your specific OS)

# Install required Python packages
pip install numpy matplotlib coloredlogs requests scipy

# Install the Python HackRF package
pip install hackrf

# Compile the transmitter (required for transmission)
./compile_transmitter.sh
```

## Usage

### Check Hardware Availability

Before running the main tool, you can check if your HackRF hardware is properly set up:

```bash
# Run the hardware checker utility
python check_hardware.py
```

This will:
- Verify if the HackRF Python package is installed
- Check if HackRF command-line tools are available
- Attempt to detect and connect to your HackRF device
- Provide troubleshooting recommendations if issues are found

### View Supported Manufacturers (No Hardware Required)

The following commands display information about supported garage door manufacturers without requiring HackRF hardware:

```bash
# List all supported manufacturers and their frequencies
python list_manufacturers.py

# Alternative method using main script
python main.py --list-manufacturers
```

### Basic Command-Line Operations (Requires HackRF Hardware)

All other operations require HackRF hardware to be connected:

```bash
# Analyze signals at a specific frequency (315 MHz)
python main.py -f 315.0 -m analyze -t 60

# Predict next codes from a captured file
python main.py -m predict -i captured_codes.dat

# Perform frequency oscillation attack on a specific manufacturer
python main.py -f 315.0 -m oscillate --manufacturer chamberlain -t 120

# Scan for active garage door frequencies
python main.py -m scan -v

# Transmit a specific code
python main.py -f 315.0 -m transmit -i code.txt
```

### Advanced Options

```bash
# Customize bandwidth, sample rate, and gain
python main.py -f 315.0 -m analyze --bandwidth 2.5 --sample-rate 4.0 --gain 25.0

# Specify a manufacturer for optimized parameters
python main.py -f 315.0 -m oscillate --manufacturer genie -t 60

# Save captured signal to file
python main.py -f 315.0 -m analyze -o captured_signal.dat

### FHSS (Frequency Hopping Spread Spectrum) Bypass

For garage door systems that use frequency hopping for increased security:

```bash
# Analyze an FHSS system at 433.92 MHz
python main.py -f 433.92 -m fhss -t 60

# Perform FHSS bypass with a file containing potential codes
python main.py -f 433.92 -m fhss -i codes.txt -t 120 --manufacturer chamberlain

# Standalone FHSS analysis with detailed output
python run_fhss_attack.py -f 433.92 -a -d 30

# Standalone FHSS attack with specific parameters
python run_fhss_attack.py -f 433.92 -b 4.0 -c 16 --codes codes.txt -d 60 --type sequential
```
