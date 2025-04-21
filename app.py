#!/usr/bin/env python3
"""
Web Interface for Garage Door Signal Analyzer with Frequency Oscillation

This script provides a web interface for analyzing and experimenting with garage
door rolling code systems through frequency analysis and signal manipulation.
For educational and security research purposes only.
"""

import os
import logging
import tempfile
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from frequency_handler import FrequencyHandler
from signal_processor import SignalProcessor
from rolling_code import RollingCodeAnalyzer
from code_predictor import CodePredictor
from utils import setup_logging, format_binary, print_legal_disclaimer

# Set up logging
setup_logging(logging.INFO)
print_legal_disclaimer()

# Create the Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", os.urandom(24))

# Initialize global components
signal_processor = SignalProcessor(sample_rate=2e6)
frequency_handler = FrequencyHandler(center_freq=315e6)
code_analyzer = RollingCodeAnalyzer()
code_predictor = CodePredictor()

# Create a directory for temporary files
temp_dir = os.path.join(tempfile.gettempdir(), 'garage_analyzer')
os.makedirs(temp_dir, exist_ok=True)

@app.route('/')
def index():
    """Render the main interface."""
    return render_template('index.html', 
                         frequencies=list(frequency_handler.common_frequencies.keys()))

@app.route('/capture', methods=['POST'])
def capture():
    """Capture a signal and extract potential codes."""
    try:
        # Get parameters from form
        frequency = float(request.form.get('frequency', 315)) * 1e6
        duration = int(request.form.get('duration', 10))
        
        # Set frequency
        frequency_handler.set_frequency(frequency)
        
        # Capture signal
        signal_data = signal_processor.capture_signal(frequency_handler, duration)
        
        # Save captured signal
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        signal_file = os.path.join(temp_dir, f'signal_{timestamp}.dat')
        signal_processor.save_signal(signal_data, signal_file)
        
        # Extract codes
        codes = code_analyzer.extract_codes(signal_data)
        formatted_codes = [format_binary(code) for code in codes]
        
        # Generate signal plots
        time_plot, freq_plot = generate_signal_plots(signal_data)
        
        return jsonify({
            'status': 'success',
            'message': f'Captured signal at {frequency/1e6} MHz for {duration} seconds',
            'codes': formatted_codes,
            'signal_file': signal_file,
            'time_plot': time_plot,
            'freq_plot': freq_plot
        })
    
    except Exception as e:
        logging.error(f"Error during capture: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Capture failed: {str(e)}'
        }), 500

@app.route('/oscillate', methods=['POST'])
def oscillate():
    """Generate a frequency oscillation signal."""
    try:
        # Get parameters from form
        frequency = float(request.form.get('frequency', 315)) * 1e6
        bandwidth = float(request.form.get('bandwidth', 0.2)) * 1e6
        oscillation_rate = float(request.form.get('rate', 10))
        duration = float(request.form.get('duration', 5))
        
        # Generate oscillating signal
        signal = frequency_handler.oscillate_frequency(
            center_freq=frequency,
            bandwidth=bandwidth,
            oscillation_rate=oscillation_rate,
            duration=duration,
            sample_rate=signal_processor.sample_rate
        )
        
        # Save generated signal
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        signal_file = os.path.join(temp_dir, f'oscillation_{timestamp}.dat')
        signal_processor.save_signal(signal, signal_file)
        
        # Generate signal plots
        time_plot, freq_plot = generate_signal_plots(signal)
        
        return jsonify({
            'status': 'success',
            'message': f'Generated oscillation signal at {frequency/1e6} MHz',
            'signal_file': signal_file,
            'time_plot': time_plot,
            'freq_plot': freq_plot
        })
    
    except Exception as e:
        logging.error(f"Error during oscillation: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Oscillation failed: {str(e)}'
        }), 500

@app.route('/sweep', methods=['POST'])
def sweep():
    """Generate a frequency sweep signal."""
    try:
        # Get parameters from form
        center_freq = float(request.form.get('frequency', 315)) * 1e6
        sweep_range = float(request.form.get('range', 10)) * 1e6
        start_freq = center_freq - sweep_range/2
        end_freq = center_freq + sweep_range/2
        duration = float(request.form.get('duration', 5))
        
        # Generate sweep signal
        signal = frequency_handler.frequency_sweep(
            start_freq=start_freq,
            end_freq=end_freq,
            sweep_time=duration,
            sample_rate=signal_processor.sample_rate
        )
        
        # Save generated signal
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        signal_file = os.path.join(temp_dir, f'sweep_{timestamp}.dat')
        signal_processor.save_signal(signal, signal_file)
        
        # Generate signal plots
        time_plot, freq_plot = generate_signal_plots(signal)
        
        return jsonify({
            'status': 'success',
            'message': f'Generated sweep from {start_freq/1e6} to {end_freq/1e6} MHz',
            'signal_file': signal_file,
            'time_plot': time_plot,
            'freq_plot': freq_plot
        })
    
    except Exception as e:
        logging.error(f"Error during sweep: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Sweep failed: {str(e)}'
        }), 500

@app.route('/attack', methods=['POST'])
def attack():
    """Perform a frequency oscillation attack."""
    try:
        # Get parameters from form
        frequency = float(request.form.get('frequency', 315)) * 1e6
        pattern = request.form.get('pattern', 'sinusoidal')
        code = request.form.get('code', '')
        duration = float(request.form.get('duration', 30))
        
        # If no code provided, use a random one
        if not code:
            code = ''.join(np.random.choice(['0', '1']) for _ in range(32))
        
        # Validate code format
        code = code.replace(' ', '')
        if not all(bit in '01' for bit in code):
            return jsonify({
                'status': 'error',
                'message': 'Invalid code format. Must be binary (0s and 1s)'
            }), 400
        
        # Perform attack simulation
        result = frequency_handler.frequency_oscillation_attack(
            target_freq=frequency,
            code_list=[code], 
            oscillation_pattern=pattern,
            attack_duration=duration
        )
        
        status = 'Possible success' if result else 'No success detected'
        
        return jsonify({
            'status': 'success',
            'message': f'Attack completed: {status}',
            'result': result,
            'code_used': format_binary(code)
        })
    
    except Exception as e:
        logging.error(f"Error during attack: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Attack failed: {str(e)}'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict next rolling codes based on observed codes."""
    try:
        # Get codes from form
        observed_codes = request.form.getlist('codes[]')
        
        # Clean and validate codes
        cleaned_codes = []
        for code in observed_codes:
            clean_code = code.replace(' ', '')
            if all(bit in '01' for bit in clean_code):
                cleaned_codes.append(clean_code)
        
        if len(cleaned_codes) < 2:
            return jsonify({
                'status': 'error',
                'message': 'Need at least 2 valid codes for prediction'
            }), 400
        
        # Analyze code sequence
        analysis = code_predictor.analyze_code_sequence(cleaned_codes)
        
        # Predict next codes
        predictions = code_predictor.predict_next_codes(cleaned_codes)
        formatted_predictions = [format_binary(code) for code in predictions]
        
        return jsonify({
            'status': 'success',
            'message': f'Generated {len(predictions)} predictions',
            'predictions': formatted_predictions,
            'analysis': analysis
        })
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

def generate_signal_plots(signal_data: np.ndarray) -> tuple:
    """Generate base64-encoded images of time and frequency domain plots."""
    try:
        # Time domain plot
        plt.figure(figsize=(10, 4))
        time = np.arange(len(signal_data)) / signal_processor.sample_rate
        plt.plot(time, np.abs(signal_data))
        plt.title('Signal Magnitude (Time Domain)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        time_plot = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Frequency domain plot
        plt.figure(figsize=(10, 4))
        spectrum = np.fft.fftshift(np.fft.fft(signal_data))
        freq = np.fft.fftshift(np.fft.fftfreq(len(signal_data), 1/signal_processor.sample_rate))
        plt.plot(freq / 1e6, 20 * np.log10(np.abs(spectrum) + 1e-10))
        plt.title('Signal Spectrum (Frequency Domain)')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (dB)')
        plt.grid(True)
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        freq_plot = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return time_plot, freq_plot
        
    except Exception as e:
        logging.error(f"Error generating plots: {e}")
        return "", ""

if __name__ == '__main__':
    # Create templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create basic HTML template if it doesn't exist
    index_template = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(index_template):
        with open(index_template, 'w') as f:
            f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Garage Door Signal Analyzer</title>
    <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .signal-plot {
            width: 100%;
            max-width: 800px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .code-display {
            font-family: monospace;
            padding: 10px;
            background-color: var(--bs-dark);
            border-radius: 5px;
            margin: 10px 0;
            overflow-x: auto;
        }
        .tab-content {
            padding: 20px;
            background-color: rgba(0, 0, 0, 0.2);
            border-radius: 5px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row">
            <div class="col-lg-12 text-center mb-4">
                <h1>Garage Door Signal Analyzer</h1>
                <p class="text-danger">For educational and security research purposes only.</p>
            </div>
        </div>
        
        <div class="row">
            <div class="col-lg-12">
                <ul class="nav nav-tabs" id="myTab" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="capture-tab" data-bs-toggle="tab" data-bs-target="#capture-tab-pane" type="button" role="tab">Signal Capture</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="oscillate-tab" data-bs-toggle="tab" data-bs-target="#oscillate-tab-pane" type="button" role="tab">Frequency Oscillation</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="sweep-tab" data-bs-toggle="tab" data-bs-target="#sweep-tab-pane" type="button" role="tab">Frequency Sweep</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="predict-tab" data-bs-toggle="tab" data-bs-target="#predict-tab-pane" type="button" role="tab">Code Prediction</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="attack-tab" data-bs-toggle="tab" data-bs-target="#attack-tab-pane" type="button" role="tab">Oscillation Attack</button>
                    </li>
                </ul>
                
                <div class="tab-content" id="myTabContent">
                    <!-- Capture Tab -->
                    <div class="tab-pane fade show active" id="capture-tab-pane" role="tabpanel" tabindex="0">
                        <div class="row">
                            <div class="col-lg-4">
                                <div class="card">
                                    <div class="card-header">Capture Settings</div>
                                    <div class="card-body">
                                        <form id="capture-form">
                                            <div class="mb-3">
                                                <label for="capture-frequency" class="form-label">Frequency (MHz)</label>
                                                <select class="form-select" id="capture-frequency" name="frequency">
                                                    {% for freq in frequencies %}
                                                    <option value="{{ freq.split('MHz')[0] }}">{{ freq }}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                            <div class="mb-3">
                                                <label for="capture-duration" class="form-label">Duration (seconds)</label>
                                                <input type="number" class="form-control" id="capture-duration" name="duration" value="10" min="1" max="60">
                                            </div>
                                            <button type="submit" class="btn btn-primary" id="capture-button">Capture Signal</button>
                                        </form>
                                    </div>
                                </div>
                            </div>
                            <div class="col-lg-8">
                                <div id="capture-results" style="display: none;">
                                    <h4>Capture Results</h4>
                                    <div id="capture-message" class="alert alert-info"></div>
                                    
                                    <h5>Signal Plots</h5>
                                    <div class="row">
                                        <div class="col-lg-12">
                                            <img id="time-plot" class="signal-plot" alt="Time Domain Plot">
                                            <img id="freq-plot" class="signal-plot" alt="Frequency Domain Plot">
                                        </div>
                                    </div>
                                    
                                    <h5>Extracted Codes</h5>
                                    <div id="extracted-codes"></div>
                                </div>
                                <div id="capture-loading" style="display: none;">
                                    <div class="d-flex justify-content-center">
                                        <div class="spinner-border text-primary" role="status">
                                            <span class="visually-hidden">Loading...</span>
                                        </div>
                                    </div>
                                    <p class="text-center mt-2">Capturing signal...</p>
                                </div>
                                <div id="capture-initial-message">
                                    <div class="alert alert-secondary">
                                        Use the form to capture radio signals and extract rolling codes.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Oscillation Tab -->
                    <div class="tab-pane fade" id="oscillate-tab-pane" role="tabpanel" tabindex="0">
                        <div class="row">
                            <div class="col-lg-4">
                                <div class="card">
                                    <div class="card-header">Oscillation Settings</div>
                                    <div class="card-body">
                                        <form id="oscillate-form">
                                            <div class="mb-3">
                                                <label for="oscillate-frequency" class="form-label">Center Frequency (MHz)</label>
                                                <select class="form-select" id="oscillate-frequency" name="frequency">
                                                    {% for freq in frequencies %}
                                                    <option value="{{ freq.split('MHz')[0] }}">{{ freq }}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                            <div class="mb-3">
                                                <label for="oscillate-bandwidth" class="form-label">Bandwidth (MHz)</label>
                                                <input type="number" class="form-control" id="oscillate-bandwidth" name="bandwidth" value="0.2" min="0.01" max="10" step="0.01">
                                            </div>
                                            <div class="mb-3">
                                                <label for="oscillate-rate" class="form-label">Oscillation Rate (Hz)</label>
                                                <input type="number" class="form-control" id="oscillate-rate" name="rate" value="10" min="0.1" max="100" step="0.1">
                                            </div>
                                            <div class="mb-3">
                                                <label for="oscillate-duration" class="form-label">Duration (seconds)</label>
                                                <input type="number" class="form-control" id="oscillate-duration" name="duration" value="5" min="1" max="30">
                                            </div>
                                            <button type="submit" class="btn btn-primary" id="oscillate-button">Generate Oscillation</button>
                                        </form>
                                    </div>
                                </div>
                            </div>
                            <div class="col-lg-8">
                                <div id="oscillate-results" style="display: none;">
                                    <h4>Oscillation Results</h4>
                                    <div id="oscillate-message" class="alert alert-info"></div>
                                    
                                    <h5>Signal Plots</h5>
                                    <div class="row">
                                        <div class="col-lg-12">
                                            <img id="oscillate-time-plot" class="signal-plot" alt="Time Domain Plot">
                                            <img id="oscillate-freq-plot" class="signal-plot" alt="Frequency Domain Plot">
                                        </div>
                                    </div>
                                </div>
                                <div id="oscillate-loading" style="display: none;">
                                    <div class="d-flex justify-content-center">
                                        <div class="spinner-border text-primary" role="status">
                                            <span class="visually-hidden">Loading...</span>
                                        </div>
                                    </div>
                                    <p class="text-center mt-2">Generating oscillation signal...</p>
                                </div>
                                <div id="oscillate-initial-message">
                                    <div class="alert alert-secondary">
                                        Generate a signal that oscillates around a center frequency.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Sweep Tab -->
                    <div class="tab-pane fade" id="sweep-tab-pane" role="tabpanel" tabindex="0">
                        <div class="row">
                            <div class="col-lg-4">
                                <div class="card">
                                    <div class="card-header">Sweep Settings</div>
                                    <div class="card-body">
                                        <form id="sweep-form">
                                            <div class="mb-3">
                                                <label for="sweep-frequency" class="form-label">Center Frequency (MHz)</label>
                                                <select class="form-select" id="sweep-frequency" name="frequency">
                                                    {% for freq in frequencies %}
                                                    <option value="{{ freq.split('MHz')[0] }}">{{ freq }}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                            <div class="mb-3">
                                                <label for="sweep-range" class="form-label">Sweep Range (MHz)</label>
                                                <input type="number" class="form-control" id="sweep-range" name="range" value="10" min="0.1" max="50" step="0.1">
                                            </div>
                                            <div class="mb-3">
                                                <label for="sweep-duration" class="form-label">Duration (seconds)</label>
                                                <input type="number" class="form-control" id="sweep-duration" name="duration" value="5" min="1" max="30">
                                            </div>
                                            <button type="submit" class="btn btn-primary" id="sweep-button">Generate Sweep</button>
                                        </form>
                                    </div>
                                </div>
                            </div>
                            <div class="col-lg-8">
                                <div id="sweep-results" style="display: none;">
                                    <h4>Sweep Results</h4>
                                    <div id="sweep-message" class="alert alert-info"></div>
                                    
                                    <h5>Signal Plots</h5>
                                    <div class="row">
                                        <div class="col-lg-12">
                                            <img id="sweep-time-plot" class="signal-plot" alt="Time Domain Plot">
                                            <img id="sweep-freq-plot" class="signal-plot" alt="Frequency Domain Plot">
                                        </div>
                                    </div>
                                </div>
                                <div id="sweep-loading" style="display: none;">
                                    <div class="d-flex justify-content-center">
                                        <div class="spinner-border text-primary" role="status">
                                            <span class="visually-hidden">Loading...</span>
                                        </div>
                                    </div>
                                    <p class="text-center mt-2">Generating sweep signal...</p>
                                </div>
                                <div id="sweep-initial-message">
                                    <div class="alert alert-secondary">
                                        Generate a linear frequency sweep (chirp) signal.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Predict Tab -->
                    <div class="tab-pane fade" id="predict-tab-pane" role="tabpanel" tabindex="0">
                        <div class="row">
                            <div class="col-lg-4">
                                <div class="card">
                                    <div class="card-header">Prediction Settings</div>
                                    <div class="card-body">
                                        <form id="predict-form">
                                            <div class="mb-3">
                                                <label class="form-label">Observed Codes</label>
                                                <div id="observed-codes-container">
                                                    <div class="input-group mb-2">
                                                        <input type="text" class="form-control code-input" name="codes[]" placeholder="Enter binary code (e.g., 10101010...)">
                                                    </div>
                                                    <div class="input-group mb-2">
                                                        <input type="text" class="form-control code-input" name="codes[]" placeholder="Enter binary code (e.g., 10101010...)">
                                                    </div>
                                                </div>
                                                <button type="button" class="btn btn-sm btn-secondary" id="add-code-btn">Add Code</button>
                                            </div>
                                            <button type="submit" class="btn btn-primary" id="predict-button">Predict Next Codes</button>
                                        </form>
                                    </div>
                                </div>
                            </div>
                            <div class="col-lg-8">
                                <div id="predict-results" style="display: none;">
                                    <h4>Prediction Results</h4>
                                    <div id="predict-message" class="alert alert-info"></div>
                                    
                                    <h5>Analysis</h5>
                                    <div id="code-analysis" class="code-display"></div>
                                    
                                    <h5>Predicted Codes</h5>
                                    <div id="predicted-codes"></div>
                                </div>
                                <div id="predict-loading" style="display: none;">
                                    <div class="d-flex justify-content-center">
                                        <div class="spinner-border text-primary" role="status">
                                            <span class="visually-hidden">Loading...</span>
                                        </div>
                                    </div>
                                    <p class="text-center mt-2">Predicting next codes...</p>
                                </div>
                                <div id="predict-initial-message">
                                    <div class="alert alert-secondary">
                                        Enter observed rolling codes to predict potential next codes.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Attack Tab -->
                    <div class="tab-pane fade" id="attack-tab-pane" role="tabpanel" tabindex="0">
                        <div class="row">
                            <div class="col-lg-4">
                                <div class="card">
                                    <div class="card-header">Attack Settings</div>
                                    <div class="card-body">
                                        <form id="attack-form">
                                            <div class="mb-3">
                                                <label for="attack-frequency" class="form-label">Target Frequency (MHz)</label>
                                                <select class="form-select" id="attack-frequency" name="frequency">
                                                    {% for freq in frequencies %}
                                                    <option value="{{ freq.split('MHz')[0] }}">{{ freq }}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                            <div class="mb-3">
                                                <label for="attack-pattern" class="form-label">Oscillation Pattern</label>
                                                <select class="form-select" id="attack-pattern" name="pattern">
                                                    <option value="linear">Linear</option>
                                                    <option value="sinusoidal" selected>Sinusoidal</option>
                                                    <option value="random">Random</option>
                                                </select>
                                            </div>
                                            <div class="mb-3">
                                                <label for="attack-code" class="form-label">Code to Transmit (optional)</label>
                                                <input type="text" class="form-control" id="attack-code" name="code" placeholder="Enter binary code or leave empty for random">
                                            </div>
                                            <div class="mb-3">
                                                <label for="attack-duration" class="form-label">Duration (seconds)</label>
                                                <input type="number" class="form-control" id="attack-duration" name="duration" value="30" min="5" max="120">
                                            </div>
                                            <button type="submit" class="btn btn-danger" id="attack-button">Simulate Attack</button>
                                        </form>
                                    </div>
                                </div>
                            </div>
                            <div class="col-lg-8">
                                <div class="alert alert-danger">
                                    <strong>WARNING:</strong> This is a simulation for educational purposes only. Actual attacks against systems without permission is illegal and unethical.
                                </div>
                                
                                <div id="attack-results" style="display: none;">
                                    <h4>Attack Results</h4>
                                    <div id="attack-message" class="alert alert-info"></div>
                                    
                                    <h5>Code Used</h5>
                                    <div id="attack-code-used" class="code-display"></div>
                                </div>
                                <div id="attack-loading" style="display: none;">
                                    <div class="d-flex justify-content-center">
                                        <div class="spinner-border text-danger" role="status">
                                            <span class="visually-hidden">Loading...</span>
                                        </div>
                                    </div>
                                    <p class="text-center mt-2">Simulating frequency oscillation attack...</p>
                                </div>
                                <div id="attack-initial-message">
                                    <div class="alert alert-secondary">
                                        Simulate a frequency oscillation attack to test rolling code security.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Capture form
            document.getElementById('capture-form').addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Show loading
                document.getElementById('capture-initial-message').style.display = 'none';
                document.getElementById('capture-results').style.display = 'none';
                document.getElementById('capture-loading').style.display = 'block';
                
                const formData = new FormData(this);
                
                fetch('/capture', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('capture-loading').style.display = 'none';
                    document.getElementById('capture-results').style.display = 'block';
                    
                    if (data.status === 'success') {
                        document.getElementById('capture-message').className = 'alert alert-success';
                        document.getElementById('capture-message').textContent = data.message;
                        
                        // Display plots
                        document.getElementById('time-plot').src = 'data:image/png;base64,' + data.time_plot;
                        document.getElementById('freq-plot').src = 'data:image/png;base64,' + data.freq_plot;
                        
                        // Display codes
                        const codesContainer = document.getElementById('extracted-codes');
                        codesContainer.innerHTML = '';
                        
                        if (data.codes.length === 0) {
                            codesContainer.innerHTML = '<div class="alert alert-warning">No codes extracted</div>';
                        } else {
                            data.codes.forEach((code, index) => {
                                const codeElem = document.createElement('div');
                                codeElem.className = 'code-display';
                                codeElem.textContent = `Code ${index + 1}: ${code}`;
                                codesContainer.appendChild(codeElem);
                            });
                        }
                    } else {
                        document.getElementById('capture-message').className = 'alert alert-danger';
                        document.getElementById('capture-message').textContent = data.message;
                    }
                })
                .catch(error => {
                    document.getElementById('capture-loading').style.display = 'none';
                    document.getElementById('capture-results').style.display = 'block';
                    document.getElementById('capture-message').className = 'alert alert-danger';
                    document.getElementById('capture-message').textContent = 'Error: ' + error.message;
                });
            });
            
            // Oscillate form
            document.getElementById('oscillate-form').addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Show loading
                document.getElementById('oscillate-initial-message').style.display = 'none';
                document.getElementById('oscillate-results').style.display = 'none';
                document.getElementById('oscillate-loading').style.display = 'block';
                
                const formData = new FormData(this);
                
                fetch('/oscillate', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('oscillate-loading').style.display = 'none';
                    document.getElementById('oscillate-results').style.display = 'block';
                    
                    if (data.status === 'success') {
                        document.getElementById('oscillate-message').className = 'alert alert-success';
                        document.getElementById('oscillate-message').textContent = data.message;
                        
                        // Display plots
                        document.getElementById('oscillate-time-plot').src = 'data:image/png;base64,' + data.time_plot;
                        document.getElementById('oscillate-freq-plot').src = 'data:image/png;base64,' + data.freq_plot;
                    } else {
                        document.getElementById('oscillate-message').className = 'alert alert-danger';
                        document.getElementById('oscillate-message').textContent = data.message;
                    }
                })
                .catch(error => {
                    document.getElementById('oscillate-loading').style.display = 'none';
                    document.getElementById('oscillate-results').style.display = 'block';
                    document.getElementById('oscillate-message').className = 'alert alert-danger';
                    document.getElementById('oscillate-message').textContent = 'Error: ' + error.message;
                });
            });
            
            // Sweep form
            document.getElementById('sweep-form').addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Show loading
                document.getElementById('sweep-initial-message').style.display = 'none';
                document.getElementById('sweep-results').style.display = 'none';
                document.getElementById('sweep-loading').style.display = 'block';
                
                const formData = new FormData(this);
                
                fetch('/sweep', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('sweep-loading').style.display = 'none';
                    document.getElementById('sweep-results').style.display = 'block';
                    
                    if (data.status === 'success') {
                        document.getElementById('sweep-message').className = 'alert alert-success';
                        document.getElementById('sweep-message').textContent = data.message;
                        
                        // Display plots
                        document.getElementById('sweep-time-plot').src = 'data:image/png;base64,' + data.time_plot;
                        document.getElementById('sweep-freq-plot').src = 'data:image/png;base64,' + data.freq_plot;
                    } else {
                        document.getElementById('sweep-message').className = 'alert alert-danger';
                        document.getElementById('sweep-message').textContent = data.message;
                    }
                })
                .catch(error => {
                    document.getElementById('sweep-loading').style.display = 'none';
                    document.getElementById('sweep-results').style.display = 'block';
                    document.getElementById('sweep-message').className = 'alert alert-danger';
                    document.getElementById('sweep-message').textContent = 'Error: ' + error.message;
                });
            });
            
            // Add code button
            document.getElementById('add-code-btn').addEventListener('click', function() {
                const container = document.getElementById('observed-codes-container');
                const inputGroup = document.createElement('div');
                inputGroup.className = 'input-group mb-2';
                inputGroup.innerHTML = `
                    <input type="text" class="form-control code-input" name="codes[]" placeholder="Enter binary code (e.g., 10101010...)">
                    <button class="btn btn-outline-danger remove-code-btn" type="button">×</button>
                `;
                container.appendChild(inputGroup);
                
                // Add event listener to the remove button
                inputGroup.querySelector('.remove-code-btn').addEventListener('click', function() {
                    container.removeChild(inputGroup);
                });
            });
            
            // Predict form
            document.getElementById('predict-form').addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Show loading
                document.getElementById('predict-initial-message').style.display = 'none';
                document.getElementById('predict-results').style.display = 'none';
                document.getElementById('predict-loading').style.display = 'block';
                
                const formData = new FormData(this);
                
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('predict-loading').style.display = 'none';
                    document.getElementById('predict-results').style.display = 'block';
                    
                    if (data.status === 'success') {
                        document.getElementById('predict-message').className = 'alert alert-success';
                        document.getElementById('predict-message').textContent = data.message;
                        
                        // Display analysis
                        const analysisContainer = document.getElementById('code-analysis');
                        let analysisHTML = '';
                        for (const [key, value] of Object.entries(data.analysis)) {
                            analysisHTML += `<div><strong>${key}:</strong> ${JSON.stringify(value)}</div>`;
                        }
                        analysisContainer.innerHTML = analysisHTML;
                        
                        // Display predicted codes
                        const codesContainer = document.getElementById('predicted-codes');
                        codesContainer.innerHTML = '';
                        
                        if (data.predictions.length === 0) {
                            codesContainer.innerHTML = '<div class="alert alert-warning">No predictions generated</div>';
                        } else {
                            data.predictions.forEach((code, index) => {
                                const codeElem = document.createElement('div');
                                codeElem.className = 'code-display';
                                codeElem.textContent = `Prediction ${index + 1}: ${code}`;
                                codesContainer.appendChild(codeElem);
                            });
                        }
                    } else {
                        document.getElementById('predict-message').className = 'alert alert-danger';
                        document.getElementById('predict-message').textContent = data.message;
                    }
                })
                .catch(error => {
                    document.getElementById('predict-loading').style.display = 'none';
                    document.getElementById('predict-results').style.display = 'block';
                    document.getElementById('predict-message').className = 'alert alert-danger';
                    document.getElementById('predict-message').textContent = 'Error: ' + error.message;
                });
            });
            
            // Attack form
            document.getElementById('attack-form').addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Show loading
                document.getElementById('attack-initial-message').style.display = 'none';
                document.getElementById('attack-results').style.display = 'none';
                document.getElementById('attack-loading').style.display = 'block';
                
                const formData = new FormData(this);
                
                fetch('/attack', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('attack-loading').style.display = 'none';
                    document.getElementById('attack-results').style.display = 'block';
                    
                    if (data.status === 'success') {
                        const alertClass = data.result ? 'alert-success' : 'alert-warning';
                        document.getElementById('attack-message').className = `alert ${alertClass}`;
                        document.getElementById('attack-message').textContent = data.message;
                        
                        // Display code used
                        document.getElementById('attack-code-used').textContent = data.code_used;
                    } else {
                        document.getElementById('attack-message').className = 'alert alert-danger';
                        document.getElementById('attack-message').textContent = data.message;
                    }
                })
                .catch(error => {
                    document.getElementById('attack-loading').style.display = 'none';
                    document.getElementById('attack-results').style.display = 'block';
                    document.getElementById('attack-message').className = 'alert alert-danger';
                    document.getElementById('attack-message').textContent = 'Error: ' + error.message;
                });
            });
        });
    </script>
</body>
</html>
            ''')
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)