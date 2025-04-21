#!/usr/bin/env python3
"""
Garage Door Security Analyzer GUI

A comprehensive graphical interface for analyzing and exploiting vulnerabilities
in garage door security systems, including both RF signal analysis and Bluetooth
proximity-based authentication bypasses.

This tool provides a unified interface to visualize signals, conduct attacks,
and analyze security vulnerabilities in modern garage door systems.

For educational and security research purposes only.
"""

import os
import sys
import time
import json
import logging
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

# Import PyQt for GUI
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox, QTextEdit, QSpinBox,
    QDoubleSpinBox, QGroupBox, QFormLayout, QSplitter, QFileDialog, QMessageBox,
    QProgressBar, QScrollArea, QFrame, QRadioButton, QButtonGroup, QGridLayout,
    QStatusBar, QAction, QMenu, QToolBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QSlider
)
from PyQt5.QtGui import QIcon, QColor, QPixmap, QPalette, QFont, QTextCursor
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, QThread

# Import PyQtGraph for signal visualization
import pyqtgraph as pg

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("GarageSecurityGUI")

# Handle conditional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Signal processing features will be limited.")

try:
    import bluepy.btle as btle
    BLUEPY_AVAILABLE = True
except ImportError:
    BLUEPY_AVAILABLE = False
    logger.warning("Bluepy module not available. BLE features will be disabled.")

try:
    import hackrf
    HACKRF_AVAILABLE = True
except ImportError:
    HACKRF_AVAILABLE = False
    logger.warning("HackRF module not available. RF features will be limited to playback/simulation.")

# Import our modules - handle gracefully if they're not available
try:
    from ble_proximity_bypass import BLEProximityBypasser, GARAGE_BLE_MANUFACTURERS
    BLE_MODULE_AVAILABLE = True
except ImportError:
    BLE_MODULE_AVAILABLE = False
    logger.warning("BLE proximity bypass module not available.")
    GARAGE_BLE_MANUFACTURERS = {}

try:
    from fhss_bypass import FHSSAnalyzer, FHSSBypass
    FHSS_MODULE_AVAILABLE = True
except ImportError:
    FHSS_MODULE_AVAILABLE = False
    logger.warning("FHSS bypass module not available.")

try:
    from manufacturer_info import MANUFACTURER_INFO
    MANUFACTURER_INFO_AVAILABLE = True
except ImportError:
    MANUFACTURER_INFO_AVAILABLE = False
    logger.warning("Manufacturer information module not available.")
    MANUFACTURER_INFO = {}


class LoggingTextEdit(QTextEdit):
    """Custom QTextEdit that can be used as a logging handler."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.NoWrap)
        font = QFont("Courier")
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)
        self.setMaximumBlockCount(5000)  # Limit to prevent memory issues
    
    def append_log(self, text, level=logging.INFO):
        """Append log text with appropriate color based on level."""
        color = "black"
        
        if level == logging.DEBUG:
            color = "gray"
        elif level == logging.WARNING:
            color = "orange"
        elif level == logging.ERROR:
            color = "red"
        elif level == logging.CRITICAL:
            color = "purple"
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.append(f'<font color="{color}">[{timestamp}] {text}</font>')
        
        # Autoscroll to bottom
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)


class WorkerThread(QThread):
    """Generic worker thread for running background tasks."""
    
    update_progress = pyqtSignal(int)
    update_status = pyqtSignal(str)
    update_result = pyqtSignal(object)
    task_completed = pyqtSignal(bool, str)
    
    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        self.result = None
    
    def run(self):
        try:
            self.result = self.task_func(*self.args, **self.kwargs)
            self.task_completed.emit(True, "Task completed successfully")
            self.update_result.emit(self.result)
        except Exception as e:
            self.task_completed.emit(False, str(e))
            logger.error(f"Error in worker thread: {e}")
            logger.debug(traceback.format_exc())


class HackRFSignalCapture(QThread):
    """Thread for capturing RF signals using HackRF."""
    
    update_progress = pyqtSignal(int)
    update_signal = pyqtSignal(object)
    capture_completed = pyqtSignal(bool, str)
    
    def __init__(self, frequency=315.0e6, sample_rate=2e6, gain=40, duration=5):
        super().__init__()
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.gain = gain
        self.duration = duration
        self.running = False
        self.data = None
    
    def run(self):
        if not HACKRF_AVAILABLE:
            self.capture_completed.emit(False, "HackRF module not available")
            return
        
        try:
            # Create HackRF device
            self.device = hackrf.HackRF()
            
            # Configure device
            self.device.set_freq(int(self.frequency))
            self.device.set_sample_rate(int(self.sample_rate))
            self.device.set_rx_gain(int(self.gain))
            
            # Start RX mode
            self.device.start_rx_mode()
            
            # Calculate number of samples to capture
            num_samples = int(self.sample_rate * self.duration)
            samples_per_update = int(self.sample_rate * 0.1)  # Update every 0.1 seconds
            
            # Capture data
            self.running = True
            self.data = np.array([], dtype=complex)
            
            captured_samples = 0
            while captured_samples < num_samples and self.running:
                # Read batch of samples
                batch = self.device.read_samples(min(samples_per_update, num_samples - captured_samples))
                self.data = np.append(self.data, batch)
                
                # Update progress
                captured_samples = len(self.data)
                progress = int(captured_samples / num_samples * 100)
                self.update_progress.emit(progress)
                
                # Emit signal data for live display
                self.update_signal.emit(self.data)
            
            # Stop RX mode
            self.device.stop_rx_mode()
            self.device.close()
            
            if self.running:
                self.capture_completed.emit(True, f"Captured {len(self.data)} samples")
            else:
                self.capture_completed.emit(False, "Capture was cancelled")
                
        except Exception as e:
            self.capture_completed.emit(False, str(e))
            logger.error(f"Error during signal capture: {e}")
            logger.debug(traceback.format_exc())
            
            # Cleanup
            try:
                if hasattr(self, 'device'):
                    self.device.stop_rx_mode()
                    self.device.close()
            except:
                pass
    
    def stop(self):
        """Stop the capture thread."""
        self.running = False


class BLEScanThread(QThread):
    """Thread for scanning Bluetooth LE devices."""
    
    update_devices = pyqtSignal(list)
    scan_completed = pyqtSignal(bool, str)
    
    def __init__(self, duration=10):
        super().__init__()
        self.duration = duration
        self.running = False
    
    def run(self):
        if not BLUEPY_AVAILABLE:
            self.scan_completed.emit(False, "Bluepy module not available")
            return
        
        try:
            # Create BLE scanner
            scanner = btle.Scanner()
            self.running = True
            
            # Scan for devices
            devices = scanner.scan(self.duration)
            
            # Process results
            found_devices = []
            for dev in devices:
                device_info = {
                    "address": dev.addr,
                    "name": "Unknown",
                    "rssi": dev.rssi,
                    "connectable": dev.connectable,
                    "manufacturer_data": None,
                    "services": []
                }
                
                # Try to get advertisement data
                for (adtype, desc, value) in dev.getScanData():
                    if adtype == 9:  # Complete Local Name
                        device_info["name"] = value
                    elif adtype == 255:  # Manufacturer Specific Data
                        device_info["manufacturer_data"] = value
                    elif adtype == 3 or adtype == 7:  # Service UUIDs
                        device_info["services"].append(value)
                
                found_devices.append(device_info)
            
            # Emit results
            self.update_devices.emit(found_devices)
            
            if self.running:
                self.scan_completed.emit(True, f"Found {len(found_devices)} devices")
            else:
                self.scan_completed.emit(False, "Scan was cancelled")
                
        except Exception as e:
            self.scan_completed.emit(False, str(e))
            logger.error(f"Error during BLE scan: {e}")
            logger.debug(traceback.format_exc())
    
    def stop(self):
        """Stop the scan thread."""
        self.running = False


class MainWindow(QMainWindow):
    """Main window for the Garage Door Security Analyzer GUI."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Garage Door Security Analyzer")
        self.setMinimumSize(1200, 800)
        
        # Setup variables
        self.signal_data = None
        self.captured_devices = []
        self.selected_device = None
        self.ble_bypasser = None
        self.hackrf_capture = None
        self.ble_scan = None
        
        # Create UI
        self.create_ui()
        self.setup_logging()
        
        # Initialize status
        self.check_hardware_status()
        
        # Setup update timer for UI refresh
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(500)  # Update every 500ms
    
    def create_ui(self):
        """Create the main user interface."""
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create RF tab
        self.rf_tab = QWidget()
        self.tabs.addTab(self.rf_tab, "RF Analysis")
        self.setup_rf_tab()
        
        # Create BLE tab
        self.ble_tab = QWidget()
        self.tabs.addTab(self.ble_tab, "BLE Proximity Bypass")
        self.setup_ble_tab()
        
        # Create FHSS tab
        self.fhss_tab = QWidget()
        self.tabs.addTab(self.fhss_tab, "FHSS Bypass")
        self.setup_fhss_tab()
        
        # Create Multi-Attack tab
        self.multi_attack_tab = QWidget()
        self.tabs.addTab(self.multi_attack_tab, "Multi-Vector Attack")
        self.setup_multi_attack_tab()
        
        # Create Settings tab
        self.settings_tab = QWidget()
        self.tabs.addTab(self.settings_tab, "Settings")
        self.setup_settings_tab()
        
        # Create menu
        self.create_menu()
        
        # Create toolbar
        self.create_toolbar()
    
    def setup_rf_tab(self):
        """Setup the RF analysis tab."""
        layout = QVBoxLayout(self.rf_tab)
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # Left side: capture controls
        capture_group = QGroupBox("Signal Capture")
        capture_layout = QFormLayout(capture_group)
        
        self.freq_input = QDoubleSpinBox()
        self.freq_input.setRange(10, 6000)
        self.freq_input.setValue(315.0)
        self.freq_input.setSuffix(" MHz")
        self.freq_input.setDecimals(3)
        self.freq_input.setSingleStep(0.1)
        capture_layout.addRow("Frequency:", self.freq_input)
        
        self.sample_rate_input = QDoubleSpinBox()
        self.sample_rate_input.setRange(1, 20)
        self.sample_rate_input.setValue(2.0)
        self.sample_rate_input.setSuffix(" MHz")
        self.sample_rate_input.setDecimals(1)
        capture_layout.addRow("Sample Rate:", self.sample_rate_input)
        
        self.gain_input = QSpinBox()
        self.gain_input.setRange(0, 60)
        self.gain_input.setValue(40)
        self.gain_input.setSuffix(" dB")
        capture_layout.addRow("Gain:", self.gain_input)
        
        self.duration_input = QSpinBox()
        self.duration_input.setRange(1, 60)
        self.duration_input.setValue(5)
        self.duration_input.setSuffix(" sec")
        capture_layout.addRow("Duration:", self.duration_input)
        
        self.capture_button = QPushButton("Capture Signal")
        self.capture_button.clicked.connect(self.start_signal_capture)
        capture_layout.addRow(self.capture_button)
        
        # Right side: manufacturer selection and analysis
        analysis_group = QGroupBox("Signal Analysis")
        analysis_layout = QFormLayout(analysis_group)
        
        self.manufacturer_combo = QComboBox()
        if MANUFACTURER_INFO_AVAILABLE:
            for mfr_name in MANUFACTURER_INFO.keys():
                self.manufacturer_combo.addItem(mfr_name)
        analysis_layout.addRow("Manufacturer:", self.manufacturer_combo)
        
        self.code_length_input = QSpinBox()
        self.code_length_input.setRange(8, 64)
        self.code_length_input.setValue(24)
        self.code_length_input.setSuffix(" bits")
        analysis_layout.addRow("Code Length:", self.code_length_input)
        
        self.predict_count_input = QSpinBox()
        self.predict_count_input.setRange(1, 100)
        self.predict_count_input.setValue(10)
        analysis_layout.addRow("Predict Count:", self.predict_count_input)
        
        self.analyze_button = QPushButton("Analyze Signal")
        self.analyze_button.clicked.connect(self.analyze_signal)
        analysis_layout.addRow(self.analyze_button)
        
        # Actions group (right side)
        actions_group = QGroupBox("Attack Actions")
        actions_layout = QFormLayout(actions_group)
        
        self.replay_button = QPushButton("Replay Signal")
        self.replay_button.clicked.connect(self.replay_signal)
        actions_layout.addRow(self.replay_button)
        
        self.oscillate_button = QPushButton("Frequency Oscillation")
        self.oscillate_button.clicked.connect(self.start_frequency_oscillation)
        actions_layout.addRow(self.oscillate_button)
        
        self.transmit_button = QPushButton("Transmit Code")
        self.transmit_button.clicked.connect(self.transmit_selected_code)
        actions_layout.addRow(self.transmit_button)
        
        # Add groups to control panel
        control_layout.addWidget(capture_group)
        control_layout.addWidget(analysis_group)
        control_layout.addWidget(actions_group)
        
        # Add control panel to main layout
        layout.addWidget(control_panel)
        
        # Create splitter for plotting area and results
        splitter = QSplitter(Qt.Vertical)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        splitter.addWidget(self.plot_widget)
        
        # Create results area
        results_area = QWidget()
        results_layout = QVBoxLayout(results_area)
        
        self.rf_results_table = QTableWidget(0, 3)
        self.rf_results_table.setHorizontalHeaderLabels(["Code", "Confidence", "Actions"])
        self.rf_results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        results_layout.addWidget(self.rf_results_table)
        
        splitter.addWidget(results_area)
        
        # Add splitter to main layout
        layout.addWidget(splitter)
        
        # Add progress bar
        self.rf_progress_bar = QProgressBar()
        self.rf_progress_bar.setRange(0, 100)
        self.rf_progress_bar.setValue(0)
        self.rf_progress_bar.setTextVisible(True)
        layout.addWidget(self.rf_progress_bar)
    
    def setup_ble_tab(self):
        """Setup the BLE proximity bypass tab."""
        layout = QVBoxLayout(self.ble_tab)
        
        # Create info box
        info_box = QGroupBox("BLE Proximity Bypass")
        info_layout = QVBoxLayout(info_box)
        
        info_text = """
        <h3>Bluetooth LE Proximity Bypass</h3>
        <p>This module analyzes and bypasses Bluetooth Low Energy (BLE) proximity-based authentication 
        used in modern garage door systems. It can scan for vulnerable devices, analyze their security,
        and attempt to bypass proximity-based security mechanisms.</p>
        
        <p><b>How to use:</b></p>
        <ol>
            <li>Click "Scan for Devices" to find nearby BLE devices</li>
            <li>Select a device from the list and click "Connect" to establish a connection</li>
            <li>Use "Analyze Security" to assess vulnerabilities</li>
            <li>Try different attack methods based on the analysis results</li>
        </ol>
        """
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        
        # Only show "Module not available" warning if BLE module is missing
        if not BLE_MODULE_AVAILABLE or not BLUEPY_AVAILABLE:
            warning_label = QLabel("<b>Warning:</b> BLE module not available. Install required packages to use this feature.")
            warning_label.setStyleSheet("color: red")
            info_layout.addWidget(warning_label)
        
        layout.addWidget(info_box)
        
        # Create scan panel
        scan_group = QGroupBox("Device Scan")
        scan_layout = QHBoxLayout(scan_group)
        
        self.scan_duration_input = QSpinBox()
        self.scan_duration_input.setRange(1, 60)
        self.scan_duration_input.setValue(10)
        self.scan_duration_input.setSuffix(" sec")
        scan_layout.addWidget(QLabel("Scan Duration:"))
        scan_layout.addWidget(self.scan_duration_input)
        
        scan_layout.addStretch()
        
        self.ble_scan_button = QPushButton("Scan for Devices")
        self.ble_scan_button.clicked.connect(self.start_ble_scan)
        self.ble_scan_button.setEnabled(BLE_MODULE_AVAILABLE and BLUEPY_AVAILABLE)
        scan_layout.addWidget(self.ble_scan_button)
        
        layout.addWidget(scan_group)
        
        # Create devices table
        self.ble_devices_table = QTableWidget(0, 5)
        self.ble_devices_table.setHorizontalHeaderLabels(["Name", "Address", "RSSI", "Manufacturer", "Connect"])
        self.ble_devices_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.ble_devices_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.ble_devices_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addWidget(self.ble_devices_table)
        
        # Create details area
        details_area = QWidget()
        details_layout = QHBoxLayout(details_area)
        
        # Left side: connection and device info
        connection_group = QGroupBox("Device Connection")
        connection_layout = QVBoxLayout(connection_group)
        
        self.device_info_text = QTextEdit()
        self.device_info_text.setReadOnly(True)
        connection_layout.addWidget(QLabel("Device Information:"))
        connection_layout.addWidget(self.device_info_text)
        
        # Put connection actions in a horizontal layout
        conn_actions = QHBoxLayout()
        self.ble_connect_button = QPushButton("Connect")
        self.ble_connect_button.clicked.connect(self.connect_to_ble_device)
        self.ble_connect_button.setEnabled(False)
        conn_actions.addWidget(self.ble_connect_button)
        
        self.ble_disconnect_button = QPushButton("Disconnect")
        self.ble_disconnect_button.clicked.connect(self.disconnect_ble_device)
        self.ble_disconnect_button.setEnabled(False)
        conn_actions.addWidget(self.ble_disconnect_button)
        
        connection_layout.addLayout(conn_actions)
        
        # Right side: security analysis and bypass
        analysis_group = QGroupBox("Security Analysis & Bypass")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.ble_analysis_button = QPushButton("Analyze Security")
        self.ble_analysis_button.clicked.connect(self.analyze_ble_security)
        self.ble_analysis_button.setEnabled(False)
        analysis_layout.addWidget(self.ble_analysis_button)
        
        self.ble_analysis_text = QTextEdit()
        self.ble_analysis_text.setReadOnly(True)
        analysis_layout.addWidget(QLabel("Analysis Results:"))
        analysis_layout.addWidget(self.ble_analysis_text)
        
        # Attack options
        attack_options = QGroupBox("Attack Options")
        attack_layout = QGridLayout(attack_options)
        
        self.ble_operation_combo = QComboBox()
        self.ble_operation_combo.addItems(["Status", "Open", "Close"])
        attack_layout.addWidget(QLabel("Operation:"), 0, 0)
        attack_layout.addWidget(self.ble_operation_combo, 0, 1)
        
        self.ble_bypass_button = QPushButton("Execute Proximity Bypass")
        self.ble_bypass_button.clicked.connect(self.execute_ble_bypass)
        self.ble_bypass_button.setEnabled(False)
        attack_layout.addWidget(self.ble_bypass_button, 1, 0, 1, 2)
        
        self.ble_amplification_button = QPushButton("Simulate Signal Amplification")
        self.ble_amplification_button.clicked.connect(self.simulate_signal_amplification)
        self.ble_amplification_button.setEnabled(False)
        attack_layout.addWidget(self.ble_amplification_button, 2, 0, 1, 2)
        
        analysis_layout.addWidget(attack_options)
        
        # Add groups to details area
        details_layout.addWidget(connection_group)
        details_layout.addWidget(analysis_group)
        
        layout.addWidget(details_area)
        
        # Add progress bar
        self.ble_progress_bar = QProgressBar()
        self.ble_progress_bar.setRange(0, 100)
        self.ble_progress_bar.setValue(0)
        self.ble_progress_bar.setTextVisible(True)
        layout.addWidget(self.ble_progress_bar)
    
    def setup_fhss_tab(self):
        """Setup the FHSS bypass tab."""
        layout = QVBoxLayout(self.fhss_tab)
        
        # Create info box
        info_box = QGroupBox("FHSS Bypass")
        info_layout = QVBoxLayout(info_box)
        
        info_text = """
        <h3>Frequency Hopping Spread Spectrum (FHSS) Bypass</h3>
        <p>This module analyzes and bypasses garage door openers that use frequency hopping spread 
        spectrum technology for enhanced security. It can detect hop patterns, predict next frequencies,
        and execute targeted attacks against FHSS-enabled garage doors.</p>
        
        <p><b>How to use:</b></p>
        <ol>
            <li>Configure the base frequency and scanning parameters</li>
            <li>Click "Scan for Hop Pattern" to detect the FHSS pattern</li>
            <li>Analyze the detected hop sequence</li>
            <li>Use "Execute Bypass" to attempt a bypass with loaded codes</li>
        </ol>
        """
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        
        # Only show "Module not available" warning if FHSS module is missing
        if not FHSS_MODULE_AVAILABLE:
            warning_label = QLabel("<b>Warning:</b> FHSS module not available. Install required packages to use this feature.")
            warning_label.setStyleSheet("color: red")
            info_layout.addWidget(warning_label)
        
        layout.addWidget(info_box)
        
        # Create parameters panel
        params_group = QGroupBox("FHSS Parameters")
        params_layout = QFormLayout(params_group)
        
        self.fhss_frequency_input = QDoubleSpinBox()
        self.fhss_frequency_input.setRange(300, 900)
        self.fhss_frequency_input.setValue(433.92)
        self.fhss_frequency_input.setSuffix(" MHz")
        self.fhss_frequency_input.setDecimals(3)
        params_layout.addRow("Base Frequency:", self.fhss_frequency_input)
        
        self.fhss_bandwidth_input = QDoubleSpinBox()
        self.fhss_bandwidth_input.setRange(1, 20)
        self.fhss_bandwidth_input.setValue(10.0)
        self.fhss_bandwidth_input.setSuffix(" MHz")
        self.fhss_bandwidth_input.setDecimals(1)
        params_layout.addRow("Hop Bandwidth:", self.fhss_bandwidth_input)
        
        self.fhss_channels_input = QSpinBox()
        self.fhss_channels_input.setRange(2, 100)
        self.fhss_channels_input.setValue(50)
        params_layout.addRow("Number of Channels:", self.fhss_channels_input)
        
        self.fhss_duration_input = QSpinBox()
        self.fhss_duration_input.setRange(5, 60)
        self.fhss_duration_input.setValue(20)
        self.fhss_duration_input.setSuffix(" sec")
        params_layout.addRow("Scan Duration:", self.fhss_duration_input)
        
        # Put scan button in its own layout to add spacing
        scan_layout = QHBoxLayout()
        scan_layout.addStretch()
        
        self.fhss_scan_button = QPushButton("Scan for Hop Pattern")
        self.fhss_scan_button.clicked.connect(self.scan_for_fhss_pattern)
        self.fhss_scan_button.setEnabled(FHSS_MODULE_AVAILABLE and HACKRF_AVAILABLE)
        scan_layout.addWidget(self.fhss_scan_button)
        
        params_layout.addRow(scan_layout)
        
        layout.addWidget(params_group)
        
        # Create results area with splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: hop sequence visualization
        hop_viz_group = QGroupBox("Hop Sequence Visualization")
        hop_viz_layout = QVBoxLayout(hop_viz_group)
        
        self.fhss_plot_widget = pg.PlotWidget()
        self.fhss_plot_widget.setBackground('w')
        self.fhss_plot_widget.setLabel('left', 'Channel')
        self.fhss_plot_widget.setLabel('bottom', 'Time', units='s')
        hop_viz_layout.addWidget(self.fhss_plot_widget)
        
        # Channel pattern details
        self.fhss_pattern_text = QTextEdit()
        self.fhss_pattern_text.setReadOnly(True)
        self.fhss_pattern_text.setMaximumHeight(100)
        hop_viz_layout.addWidget(QLabel("Detected Pattern:"))
        hop_viz_layout.addWidget(self.fhss_pattern_text)
        
        splitter.addWidget(hop_viz_group)
        
        # Right side: analysis and bypass
        analysis_group = QGroupBox("Analysis & Bypass")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.fhss_analysis_text = QTextEdit()
        self.fhss_analysis_text.setReadOnly(True)
        analysis_layout.addWidget(QLabel("Analysis Results:"))
        analysis_layout.addWidget(self.fhss_analysis_text)
        
        # Bypass options
        bypass_options = QGroupBox("Bypass Options")
        bypass_layout = QFormLayout(bypass_options)
        
        self.fhss_code_file_button = QPushButton("Load Codes File")
        self.fhss_code_file_button.clicked.connect(self.load_fhss_codes_file)
        bypass_layout.addRow("Codes File:", self.fhss_code_file_button)
        
        self.fhss_codes_label = QLabel("No codes loaded")
        bypass_layout.addRow("Status:", self.fhss_codes_label)
        
        self.fhss_bypass_button = QPushButton("Execute FHSS Bypass")
        self.fhss_bypass_button.clicked.connect(self.execute_fhss_bypass)
        self.fhss_bypass_button.setEnabled(False)
        bypass_layout.addRow(self.fhss_bypass_button)
        
        analysis_layout.addWidget(bypass_options)
        
        splitter.addWidget(analysis_group)
        
        layout.addWidget(splitter)
        
        # Add progress bar
        self.fhss_progress_bar = QProgressBar()
        self.fhss_progress_bar.setRange(0, 100)
        self.fhss_progress_bar.setValue(0)
        self.fhss_progress_bar.setTextVisible(True)
        layout.addWidget(self.fhss_progress_bar)
    
    def setup_multi_attack_tab(self):
        """Setup the multi-vector attack tab."""
        layout = QVBoxLayout(self.multi_attack_tab)
        
        # Create info box
        info_box = QGroupBox("Multi-Vector Attack")
        info_layout = QVBoxLayout(info_box)
        
        info_text = """
        <h3>Multi-Vector Attack Framework</h3>
        <p>This module combines RF, BLE, and FHSS attack techniques to create a comprehensive
        security assessment of modern garage door systems. It can detect which attack vectors 
        are applicable to the target system and execute them in sequence or simultaneously.</p>
        
        <p><b>How to use:</b></p>
        <ol>
            <li>Configure the attack parameters for each vector</li>
            <li>Click "Start Analysis" to detect applicable vulnerabilities</li>
            <li>Review the suggested attack plan</li>
            <li>Execute the attack sequence with the "Launch Attack" button</li>
        </ol>
        """
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        
        layout.addWidget(info_box)
        
        # Create attack configuration panel
        config_group = QGroupBox("Attack Configuration")
        config_layout = QGridLayout(config_group)
        
        # RF vector
        rf_check = QCheckBox("RF Signal Analysis")
        rf_check.setChecked(True)
        config_layout.addWidget(rf_check, 0, 0)
        
        rf_freq = QDoubleSpinBox()
        rf_freq.setRange(300, 900)
        rf_freq.setValue(315.0)
        rf_freq.setSuffix(" MHz")
        config_layout.addWidget(QLabel("Frequency:"), 0, 1)
        config_layout.addWidget(rf_freq, 0, 2)
        
        # BLE vector
        ble_check = QCheckBox("BLE Proximity Bypass")
        ble_check.setChecked(True)
        config_layout.addWidget(ble_check, 1, 0)
        
        ble_scan = QSpinBox()
        ble_scan.setRange(5, 30)
        ble_scan.setValue(10)
        ble_scan.setSuffix(" sec")
        config_layout.addWidget(QLabel("Scan Time:"), 1, 1)
        config_layout.addWidget(ble_scan, 1, 2)
        
        # FHSS vector
        fhss_check = QCheckBox("FHSS Bypass")
        fhss_check.setChecked(True)
        config_layout.addWidget(fhss_check, 2, 0)
        
        fhss_freq = QDoubleSpinBox()
        fhss_freq.setRange(300, 900)
        fhss_freq.setValue(433.92)
        fhss_freq.setSuffix(" MHz")
        config_layout.addWidget(QLabel("Frequency:"), 2, 1)
        config_layout.addWidget(fhss_freq, 2, 2)
        
        # Attack options
        options_group = QGroupBox("Attack Options")
        options_layout = QVBoxLayout(options_group)
        
        sequential_radio = QRadioButton("Sequential (one vector at a time)")
        sequential_radio.setChecked(True)
        options_layout.addWidget(sequential_radio)
        
        parallel_radio = QRadioButton("Parallel (simultaneous vectors)")
        options_layout.addWidget(parallel_radio)
        
        self.attack_mode_group = QButtonGroup()
        self.attack_mode_group.addButton(sequential_radio, 0)
        self.attack_mode_group.addButton(parallel_radio, 1)
        
        config_layout.addWidget(options_group, 0, 3, 3, 1)
        
        # Analysis button
        analyze_button = QPushButton("Start Multi-Vector Analysis")
        analyze_button.clicked.connect(self.start_multi_vector_analysis)
        config_layout.addWidget(analyze_button, 3, 0, 1, 4)
        
        layout.addWidget(config_group)
        
        # Create attack plan and results area
        attack_area = QSplitter(Qt.Vertical)
        
        # Attack plan
        plan_group = QGroupBox("Attack Plan")
        plan_layout = QVBoxLayout(plan_group)
        
        self.attack_plan_text = QTextEdit()
        self.attack_plan_text.setReadOnly(True)
        plan_layout.addWidget(self.attack_plan_text)
        
        launch_button = QPushButton("Launch Attack Sequence")
        launch_button.clicked.connect(self.launch_multi_vector_attack)
        plan_layout.addWidget(launch_button)
        
        attack_area.addWidget(plan_group)
        
        # Attack progress and results
        results_group = QGroupBox("Attack Progress & Results")
        results_layout = QVBoxLayout(results_group)
        
        self.attack_progress_text = QTextEdit()
        self.attack_progress_text.setReadOnly(True)
        results_layout.addWidget(self.attack_progress_text)
        
        attack_area.addWidget(results_group)
        
        layout.addWidget(attack_area)
        
        # Add progress bar
        self.multi_attack_progress_bar = QProgressBar()
        self.multi_attack_progress_bar.setRange(0, 100)
        self.multi_attack_progress_bar.setValue(0)
        self.multi_attack_progress_bar.setTextVisible(True)
        layout.addWidget(self.multi_attack_progress_bar)
    
    def setup_settings_tab(self):
        """Setup the settings tab."""
        layout = QVBoxLayout(self.settings_tab)
        
        # Create hardware status box
        hardware_group = QGroupBox("Hardware Status")
        hardware_layout = QGridLayout(hardware_group)
        
        # HackRF status
        self.hackrf_status_label = QLabel("Checking...")
        hardware_layout.addWidget(QLabel("HackRF:"), 0, 0)
        hardware_layout.addWidget(self.hackrf_status_label, 0, 1)
        
        # BLE status
        self.ble_status_label = QLabel("Checking...")
        hardware_layout.addWidget(QLabel("Bluetooth LE:"), 1, 0)
        hardware_layout.addWidget(self.ble_status_label, 1, 1)
        
        # Check hardware button
        check_button = QPushButton("Check Hardware")
        check_button.clicked.connect(self.check_hardware_status)
        hardware_layout.addWidget(check_button, 2, 0, 1, 2)
        
        layout.addWidget(hardware_group)
        
        # Create logging settings
        logging_group = QGroupBox("Logging")
        logging_layout = QVBoxLayout(logging_group)
        
        # Log level selection
        log_level_layout = QHBoxLayout()
        log_level_layout.addWidget(QLabel("Log Level:"))
        
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentIndex(1)  # Default to INFO
        self.log_level_combo.currentIndexChanged.connect(self.change_log_level)
        log_level_layout.addWidget(self.log_level_combo)
        
        logging_layout.addLayout(log_level_layout)
        
        # Log display
        self.log_display = LoggingTextEdit()
        logging_layout.addWidget(self.log_display)
        
        # Clear logs button
        clear_button = QPushButton("Clear Logs")
        clear_button.clicked.connect(self.log_display.clear)
        logging_layout.addWidget(clear_button)
        
        layout.addWidget(logging_group)
        
        # Create application settings
        app_settings_group = QGroupBox("Application Settings")
        app_settings_layout = QFormLayout(app_settings_group)
        
        # Auto-save results checkbox
        self.auto_save_check = QCheckBox()
        self.auto_save_check.setChecked(False)
        app_settings_layout.addRow("Auto-save Results:", self.auto_save_check)
        
        # Dark mode
        self.dark_mode_check = QCheckBox()
        self.dark_mode_check.setChecked(False)
        self.dark_mode_check.stateChanged.connect(self.toggle_dark_mode)
        app_settings_layout.addRow("Dark Mode:", self.dark_mode_check)
        
        # Save settings button
        save_settings_button = QPushButton("Save Settings")
        save_settings_button.clicked.connect(self.save_settings)
        app_settings_layout.addRow(save_settings_button)
        
        layout.addWidget(app_settings_group)
        
        # Version info
        version_label = QLabel("Garage Door Security Analyzer v1.0")
        layout.addWidget(version_label)
    
    def create_menu(self):
        """Create application menu."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Signal", self)
        open_action.triggered.connect(self.open_signal_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Signal", self)
        save_action.triggered.connect(self.save_signal_file)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        rf_tool_action = QAction("RF Signal Analysis", self)
        rf_tool_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        tools_menu.addAction(rf_tool_action)
        
        ble_tool_action = QAction("BLE Proximity Bypass", self)
        ble_tool_action.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        tools_menu.addAction(ble_tool_action)
        
        fhss_tool_action = QAction("FHSS Bypass", self)
        fhss_tool_action.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        tools_menu.addAction(fhss_tool_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)
    
    def create_toolbar(self):
        """Create application toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # RF capture button
        capture_action = QAction("Capture RF", self)
        capture_action.triggered.connect(self.start_signal_capture)
        toolbar.addAction(capture_action)
        
        # BLE scan button
        ble_scan_action = QAction("Scan BLE", self)
        ble_scan_action.triggered.connect(self.start_ble_scan)
        toolbar.addAction(ble_scan_action)
        
        toolbar.addSeparator()
        
        # Save button
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_signal_file)
        toolbar.addAction(save_action)
        
        # Open button
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_signal_file)
        toolbar.addAction(open_action)
    
    def setup_logging(self):
        """Setup logging to route to the log display widget."""
        class LogHandler(logging.Handler):
            def __init__(self, widget):
                super().__init__()
                self.widget = widget
            
            def emit(self, record):
                msg = self.format(record)
                self.widget.append_log(msg, record.levelno)
        
        # Create custom handler
        handler = LogHandler(self.log_display)
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
        # Get the root logger and add our handler
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        
        # Set initial level
        self.change_log_level(1)  # INFO level by default
    
    def check_hardware_status(self):
        """Check the status of required hardware components."""
        # Check HackRF
        if HACKRF_AVAILABLE:
            try:
                device = hackrf.HackRF()
                serial = device.serial_number()
                device.close()
                self.hackrf_status_label.setText(f"Connected (S/N: {serial})")
                self.hackrf_status_label.setStyleSheet("color: green")
                self.log_display.append_log("HackRF device detected and operational", logging.INFO)
            except Exception as e:
                self.hackrf_status_label.setText("Error: " + str(e))
                self.hackrf_status_label.setStyleSheet("color: red")
                self.log_display.append_log(f"HackRF error: {e}", logging.ERROR)
        else:
            self.hackrf_status_label.setText("Not Available")
            self.hackrf_status_label.setStyleSheet("color: red")
            self.log_display.append_log("HackRF module not available", logging.WARNING)
        
        # Check BLE
        if BLUEPY_AVAILABLE:
            try:
                # Don't actually scan, just check if we can create a scanner
                btle.Scanner()
                self.ble_status_label.setText("Available")
                self.ble_status_label.setStyleSheet("color: green")
                self.log_display.append_log("Bluetooth LE module is available", logging.INFO)
            except Exception as e:
                self.ble_status_label.setText("Error: " + str(e))
                self.ble_status_label.setStyleSheet("color: orange")
                self.log_display.append_log(f"Bluetooth LE error: {e}", logging.WARNING)
        else:
            self.ble_status_label.setText("Not Available")
            self.ble_status_label.setStyleSheet("color: red")
            self.log_display.append_log("Bluetooth LE module not available", logging.WARNING)
    
    def change_log_level(self, index):
        """Change the logging level based on combobox selection."""
        levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
        selected_level = levels[index]
        
        # Set level on the root logger
        logging.getLogger().setLevel(selected_level)
        
        # Log the change
        level_names = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self.log_display.append_log(f"Log level changed to {level_names[index]}", logging.INFO)
    
    def toggle_dark_mode(self, state):
        """Toggle dark mode for the application."""
        if state == Qt.Checked:
            # Set dark mode
            self.setStyleSheet("""
                QWidget {
                    background-color: #2D2D30;
                    color: #E1E1E1;
                }
                QMenuBar, QMenu {
                    background-color: #1E1E1E;
                    color: #E1E1E1;
                }
                QMenuBar::item:selected, QMenu::item:selected {
                    background-color: #3E3E40;
                }
                QTabWidget::pane {
                    border: 1px solid #3E3E40;
                }
                QTabBar::tab {
                    background-color: #2D2D30;
                    color: #E1E1E1;
                    border: 1px solid #3E3E40;
                    padding: 6px;
                }
                QTabBar::tab:selected {
                    background-color: #3E3E40;
                }
                QPushButton {
                    background-color: #0E639C;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                }
                QPushButton:hover {
                    background-color: #1177BB;
                }
                QPushButton:pressed {
                    background-color: #0D5A8C;
                }
                QPushButton:disabled {
                    background-color: #2D2D30;
                    color: #6D6D6F;
                }
                QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                    background-color: #1E1E1E;
                    color: #E1E1E1;
                    border: 1px solid #3E3E40;
                }
                QGroupBox {
                    border: 1px solid #3E3E40;
                    margin-top: 0.5em;
                    color: #E1E1E1;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 3px 0 3px;
                }
                QTableWidget {
                    background-color: #1E1E1E;
                    color: #E1E1E1;
                    gridline-color: #3E3E40;
                }
                QHeaderView::section {
                    background-color: #2D2D30;
                    color: #E1E1E1;
                    border: 1px solid #3E3E40;
                }
                QScrollBar {
                    background-color: #2D2D30;
                }
            """)
            
            # Update plot backgrounds
            self.plot_widget.setBackground('#1E1E1E')
            self.plot_widget.getAxis('bottom').setPen('w')
            self.plot_widget.getAxis('left').setPen('w')
            
            self.fhss_plot_widget.setBackground('#1E1E1E')
            self.fhss_plot_widget.getAxis('bottom').setPen('w')
            self.fhss_plot_widget.getAxis('left').setPen('w')
        else:
            # Restore default style
            self.setStyleSheet("")
            
            # Update plot backgrounds
            self.plot_widget.setBackground('w')
            self.plot_widget.getAxis('bottom').setPen('k')
            self.plot_widget.getAxis('left').setPen('k')
            
            self.fhss_plot_widget.setBackground('w')
            self.fhss_plot_widget.getAxis('bottom').setPen('k')
            self.fhss_plot_widget.getAxis('left').setPen('k')
    
    def save_settings(self):
        """Save application settings."""
        settings = {
            "log_level": self.log_level_combo.currentIndex(),
            "auto_save": self.auto_save_check.isChecked(),
            "dark_mode": self.dark_mode_check.isChecked(),
            "rf_frequency": self.freq_input.value(),
            "rf_sample_rate": self.sample_rate_input.value(),
            "rf_gain": self.gain_input.value(),
            "rf_duration": self.duration_input.value(),
            "ble_scan_duration": self.scan_duration_input.value(),
            "fhss_frequency": self.fhss_frequency_input.value(),
            "fhss_bandwidth": self.fhss_bandwidth_input.value(),
            "fhss_channels": self.fhss_channels_input.value(),
            "fhss_duration": self.fhss_duration_input.value()
        }
        
        try:
            os.makedirs('config', exist_ok=True)
            with open('config/settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            
            self.log_display.append_log("Settings saved successfully", logging.INFO)
            QMessageBox.information(self, "Settings", "Settings saved successfully")
        except Exception as e:
            self.log_display.append_log(f"Error saving settings: {e}", logging.ERROR)
            QMessageBox.warning(self, "Settings", f"Error saving settings: {e}")
    
    def load_settings(self):
        """Load application settings."""
        try:
            if os.path.exists('config/settings.json'):
                with open('config/settings.json', 'r') as f:
                    settings = json.load(f)
                
                # Apply settings
                self.log_level_combo.setCurrentIndex(settings.get("log_level", 1))
                self.auto_save_check.setChecked(settings.get("auto_save", False))
                self.dark_mode_check.setChecked(settings.get("dark_mode", False))
                
                self.freq_input.setValue(settings.get("rf_frequency", 315.0))
                self.sample_rate_input.setValue(settings.get("rf_sample_rate", 2.0))
                self.gain_input.setValue(settings.get("rf_gain", 40))
                self.duration_input.setValue(settings.get("rf_duration", 5))
                
                self.scan_duration_input.setValue(settings.get("ble_scan_duration", 10))
                
                self.fhss_frequency_input.setValue(settings.get("fhss_frequency", 433.92))
                self.fhss_bandwidth_input.setValue(settings.get("fhss_bandwidth", 10.0))
                self.fhss_channels_input.setValue(settings.get("fhss_channels", 50))
                self.fhss_duration_input.setValue(settings.get("fhss_duration", 20))
                
                self.log_display.append_log("Settings loaded successfully", logging.INFO)
        except Exception as e:
            self.log_display.append_log(f"Error loading settings: {e}", logging.ERROR)
    
    def update_ui(self):
        """Update UI elements based on current state."""
        # Update the progress bars if tasks are running
        pass
    
    def start_signal_capture(self):
        """Start capturing RF signal using HackRF."""
        if self.hackrf_capture and self.hackrf_capture.isRunning():
            self.hackrf_capture.stop()
            self.capture_button.setText("Capture Signal")
            self.rf_progress_bar.setValue(0)
            return
        
        if not HACKRF_AVAILABLE:
            QMessageBox.warning(self, "Hardware Error", "HackRF hardware not available")
            self.log_display.append_log("Cannot capture signal: HackRF hardware not available", logging.ERROR)
            return
        
        # Get parameters
        frequency = self.freq_input.value() * 1e6  # Convert MHz to Hz
        sample_rate = self.sample_rate_input.value() * 1e6  # Convert MHz to Hz
        gain = self.gain_input.value()
        duration = self.duration_input.value()
        
        self.log_display.append_log(f"Starting signal capture at {frequency/1e6} MHz for {duration} seconds", logging.INFO)
        
        # Create and start capture thread
        self.hackrf_capture = HackRFSignalCapture(
            frequency=frequency,
            sample_rate=sample_rate,
            gain=gain,
            duration=duration
        )
        
        # Connect signals
        self.hackrf_capture.update_progress.connect(self.rf_progress_bar.setValue)
        self.hackrf_capture.update_signal.connect(self.update_signal_plot)
        self.hackrf_capture.capture_completed.connect(self.on_capture_completed)
        
        # Start capture
        self.hackrf_capture.start()
        
        # Update UI
        self.capture_button.setText("Stop Capture")
    
    def update_signal_plot(self, data):
        """Update the signal plot with new data."""
        if data is None or len(data) == 0:
            return
        
        # Store the signal data
        self.signal_data = data
        
        # Clear the plot
        self.plot_widget.clear()
        
        # Create time axis
        sample_rate = self.sample_rate_input.value() * 1e6
        time_axis = np.arange(len(data)) / sample_rate
        
        # Plot magnitude of complex signal
        magnitude = np.abs(data)
        self.plot_widget.plot(time_axis, magnitude, pen='b')
        
        # Set appropriate range
        self.plot_widget.setYRange(0, np.max(magnitude) * 1.1)
        self.plot_widget.setXRange(0, time_axis[-1])
    
    def on_capture_completed(self, success, message):
        """Handle signal capture completion."""
        self.capture_button.setText("Capture Signal")
        
        if success:
            self.log_display.append_log(f"Signal capture completed: {message}", logging.INFO)
            
            # If auto-save is enabled, save the signal
            if self.auto_save_check.isChecked():
                self.save_signal_file()
        else:
            self.log_display.append_log(f"Signal capture failed: {message}", logging.ERROR)
            QMessageBox.warning(self, "Capture Error", f"Signal capture failed: {message}")
    
    def analyze_signal(self):
        """Analyze the captured RF signal."""
        if self.signal_data is None or len(self.signal_data) == 0:
            QMessageBox.warning(self, "Analysis Error", "No signal data available. Capture a signal first.")
            return
        
        self.log_display.append_log("Starting signal analysis...", logging.INFO)
        
        # Mock analysis for demonstration
        # In a real implementation, this would use the actual rolling code analysis
        # from the existing modules
        
        # Clear results table
        self.rf_results_table.setRowCount(0)
        
        # Generate some example codes for demonstration
        num_codes = 5
        for i in range(num_codes):
            code = ''.join(np.random.choice(['0', '1']) for _ in range(24))
            confidence = np.random.uniform(0.5, 0.99)
            
            # Add to table
            row = self.rf_results_table.rowCount()
            self.rf_results_table.insertRow(row)
            
            self.rf_results_table.setItem(row, 0, QTableWidgetItem(code))
            self.rf_results_table.setItem(row, 1, QTableWidgetItem(f"{confidence:.2f}"))
            
            # Add action button
            transmit_button = QPushButton("Transmit")
            transmit_button.clicked.connect(lambda _, c=code: self.transmit_code(c))
            self.rf_results_table.setCellWidget(row, 2, transmit_button)
        
        self.log_display.append_log(f"Signal analysis completed. Found {num_codes} potential codes.", logging.INFO)
    
    def transmit_code(self, code):
        """Transmit a specific code."""
        if not HACKRF_AVAILABLE:
            QMessageBox.warning(self, "Hardware Error", "HackRF hardware not available")
            return
        
        self.log_display.append_log(f"Transmitting code: {code}", logging.INFO)
        
        # In a real implementation, this would use the actual code transmission
        # functionality from the existing modules
        
        QMessageBox.information(self, "Transmission", f"Code transmitted: {code}")
    
    def transmit_selected_code(self):
        """Transmit the selected code from the results table."""
        selected_rows = self.rf_results_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "Selection Error", "No code selected. Select a code from the results table.")
            return
        
        # Get the code from the first column of the selected row
        selected_row = selected_rows[0].row()
        code = self.rf_results_table.item(selected_row, 0).text()
        
        self.transmit_code(code)
    
    def replay_signal(self):
        """Replay the captured signal."""
        if self.signal_data is None or len(self.signal_data) == 0:
            QMessageBox.warning(self, "Replay Error", "No signal data available. Capture a signal first.")
            return
        
        if not HACKRF_AVAILABLE:
            QMessageBox.warning(self, "Hardware Error", "HackRF hardware not available")
            return
        
        self.log_display.append_log("Replaying captured signal...", logging.INFO)
        
        # In a real implementation, this would use the actual signal replay
        # functionality from the existing modules
        
        QMessageBox.information(self, "Replay", "Signal replayed successfully")
    
    def start_frequency_oscillation(self):
        """Start frequency oscillation attack."""
        if not HACKRF_AVAILABLE:
            QMessageBox.warning(self, "Hardware Error", "HackRF hardware not available")
            return
        
        frequency = self.freq_input.value()
        
        self.log_display.append_log(f"Starting frequency oscillation attack at {frequency} MHz...", logging.INFO)
        
        # In a real implementation, this would use the actual frequency oscillation
        # functionality from the existing modules
        
        QMessageBox.information(self, "Oscillation", "Frequency oscillation attack completed")
    
    def start_ble_scan(self):
        """Start scanning for BLE devices."""
        if self.ble_scan and self.ble_scan.isRunning():
            self.ble_scan.stop()
            self.ble_scan_button.setText("Scan for Devices")
            self.ble_progress_bar.setValue(0)
            return
        
        if not BLUEPY_AVAILABLE:
            QMessageBox.warning(self, "Error", "Bluetooth LE module not available")
            self.log_display.append_log("Cannot scan for BLE devices: Bluepy module not available", logging.ERROR)
            return
        
        # Get scan duration
        duration = self.scan_duration_input.value()
        
        self.log_display.append_log(f"Starting BLE scan for {duration} seconds...", logging.INFO)
        
        # Clear devices table
        self.ble_devices_table.setRowCount(0)
        
        # Create and start scan thread
        self.ble_scan = BLEScanThread(duration=duration)
        
        # Connect signals
        self.ble_scan.update_devices.connect(self.update_ble_devices)
        self.ble_scan.scan_completed.connect(self.on_ble_scan_completed)
        
        # Start scan
        self.ble_scan.start()
        
        # Update UI
        self.ble_scan_button.setText("Stop Scan")
        
        # Start progress timer
        self.ble_progress = 0
        self.ble_progress_timer = QTimer()
        self.ble_progress_timer.timeout.connect(self.update_ble_progress)
        self.ble_progress_timer.start(100)  # Update every 100ms
    
    def update_ble_progress(self):
        """Update BLE scan progress bar."""
        if not self.ble_scan or not self.ble_scan.isRunning():
            self.ble_progress_timer.stop()
            return
        
        duration = self.scan_duration_input.value() * 1000  # Convert to ms
        self.ble_progress += 100  # 100ms increment
        progress = min(100, int(self.ble_progress / duration * 100))
        self.ble_progress_bar.setValue(progress)
    
    def update_ble_devices(self, devices):
        """Update the BLE devices table with found devices."""
        self.captured_devices = devices
        
        # Clear table
        self.ble_devices_table.setRowCount(0)
        
        for device in devices:
            row = self.ble_devices_table.rowCount()
            self.ble_devices_table.insertRow(row)
            
            name = device.get("name", "Unknown")
            address = device.get("address", "")
            rssi = device.get("rssi", 0)
            
            # Determine manufacturer
            mfr = "Unknown"
            if BLE_MODULE_AVAILABLE:
                for mfr_name, info in GARAGE_BLE_MANUFACTURERS.items():
                    name_match = any(pattern.lower() in name.lower() for pattern in info["name_patterns"])
                    if name_match:
                        mfr = mfr_name.capitalize()
                        break
            
            self.ble_devices_table.setItem(row, 0, QTableWidgetItem(name))
            self.ble_devices_table.setItem(row, 1, QTableWidgetItem(address))
            self.ble_devices_table.setItem(row, 2, QTableWidgetItem(str(rssi)))
            self.ble_devices_table.setItem(row, 3, QTableWidgetItem(mfr))
            
            # Add connect button
            connect_button = QPushButton("Connect")
            connect_button.clicked.connect(lambda _, addr=address: self.set_ble_device(addr))
            self.ble_devices_table.setCellWidget(row, 4, connect_button)
    
    def on_ble_scan_completed(self, success, message):
        """Handle BLE scan completion."""
        self.ble_scan_button.setText("Scan for Devices")
        self.ble_progress_bar.setValue(0)
        
        if success:
            self.log_display.append_log(f"BLE scan completed: {message}", logging.INFO)
        else:
            self.log_display.append_log(f"BLE scan failed: {message}", logging.ERROR)
            QMessageBox.warning(self, "Scan Error", f"BLE scan failed: {message}")
    
    def set_ble_device(self, address):
        """Set the selected BLE device."""
        self.selected_device = address
        
        # Find device info
        device_info = None
        for device in self.captured_devices:
            if device.get("address") == address:
                device_info = device
                break
        
        if not device_info:
            return
        
        # Display device info
        info_text = f"""
        <h3>Device Information</h3>
        <p><b>Name:</b> {device_info.get('name', 'Unknown')}</p>
        <p><b>Address:</b> {address}</p>
        <p><b>RSSI:</b> {device_info.get('rssi', 0)} dBm</p>
        <p><b>Connectable:</b> {'Yes' if device_info.get('connectable', False) else 'No'}</p>
        """
        
        if device_info.get('services'):
            info_text += "<p><b>Services:</b></p><ul>"
            for service in device_info.get('services', []):
                info_text += f"<li>{service}</li>"
            info_text += "</ul>"
        
        self.device_info_text.setHtml(info_text)
        
        # Enable connect button
        self.ble_connect_button.setEnabled(True)
    
    def connect_to_ble_device(self):
        """Connect to the selected BLE device."""
        if not self.selected_device:
            QMessageBox.warning(self, "Connection Error", "No device selected")
            return
        
        if not BLE_MODULE_AVAILABLE:
            QMessageBox.warning(self, "Module Error", "BLE bypass module not available")
            return
        
        self.log_display.append_log(f"Connecting to device: {self.selected_device}...", logging.INFO)
        
        try:
            # Create BLE bypasser if not already created
            if not self.ble_bypasser:
                self.ble_bypasser = BLEProximityBypasser()
            
            # Connect in a thread to avoid freezing UI
            worker = WorkerThread(self.ble_bypasser.connect_to_device, self.selected_device)
            worker.task_completed.connect(self.on_ble_connect_completed)
            worker.start()
            
            # Show progress
            self.ble_progress_bar.setRange(0, 0)  # Indeterminate progress
        except Exception as e:
            self.log_display.append_log(f"Error connecting to device: {e}", logging.ERROR)
            QMessageBox.warning(self, "Connection Error", f"Error connecting to device: {e}")
    
    def on_ble_connect_completed(self, success, message):
        """Handle BLE connection completion."""
        self.ble_progress_bar.setRange(0, 100)  # Restore determinate progress
        
        if success:
            self.log_display.append_log(f"Connected to device: {self.selected_device}", logging.INFO)
            
            # Enable buttons
            self.ble_connect_button.setEnabled(False)
            self.ble_disconnect_button.setEnabled(True)
            self.ble_analysis_button.setEnabled(True)
            self.ble_bypass_button.setEnabled(True)
            self.ble_amplification_button.setEnabled(True)
        else:
            self.log_display.append_log(f"Connection failed: {message}", logging.ERROR)
            QMessageBox.warning(self, "Connection Error", f"Failed to connect: {message}")
    
    def disconnect_ble_device(self):
        """Disconnect from the BLE device."""
        if not self.ble_bypasser:
            return
        
        try:
            self.ble_bypasser.disconnect()
            
            # Update UI
            self.ble_connect_button.setEnabled(True)
            self.ble_disconnect_button.setEnabled(False)
            self.ble_analysis_button.setEnabled(False)
            self.ble_bypass_button.setEnabled(False)
            self.ble_amplification_button.setEnabled(False)
            
            self.log_display.append_log("Disconnected from device", logging.INFO)
        except Exception as e:
            self.log_display.append_log(f"Error disconnecting: {e}", logging.ERROR)
    
    def analyze_ble_security(self):
        """Analyze security of the connected BLE device."""
        if not self.ble_bypasser:
            QMessageBox.warning(self, "Analysis Error", "Not connected to a device")
            return
        
        self.log_display.append_log("Analyzing device security...", logging.INFO)
        
        try:
            # Run analysis in a thread
            worker = WorkerThread(self.ble_bypasser.analyze_security)
            worker.update_result.connect(self.display_ble_analysis)
            worker.task_completed.connect(self.on_ble_analysis_completed)
            worker.start()
            
            # Show progress
            self.ble_progress_bar.setRange(0, 0)  # Indeterminate progress
        except Exception as e:
            self.log_display.append_log(f"Error analyzing security: {e}", logging.ERROR)
            QMessageBox.warning(self, "Analysis Error", f"Error analyzing security: {e}")
    
    def display_ble_analysis(self, results):
        """Display BLE security analysis results."""
        if not results or "error" in results:
            error_msg = results.get("error", "Unknown error") if results else "Unknown error"
            self.ble_analysis_text.setHtml(f"<p style='color:red'>Error: {error_msg}</p>")
            return
        
        # Format results as HTML
        html = f"""
        <h3>Security Analysis Results</h3>
        <p><b>Manufacturer:</b> {results.get('manufacturer', 'Unknown').capitalize()}</p>
        <p><b>Authentication Type:</b> {results.get('auth_type', 'Unknown')}</p>
        <p><b>Security Rating:</b> {results.get('security_rating', 0)}/10</p>
        
        <h4>Vulnerabilities:</h4>
        <ul>
        """
        
        vulnerabilities = results.get('vulnerabilities', {})
        for name, vuln in vulnerabilities.items():
            color = "red" if vuln.get('vulnerable', False) else "green"
            status = "Vulnerable" if vuln.get('vulnerable', False) else "Not Vulnerable"
            html += f"<li><b>{name.replace('_', ' ').title()}:</b> <span style='color:{color}'>{status}</span><br>{vuln.get('details', '')}</li>"
        
        html += "</ul>"
        
        if results.get('security_rating', 0) <= 3:
            html += "<p><b>Recommendation:</b> <span style='color:red'>This device has critical security vulnerabilities and should be replaced with a more secure model.</span></p>"
        elif results.get('security_rating', 0) <= 6:
            html += "<p><b>Recommendation:</b> <span style='color:orange'>This device has security weaknesses. Consider firmware updates or additional security measures.</span></p>"
        else:
            html += "<p><b>Recommendation:</b> <span style='color:green'>This device has reasonable security. Continue monitoring for firmware updates.</span></p>"
        
        self.ble_analysis_text.setHtml(html)
    
    def on_ble_analysis_completed(self, success, message):
        """Handle BLE security analysis completion."""
        self.ble_progress_bar.setRange(0, 100)  # Restore determinate progress
        
        if success:
            self.log_display.append_log("Security analysis completed", logging.INFO)
        else:
            self.log_display.append_log(f"Security analysis failed: {message}", logging.ERROR)
            QMessageBox.warning(self, "Analysis Error", f"Security analysis failed: {message}")
    
    def execute_ble_bypass(self):
        """Execute BLE proximity bypass attack."""
        if not self.ble_bypasser:
            QMessageBox.warning(self, "Bypass Error", "Not connected to a device")
            return
        
        # Get operation
        operation = self.ble_operation_combo.currentText().lower()
        
        self.log_display.append_log(f"Executing proximity bypass for operation: {operation}...", logging.INFO)
        
        try:
            # Run bypass in a thread
            worker = WorkerThread(self.ble_bypasser.execute_proximity_bypass, operation)
            worker.task_completed.connect(self.on_ble_bypass_completed)
            worker.start()
            
            # Show progress
            self.ble_progress_bar.setRange(0, 0)  # Indeterminate progress
        except Exception as e:
            self.log_display.append_log(f"Error executing bypass: {e}", logging.ERROR)
            QMessageBox.warning(self, "Bypass Error", f"Error executing bypass: {e}")
    
    def on_ble_bypass_completed(self, success, message):
        """Handle BLE bypass completion."""
        self.ble_progress_bar.setRange(0, 100)  # Restore determinate progress
        
        if success:
            self.log_display.append_log("Proximity bypass completed successfully", logging.INFO)
            QMessageBox.information(self, "Bypass", "Proximity bypass executed successfully")
        else:
            self.log_display.append_log(f"Proximity bypass failed: {message}", logging.ERROR)
            QMessageBox.warning(self, "Bypass Error", f"Proximity bypass failed: {message}")
    
    def simulate_signal_amplification(self):
        """Simulate signal amplification attack."""
        if not self.ble_bypasser:
            QMessageBox.warning(self, "Simulation Error", "Not connected to a device")
            return
        
        self.log_display.append_log("Simulating signal amplification attack...", logging.INFO)
        
        try:
            # Run simulation in a thread
            worker = WorkerThread(self.ble_bypasser.perform_signal_amplification_simulation)
            worker.update_result.connect(self.display_amplification_results)
            worker.task_completed.connect(self.on_amplification_completed)
            worker.start()
            
            # Show progress
            self.ble_progress_bar.setRange(0, 0)  # Indeterminate progress
        except Exception as e:
            self.log_display.append_log(f"Error simulating amplification: {e}", logging.ERROR)
            QMessageBox.warning(self, "Simulation Error", f"Error simulating amplification: {e}")
    
    def display_amplification_results(self, results):
        """Display signal amplification simulation results."""
        if not results or "error" in results:
            error_msg = results.get("error", "Unknown error") if results else "Unknown error"
            QMessageBox.warning(self, "Simulation Error", f"Error: {error_msg}")
            return
        
        # Format results as message box
        vuln = results.get('vulnerability', {})
        status = "Vulnerable" if vuln.get('vulnerable', False) else "Not Vulnerable"
        
        msg = f"""
        Signal Amplification Simulation Results:
        
        Actual Signal Strength: {results.get('actual_rssi', 0)} dBm
        Estimated Threshold: {results.get('signal_threshold', 0)} dBm
        Amplification Needed: {results.get('amplification_needed', 0)} dBm
        
        Status: {status}
        Bypass Difficulty: {results.get('bypass_difficulty', 0)}/10
        
        Details: {vuln.get('details', '')}
        """
        
        QMessageBox.information(self, "Amplification Simulation", msg)
    
    def on_amplification_completed(self, success, message):
        """Handle signal amplification simulation completion."""
        self.ble_progress_bar.setRange(0, 100)  # Restore determinate progress
        
        if success:
            self.log_display.append_log("Signal amplification simulation completed", logging.INFO)
        else:
            self.log_display.append_log(f"Signal amplification simulation failed: {message}", logging.ERROR)
            QMessageBox.warning(self, "Simulation Error", f"Simulation failed: {message}")
    
    def scan_for_fhss_pattern(self):
        """Scan for FHSS hop pattern."""
        if not FHSS_MODULE_AVAILABLE:
            QMessageBox.warning(self, "Module Error", "FHSS bypass module not available")
            return
        
        if not HACKRF_AVAILABLE:
            QMessageBox.warning(self, "Hardware Error", "HackRF hardware not available")
            return
        
        # Get parameters
        frequency = self.fhss_frequency_input.value() * 1e6  # Convert MHz to Hz
        bandwidth = self.fhss_bandwidth_input.value() * 1e6  # Convert MHz to Hz
        channels = self.fhss_channels_input.value()
        duration = self.fhss_duration_input.value()
        
        self.log_display.append_log(f"Scanning for FHSS pattern at {frequency/1e6} MHz with {channels} channels...", logging.INFO)
        
        try:
            # Create FHSS analyzer
            analyzer = FHSSAnalyzer(
                base_frequency=frequency,
                hop_bandwidth=bandwidth,
                num_channels=channels,
                sample_rate=20e6  # Fixed sample rate for FHSS scan
            )
            
            # Run scan in a thread
            worker = WorkerThread(analyzer.scan_for_hop_pattern, duration=duration)
            worker.update_result.connect(self.display_fhss_scan_results)
            worker.task_completed.connect(self.on_fhss_scan_completed)
            worker.start()
            
            # Show progress
            self.fhss_progress_bar.setRange(0, 100)
            
            # Start progress timer
            self.fhss_progress = 0
            self.fhss_progress_timer = QTimer()
            self.fhss_progress_timer.timeout.connect(self.update_fhss_progress)
            self.fhss_progress_timer.start(100)  # Update every 100ms
        except Exception as e:
            self.log_display.append_log(f"Error scanning for FHSS pattern: {e}", logging.ERROR)
            QMessageBox.warning(self, "Scan Error", f"Error scanning for FHSS pattern: {e}")
    
    def update_fhss_progress(self):
        """Update FHSS scan progress bar."""
        duration = self.fhss_duration_input.value() * 1000  # Convert to ms
        self.fhss_progress += 100  # 100ms increment
        progress = min(100, int(self.fhss_progress / duration * 100))
        self.fhss_progress_bar.setValue(progress)
        
        if progress >= 100:
            self.fhss_progress_timer.stop()
    
    def display_fhss_scan_results(self, hop_sequence):
        """Display FHSS scan results."""
        if not hop_sequence or len(hop_sequence) == 0:
            self.fhss_pattern_text.setPlainText("No hop pattern detected")
            return
        
        # Format the sequence
        sequence_text = f"Detected {len(hop_sequence)} hops: {hop_sequence}"
        self.fhss_pattern_text.setPlainText(sequence_text)
        
        # Update the plot
        self.fhss_plot_widget.clear()
        
        # Create time axis assuming 10 hops per second (default)
        hop_rate = 10.0  # Default
        time_axis = np.arange(len(hop_sequence)) / hop_rate
        
        # Plot the hop sequence
        self.fhss_plot_widget.plot(time_axis, hop_sequence, pen=None, symbol='o', symbolPen='b', symbolBrush='b')
        
        # Set appropriate range
        self.fhss_plot_widget.setYRange(0, max(hop_sequence) + 1)
        self.fhss_plot_widget.setXRange(0, time_axis[-1])
        
        # Enable bypass button
        self.fhss_bypass_button.setEnabled(True)
        
        # Mock analysis (in real implementation, this would analyze the pattern)
        analysis_text = f"""
        FHSS Pattern Analysis:
        
        Hop Sequence Length: {len(hop_sequence)}
        Estimated Hop Rate: 10.0 hops/second
        
        Pattern Type: {"Sequential" if all(hop_sequence[i] == hop_sequence[i-1] + 1 for i in range(1, len(hop_sequence))) else "Random"}
        
        Channel Statistics:
        - Minimum Channel: {min(hop_sequence)}
        - Maximum Channel: {max(hop_sequence)}
        - Average Channel: {sum(hop_sequence) / len(hop_sequence):.1f}
        
        Attack Difficulty: {"Easy" if len(set(hop_sequence)) < 10 else "Medium" if len(set(hop_sequence)) < 20 else "Hard"}
        """
        
        self.fhss_analysis_text.setPlainText(analysis_text)
    
    def on_fhss_scan_completed(self, success, message):
        """Handle FHSS scan completion."""
        self.fhss_progress_bar.setValue(0)
        
        if success:
            self.log_display.append_log(f"FHSS pattern scan completed: {message}", logging.INFO)
        else:
            self.log_display.append_log(f"FHSS pattern scan failed: {message}", logging.ERROR)
            QMessageBox.warning(self, "Scan Error", f"FHSS pattern scan failed: {message}")
    
    def load_fhss_codes_file(self):
        """Load codes file for FHSS bypass."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Codes File", "", "Text Files (*.txt);;All Files (*)")
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                codes = [line.strip() for line in f.readlines() if line.strip()]
            
            if not codes:
                QMessageBox.warning(self, "File Error", "No valid codes found in file")
                return
            
            self.fhss_codes = codes
            self.fhss_codes_label.setText(f"Loaded {len(codes)} codes")
            self.log_display.append_log(f"Loaded {len(codes)} codes from {file_path}", logging.INFO)
        except Exception as e:
            self.log_display.append_log(f"Error loading codes file: {e}", logging.ERROR)
            QMessageBox.warning(self, "File Error", f"Error loading codes file: {e}")
    
    def execute_fhss_bypass(self):
        """Execute FHSS bypass attack."""
        if not hasattr(self, 'fhss_codes') or not self.fhss_codes:
            QMessageBox.warning(self, "Bypass Error", "No codes loaded. Load a codes file first.")
            return
        
        if not FHSS_MODULE_AVAILABLE:
            QMessageBox.warning(self, "Module Error", "FHSS bypass module not available")
            return
        
        if not HACKRF_AVAILABLE:
            QMessageBox.warning(self, "Hardware Error", "HackRF hardware not available")
            return
        
        # Get parameters
        frequency = self.fhss_frequency_input.value() * 1e6  # Convert MHz to Hz
        bandwidth = self.fhss_bandwidth_input.value() * 1e6  # Convert MHz to Hz
        channels = self.fhss_channels_input.value()
        
        self.log_display.append_log(f"Executing FHSS bypass with {len(self.fhss_codes)} codes...", logging.INFO)
        
        try:
            # Create FHSS bypass
            bypass = FHSSBypass(
                target_frequency=frequency,
                hop_bandwidth=bandwidth,
                num_channels=channels
            )
            
            # Run bypass in a thread
            worker = WorkerThread(bypass.execute_rolling_attack, self.fhss_codes, attack_duration=30.0)
            worker.task_completed.connect(self.on_fhss_bypass_completed)
            worker.start()
            
            # Show progress
            self.fhss_progress_bar.setRange(0, 0)  # Indeterminate progress
        except Exception as e:
            self.log_display.append_log(f"Error executing FHSS bypass: {e}", logging.ERROR)
            QMessageBox.warning(self, "Bypass Error", f"Error executing FHSS bypass: {e}")
    
    def on_fhss_bypass_completed(self, success, message):
        """Handle FHSS bypass completion."""
        self.fhss_progress_bar.setRange(0, 100)  # Restore determinate progress
        
        if success:
            self.log_display.append_log("FHSS bypass completed successfully", logging.INFO)
            QMessageBox.information(self, "Bypass", "FHSS bypass executed successfully")
        else:
            self.log_display.append_log(f"FHSS bypass failed: {message}", logging.ERROR)
            QMessageBox.warning(self, "Bypass Error", f"FHSS bypass failed: {message}")
    
    def start_multi_vector_analysis(self):
        """Start multi-vector analysis."""
        self.log_display.append_log("Starting multi-vector analysis...", logging.INFO)
        
        # Generate attack plan (mock implementation)
        plan = """
        Multi-Vector Attack Plan:
        
        1. RF Signal Analysis
           - Capture at 315.0 MHz for 5 seconds
           - Extract rolling codes
           - Estimated success probability: 65%
        
        2. BLE Proximity Bypass
           - Scan for BLE devices for 10 seconds
           - Target manufacturer: Chamberlain
           - Use signal amplification technique
           - Estimated success probability: 80%
        
        3. FHSS Bypass (if needed)
           - Analyze at 433.92 MHz with 50 channels
           - Execute bypass with extracted codes
           - Estimated success probability: 45%
        
        Recommended attack order:
        1. BLE Proximity Bypass (highest success probability)
        2. RF Signal Analysis
        3. FHSS Bypass
        
        Click "Launch Attack Sequence" to begin the attack.
        """
        
        self.attack_plan_text.setPlainText(plan)
    
    def launch_multi_vector_attack(self):
        """Launch multi-vector attack sequence."""
        self.log_display.append_log("Launching multi-vector attack sequence...", logging.INFO)
        
        # Mock implementation - in a real scenario, this would execute
        # the actual attack sequence
        
        self.attack_progress_text.clear()
        
        # Simulate attack progress
        self.attack_step = 0
        self.attack_timer = QTimer()
        self.attack_timer.timeout.connect(self.update_attack_progress)
        self.attack_timer.start(1000)  # Update every second
    
    def update_attack_progress(self):
        """Update multi-vector attack progress."""
        steps = [
            "Starting BLE proximity bypass...",
            "Scanning for BLE devices...",
            "Found 3 potential targets",
            "Connecting to strongest signal device...",
            "Connection established",
            "Analyzing security vulnerabilities...",
            "Executing proximity bypass...",
            "BLE bypass successful!",
            "Starting RF signal analysis...",
            "Capturing signals at 315.0 MHz...",
            "Analyzing captured signals...",
            "Extracted 8 potential rolling codes",
            "Testing codes...",
            "RF attack successful!",
            "Multi-vector attack completed successfully!"
        ]
        
        if self.attack_step < len(steps):
            self.attack_progress_text.append(steps[self.attack_step])
            self.attack_step += 1
            
            # Update progress bar
            progress = int(self.attack_step / len(steps) * 100)
            self.multi_attack_progress_bar.setValue(progress)
        else:
            self.attack_timer.stop()
            QMessageBox.information(self, "Attack Complete", "Multi-vector attack sequence completed successfully")
    
    def open_signal_file(self):
        """Open a signal file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Signal File", "", "Data Files (*.dat);;NumPy Files (*.npy);;All Files (*)")
        if not file_path:
            return
        
        try:
            if file_path.endswith('.npy'):
                data = np.load(file_path)
            else:
                with open(file_path, 'rb') as f:
                    data = np.frombuffer(f.read(), dtype=np.complex64)
            
            self.signal_data = data
            self.update_signal_plot(data)
            self.log_display.append_log(f"Loaded signal from {file_path}", logging.INFO)
        except Exception as e:
            self.log_display.append_log(f"Error loading signal file: {e}", logging.ERROR)
            QMessageBox.warning(self, "File Error", f"Error loading signal file: {e}")
    
    def save_signal_file(self):
        """Save the current signal to a file."""
        if self.signal_data is None or len(self.signal_data) == 0:
            QMessageBox.warning(self, "Save Error", "No signal data to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Signal File", "", "Data Files (*.dat);;NumPy Files (*.npy);;All Files (*)")
        if not file_path:
            return
        
        try:
            if file_path.endswith('.npy'):
                np.save(file_path, self.signal_data)
            else:
                with open(file_path, 'wb') as f:
                    f.write(self.signal_data.tobytes())
            
            self.log_display.append_log(f"Saved signal to {file_path}", logging.INFO)
        except Exception as e:
            self.log_display.append_log(f"Error saving signal file: {e}", logging.ERROR)
            QMessageBox.warning(self, "File Error", f"Error saving signal file: {e}")
    
    def show_about_dialog(self):
        """Show about dialog."""
        QMessageBox.about(self, "About", """
        <h3>Garage Door Security Analyzer</h3>
        <p>Version 1.0</p>
        <p>A comprehensive tool for analyzing and testing security vulnerabilities in garage door systems.</p>
        <p>This tool implements multiple attack vectors:</p>
        <ul>
            <li>RF Signal Analysis and Replay</li>
            <li>BLE Proximity Bypass</li>
            <li>FHSS Bypass</li>
        </ul>
        <p><b>Important:</b> This tool is for educational and security research purposes only.</p>
        """)
    
    def show_documentation(self):
        """Show documentation."""
        QMessageBox.information(self, "Documentation", """
        Documentation is available in the README.md file included with this tool.
        
        For further assistance, please refer to the individual module documentation:
        - fhss_bypass.py
        - ble_proximity_bypass.py
        - rolling_code.py
        """)


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    
    # Apply default style
    app.setStyle("Fusion")
    
    # Create main window
    window = MainWindow()
    window.show()
    
    # Load settings
    window.load_settings()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()