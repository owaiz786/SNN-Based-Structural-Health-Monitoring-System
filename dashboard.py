"""
Streamlit Dashboard for SHM-SNN Monitor (Serial Version)
With proper debugging and error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
import queue
import serial
import torch
import pickle
from datetime import datetime
from collections import deque
import sys
import os
import traceback
import socket
import threading

# Import SNN components from your monitor
from serial_snn_monitor import ImprovedSNN, rate_encode_sample, extract_features, normalize_features

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="SHM-SNN Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# DATA BUFFER CLASS
# =====================================================================
class SerialDataBuffer:
    """Thread-safe buffer for serial data"""
    
    def __init__(self, max_samples=500):
        self.max_samples = max_samples
        self.data_queue = queue.Queue(maxsize=100)
        
        # Data storage
        self.timestamps = deque(maxlen=max_samples)
        self.ax_values = deque(maxlen=max_samples)
        self.ay_values = deque(maxlen=max_samples)
        self.az_values = deque(maxlen=max_samples)
        self.mag_values = deque(maxlen=max_samples)
        self.anomaly_scores = deque(maxlen=max_samples)
        self.predictions = deque(maxlen=max_samples)
        
        # Latest values
        self.latest_data = None
        self.latest_prediction = None
        self.latest_probs = None
        self.latest_raw = None
        
        # Statistics
        self.total_samples = 0
        self.anomaly_count = 0
        self.start_time = time.time()
        
        # Status
        self.last_error = None
        self.is_ready = False
        
    def add_data(self, raw_features, prediction, probs):
        """Add a new data point to the buffer"""
        self.total_samples += 1
        if prediction == 1:
            self.anomaly_count += 1
            
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        self.timestamps.append(timestamp)
        self.ax_values.append(raw_features[0])
        self.ay_values.append(raw_features[1])
        self.az_values.append(raw_features[2])
        self.mag_values.append(raw_features[3])
        self.anomaly_scores.append(probs[1])
        self.predictions.append(prediction)
        
        self.latest_data = {
            'timestamp': timestamp,
            'ax': raw_features[0],
            'ay': raw_features[1],
            'az': raw_features[2],
            'mag': raw_features[3],
            'sum_abs': raw_features[4]
        }
        self.latest_prediction = prediction
        self.latest_probs = probs
        self.latest_raw = raw_features
        self.is_ready = True
        
    def get_dataframe(self):
        """Get current data as pandas DataFrame"""
        if len(self.timestamps) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'Timestamp': list(self.timestamps),
            'AX': list(self.ax_values),
            'AY': list(self.ay_values),
            'AZ': list(self.az_values),
            'Magnitude': list(self.mag_values),
            'Anomaly Score': list(self.anomaly_scores),
            'Prediction': list(self.predictions)
        })
    
    def get_stats(self):
        """Get current statistics"""
        elapsed = time.time() - self.start_time
        return {
            'total_samples': self.total_samples,
            'anomaly_count': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / self.total_samples * 100 if self.total_samples > 0 else 0,
            'uptime': elapsed,
            'rate': self.total_samples / elapsed if elapsed > 0 else 0
        }

# =====================================================================
# SERIAL READER THREAD
# =====================================================================
class SerialReader:
    """Reads data from serial port in background thread"""
    
    def __init__(self, port='COM6', baudrate=115200, buffer=None):
        self.port = port
        self.baudrate = baudrate
        self.buffer = buffer
        self.running = False
        self.serial_conn = None
        self.error = None
        self.model_loaded = False
        self.status_message = "Initializing..."
        
    def start(self):
        """Start serial reading thread"""
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop serial reading"""
        self.running = False
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except:
                pass
            
    def _load_model(self):
        """Load SNN model and scaler"""
        try:
            self.status_message = "Loading scaler..."
            # Load scaler
            scaler_path = 'data/scaler_esp32.pkl'
            if not os.path.exists(scaler_path):
                self.error = f"Scaler not found: {scaler_path}"
                self.status_message = self.error
                return False
                
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.status_message = "Loading model..."
            # Load model
            model_path = 'data/snn_model_optimized.pth'
            if not os.path.exists(model_path):
                self.error = f"Model not found: {model_path}"
                self.status_message = self.error
                return False
                
            torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.ndarray])
            checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
            
            self.config = checkpoint.get('config', {})
            self.time_steps = self.config.get('time_steps', 120)
            self.threshold = checkpoint.get('optimal_threshold', 0.5)
            
            input_size = checkpoint['model_state_dict']['fc1.weight'].shape[1]
            
            self.model = ImprovedSNN(
                input_size=input_size,
                hidden_size_1=self.config.get('hidden_size_1', 128),
                hidden_size_2=self.config.get('hidden_size_2', 64),
                hidden_size_3=self.config.get('hidden_size_3', 32),
                output_size=2,
                dropout=self.config.get('dropout', 0.3)
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.input_size = input_size
            self.status_message = f"✅ Model loaded: {input_size} features"
            return True
            
        except Exception as e:
            self.error = f"Model loading error: {e}"
            self.status_message = self.error
            traceback.print_exc()
            return False
            
    def _read_loop(self):
        """Main reading loop"""
        try:
            # Load model first
            if not self._load_model():
                self.status_message = f"Failed to load model: {self.error}"
                return
                
            self.status_message = f"Connecting to {self.port}..."
            
            # Connect to serial
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2)
            self.serial_conn.reset_input_buffer()
            
            self.status_message = f"✅ Connected to {self.port}. Waiting for data..."
            
            # Main loop
            while self.running:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self._process_line(line)
                        self.status_message = f"Receiving data... ({self.buffer.total_samples if self.buffer else 0} samples)"
                else:
                    time.sleep(0.05)
                        
        except Exception as e:
            self.error = str(e)
            self.status_message = f"Serial error: {e}"
            print(f"Serial error: {e}")
            traceback.print_exc()
        finally:
            if self.serial_conn:
                try:
                    self.serial_conn.close()
                except:
                    pass
            self.status_message = "Disconnected"
                
    def _process_line(self, line):
        """Process a single line of serial data"""
        try:
            parts = line.split(',')
            if len(parts) < 8:
                return
                
            # Extract features
            raw_features = extract_features(parts, self.input_size)
            
            # Normalize
            raw_reshaped = raw_features.reshape(1, -1)
            scaled = self.scaler.transform(raw_reshaped)[0]
            normalized = 1.0 / (1.0 + np.exp(-scaled))
            
            # Encode to spikes
            spike_tensor = rate_encode_sample(normalized, self.time_steps)
            
            # Predict
            with torch.no_grad():
                output = self.model(spike_tensor)
                probs = torch.softmax(output, dim=1)[0].numpy()
                prediction = 1 if probs[1] >= self.threshold else 0
                
            # Add to buffer
            if self.buffer:
                self.buffer.add_data(raw_features, prediction, probs)
                
        except Exception as e:
            print(f"Processing error: {e}")

# =====================================================================
# DASHBOARD UI
# =====================================================================
def main():
    st.title("🏗️ Structural Health Monitoring System")
    st.markdown("Real-time anomaly detection using **Spiking Neural Networks**")
    st.markdown("---")
    
    # Initialize session state
    if 'buffer' not in st.session_state:
        st.session_state.buffer = SerialDataBuffer()
        st.session_state.serial_reader = None
        st.session_state.connected = False
        st.session_state.status_text = "Not connected"
        
    # Sidebar
    with st.sidebar:
        st.header("📡 Connection Settings")
        
        # Serial port selection
        port = st.text_input("COM Port", value="COM6")
        baudrate = st.selectbox("Baud Rate", [9600, 115200, 230400], index=1)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔌 Connect", type="primary", use_container_width=True):
                if st.session_state.serial_reader:
                    st.session_state.serial_reader.stop()
                st.session_state.buffer = SerialDataBuffer()
                st.session_state.serial_reader = SerialReader(port, baudrate, st.session_state.buffer)
                st.session_state.serial_reader.start()
                st.session_state.connected = True
                st.rerun()
                
        with col2:
            if st.button("⏹️ Disconnect", use_container_width=True):
                if st.session_state.serial_reader:
                    st.session_state.serial_reader.stop()
                st.session_state.connected = False
                st.rerun()
        
        # Status display
        st.markdown("---")
        st.header("📡 Status")
        
        if st.session_state.serial_reader:
            status = st.session_state.serial_reader.status_message
            st.info(f"🔄 {status}")
            
            if st.session_state.serial_reader.error:
                st.error(f"❌ {st.session_state.serial_reader.error}")
        else:
            st.warning("⏸️ Not connected")
        
        st.markdown("---")
        
        # Statistics (only if connected and have data)
        if st.session_state.connected and st.session_state.buffer.total_samples > 0:
            stats = st.session_state.buffer.get_stats()
            st.header("📊 Statistics")
            st.metric("Total Samples", f"{stats['total_samples']:,}")
            st.metric("Anomaly Rate", f"{stats['anomaly_rate']:.1f}%")
            st.metric("Processing Rate", f"{stats['rate']:.1f} Hz")
            st.metric("Uptime", f"{stats['uptime']/60:.1f} min")
        
        st.markdown("---")
        st.caption(f"Dashboard updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Main content area
    if not st.session_state.connected:
        st.info("👈 Click **Connect** to start monitoring")
        st.markdown("""
        ### Instructions:
        1. Make sure your ESP32 is connected via USB
        2. Select the correct COM port (usually COM3, COM4, COM5, or COM6)
        3. Click **Connect**
        4. The dashboard will automatically display data when received
        """)
        
        # Show available COM ports
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            if ports:
                st.markdown("### Available COM Ports:")
                for p in ports:
                    st.code(f"{p.device} - {p.description}")
        except:
            pass
        return
    
    buffer = st.session_state.buffer
    
    # Check if we have data
    if not buffer.is_ready:
        st.warning("⏳ Waiting for data from ESP32...")
        st.info("Make sure your ESP32 is sending data. You should see messages in the serial monitor.")
        
        # Show last status
        if st.session_state.serial_reader:
            st.code(f"Status: {st.session_state.serial_reader.status_message}")
        return
    
    # Alert banner
    if buffer.latest_prediction == 1:
        st.error(f"""
        ## ⚠️ ANOMALY DETECTED!
        **Confidence:** {buffer.latest_probs[1]*100:.1f}%
        **Timestamp:** {buffer.latest_data['timestamp']}
        
        Unusual vibration pattern detected. Please inspect the structure.
        """)
    else:
        st.success(f"""
        ## ✅ NORMAL OPERATION
        **Confidence:** {(1-buffer.latest_probs[1])*100:.1f}%
        **Timestamp:** {buffer.latest_data['timestamp']}
        
        Structure is operating within normal parameters.
        """)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AX", f"{buffer.latest_data['ax']:.2f} m/s²")
    with col2:
        st.metric("AY", f"{buffer.latest_data['ay']:.2f} m/s²")
    with col3:
        st.metric("AZ", f"{buffer.latest_data['az']:.2f} m/s²")
    with col4:
        st.metric("Magnitude", f"{buffer.latest_data['mag']:.2f} m/s²")
    
    # Real-time plot
    st.subheader("📈 Real-time Sensor Data")
    
    # Create figure
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Acceleration (X, Y, Z)', 'Magnitude', 'Anomaly Score'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Add traces if we have data
    if len(buffer.ax_values) > 0:
        x_axis = list(range(len(buffer.ax_values)))
        
        # Acceleration
        fig.add_trace(go.Scatter(y=list(buffer.ax_values), mode='lines', name='AX',
                                line=dict(color='#ff6b6b', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(y=list(buffer.ay_values), mode='lines', name='AY',
                                line=dict(color='#4ecdc4', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(y=list(buffer.az_values), mode='lines', name='AZ',
                                line=dict(color='#45b7d1', width=2)), row=1, col=1)
        
        # Magnitude
        fig.add_trace(go.Scatter(y=list(buffer.mag_values), mode='lines', name='Magnitude',
                                line=dict(color='#96ceb4', width=2)), row=2, col=1)
        
        # Anomaly score
        fig.add_trace(go.Scatter(y=list(buffer.anomaly_scores), mode='lines', name='Anomaly Score',
                                line=dict(color='#ff6b6b', width=2), fill='tozeroy'), row=3, col=1)
        
        # Threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=3, col=1,
                     annotation_text="Threshold")
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
        fig.update_yaxes(title_text="Magnitude (m/s²)", row=2, col=1)
        fig.update_yaxes(title_text="Score", range=[0, 1], row=3, col=1)
        fig.update_xaxes(title_text="Sample", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent data table
    st.subheader("📋 Recent Data")
    df = buffer.get_dataframe()
    if not df.empty:
        st.dataframe(df.tail(20), use_container_width=True)
    
    # Auto-refresh
    time.sleep(0.5)
    st.rerun()

if __name__ == "__main__":
    main()