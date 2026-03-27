# wifi_data_server.py
import socket
import threading
import queue
import json
import time
import numpy as np
from datetime import datetime
import pickle
import torch

class WiFiDataServer:
    """Receives sensor data from ESP32 over WiFi and makes available for dashboard"""
    
    def __init__(self, host='0.0.0.0', port=12345, max_samples=1000):
        self.host = host
        self.port = port
        self.data_queue = queue.Queue(maxsize=1000)
        self.running = True
        
        # Store historical data
        self.data_buffer = []
        self.max_samples = max_samples
        self.connections = []
        
        # Statistics
        self.total_samples = 0
        self.anomaly_count = 0
        self.start_time = time.time()
        
        # Latest data point
        self.latest_data = None
        self.latest_prediction = None
        self.latest_probs = None
        
    def start_server(self):
        """Start TCP server in background thread"""
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        print(f"📡 WiFi data server running on {self.host}:{self.port}")
        
    def _run_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(5)
        
        while self.running:
            try:
                client, addr = server.accept()
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client, addr),
                    daemon=True
                )
                client_thread.start()
            except:
                break
        
        server.close()
    
    def _handle_client(self, client, addr):
        """Handle incoming data from ESP32"""
        buffer = ""
        while self.running:
            try:
                data = client.recv(1024).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self._process_data(line.strip())
                    
            except Exception as e:
                print(f"Client error: {e}")
                break
        
        client.close()
    
    def _process_data(self, data_string):
        """Parse and store incoming sensor data"""
        try:
            parts = data_string.split(',')
            if len(parts) >= 8:
                # Parse sensor data
                timestamp = float(parts[0]) if parts[0].replace('.', '').isdigit() else time.time()
                ax = float(parts[1])
                ay = float(parts[2])
                az = float(parts[3])
                gx = float(parts[4]) if len(parts) > 4 else 0
                gy = float(parts[5]) if len(parts) > 5 else 0
                gz = float(parts[6]) if len(parts) > 6 else 0
                mag = float(parts[7])
                
                # Calculate derived features
                sum_abs = abs(ax) + abs(ay) + abs(az)
                
                # Create data point
                data_point = {
                    'timestamp': timestamp,
                    'datetime': datetime.now().strftime('%H:%M:%S.%f')[:-3],
                    'ax': ax, 'ay': ay, 'az': az,
                    'gx': gx, 'gy': gy, 'gz': gz,
                    'mag': mag,
                    'sum_abs': sum_abs,
                    'raw_features': np.array([ax, ay, az, mag, sum_abs], dtype=np.float32)
                }
                
                self.data_buffer.append(data_point)
                if len(self.data_buffer) > self.max_samples:
                    self.data_buffer.pop(0)
                
                self.total_samples += 1
                self.latest_data = data_point
                
        except Exception as e:
            print(f"Parse error: {e}")
    
    def update_prediction(self, prediction, probs):
        """Update latest prediction (called by SNN processor)"""
        self.latest_prediction = prediction
        self.latest_probs = probs
        if prediction == 1:
            self.anomaly_count += 1
    
    def get_data(self):
        """Get current data for dashboard"""
        return {
            'latest_data': self.latest_data,
            'latest_prediction': self.latest_prediction,
            'latest_probs': self.latest_probs,
            'history': self.data_buffer[-200:],  # Last 200 samples for plotting
            'total_samples': self.total_samples,
            'anomaly_count': self.anomaly_count,
            'uptime': time.time() - self.start_time
        }
    
    def stop(self):
        self.running = False