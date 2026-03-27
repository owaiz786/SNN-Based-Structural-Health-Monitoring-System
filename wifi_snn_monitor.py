# wifi_snn_monitor_corrected.py
import socket
import torch
import torch.nn as nn
import numpy as np
import time
import sys
import pickle
from datetime import datetime

# =====================================================================
# NORSE IMPORT (exactly as in serial_snn_monitor.py)
# =====================================================================
try:
    import norse.torch as norse
except ImportError:
    print("❌ Norse library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "norse"])
    import norse.torch as norse


# =====================================================================
# MODEL ARCHITECTURE - EXACT COPY from serial_snn_monitor.py
# =====================================================================
class ImprovedSNN(nn.Module):
    def __init__(self, input_size, hidden_size_1=128, hidden_size_2=64,
                 hidden_size_3=32, output_size=2, dropout=0.3):
        super().__init__()

        self.bn_input = nn.BatchNorm1d(input_size)

        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.lif1 = norse.LIFCell()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = nn.BatchNorm1d(hidden_size_2)
        self.lif2 = norse.LIFCell()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.bn3 = nn.BatchNorm1d(hidden_size_3)
        self.lif3 = norse.LIFCell()
        self.dropout3 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_size_3, output_size)
        self.lif_out = norse.LIFCell()

        self.output_size = output_size

    def forward(self, spike_input):
        """spike_input shape: (batch, features, time_steps)"""
        batch_size = spike_input.size(0)
        state1, state2, state3, state_out = None, None, None, None
        output_accumulator = torch.zeros(batch_size, self.output_size,
                                         device=spike_input.device)

        for t in range(spike_input.size(2)):
            x_t = spike_input[:, :, t]              # (batch, features)
            x_t = self.bn_input(x_t)

            c1 = self.bn1(self.fc1(x_t))
            s1, state1 = self.lif1(c1, state1)
            s1 = self.dropout1(s1)

            c2 = self.bn2(self.fc2(s1))
            s2, state2 = self.lif2(c2, state2)
            s2 = self.dropout2(s2)

            c3 = self.bn3(self.fc3(s2))
            s3, state3 = self.lif3(c3, state3)
            s3 = self.dropout3(s3)

            c_out = self.fc_out(s3)
            s_out, state_out = self.lif_out(c_out, state_out)
            output_accumulator += s_out

        return output_accumulator


# =====================================================================
# SPIKE ENCODING - EXACT COPY from serial_snn_monitor.py
# =====================================================================
def rate_encode_sample(features_normalized: np.ndarray, time_steps: int) -> torch.Tensor:
    """
    Convert a single normalized feature vector → spike train tensor.
    Input  : (num_features,) values in [0, 1] after normalization
    Output : (1, num_features, time_steps) float32 {0, 1}
    """
    # Ensure values are in [0, 1] range
    features_normalized = np.clip(features_normalized, 0.0, 1.0)
    # Replace any NaN with 0
    features_normalized = np.nan_to_num(features_normalized, nan=0.0)
    
    probs = torch.tensor(features_normalized, dtype=torch.float32)          # (F,)
    probs = probs.unsqueeze(-1).expand(-1, time_steps)                      # (F, T)
    random_matrix = torch.rand_like(probs)
    spikes = (random_matrix < probs).float()                                # (F, T)
    return spikes.unsqueeze(0)                                              # (1, F, T)


# =====================================================================
# FEATURE EXTRACTION - EXACT COPY from serial_snn_monitor.py
# =====================================================================
def extract_features(parts, num_features=5) -> np.ndarray:
    """
    Parse one CSV line from ESP32 and return raw feature array.
    Expected CSV format: idx,ax,ay,az,gx,gy,gz,mag,...
    
    Features (5 total):
        0: ax (accel X)
        1: ay (accel Y)
        2: az (accel Z)
        3: mag (magnitude from ESP32 or computed)
        4: sum_abs (|ax| + |ay| + |az|)
    """
    try:
        # Extract basic values
        ax = float(parts[1])
        ay = float(parts[2])
        az = float(parts[3])
        
        # Try to get mag from CSV, otherwise compute it
        if len(parts) > 7:
            mag = float(parts[7])  # Use mag from ESP32 if available
        else:
            mag = np.sqrt(ax**2 + ay**2 + az**2)  # Compute magnitude
        
        # Compute sum of absolute values (derived feature)
        sum_abs = abs(ax) + abs(ay) + abs(az)
        
        # Build feature vector (5 features as per training)
        features = np.array([ax, ay, az, mag, sum_abs], dtype=np.float32)
        
        # Check for NaN or Inf
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print(f"⚠️  Invalid features detected: {features}")
            # Return default values
            features = np.zeros(num_features, dtype=np.float32)
        
        return features
        
    except (ValueError, IndexError) as e:
        print(f"⚠️  Error parsing features: {e}")
        # Return zeros as fallback
        return np.zeros(num_features, dtype=np.float32)


# =====================================================================
# NORMALIZE FEATURES - EXACT COPY from serial_snn_monitor.py
# =====================================================================
def normalize_features(raw_features, scaler):
    """
    Normalize features using the saved StandardScaler.
    Handles NaN values properly.
    """
    # Check for NaN in raw features
    if np.any(np.isnan(raw_features)):
        print(f"⚠️  NaN in raw features: {raw_features}")
        raw_features = np.nan_to_num(raw_features, nan=0.0)
    
    # Reshape for sklearn scaler
    raw_reshaped = raw_features.reshape(1, -1)
    
    try:
        # Apply StandardScaler (z-score normalization)
        scaled = scaler.transform(raw_reshaped)[0]
        
        # Apply sigmoid to map to [0, 1] range for spike encoding
        # This matches what the model was trained on
        normalized = 1.0 / (1.0 + np.exp(-scaled))
        
        # Handle any NaN/Inf from transformation
        normalized = np.nan_to_num(normalized, nan=0.5, posinf=1.0, neginf=0.0)
        
        return normalized
        
    except Exception as e:
        print(f"⚠️  Normalization error: {e}")
        # Fallback: simple min-max scaling using default range
        print(f"   Using fallback normalization")
        return np.clip(raw_features / 10.0, 0.0, 1.0)  # Assume range [-10, 10]


# =====================================================================
# LOAD SCALER - EXACT COPY from serial_snn_monitor.py
# =====================================================================
def load_scaler(scaler_path):
    """Load scaler with error handling"""
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler loaded from {scaler_path}")
        print(f"   Expected features: {scaler.mean_.shape[0]}")
        return scaler
    except FileNotFoundError:
        print(f"⚠️  Scaler file not found: {scaler_path}")
        return None
    except Exception as e:
        print(f"⚠️  Could not load scaler: {e}")
        return None


# =====================================================================
# WiFi SNN MONITOR
# =====================================================================
class WiFiSNNMonitor:
    def __init__(self, host='0.0.0.0', port=8080,
                 model_path='data/snn_model_optimized.pth',
                 scaler_path='data/scaler_esp32.pkl'):

        self.host = host
        self.port = port
        self.model_path = model_path
        self.scaler_path = scaler_path

        # Load model and scaler
        self.load_model()

        # Statistics
        self.total_samples = 0
        self.anomaly_count = 0
        self.start_time = time.time()
        self.debug_samples = 0

    # ------------------------------------------------------------------
    def load_model(self):
        """Load the trained SNN model and scaler - EXACT same as serial version"""
        import os
        
        print("\n" + "=" * 60)
        print("FIXED SNN MONITOR - WiFi Version")
        print("=" * 60)
        
        print(f"\n🔄 Loading model from {self.model_path} ...")

        # Add NumPy to safe globals for PyTorch 2.6+ if needed
        try:
            torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.ndarray])
        except:
            pass

        checkpoint = None
        load_error = None

        # Try different loading methods (same as serial version)
        loading_methods = [
            lambda: torch.load(self.model_path, weights_only=False, 
                              map_location='cpu', encoding='latin1'),
            lambda: torch.load(self.model_path, weights_only=True, map_location='cpu'),
            lambda: torch.load(self.model_path, weights_only=False, map_location='cpu'),
        ]

        for i, load_method in enumerate(loading_methods, 1):
            try:
                checkpoint = load_method()
                print(f"✅ Model loaded successfully (method {i})")
                break
            except Exception as e:
                load_error = e
                continue

        if checkpoint is None:
            print(f"❌ Error loading model: {load_error}")
            print("   Please retrain the model: python snn_model.py")
            sys.exit(1)

        # Get configuration from checkpoint (same as serial version)
        config = checkpoint.get('config', {})
        self.time_steps = config.get('time_steps', 120)
        hidden_1 = config.get('hidden_size_1', 128)
        hidden_2 = config.get('hidden_size_2', 64)
        hidden_3 = config.get('hidden_size_3', 32)
        dropout_rate = config.get('dropout', 0.3)
        self.threshold = checkpoint.get('optimal_threshold', 0.5)

        # Infer input size from saved weights (same as serial version)
        input_size = checkpoint['model_state_dict']['fc1.weight'].shape[1]
        
        print(f"\n📊 Model Configuration:")
        print(f"   Input features : {input_size}")
        print(f"   Time steps     : {self.time_steps}")
        print(f"   Threshold      : {self.threshold:.3f}")
        print(f"   Hidden layers  : {hidden_1} → {hidden_2} → {hidden_3}")

        # Create and load model (same as serial version)
        self.model = ImprovedSNN(
            input_size=input_size,
            hidden_size_1=hidden_1,
            hidden_size_2=hidden_2,
            hidden_size_3=hidden_3,
            output_size=2,
            dropout=dropout_rate
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("✅ Model architecture loaded!\n")

        # Load scaler (same as serial version)
        self.scaler = load_scaler(self.scaler_path)
        if self.scaler is None:
            print("❌ Scaler not available. Cannot run monitor.")
            print("   Please run: python recreate_scaler.py")
            sys.exit(1)

        # Verify feature count matches
        if self.scaler.mean_.shape[0] != input_size:
            print(f"⚠️  WARNING: Feature count mismatch!")
            print(f"   Scaler expects: {self.scaler.mean_.shape[0]} features")
            print(f"   Model expects: {input_size} features")
            print("   This may cause errors during normalization!")
            print("   Please recreate scaler with correct feature count.")
            sys.exit(1)

        print(f"✅ System ready! Features: {input_size}, Time steps: {self.time_steps}\n")

    # ------------------------------------------------------------------
    def process_line(self, line):
        """Process one line of data - EXACT same logic as serial version"""
        parts = line.split(',')
        if len(parts) < 8:  # Need at least idx,ax,ay,az,gx,gy,gz,mag
            return

        # 1. Extract raw features (5 features)
        raw_features = extract_features(parts, 5)
        
        # 2. Normalize using saved scaler
        norm_features = normalize_features(raw_features, self.scaler)
        
        # 3. Encode to spike train (1, features, time_steps)
        spike_tensor = rate_encode_sample(norm_features, self.time_steps)
        
        # 4. Inference
        with torch.no_grad():
            output = self.model(spike_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()
            prediction = 1 if probs[1] >= self.threshold else 0
        
        # 5. Stats & display (same as serial version)
        self.total_samples += 1
        if prediction == 1:
            self.anomaly_count += 1
        
        elapsed = time.time() - self.start_time
        rate = self.total_samples / elapsed if elapsed > 0 else 0
        mag_val = raw_features[3]  # Magnitude
        status = "⚠️  ANOMALY" if prediction else "✅ NORMAL"
        
        # Determine confidence color indicator
        if probs[1] >= 0.8:
            conf_indicator = "🔴"
        elif probs[1] >= 0.5:
            conf_indicator = "🟡"
        else:
            conf_indicator = "🟢"
        
        # Display with confidence (same format as serial version)
        print(f"[{self.total_samples:4d}] {status} {conf_indicator} | "
              f"X={raw_features[0]:+6.2f} Y={raw_features[1]:+6.2f} Z={raw_features[2]:+6.2f} | "
              f"Mag={mag_val:5.2f} | "
              f"P(anom)={probs[1]*100:5.1f}% | "
              f"Rate={rate:.2f}Hz")
        
        # Debug print for first few samples (same as serial version)
        if self.debug_samples < 5:
            print(f"   [DEBUG] Raw: {raw_features}")
            print(f"   [DEBUG] Norm: {norm_features}")
            self.debug_samples += 1
        elif self.total_samples % 500 == 0:
            print(f"\n[STATS] Sample #{self.total_samples}: Anomaly rate: {self.anomaly_count/self.total_samples*100:.1f}%")

    # ------------------------------------------------------------------
    def handle_client(self, client, addr):
        """Handle data from connected ESP32"""
        buffer = ""
        print(f"\n✅ ESP32 connected from {addr}")
        print("=" * 60 + "\n")

        while True:
            try:
                data = client.recv(1024).decode('utf-8', errors='ignore')
                if not data:
                    break

                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        try:
                            self.process_line(line)
                        except Exception as e:
                            print(f"⚠️  Processing error: {e}")

            except Exception as e:
                print(f"Connection error: {e}")
                break

        client.close()
        print(f"\n📴 ESP32 {addr} disconnected")

    # ------------------------------------------------------------------
    def start_server(self):
        """Start WiFi server and listen for ESP32 data"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(5)

        print("=" * 60)
        print("🚀 MONITORING STARTED (WiFi)")
        print(f"   Threshold : {self.threshold:.3f}  (anomaly if P(anomaly) ≥ threshold)")
        print(f"   Features  : {self.scaler.mean_.shape[0]}")
        print(f"   Listening on {self.host}:{self.port}")
        print("   Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        try:
            while True:
                client, addr = server.accept()
                self.handle_client(client, addr)

        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("⏹️  MONITORING STOPPED")
            print("=" * 60)
            self.print_summary()

        finally:
            server.close()

    # ------------------------------------------------------------------
    def print_summary(self):
        """Print final statistics - same as serial version"""
        if self.total_samples > 0:
            elapsed = time.time() - self.start_time
            print(f"📊 Session Summary:")
            print(f"   Total samples : {self.total_samples}")
            print(f"   Anomalies     : {self.anomaly_count}  ({self.anomaly_count/self.total_samples*100:.1f}%)")
            print(f"   Normal        : {self.total_samples - self.anomaly_count}  ({(self.total_samples-self.anomaly_count)/self.total_samples*100:.1f}%)")
            print(f"   Duration      : {elapsed:.1f} s")
            print(f"   Actual rate   : {self.total_samples/elapsed:.2f} Hz")
            
            # Show final verdict
            if self.anomaly_count / self.total_samples > 0.5:
                print(f"\n⚠️  WARNING: High anomaly rate detected!")
                print("   This may indicate actual structural issues or scaling problems.")
            print("=" * 60)


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🏗️  SNN Structural Health Monitor — WiFi Version")
    print("=" * 60)

    # Get local IP address reliably
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "0.0.0.0"

    local_ip = get_local_ip()
    print(f"\n💻 Your PC's IP  : {local_ip}")
    print(f"📡 Use this in ESP32 code : server_ip = \"{local_ip}\"")
    print(f"🔌 Port          : 8080")
    print("\n⚠️  Firewall rule (run once as Admin if not done):")
    print("   netsh advfirewall firewall add rule name=\"SNN Monitor 8080\" dir=in action=allow protocol=TCP localport=8080")
    print()

    try:
        monitor = WiFiSNNMonitor()
        monitor.start_server()

    except FileNotFoundError as e:
        import os
        print(f"\n❌ File not found: {e}")
        print("\nExpected files in data/ folder:")
        print("   - snn_model_optimized.pth")
        print("   - scaler_esp32.pkl")
        print(f"\nWorking directory: {os.getcwd()}")
        if os.path.exists('data'):
            print("data/ contents:")
            for f in os.listdir('data'):
                print(f"   {f}")
        else:
            print("❌ data/ folder not found!")

    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()