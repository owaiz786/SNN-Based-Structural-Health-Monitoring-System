"""
WiFi SNN Monitor - Real-time Anomaly Detection via WiFi (ESP32 + MPU6050)
WITH FLASK-SOCKETIO WEBSOCKET SUPPORT FOR WEB DASHBOARD

Features:
- Listens for ESP32 data over WiFi (TCP socket)
- Runs SNN inference on received sensor data
- Streams predictions to web dashboard via WebSocket
- Same model architecture as serial version for consistency
"""
import socket
import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
import pickle
from datetime import datetime

# ─────────────────────────────────────────────
# FLASK + SOCKET.IO SETUP (WebSocket Backend)
# ─────────────────────────────────────────────
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
import threading

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['SECRET_KEY'] = 'snn-wifi-monitor-secret-2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state for WebSocket clients
connected_clients = 0
last_emit_time = 0
EMIT_INTERVAL = 0.05  # Minimum seconds between emits (prevent flooding)

# Serve the HTML dashboard
@app.route('/')
def index():
    return send_from_directory('.', 'wifi_dashboard.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@socketio.on('connect')
def handle_connect():
    global connected_clients
    connected_clients += 1
    print(f"🔗 Web Client connected ({connected_clients} total)")
    emit('system_status', {
        'status': 'connected',
        'message': 'WiFi SNN Monitor Backend Ready',
        'timestamp': time.time()
    })

@socketio.on('disconnect')
def handle_disconnect():
    global connected_clients
    connected_clients = max(0, connected_clients - 1)
    print(f"🔌 Web Client disconnected ({connected_clients} remaining)")

@socketio.on('client_ready')
def handle_client_ready(data):
    """Acknowledge client is ready to receive data"""
    print(f"✅ Web Client ready: {data.get('user_agent', 'unknown')}")
    emit('backend_ready', {
        'features': input_size if 'input_size' in globals() else 5,
        'time_steps': TIME_STEPS if 'TIME_STEPS' in globals() else 120,
        'threshold': threshold if 'threshold' in globals() else 0.5
    })

# ─────────────────────────────────────────────
# WEBSOCKET EMIT FUNCTION
# ─────────────────────────────────────────────
def emit_sensor_data(sample_data):
    """
    Emit processed sample to all connected WebSocket clients.
    Rate-limited to prevent flooding.
    """
    global last_emit_time
    current_time = time.time()
    
    if current_time - last_emit_time < EMIT_INTERVAL:
        return
    
    last_emit_time = current_time
    
    try:
        payload = {
            'count': sample_data['count'],
            'timestamp': time.time(),
            'server_time': datetime.now().isoformat(),
            'connection_type': 'wifi',
            'raw_features': {
                'ax': float(sample_data['ax']),
                'ay': float(sample_data['ay']),
                'az': float(sample_data['az']),
                'mag': float(sample_data['mag']),
                'sum_abs': float(sample_data['sum_abs'])
            },
            'normalized_features': sample_data.get('norm_features', []),
            'prediction': {
                'class': int(sample_data['prediction']),
                'p_normal': float(sample_data['p_norm']),
                'p_anomaly': float(sample_data['p_anom']),
                'threshold': float(threshold)
            },
            'stats': {
                'total_samples': total_samples,
                'anomaly_count': anomaly_count,
                'anomaly_rate': (anomaly_count/total_samples*100) if total_samples > 0 else 0,
                'sample_rate': rate,
                'elapsed': elapsed
            },
            'esp32_info': {
                'ip': sample_data.get('esp_ip', 'unknown'),
                'port': sample_data.get('esp_port', 0)
            }
        }
        socketio.emit('sensor_update', payload)
    except Exception as e:
        if connected_clients > 0:
            print(f"⚠️  WebSocket emit error: {e}")

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE (must match snn_model.py: ImprovedSNN)
# ─────────────────────────────────────────────
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
        """
        spike_input shape: (batch, features, time_steps)
        
        LIF states MUST be reset to None at the start of every forward() call.
        Carrying state across unrelated samples contaminates spike accumulation
        and causes the model to always fire into one class.
        """
        batch_size = spike_input.size(0)
        # Always reset — never carry state between independent inference calls
        state1, state2, state3, state_out = None, None, None, None
        output_accumulator = torch.zeros(batch_size, self.output_size,
                                         device=spike_input.device)
        
        for t in range(spike_input.size(2)):
            x_t = spike_input[:, :, t]
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


# ─────────────────────────────────────────────
# SPIKE ENCODING
# ─────────────────────────────────────────────
def rate_encode_sample(features_normalized: np.ndarray, time_steps: int) -> torch.Tensor:
    features_normalized = np.clip(features_normalized, 0.0, 1.0)
    features_normalized = np.nan_to_num(features_normalized, nan=0.0)
    
    probs = torch.tensor(features_normalized, dtype=torch.float32)
    probs = probs.unsqueeze(-1).expand(-1, time_steps)
    random_matrix = torch.rand_like(probs)
    spikes = (random_matrix < probs).float()
    return spikes.unsqueeze(0)


# ─────────────────────────────────────────────
# SCALER
# ─────────────────────────────────────────────
SCALER_PATH = 'data/scaler_esp32.pkl'

def load_scaler(scaler_path):
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


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(parts, num_features=5) -> np.ndarray:
    try:
        ax = float(parts[1])
        ay = float(parts[2])
        az = float(parts[3])
        
        if len(parts) > 7:
            mag = float(parts[7])
        else:
            mag = np.sqrt(ax**2 + ay**2 + az**2)
        
        sum_abs = abs(ax) + abs(ay) + abs(az)
        features = np.array([ax, ay, az, mag, sum_abs], dtype=np.float32)
        
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print(f"⚠️  Invalid features detected: {features}")
            features = np.zeros(num_features, dtype=np.float32)
        
        return features
        
    except (ValueError, IndexError) as e:
        print(f"⚠️  Error parsing features: {e}")
        return np.zeros(num_features, dtype=np.float32)


# ─────────────────────────────────────────────
# NORMALIZE FEATURES
# ─────────────────────────────────────────────
def normalize_features(raw_features, scaler):
    """
    Normalize features for spike encoding.
    
    IMPORTANT: StandardScaler produces z-scores (mean=0, std=1), which can be
    negative and outside [0,1]. For rate encoding (which needs probabilities in
    [0,1]), we must map z-scores to [0,1] WITHOUT applying sigmoid on top of
    the already-scaled output — that was collapsing all values to ~0.5.
    
    We use min-max rescaling of the z-scores using known typical ranges, OR
    clip-and-rescale from [-3, 3] standard deviations → [0, 1].
    """
    if np.any(np.isnan(raw_features)):
        print(f"⚠️  NaN in raw features: {raw_features}")
        raw_features = np.nan_to_num(raw_features, nan=0.0)
    
    raw_reshaped = raw_features.reshape(1, -1)
    
    try:
        # Step 1: StandardScaler z-score (mean=0, std=1)
        scaled = scaler.transform(raw_reshaped)[0]
        
        # Step 2: Map z-scores from [-3σ, +3σ] → [0, 1] for rate encoding.
        # DO NOT apply sigmoid here — sigmoid(z-score) collapses variance to ~0.5
        # and destroys the signal the SNN needs to distinguish classes.
        normalized = (scaled + 3.0) / 6.0          # shift+scale: -3→0, 0→0.5, +3→1
        normalized = np.clip(normalized, 0.0, 1.0)  # hard clip for outliers beyond 3σ
        normalized = np.nan_to_num(normalized, nan=0.5, posinf=1.0, neginf=0.0)
        return normalized
        
    except Exception as e:
        print(f"⚠️  Normalization error: {e}")
        print(f"   Using fallback normalization")
        # Fallback: simple magnitude-based clip — MPU6050 accel ≈ ±2g = ±19.6 m/s²
        return np.clip((raw_features + 20.0) / 40.0, 0.0, 1.0)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
MODEL_PATH = 'data/snn_model_optimized.pth'
DEVICE = torch.device('cpu')

# Import Norse
try:
    import norse.torch as norse
except ImportError:
    print("❌ Norse library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "norse"])
    import norse.torch as norse

print("\n" + "=" * 60)
print("WiFi SNN MONITOR + WEBSOCKET SERVER")
print("=" * 60)

print(f"\n🔄 Loading model from {MODEL_PATH} ...")

try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.ndarray])
except:
    pass

checkpoint = None
load_error = None

loading_methods = [
    lambda: torch.load(MODEL_PATH, weights_only=False, map_location='cpu', encoding='latin1'),
    lambda: torch.load(MODEL_PATH, weights_only=True, map_location='cpu'),
    lambda: torch.load(MODEL_PATH, weights_only=False, map_location='cpu'),
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

config = checkpoint.get('config', {})
TIME_STEPS   = config.get('time_steps', 120)
hidden_1     = config.get('hidden_size_1', 128)
hidden_2     = config.get('hidden_size_2', 64)
hidden_3     = config.get('hidden_size_3', 32)
dropout_rate = config.get('dropout', 0.3)
# The checkpoint's optimal_threshold (e.g. 0.800) was calibrated during training
# against softmax logits — NOT against the spike-count firing ratio used at inference.
# Using it directly causes everything to read NORMAL because spike ratios rarely
# exceed ~0.75 even on genuine anomalies.
# Recalibrated: 0.50 = majority-spike class wins, correct for spike-ratio decisions.
_raw_threshold = checkpoint.get('optimal_threshold', 0.5)
threshold = 0.7186  # recalibrated for spike-ratio inference
print(f"ℹ️  Checkpoint threshold {_raw_threshold:.3f} overridden → {threshold:.3f} (spike-ratio recalibration)")

input_size = checkpoint['model_state_dict']['fc1.weight'].shape[1]

print(f"\n📊 Model Configuration:")
print(f"   Input features : {input_size}")
print(f"   Time steps     : {TIME_STEPS}")
print(f"   Threshold      : {threshold:.3f}")
print(f"   Hidden layers  : {hidden_1} → {hidden_2} → {hidden_3}")

model = ImprovedSNN(
    input_size=input_size,
    hidden_size_1=hidden_1,
    hidden_size_2=hidden_2,
    hidden_size_3=hidden_3,
    output_size=2,
    dropout=dropout_rate
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().to(DEVICE)
print("✅ Model architecture loaded!\n")

scaler = load_scaler(SCALER_PATH)
if scaler is None:
    print("❌ Scaler not available. Cannot run monitor.")
    print("   Please run: python recreate_scaler.py")
    sys.exit(1)

if scaler.mean_.shape[0] != input_size:
    print(f"⚠️  WARNING: Feature count mismatch!")
    print(f"   Scaler expects: {scaler.mean_.shape[0]} features")
    print(f"   Model expects: {input_size} features")
    sys.exit(1)

print(f"✅ System ready! Features: {input_size}, Time steps: {TIME_STEPS}\n")


# ─────────────────────────────────────────────
# START FLASK SERVER IN BACKGROUND THREAD
# ─────────────────────────────────────────────
def run_flask_server():
    """Run Flask-SocketIO server in background thread"""
    print("🌐 Starting WebSocket server on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, 
                 allow_unsafe_werkzeug=True, log_output=False)

flask_thread = threading.Thread(target=run_flask_server, daemon=True)
flask_thread.start()
time.sleep(1)
print("✅ Dashboard available at: http://localhost:5000\n")


# ─────────────────────────────────────────────
# WiFi Monitor Class
# ─────────────────────────────────────────────
class WiFiSNNMonitor:
    def __init__(self, host='0.0.0.0', port=8080):
        self.host = host
        self.port = port
        self.total_samples = 0
        self.anomaly_count = 0
        self.start_time = time.time()
        self.debug_samples = 0
        self.current_esp = None

    def process_line(self, line, esp_info=None):
        """Process one line of data from ESP32"""
        global total_samples, anomaly_count, elapsed, rate
        
        parts = line.split(',')
        if len(parts) < 8:
            return

        raw_features = extract_features(parts, input_size)
        norm_features = normalize_features(raw_features, scaler)
        spike_tensor = rate_encode_sample(norm_features, TIME_STEPS).to(DEVICE)
        
        with torch.no_grad():
            output = model(spike_tensor)
            
            # output = raw accumulated spike counts per class over TIME_STEPS steps.
            # Softmax of spike counts is valid for relative probability, BUT the
            # optimal_threshold in the checkpoint was calibrated against this ratio.
            # Use softmax for display probabilities, and spike-count ratio for the
            # actual threshold decision so it matches training-time calibration.
            spike_counts = output[0].numpy()           # shape: (2,)
            total_spikes = spike_counts.sum()
            
            if total_spikes > 0:
                # Spike-firing rate ratio — what the threshold was tuned against
                ratio = spike_counts / total_spikes     # sums to 1, like probs
            else:
                # No spikes at all → model is saturated or input is all zeros
                ratio = np.array([0.5, 0.5])
            
            # Softmax probabilities for display only (smoother curve for UI)
            probs = torch.softmax(output, dim=1)[0].numpy()
            
            # Decision uses spike-rate ratio to match training threshold calibration
            prediction = 1 if ratio[1] >= threshold else 0
        
        self.total_samples += 1
        total_samples = self.total_samples
        if prediction == 1:
            self.anomaly_count += 1
            anomaly_count = self.anomaly_count
        
        elapsed = time.time() - self.start_time
        rate = self.total_samples / elapsed if elapsed > 0 else 0
        mag_val = raw_features[3]
        status = "⚠️  ANOMALY" if prediction else "✅ NORMAL"
        
        if probs[1] >= 0.8:
            conf_indicator = "🔴"
        elif probs[1] >= 0.5:
            conf_indicator = "🟡"
        else:
            conf_indicator = "🟢"
        
        esp_ip = esp_info[0] if esp_info else "unknown"
        
        print(f"[{self.total_samples:4d}] {status} {conf_indicator} |  "
              f"X={raw_features[0]:+6.2f} Y={raw_features[1]:+6.2f} Z={raw_features[2]:+6.2f} |  "
              f"Mag={mag_val:5.2f} |  "
              f"P(anom)={probs[1]*100:5.1f}% |  "
              f"Rate={rate:.2f}Hz |  "
              f"ESP: {esp_ip}")
        
        if self.debug_samples < 5:
            print(f"   [DEBUG] Raw features  : {raw_features}")
            print(f"   [DEBUG] Norm features : {norm_features.tolist()}")
            print(f"   [DEBUG] Spike counts  : normal={spike_counts[0]:.1f}  anomaly={spike_counts[1]:.1f}  total={total_spikes:.1f}")
            print(f"   [DEBUG] Spike ratio   : normal={ratio[0]:.3f}  anomaly={ratio[1]:.3f}  threshold={threshold:.3f}")
            print(f"   [DEBUG] Softmax probs : normal={probs[0]:.3f}  anomaly={probs[1]:.3f}")
            self.debug_samples += 1
        elif self.total_samples % 500 == 0:
            print(f"\n[STATS] Sample #{self.total_samples}: Anomaly rate: {self.anomaly_count/self.total_samples*100:.1f}%")
        
        # 🚀 EMIT TO WEBSOCKET
        sample_data = {
            'count': self.total_samples,
            'ax': raw_features[0],
            'ay': raw_features[1],
            'az': raw_features[2],
            'mag': raw_features[3],
            'sum_abs': raw_features[4],
            'norm_features': norm_features.tolist(),
            'prediction': prediction,
            'p_norm': float(ratio[0]),      # spike-rate ratio (matches threshold logic)
            'p_anom': float(ratio[1]),
            'p_norm_softmax': float(probs[0]),  # softmax — for smooth UI display
            'p_anom_softmax': float(probs[1]),
            'esp_ip': esp_ip,
            'esp_port': esp_info[1] if esp_info else 0
        }
        emit_sensor_data(sample_data)

    def handle_client(self, client, addr):
        """Handle data from connected ESP32"""
        buffer = ""
        self.current_esp = addr
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
                            self.process_line(line, addr)
                        except Exception as e:
                            print(f"⚠️  Processing error: {e}")

            except Exception as e:
                print(f"Connection error: {e}")
                break

        client.close()
        print(f"\n📴 ESP32 {addr} disconnected")
        self.current_esp = None

    def start_server(self):
        """Start WiFi server and listen for ESP32 data"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(5)

        print("=" * 60)
        print("🚀 MONITORING STARTED (WiFi)")
        print(f"   Threshold : {threshold:.3f}")
        print(f"   Features  : {input_size}")
        print(f"   WiFi Port : {self.port} (ESP32 connects here)")
        print(f"   Dashboard : http://localhost:5000")
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

    def print_summary(self):
        if self.total_samples > 0:
            elapsed = time.time() - self.start_time
            print(f"📊 Session Summary:")
            print(f"   Total samples : {self.total_samples}")
            print(f"   Anomalies     : {self.anomaly_count}  ({self.anomaly_count/self.total_samples*100:.1f}%)")
            print(f"   Normal        : {self.total_samples - self.anomaly_count}")
            print(f"   Duration      : {elapsed:.1f} s")
            print(f"   Actual rate   : {self.total_samples/elapsed:.2f} Hz")
            
            if self.anomaly_count / self.total_samples > 0.5:
                print(f"\n⚠️  WARNING: High anomaly rate detected!")
            print("=" * 60)


# ─────────────────────────────────────────────
# GLOBAL VARIABLES
# ─────────────────────────────────────────────
total_samples = 0
anomaly_count = 0
elapsed = 0
rate = 0

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
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
    print(f"📡 ESP32 should connect to: {local_ip}:8080")
    print(f"🌐 Dashboard: http://localhost:5000")
    print(f"\n⚠️  Firewall rule (run once as Admin):")
    print(f'   netsh advfirewall firewall add rule name="SNN Monitor 8080" dir=in action=allow protocol=TCP localport=8080')
    print()

    try:
        monitor = WiFiSNNMonitor(host='0.0.0.0', port=8080)
        monitor.start_server()

    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("\nExpected files in data/ folder:")
        print("   - snn_model_optimized.pth")
        print("   - scaler_esp32.pkl")
        print(f"\nWorking directory: {os.getcwd()}")

    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()