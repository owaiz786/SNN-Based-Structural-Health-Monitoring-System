"""
FIXED SNN Monitor - Real-time Anomaly Detection via Serial (ESP32 + MPU6050)

ROOT CAUSE FIXES:
  1. Uses ImprovedSNN (correct architecture matching snn_model.py)
  2. Applies StandardScaler normalization BEFORE spike encoding
  3. Converts features to proper spike trains (shape: 1 x features x time_steps)
  4. Loads scaler from disk (saved during training)
  5. Uses the optimal_threshold saved in the checkpoint
  6. FIXED: Proper feature extraction for 5 features
  7. FIXED: NaN handling in normalization
  8. FIXED: PyTorch 2.6+ compatibility with weights_only parameter
"""

import serial
import torch
import torch.nn as nn
import norse.torch as norse
import numpy as np
import time
import sys
import os
import pickle

# =====================================================================
# CORRECT MODEL ARCHITECTURE (must match snn_model.py : ImprovedSNN)
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
# SPIKE ENCODING (matches RobustSpikeEncoder in snn_model.py)
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
# SCALER - Load saved sklearn scaler
# =====================================================================
SCALER_PATH = 'data/scaler_esp32.pkl'

def load_scaler():
    """Load scaler with error handling"""
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler loaded from {SCALER_PATH}")
        print(f"   Expected features: {scaler.mean_.shape[0]}")
        return scaler
    except FileNotFoundError:
        print(f"⚠️  Scaler file not found: {SCALER_PATH}")
        print(f"   Please run: python recreate_scaler.py")
        return None
    except Exception as e:
        print(f"⚠️  Could not load scaler: {e}")
        print(f"   Please run: python recreate_scaler.py")
        return None


# =====================================================================
# FEATURE EXTRACTION - FIXED for 5 features
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
# NORMALIZE FEATURES - Using saved scaler
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
# LOAD MODEL - FIXED for PyTorch 2.6+
# =====================================================================
MODEL_PATH = 'data/snn_model_optimized.pth'
DEVICE = torch.device('cpu')

print("\n" + "=" * 60)
print("FIXED SNN MONITOR")
print("=" * 60)

# ---- Load checkpoint with PyTorch 2.6+ compatibility ----
print(f"\n🔄 Loading model from {MODEL_PATH} ...")

# Add NumPy to safe globals for PyTorch 2.6+ if needed
try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.ndarray])
except:
    pass

checkpoint = None
load_error = None

# Try different loading methods
loading_methods = [
    # Method 1: weights_only=False (most compatible)
    lambda: torch.load(MODEL_PATH, weights_only=False, map_location='cpu', encoding='latin1'),
    # Method 2: weights_only=True with safe globals
    lambda: torch.load(MODEL_PATH, weights_only=True, map_location='cpu'),
    # Method 3: without encoding parameter
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

# Get configuration from checkpoint
config = checkpoint.get('config', {})
TIME_STEPS   = config.get('time_steps', 120)
hidden_1     = config.get('hidden_size_1', 128)
hidden_2     = config.get('hidden_size_2', 64)
hidden_3     = config.get('hidden_size_3', 32)
dropout_rate = config.get('dropout', 0.3)
threshold    = checkpoint.get('optimal_threshold', 0.5)

# Infer input size from saved weights
input_size = checkpoint['model_state_dict']['fc1.weight'].shape[1]
print(f"\n📊 Model Configuration:")
print(f"   Input features : {input_size}")
print(f"   Time steps     : {TIME_STEPS}")
print(f"   Threshold      : {threshold:.3f}")
print(f"   Hidden layers  : {hidden_1} → {hidden_2} → {hidden_3}")

# Create and load model
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

# ---- Load scaler ----
scaler = load_scaler()
if scaler is None:
    print("❌ Scaler not available. Cannot run monitor.")
    print("   Please run: python recreate_scaler.py")
    sys.exit(1)

# Verify feature count matches
if scaler.mean_.shape[0] != input_size:
    print(f"⚠️  WARNING: Feature count mismatch!")
    print(f"   Scaler expects: {scaler.mean_.shape[0]} features")
    print(f"   Model expects: {input_size} features")
    print("   This may cause errors during normalization!")
    print("   Please recreate scaler with correct feature count.")
    sys.exit(1)

print(f"✅ System ready! Features: {input_size}, Time steps: {TIME_STEPS}\n")

# =====================================================================
# SERIAL CONNECTION
# =====================================================================
SERIAL_PORT = 'COM6'      # ← CHANGE THIS TO YOUR COM PORT
BAUD_RATE   = 115200

print(f"🔌 Connecting to {SERIAL_PORT} ...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
    ser.dtr = False
    time.sleep(0.5)
    ser.dtr = True
    time.sleep(3)
    ser.reset_input_buffer()
    print("✅ Connected!\n")
except Exception as e:
    print(f"❌ Serial connection failed: {e}")
    print("\nIf you don't have ESP32 connected, use test mode:")
    print("  python serial_snn_monitor_test.py")
    sys.exit(1)


# =====================================================================
# MAIN LOOP
# =====================================================================
print("=" * 60)
print("🚀 MONITORING STARTED")
print(f"   Threshold : {threshold:.3f}  (anomaly if P(anomaly) ≥ threshold)")
print(f"   Features  : {input_size}")
print("   Press Ctrl+C to stop")
print("=" * 60 + "\n")

count         = 0
anomaly_count = 0
start_time    = time.time()
debug_samples = 0  # Counter for debug prints

try:
    while True:
        if ser.in_waiting == 0:
            time.sleep(0.05)
            continue

        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue

        try:
            parts = line.split(',')
            if len(parts) < 8:  # Need at least idx,ax,ay,az,gx,gy,gz,mag
                continue

            # 1. Extract raw features (5 features)
            raw_features = extract_features(parts, input_size)
            
            # 2. Normalize using saved scaler
            norm_features = normalize_features(raw_features, scaler)
            
            # 3. Encode to spike train (1, features, time_steps)
            spike_tensor = rate_encode_sample(norm_features, TIME_STEPS).to(DEVICE)
            
            # 4. Inference
            with torch.no_grad():
                output = model(spike_tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                prediction = 1 if probs[1] >= threshold else 0
            
            # 5. Stats & display
            count += 1
            if prediction == 1:
                anomaly_count += 1
            
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0
            mag_val = raw_features[3]  # Magnitude
            status = "⚠️  ANOMALY" if prediction else "✅ NORMAL"
            
            # Determine confidence color indicator
            if probs[1] >= 0.8:
                conf_indicator = "🔴"
            elif probs[1] >= 0.5:
                conf_indicator = "🟡"
            else:
                conf_indicator = "🟢"
            
            # Display with confidence
            print(f"[{count:4d}] {status} {conf_indicator} | "
                  f"X={raw_features[0]:+6.2f} Y={raw_features[1]:+6.2f} Z={raw_features[2]:+6.2f} | "
                  f"Mag={mag_val:5.2f} | "
                  f"P(anom)={probs[1]*100:5.1f}% | "
                  f"Rate={rate:.2f}Hz")
            
            # Debug print for first few samples or when confidence changes significantly
            if debug_samples < 5:
                print(f"   [DEBUG] Raw: {raw_features}")
                print(f"   [DEBUG] Norm: {norm_features}")
                debug_samples += 1
            elif count % 500 == 0:
                print(f"\n[STATS] Sample #{count}: Anomaly rate: {anomaly_count/count*100:.1f}%")

        except (ValueError, IndexError) as e:
            # Skip malformed lines silently
            continue
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}")
            continue

        time.sleep(0.02)   # Yield CPU

except KeyboardInterrupt:
    print("\n\n" + "=" * 60)
    print("⏹️  MONITORING STOPPED")
    print("=" * 60)
    
    if count > 0:
        elapsed = time.time() - start_time
        print(f"📊 Session Summary:")
        print(f"   Total samples : {count}")
        print(f"   Anomalies     : {anomaly_count}  ({anomaly_count/count*100:.1f}%)")
        print(f"   Normal        : {count - anomaly_count}  ({(count-anomaly_count)/count*100:.1f}%)")
        print(f"   Duration      : {elapsed:.1f} s")
        print(f"   Actual rate   : {count/elapsed:.2f} Hz")
        
        # Show final verdict
        if anomaly_count / count > 0.5:
            print(f"\n⚠️  WARNING: High anomaly rate detected!")
            print("   This may indicate actual structural issues or scaling problems.")
    print("=" * 60)

finally:
    ser.close()
    print("✅ Serial port closed")