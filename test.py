# test_new_scaler.py
import numpy as np
import pickle
import torch
import sys

# Add safe globals for PyTorch 2.6+
torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.ndarray])

print("=" * 60)
print("Testing New ESP32 Scaler")
print("=" * 60)

# Load the new scaler
print("\n📂 Loading scaler_esp32.pkl...")
with open('data/scaler_esp32.pkl', 'rb') as f:
    scaler = pickle.load(f)
print(f"✅ Scaler loaded")
print(f"   Mean: {scaler.mean_}")
print(f"   Scale: {scaler.scale_}")

# Load model
print("\n📂 Loading model...")
checkpoint = torch.load('data/snn_model_optimized.pth', weights_only=False, map_location='cpu')
print(f"✅ Model loaded")

# Import model architecture
from serial_snn_monitor import ImprovedSNN

model = ImprovedSNN(
    input_size=5,
    hidden_size_1=128,
    hidden_size_2=64,
    hidden_size_3=32,
    output_size=2,
    dropout=0.3
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test with actual samples
print("\n" + "=" * 60)
print("Testing with actual ESP32 samples")
print("=" * 60)

test_samples = [
    np.array([-4.92, 4.35, -6.99, 9.59, 16.26]),
    np.array([-4.85, 4.40, -6.89, 9.50, 16.14]),
    np.array([-5.02, 4.44, -6.83, 9.56, 16.29]),
]

time_steps = 120

for i, sample in enumerate(test_samples):
    print(f"\n📊 Test Sample {i+1}:")
    print(f"   Raw: {sample}")
    
    # Normalize using new scaler
    sample_reshaped = sample.reshape(1, -1)
    scaled = scaler.transform(sample_reshaped)[0]
    normalized = 1.0 / (1.0 + np.exp(-scaled))
    print(f"   Normalized: {normalized}")
    
    # Create spike train
    probs = torch.tensor(normalized, dtype=torch.float32)
    probs = probs.unsqueeze(-1).expand(-1, time_steps)
    random_matrix = torch.rand_like(probs)
    spikes = (random_matrix < probs).float()
    spikes = spikes.unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(spikes)
        probs_out = torch.softmax(output, dim=1)[0].numpy()
        
    print(f"   Model prediction:")
    print(f"     Normal: {probs_out[0]*100:.1f}%")
    print(f"     Anomaly: {probs_out[1]*100:.1f}%")
    
    if probs_out[1] > 0.5:
        print(f"     Result: ⚠️  ANOMALY")
    else:
        print(f"     Result: ✅ NORMAL")