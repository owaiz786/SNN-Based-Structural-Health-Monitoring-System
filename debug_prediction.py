# debug_prediction.py
import numpy as np
import pickle
import torch
import sys

# Load scaler
print("Loading scaler...")
with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print(f"Scaler mean: {scaler.mean_}")
print(f"Scaler scale: {scaler.scale_}")

# Load model
print("\nLoading model...")
# Add safe globals for PyTorch 2.6+
torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.ndarray])
checkpoint = torch.load('data/snn_model_optimized.pth', weights_only=False, map_location='cpu')
print(f"Model input features: {checkpoint['model_state_dict']['fc1.weight'].shape[1]}")
print(f"Optimal threshold: {checkpoint.get('optimal_threshold', 0.5)}")

# Create a test sample from your actual ESP32 data (from your output)
print("\n" + "="*60)
print("TESTING WITH ACTUAL SENSOR READINGS")
print("="*60)

# These are from your monitor output
test_samples = [
    np.array([-4.92, 4.35, -6.99, 9.59, abs(-4.92)+abs(4.35)+abs(-6.99)], dtype=np.float32),
    np.array([-4.85, 4.40, -6.89, 9.50, abs(-4.85)+abs(4.40)+abs(-6.89)], dtype=np.float32),
    np.array([-5.02, 4.44, -6.83, 9.56, abs(-5.02)+abs(4.44)+abs(-6.83)], dtype=np.float32),
]

for i, sample in enumerate(test_samples):
    print(f"\n📊 Test Sample {i+1}:")
    print(f"   Raw features: {sample}")
    
    # Normalize
    sample_reshaped = sample.reshape(1, -1)
    scaled = scaler.transform(sample_reshaped)[0]
    print(f"   Scaled (z-score): {scaled}")
    
    # Apply sigmoid
    normalized = 1.0 / (1.0 + np.exp(-scaled))
    print(f"   Normalized [0,1]: {normalized}")
    
    # Check if normalized values are reasonable
    if np.any(normalized < 0) or np.any(normalized > 1):
        print(f"   ⚠️  Normalized values out of range!")
    
    # Check spike probability
    print(f"   Spike probabilities: {normalized}")
    
    # Now test with the actual model
    # Create spike train
    time_steps = 120
    probs = torch.tensor(normalized, dtype=torch.float32)
    probs = probs.unsqueeze(-1).expand(-1, time_steps)
    random_matrix = torch.rand_like(probs)
    spikes = (random_matrix < probs).float()
    spikes = spikes.unsqueeze(0)  # Add batch dimension
    
    # Load model architecture
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
    
    # Predict
    with torch.no_grad():
        output = model(spikes)
        probs_out = torch.softmax(output, dim=1)[0].numpy()
        print(f"   Model output: Normal={probs_out[0]*100:.1f}%, Anomaly={probs_out[1]*100:.1f}%")