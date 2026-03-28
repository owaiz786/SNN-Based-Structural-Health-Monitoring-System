# test_cpp_inference.py
# Verify C++ inference matches PyTorch

import torch
import numpy as np
import sys
import pickle
import os

# Load model
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

print("Loading model...")
checkpoint = torch.load('D:\SHM-SNN - Working\data\snn_model_optimized.pth', weights_only=False, map_location='cpu')
model_state = checkpoint['model_state_dict']
threshold = 0.7999999999999998

# Import model architecture
try:
    # Add parent directory to path for imports
    sys.path.insert(0, parent_dir)
    from serial_snn_monitor import ImprovedSNN, rate_encode_sample
    
    # Create model
    input_size = 5
    model = ImprovedSNN(
        input_size=input_size,
        hidden_size_1=128,
        hidden_size_2=64,
        hidden_size_3=32,
        output_size=2
    )
    model.load_state_dict(model_state)
    model.eval()
    
    # Load scaler
    with open(os.path.join(parent_dir, 'data', 'scaler_esp32.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Create dummy input (simulate stationary sensor)
    # Features: [ax, ay, az, mag, sum_abs]
    test_input = np.array([[0.0, 0.0, 9.81, 9.81, 9.81]], dtype=np.float32)
    
    # Normalize
    scaled = scaler.transform(test_input)[0]
    normalized = 1.0 / (1.0 + np.exp(-scaled))
    
    # Encode to spikes
    spikes = rate_encode_sample(normalized, 120)
    
    # Run inference
    with torch.no_grad():
        output = model(spikes)
        probs = torch.softmax(output, dim=1)[0].numpy()
        prediction = int(probs[1] >= threshold)
    
    print(f"\n============================================================")
    print(f"PyTorch Inference Test")
    print(f"============================================================")
    print(f"Input: {test_input[0]}")
    print(f"Normalized: {normalized}")
    print(f"Output: Normal={probs[0]*100:.1f}%, Anomaly={probs[1]*100:.1f}%")
    print(f"Prediction: {'ANOMALY' if prediction else 'NORMAL'}")
    print(f"============================================================\n")
    
    # Save test input for C++ verification
    np.save(os.path.join(script_dir, 'test_input.npy'), test_input)
    np.save(os.path.join(script_dir, 'test_output.npy'), probs)
    print("Test data saved for C++ verification")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure serial_snn_monitor.py is in the parent directory")
    import traceback
    traceback.print_exc()
    sys.exit(1)
