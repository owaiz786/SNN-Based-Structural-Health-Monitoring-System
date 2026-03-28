# test_cpp_inference.py
# Verify C++ inference matches PyTorch

import torch
import numpy as np
import sys
import pickle
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# The model is in the parent directory's data folder
model_path = os.path.join(parent_dir, 'data', 'snn_model_optimized.pth')

if not os.path.exists(model_path):
    print(f"❌ Error: Model not found at {model_path}")
    print("\nSearched locations:")
    print(f"  - {model_path}")
    print(f"  - {os.path.join(script_dir, 'data', 'snn_model_optimized.pth')}")
    print("\nPlease ensure you have trained the model first by running:")
    print("  cd .. && python serial_snn_monitor.py")
    sys.exit(1)

print(f"✓ Found model: {model_path}")

# Load model
print("Loading model...")
checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
model_state = checkpoint['model_state_dict']
threshold = checkpoint.get('optimal_threshold', 0.5)

print(f"✓ Loaded model with threshold: {threshold}")

# Import model architecture
try:
    # Add parent directory to path to import serial_snn_monitor
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from serial_snn_monitor import ImprovedSNN, rate_encode_sample
    
    # Create model (architecture from the saved model)
    input_size = model_state['fc1.weight'].shape[1]  # Should be 5
    hidden_1 = model_state['fc1.weight'].shape[0]    # 128
    hidden_2 = model_state['fc2.weight'].shape[0]    # 64
    hidden_3 = model_state['fc3.weight'].shape[0]    # 32
    output_size = model_state['fc_out.weight'].shape[0]  # 2
    
    print(f"✓ Model architecture: {input_size}→{hidden_1}→{hidden_2}→{hidden_3}→{output_size}")
    
    model = ImprovedSNN(
        input_size=input_size,
        hidden_size_1=hidden_1,
        hidden_size_2=hidden_2,
        hidden_size_3=hidden_3,
        output_size=output_size
    )
    model.load_state_dict(model_state)
    model.eval()
    
    # Load scaler
    scaler_path = os.path.join(parent_dir, 'data', 'scaler_esp32.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(parent_dir, 'data', 'scaler.pkl')
    
    if not os.path.exists(scaler_path):
        print(f"❌ Error: Scaler not found at {scaler_path}")
        sys.exit(1)
    
    print(f"✓ Found scaler: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Create dummy input (simulate stationary sensor)
    # Features: [ax, ay, az, mag, sum_abs]
    test_input = np.array([[0.0, 0.0, 9.81, 9.81, 9.81]], dtype=np.float32)
    
    # Normalize
    scaled = scaler.transform(test_input)[0]
    normalized = 1.0 / (1.0 + np.exp(-scaled))
    
    print(f"\nTest input: {test_input[0]}")
    print(f"Normalized: {normalized}")
    
    # Encode to spikes
    time_steps = 120
    spikes = rate_encode_sample(normalized, time_steps)
    
    print(f"Spikes shape: {spikes.shape}")
    
    # Run inference
    with torch.no_grad():
        output = model(spikes)
        probs = torch.softmax(output, dim=1)[0].numpy()
        prediction = int(probs[1] >= threshold)
    
    print(f"\n{'='*60}")
    print(f"PyTorch Inference Test Results")
    print(f"{'='*60}")
    print(f"Output: Normal={probs[0]*100:.1f}%, Anomaly={probs[1]*100:.1f}%")
    print(f"Prediction: {'ANOMALY' if prediction else 'NORMAL'}")
    print(f"Threshold: {threshold:.4f}")
    print(f"{'='*60}\n")
    
    # Save test input for C++ verification
    output_dir = script_dir
    np.save(os.path.join(output_dir, 'test_input.npy'), test_input)
    np.save(os.path.join(output_dir, 'test_output.npy'), probs)
    print(f"✓ Test data saved to:")
    print(f"  - {os.path.join(output_dir, 'test_input.npy')}")
    print(f"  - {os.path.join(output_dir, 'test_output.npy')}")
    
    print("\n✓ Test completed successfully!")
    
except ImportError as e:
    print(f"Error importing serial_snn_monitor: {e}")
    print(f"\nExpected location: {os.path.join(parent_dir, 'serial_snn_monitor.py')}")
    print("Please make sure serial_snn_monitor.py exists in the parent directory")
    sys.exit(1)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)