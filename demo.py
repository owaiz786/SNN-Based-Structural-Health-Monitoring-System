#!/usr/bin/env python3
"""
One-click demo: Load model and predict on new sensor data
"""
import torch
import numpy as np
from snn_model import ImprovedSNN, RobustSpikeEncoder

def predict_new_reading(sensor_values):
    """
    Predict if a new sensor reading indicates anomaly
    
    Args:
        sensor_values: dict with keys ['Accel_X', 'Accel_Y', 'Accel_Z', 'Strain', 'Temp']
    
    Returns:
        dict with prediction, confidence, and recommendation
    """
    # Load model
    checkpoint = torch.load('data/snn_model_final.pth', weights_only=False)
    model = ImprovedSNN(input_size=5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare input
    features = np.array([[
        sensor_values['Accel_X'],
        sensor_values['Accel_Y'], 
        sensor_values['Accel_Z'],
        sensor_values['Strain'],
        sensor_values['Temp']
    ]])
    
    # Encode to spikes
    encoder = RobustSpikeEncoder()
    spikes = encoder.rate_encode(features, time_steps=120)
    
    # Predict
    with torch.no_grad():
        output = model(spikes)
        probs = torch.softmax(output, dim=1)[0]
        prediction = (probs[1] >= checkpoint['optimal_threshold']).item()
    
    # Return result
    return {
        'prediction': 'ANOMALY DETECTED ⚠️' if prediction else 'Normal ✓',
        'confidence': f"{max(probs).item()*100:.1f}%",
        'anomaly_probability': f"{probs[1].item()*100:.1f}%",
        'recommendation': 'Inspect structure immediately' if prediction else 'Continue monitoring'
    }

# Example usage
if __name__ == "__main__":
    test_reading = {
        'Accel_X': 0.85, 'Accel_Y': -0.32, 'Accel_Z': 0.12,
        'Strain': 145.2, 'Temp': 23.5
    }
    result = predict_new_reading(test_reading)
    
    print("\n🔍 Structural Health Assessment")
    print("=" * 40)
    for key, value in result.items():
        print(f"{key:25s}: {value}")
    print("=" * 40 + "\n")