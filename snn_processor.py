# snn_processor.py
import torch
import numpy as np
import pickle
from serial_snn_monitor import ImprovedSNN, rate_encode_sample

class SNNProcessor:
    """Processes sensor data through the SNN model"""
    
    def __init__(self, model_path='data/snn_model_optimized.pth', 
                 scaler_path='data/scaler_esp32.pkl'):  # ← FIXED: consistent filename
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.load_model()
        
    def load_model(self):
        """Load the trained SNN model and scaler"""
        print("🤖 Loading SNN model...")
        
        # Add safe globals for PyTorch 2.6+
        torch.serialization.add_safe_globals([np.ndarray])
        
        # Load model checkpoint
        checkpoint = torch.load(self.model_path, weights_only=False, map_location='cpu')
        
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
        
        # Load scaler
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✅ Model ready: {input_size} features, threshold={self.threshold:.3f}")
        
    def predict(self, raw_features):
        """Predict anomaly for a single sample"""
        try:
            # Normalize
            raw_reshaped = raw_features.reshape(1, -1)
            scaled = self.scaler.transform(raw_reshaped)[0]
            normalized = 1.0 / (1.0 + np.exp(-scaled))
            
            # Encode to spikes
            spike_tensor = rate_encode_sample(normalized, self.time_steps)
            
            # Inference
            with torch.no_grad():
                output = self.model(spike_tensor)
                probs = torch.softmax(output, dim=1)[0].numpy()
                prediction = 1 if probs[1] >= self.threshold else 0
            
            return prediction, probs
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, np.array([0.5, 0.5])