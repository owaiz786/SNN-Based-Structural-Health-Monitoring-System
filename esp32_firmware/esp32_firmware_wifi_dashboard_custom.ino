"""
Export trained SNN weights to C++ compatible format
Converts PyTorch model → Header files for ESP32
"""
import torch
import numpy as np
import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Try to find the model file in multiple locations
model_paths = [
    os.path.join(parent_dir, 'data', 'snn_model_optimized.pth'),
    os.path.join(script_dir, 'data', 'snn_model_optimized.pth'),
    'data/snn_model_optimized.pth',
    'snn_model_optimized.pth',
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    print("❌ Error: Could not find trained model file!")
    print("Searched in:")
    for path in model_paths:
        print(f"  - {path}")
    print("\nPlease ensure you have trained the model first using:")
    print("  python serial_snn_monitor.py")
    sys.exit(1)

print(f"✓ Found model: {model_path}")

# Create output directory
os.makedirs(os.path.join(script_dir, 'weights'), exist_ok=True)

# Load your trained model
print("🔄 Loading trained SNN model...")
checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
model_state = checkpoint['model_state_dict']
threshold = checkpoint.get('optimal_threshold', 0.5)

# Get actual input size from model
input_size = model_state['fc1.weight'].shape[1]
hidden_1 = model_state['fc1.weight'].shape[0]
hidden_2 = model_state['fc2.weight'].shape[0]
hidden_3 = model_state['fc3.weight'].shape[0]
output_size = model_state['fc_out.weight'].shape[0]

# Model architecture
ARCHITECTURE = {
    'input_size': input_size,
    'hidden_1': hidden_1,
    'hidden_2': hidden_2,
    'hidden_3': hidden_3,
    'output_size': output_size,
    'time_steps': 120,
}

print(f"✓ Model architecture: {ARCHITECTURE}")

# Extract weights and biases
def extract_weights(state_dict, layer_name):
    weight = state_dict[f'{layer_name}.weight'].cpu().numpy()
    bias = state_dict[f'{layer_name}.bias'].cpu().numpy()
    return weight, bias

# Extract all layer weights
layers = {
    'fc1': extract_weights(model_state, 'fc1'),
    'fc2': extract_weights(model_state, 'fc2'),
    'fc3': extract_weights(model_state, 'fc3'),
    'fc_out': extract_weights(model_state, 'fc_out'),
}

# Quantize weights to int8
def quantize_to_int8(float_array, scale_factor=127.0):
    max_val = np.max(np.abs(float_array))
    if max_val > 0:
        scale = 127.0 / max_val
    else:
        scale = scale_factor
    
    int8_array = np.clip(float_array * scale, -128, 127).astype(np.int8)
    return int8_array, scale

print("\n🔧 Quantizing weights to int8...")
quantized_layers = {}
for name, (weight, bias) in layers.items():
    q_weight, w_scale = quantize_to_int8(weight)
    q_bias, b_scale = quantize_to_int8(bias)
    quantized_layers[name] = {
        'weight': q_weight,
        'bias': q_bias,
        'weight_scale': w_scale,
        'bias_scale': b_scale,
        'shape': weight.shape
    }
    print(f"  {name}: {weight.shape} -> int8 (weight_scale={w_scale:.2f}, bias_scale={b_scale:.2f})")

# Generate files
weights_dir = os.path.join(script_dir, 'weights')

def generate_header_file():
    """Generate main configuration header"""
    content = f"""// snn_config.h
// Auto-generated configuration for SNN-SHM ESP32 Firmware
// Generated from PyTorch model: {os.path.basename(model_path)}

#ifndef SNN_CONFIG_H
#define SNN_CONFIG_H

// Model Architecture
#define INPUT_SIZE {ARCHITECTURE['input_size']}
#define HIDDEN_1_SIZE {ARCHITECTURE['hidden_1']}
#define HIDDEN_2_SIZE {ARCHITECTURE['hidden_2']}
#define HIDDEN_3_SIZE {ARCHITECTURE['hidden_3']}
#define OUTPUT_SIZE {ARCHITECTURE['output_size']}
#define TIME_STEPS {ARCHITECTURE['time_steps']}

// LIF Neuron Parameters (Q15 fixed-point)
#define LIF_THRESHOLD 16384      // 1.0 in Q15
#define LIF_RESET 0              // 0.0 in Q15
#define LIF_DECAY 32678          // 0.995 in Q15
#define LIF_SCALE 127            // Weight quantization scale

// Classification Threshold (Q15 format)
#define ANOMALY_THRESHOLD {int(threshold * 32768)}  // {threshold:.4f} in Q15

// Hardware Pins
#define MPU_SDA 21
#define MPU_SCL 22
#define LED_PIN 2
#define BUZZER_PIN 4

// Sampling
#define SAMPLE_INTERVAL_MS 500

#endif // SNN_CONFIG_H
"""
    with open(os.path.join(weights_dir, 'snn_config.h'), 'w', encoding='utf-8') as f:
        f.write(content)
    print("✓ Generated: snn_config.h")

def generate_weights_header():
    """Generate weights header file with complete arrays"""
    content = """// snn_weights.h
// Auto-generated weight matrices for SNN-SHM
// WARNING: Large file - stored in PROGMEM

#ifndef SNN_WEIGHTS_H
#define SNN_WEIGHTS_H

#include <pgmspace.h>

"""
    
    for name, data in quantized_layers.items():
        weight = data['weight']
        bias = data['bias']
        shape = data['shape']
        
        # Weight matrix - FULL array
        content += f"// {name} weights: {shape[0]} x {shape[1]}\n"
        content += f"const int8_t {name}_weight[{shape[0]}][{shape[1]}] PROGMEM = {{\n"
        
        for i in range(shape[0]):
            row = ', '.join(str(int(w)) for w in weight[i])
            content += f"  {{{row}}}"
            if i < shape[0] - 1:
                content += ",\n"
            else:
                content += "\n"
        
        content += "};\n\n"
        
        # Bias vector - FULL array
        content += f"// {name} bias: {len(bias)}\n"
        content += f"const int8_t {name}_bias[{len(bias)}] PROGMEM = {{\n  "
        content += ', '.join(str(int(b)) for b in bias)
        content += "\n};\n\n"
    
    content += "#endif // SNN_WEIGHTS_H\n"
    
    with open(os.path.join(weights_dir, 'snn_weights.h'), 'w', encoding='utf-8') as f:
        f.write(content)
    
    file_size = os.path.getsize(os.path.join(weights_dir, 'snn_weights.h'))
    print(f"✓ Generated: snn_weights.h ({file_size / 1024:.1f} KB)")

def generate_weights_info():
    """Generate a text file with weights information"""
    content = f"""SNN Model Weights Information
{'='*60}

Model Architecture:
  Input size:  {ARCHITECTURE['input_size']}
  Hidden 1:    {ARCHITECTURE['hidden_1']}
  Hidden 2:    {ARCHITECTURE['hidden_2']}
  Hidden 3:    {ARCHITECTURE['hidden_3']}
  Output size: {ARCHITECTURE['output_size']}
  Time steps:  {ARCHITECTURE['time_steps']}

Layer Details:
"""
    for name, data in quantized_layers.items():
        shape = data['shape']
        content += f"""
{name}:
  Shape: {shape[0]} x {shape[1]}
  Weight scale: {data['weight_scale']:.4f}
  Bias scale: {data['bias_scale']:.4f}
  Weight range: [{data['weight'].min()}, {data['weight'].max()}]
  Bias range: [{data['bias'].min()}, {data['bias'].max()}]
"""

    content += f"""
Threshold: {threshold:.4f}

Memory Requirements:
  Total weights: {sum(w['weight'].size for w in quantized_layers.values())} bytes
  Total biases: {sum(w['bias'].size for w in quantized_layers.values())} bytes
  Total: ~{(sum(w['weight'].size for w in quantized_layers.values()) + sum(w['bias'].size for w in quantized_layers.values())) / 1024:.1f} KB
"""
    
    with open(os.path.join(weights_dir, 'weights_info.txt'), 'w', encoding='utf-8') as f:
        f.write(content)
    print("✓ Generated: weights_info.txt")

# Generate all files
generate_header_file()
generate_weights_header()
generate_weights_info()

print("\n" + "="*60)
print("WEIGHT EXPORT COMPLETE!")
print("="*60)
print(f"\nGenerated files in: {weights_dir}")
print(f"  - snn_config.h")
print(f"  - snn_weights.h")
print(f"  - weights_info.txt")

total_size = 0
for root, dirs, files in os.walk(weights_dir):
    for file in files:
        filepath = os.path.join(root, file)
        total_size += os.path.getsize(filepath)

print(f"\n📊 Total exported files size: {total_size / 1024:.1f} KB")
print("="*60)
