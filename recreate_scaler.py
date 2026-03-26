# create_esp32_scaler.py
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os

print("=" * 60)
print("Creating Scaler for ESP32 Raw Sensor Data")
print("=" * 60)

# Based on your monitor output, your ESP32 sensor ranges:
# ax: -5 to -9 (typical)
# ay: 4 to 5 (typical)  
# az: -7 to 10 (typical)
# mag: 9-10
# sum_abs: 15-20

# Create realistic synthetic data matching ESP32 MPU6050 ranges
np.random.seed(42)

# Generate 2000 samples that represent your actual sensor ranges
n_samples = 2000

# Normal operating range (based on your monitor output)
ax_normal = np.random.uniform(-10, -2, n_samples)
ay_normal = np.random.uniform(2, 6, n_samples)
az_normal = np.random.uniform(-8, 12, n_samples)

# Calculate derived features
mag_normal = np.sqrt(ax_normal**2 + ay_normal**2 + az_normal**2)
sum_abs_normal = np.abs(ax_normal) + np.abs(ay_normal) + np.abs(az_normal)

# Combine features
X_raw = np.column_stack([ax_normal, ay_normal, az_normal, mag_normal, sum_abs_normal])

print(f"\n📊 Generated synthetic data shape: {X_raw.shape}")
print(f"\nFeature ranges (raw ESP32 values):")
feature_names = ['ax', 'ay', 'az', 'mag', 'sum_abs']
for i, name in enumerate(feature_names):
    print(f"  {name}: [{X_raw[:,i].min():.2f}, {X_raw[:,i].max():.2f}]")
    print(f"         mean={X_raw[:,i].mean():.2f}, std={X_raw[:,i].std():.2f}")

# Create and fit scaler
print("\n🔄 Fitting StandardScaler...")
scaler = StandardScaler()
scaler.fit(X_raw)

print(f"\n✅ Scaler created with:")
print(f"   Mean: {scaler.mean_}")
print(f"   Scale: {scaler.scale_}")

# Save the scaler
os.makedirs('data', exist_ok=True)
with open('data/scaler_esp32.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("\n✅ Saved: data/scaler_esp32.pkl")

# Test with your actual samples
print("\n" + "=" * 60)
print("Testing with your actual ESP32 samples")
print("=" * 60)

test_samples = [
    np.array([-4.92, 4.35, -6.99, 9.59, 16.26]),
    np.array([-4.85, 4.40, -6.89, 9.50, 16.14]),
    np.array([-5.02, 4.44, -6.83, 9.56, 16.29]),
]

for i, sample in enumerate(test_samples):
    sample_reshaped = sample.reshape(1, -1)
    scaled = scaler.transform(sample_reshaped)[0]
    normalized = 1.0 / (1.0 + np.exp(-scaled))
    
    print(f"\n📊 Test Sample {i+1}:")
    print(f"   Raw: {sample}")
    print(f"   Scaled (z-score): {scaled}")
    print(f"   Normalized: {normalized}")
    print(f"   Spike probability range: [{normalized.min():.3f}, {normalized.max():.3f}]")

print("\n" + "=" * 60)
print("✅ Scaler created successfully!")
print("=" * 60)
print("\nTo use this scaler, update serial_snn_monitor.py:")
print("  SCALER_PATH = 'data/scaler_esp32.pkl'")