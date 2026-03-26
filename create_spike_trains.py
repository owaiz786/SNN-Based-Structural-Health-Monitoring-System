import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

print("=" * 60)
print("SPIKE ENCODING FOR SNN")
print("=" * 60)

# ============================================================
# STEP 1: Load Preprocessed Data
# ============================================================
print("\n📂 Loading preprocessed data...")

X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

print(f"✓ Training data shape: {X_train.shape}")
print(f"✓ Testing data shape: {X_test.shape}")
print(f"✓ Training labels shape: {y_train.shape}")
print(f"✓ Testing labels shape: {y_test.shape}")

# ============================================================
# STEP 2: Define Spike Encoding Parameters
# ============================================================
print("\n" + "=" * 60)
print("SPIKE ENCODING PARAMETERS")
print("=" * 60)

# Time steps for simulation
TIME_STEPS = 50  # Number of time steps per sample
print(f"\n⏱️  Time steps per sample: {TIME_STEPS}")

# Encoding method: Rate Encoding
print(f"\n📊 Encoding method: Rate Encoding")
print(f"   - Higher input values → Higher spike probability")
print(f"   - Input range: [0, 1] after normalization")

# ============================================================
# STEP 3: Spike Encoding Function
# ============================================================

def rate_encode(data, time_steps, seed=None):
    """
    Convert normalized data to spike trains using rate encoding.
    
    Parameters:
    -----------
    data : numpy array of shape (num_samples, num_features)
        Normalized sensor data
    time_steps : int
        Number of time steps for the spike train
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    spike_trains : torch tensor of shape (num_samples, num_features, time_steps)
        Binary spike trains (0 or 1 at each time step)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    num_samples, num_features = data.shape
    
    # Create spike trains tensor
    spike_trains = torch.zeros(num_samples, num_features, time_steps)
    
    # Generate spikes based on input intensity
    # For each sample and feature, generate spikes with probability = input value
    for i in tqdm(range(num_samples), desc="Encoding samples"):
        for j in range(num_features):
            # Get the normalized value (between 0 and 1 approximately)
            intensity = data[i, j]
            
            # Normalize to [0, 1] range for probability
            # Since we used StandardScaler, values can be negative
            # Apply sigmoid to map to [0, 1]
            prob = 1 / (1 + np.exp(-intensity))  # Sigmoid function
            
            # Generate random spikes with probability proportional to input
            spikes = (torch.rand(time_steps) < prob).float()
            spike_trains[i, j, :] = spikes
    
    return spike_trains

# Alternative: More efficient vectorized version
def rate_encode_vectorized(data, time_steps, seed=None):
    """
    Vectorized version of rate encoding (faster).
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    num_samples, num_features = data.shape
    
    # Convert to torch tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    # Apply sigmoid to get probabilities in [0, 1]
    probabilities = torch.sigmoid(data_tensor)  # Shape: (num_samples, num_features)
    
    # Expand to match time steps
    probabilities = probabilities.unsqueeze(-1).expand(-1, -1, time_steps)
    
    # Generate random spikes
    random_matrix = torch.rand(num_samples, num_features, time_steps)
    
    # Create spike trains: 1 if random < probability, else 0
    spike_trains = (random_matrix < probabilities).float()
    
    return spike_trains

# ============================================================
# STEP 4: Generate Spike Trains
# ============================================================
print("\n" + "=" * 60)
print("GENERATING SPIKE TRAINS")
print("=" * 60)

print("\n🔄 Encoding training data...")
X_train_spikes = rate_encode_vectorized(X_train, TIME_STEPS, seed=42)

print("\n🔄 Encoding testing data...")
X_test_spikes = rate_encode_vectorized(X_test, TIME_STEPS, seed=42)

print(f"\n✓ Training spike trains shape: {X_train_spikes.shape}")
print(f"  - Samples: {X_train_spikes.shape[0]}")
print(f"  - Features: {X_train_spikes.shape[1]}")
print(f"  - Time steps: {X_train_spikes.shape[2]}")

print(f"\n✓ Testing spike trains shape: {X_test_spikes.shape}")
print(f"  - Samples: {X_test_spikes.shape[0]}")
print(f"  - Features: {X_test_spikes.shape[1]}")
print(f"  - Time steps: {X_test_spikes.shape[2]}")

# ============================================================
# STEP 5: Analyze Spike Statistics
# ============================================================
print("\n" + "=" * 60)
print("SPIKE STATISTICS")
print("=" * 60)

# Calculate firing rates
train_firing_rate = X_train_spikes.mean().item() * 100
test_firing_rate = X_test_spikes.mean().item() * 100

print(f"\n📊 Average firing rate:")
print(f"  - Training set: {train_firing_rate:.2f}%")
print(f"  - Testing set: {test_firing_rate:.2f}%")

# Firing rate per feature
feature_names = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Strain', 'Temp']
print(f"\n📊 Firing rate per feature (Training):")
for i, name in enumerate(feature_names):
    rate = X_train_spikes[:, i, :].mean().item() * 100
    print(f"  - {name}: {rate:.2f}%")

# ============================================================
# STEP 6: Save Spike Trains
# ============================================================
print("\n" + "=" * 60)
print("SAVING SPIKE TRAINS")
print("=" * 60)

# Save as PyTorch tensors
torch.save(X_train_spikes, 'data/X_train_spikes.pt')
torch.save(X_test_spikes, 'data/X_test_spikes.pt')

# Also save labels as tensors
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

torch.save(y_train_tensor, 'data/y_train.pt')
torch.save(y_test_tensor, 'data/y_test.pt')

print("\n✓ Saved files:")
print("  - data/X_train_spikes.pt")
print("  - data/X_test_spikes.pt")
print("  - data/y_train.pt")
print("  - data/y_test.pt")

# ============================================================
# STEP 7: Visualize Spike Trains
# ============================================================
print("\n" + "=" * 60)
print("VISUALIZING SPIKE TRAINS")
print("=" * 60)

def visualize_spike_train(spike_tensor, sample_idx, title=""):
    """Visualize spike train for a single sample."""
    
    fig, axes = plt.subplots(spike_tensor.shape[1], 1, figsize=(12, 8))
    if spike_tensor.shape[1] == 1:
        axes = [axes]
    
    for i in range(spike_tensor.shape[1]):
        spikes = spike_tensor[sample_idx, i, :].numpy()
        spike_times = np.where(spikes == 1)[0]
        
        axes[i].eventplot(spike_times, orientation='horizontal', 
                         colors='blue', linelengths=0.8)
        axes[i].set_xlim(0, spike_tensor.shape[2])
        axes[i].set_ylabel(feature_names[i], fontsize=10)
        axes[i].set_yticks([])
        axes[i].grid(True, axis='x', alpha=0.3)
        
        if i == 0:
            axes[i].set_title(f'Sample {sample_idx} - {title}', fontsize=12, fontweight='bold')
    
    axes[-1].set_xlabel('Time Step', fontsize=10)
    plt.tight_layout()
    return fig

# Visualize a normal sample
normal_idx = np.where(y_train == 0)[0][0]
fig1 = visualize_spike_train(X_train_spikes, normal_idx, "Normal (Healthy)")
plt.savefig('data/spike_train_normal.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: data/spike_train_normal.png")
plt.show()

# Visualize an anomaly sample
anomaly_idx = np.where(y_train == 1)[0][0]
fig2 = visualize_spike_train(X_train_spikes, anomaly_idx, "Anomaly (Damaged)")
plt.savefig('data/spike_train_anomaly.png', dpi=300, bbox_inches='tight')
print("✓ Saved: data/spike_train_anomaly.png")
plt.show()

# Visualize comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Normal sample
ax1 = axes[0]
for i in range(X_train_spikes.shape[1]):
    spikes = X_train_spikes[normal_idx, i, :].numpy()
    spike_times = np.where(spikes == 1)[0]
    ax1.eventplot(spike_times, orientation='horizontal', 
                 colors='green', linelengths=0.8, linewidths=0.5)
ax1.set_xlim(0, TIME_STEPS)
ax1.set_ylabel('Features', fontsize=10)
ax1.set_yticks(range(len(feature_names)))
ax1.set_yticklabels(feature_names)
ax1.set_title('Normal Sample (Healthy Structure)', fontsize=11, fontweight='bold', color='green')
ax1.grid(True, axis='x', alpha=0.3)

# Anomaly sample
ax2 = axes[1]
for i in range(X_train_spikes.shape[1]):
    spikes = X_train_spikes[anomaly_idx, i, :].numpy()
    spike_times = np.where(spikes == 1)[0]
    ax2.eventplot(spike_times, orientation='horizontal', 
                 colors='red', linelengths=0.8, linewidths=0.5)
ax2.set_xlim(0, TIME_STEPS)
ax2.set_xlabel('Time Step', fontsize=10)
ax2.set_ylabel('Features', fontsize=10)
ax2.set_yticks(range(len(feature_names)))
ax2.set_yticklabels(feature_names)
ax2.set_title('Anomaly Sample (Damaged Structure)', fontsize=11, fontweight='bold', color='red')
ax2.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('data/spike_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: data/spike_comparison.png")
plt.show()

# ============================================================
# STEP 8: Verify Data Integrity
# ============================================================
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

# Load and verify
loaded_train = torch.load('data/X_train_spikes.pt')
loaded_test = torch.load('data/X_test_spikes.pt')

print(f"\n✓ Verification complete:")
print(f"  - Training spikes: {loaded_train.shape}")
print(f"  - Testing spikes: {loaded_test.shape}")
print(f"  - Data type: {loaded_train.dtype}")
print(f"  - Contains only binary values: {torch.all((loaded_train == 0) | (loaded_train == 1)).item()}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("✅ SPIKE ENCODING COMPLETE!")
print("=" * 60)
print(f"\n📊 Summary:")
print(f"  - Time steps: {TIME_STEPS}")
print(f"  - Training samples: {X_train_spikes.shape[0]}")
print(f"  - Testing samples: {X_test_spikes.shape[0]}")
print(f"  - Spike tensor shape: (samples, features, time_steps)")
print(f"  - Average firing rate: {train_firing_rate:.2f}%")
print(f"\n🎯 Ready to build the SNN model!")
print("=" * 60 + "\n")