"""
OPTIMIZED SNN - Target: 85%+ Accuracy
Fixes: Focal Loss, Class Weights, Threshold Tuning, Better Scheduler
"""
import torch
import torch.nn as nn
import norse.torch as norse
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import warnings
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_auc_score, f1_score
)
warnings.filterwarnings('ignore')

# ============================================================
# RANDOM SEEDS
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print("=" * 60)
print("OPTIMIZED SNN (TARGET: 85%+)")
print("=" * 60)

# ============================================================
# CRITICAL FIX 1: FOCAL LOSS FOR IMBALANCED DATA
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss - focuses on hard examples, excellent for imbalanced data
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ============================================================
# CONFIGURATION - OPTIMIZED
# ============================================================
CONFIG = {
    # Architecture
    'time_steps': 120,              # ↑ More temporal info
    'hidden_size_1': 128,
    'hidden_size_2': 64,
    'hidden_size_3': 32,
    'output_size': 2,
    
    # Training - CRITICAL FIXES
    'batch_size': 16,               # ↓ Smaller = better generalization
    'learning_rate': 0.0005,        # Tuned for stability
    'num_epochs': 50,               # ↑ More epochs
    'weight_decay': 1e-4,           # ↑ Stronger regularization
    
    # Regularization
    'dropout': 0.3,                 # ↑ More dropout
    'grad_clip': 0.5,               # ↓ More conservative
    
    # Data augmentation - MORE CONSERVATIVE
    'augmentation_prob': 0.5,
    'noise_prob': 0.02,             # ↓ Less noise
    
    # Early stopping - LESS AGGRESSIVE
    'early_stop_patience': 12,      # ↑ Allow more epochs
    
    # CRITICAL: Use Focal Loss
    'use_focal_loss': True,
    'focal_gamma': 2.0,
}

print("\n⚙️  OPTIMIZED Configuration:")
for key, value in CONFIG.items():
    print(f"  - {key}: {value}")

# ============================================================
# ROBUST SPIKE ENCODING
# ============================================================
class RobustSpikeEncoder:
    @staticmethod
    def normalize_data(data):
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        min_val = data.min(axis=0, keepdims=True)
        max_val = data.max(axis=0, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        normalized = (data - min_val) / range_val
        return np.clip(normalized, 0, 1)
    
    @staticmethod
    def rate_encode(data, time_steps):
        normalized_data = RobustSpikeEncoder.normalize_data(data)
        data_tensor = torch.tensor(normalized_data, dtype=torch.float32)
        probabilities = torch.clamp(data_tensor, 0.0, 1.0)
        probabilities = probabilities.unsqueeze(-1).expand(-1, -1, time_steps)
        random_matrix = torch.rand_like(probabilities)
        spike_trains = (random_matrix < probabilities).float()
        return spike_trains

# ============================================================
# CRITICAL FIX 2: FIXED DATASET (XOR BUG REMOVED)
# ============================================================
class AugmentedDataset(TensorDataset):
    def __init__(self, spikes, labels, noise_prob=0.02, training_mode=True):
        super().__init__(spikes, labels)
        self.noise_prob = noise_prob
        self.training_mode = training_mode
    
    def set_training_mode(self, training=True):
        self.training_mode = training
    
    def __getitem__(self, idx):
        spike, label = super().__getitem__(idx)
        if self.training_mode and random.random() < CONFIG['augmentation_prob']:
            noise_mask = torch.rand_like(spike) < self.noise_prob
            spike = spike.clone()
            # FIX: Use arithmetic instead of XOR for float tensors
            spike[noise_mask] = 1.0 - spike[noise_mask]
        return spike, label

# ============================================================
# IMPROVED SNN ARCHITECTURE
# ============================================================
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
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, spike_input):
        batch_size = spike_input.size(0)
        state1, state2, state3, state_out = None, None, None, None
        output_accumulator = torch.zeros(batch_size, self.output_size, 
                                         device=spike_input.device)
        
        for t in range(spike_input.size(2)):
            x_t = spike_input[:, :, t]
            x_t = self.bn_input(x_t)
            
            current1 = self.fc1(x_t)
            current1 = self.bn1(current1)
            spikes1, state1 = self.lif1(current1, state1)
            spikes1 = self.dropout1(spikes1)
            
            current2 = self.fc2(spikes1)
            current2 = self.bn2(current2)
            spikes2, state2 = self.lif2(current2, state2)
            spikes2 = self.dropout2(spikes2)
            
            current3 = self.fc3(spikes2)
            current3 = self.bn3(current3)
            spikes3, state3 = self.lif3(current3, state3)
            spikes3 = self.dropout3(spikes3)
            
            current_out = self.fc_out(spikes3)
            spikes_out, state_out = self.lif_out(current_out, state_out)
            
            # Temporal weighting
            weight = 0.995 ** (spike_input.size(2) - t - 1)
            output_accumulator += spikes_out * weight
        
        return output_accumulator

# ============================================================
# CRITICAL FIX 3: THRESHOLD TUNING
# ============================================================
def find_optimal_threshold(model, val_loader, device):
    """Find threshold that maximizes F1-score for anomaly class"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for spikes, labels in val_loader:
            spikes, labels = spikes.to(device), labels.to(device)
            outputs = model(spikes)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.2, 0.8, 0.05):
        predictions = (all_probs >= threshold).astype(int)
        f1 = f1_score(all_labels, predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n🎯 Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.3f})")
    return best_threshold

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(model, train_loader, val_loader, test_loader, device, dataset_train):
    # Calculate class weights
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    class_counts = np.bincount(all_labels)
    
    # CRITICAL FIX: Calculate proper class weights
    class_weights = len(all_labels) / (2 * class_counts)
    class_weights = class_weights / class_weights.sum() * 2
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"\n📊 Class distribution:")
    print(f"  - Normal (0): {class_counts[0]} ({class_counts[0]/len(all_labels)*100:.1f}%)")
    print(f"  - Anomaly (1): {class_counts[1]} ({class_counts[1]/len(all_labels)*100:.1f}%)")
    print(f"  - Class weights: {class_weights}")
    
    # CRITICAL FIX: Use Focal Loss
    if CONFIG['use_focal_loss']:
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=CONFIG['focal_gamma'])
        print(f"✓ Using Focal Loss (gamma={CONFIG['focal_gamma']})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print("✓ Using Weighted CrossEntropy")
    
    # Optimizer with AdamW (better regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # CRITICAL FIX: Better scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training loop
    best_val_acc = 0
    best_state = None
    patience_counter = 0
    train_losses, train_accs, val_accs = [], [], []
    
    print("\n🚀 Starting optimized training...\n")
    
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        dataset_train.set_training_mode(True)
        
        train_loss, train_correct, train_total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["num_epochs"]}', leave=False)
        for spikes, labels in pbar:
            spikes, labels = spikes.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(spikes)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        dataset_train.set_training_mode(False)
        
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for spikes, labels in val_loader:
                spikes, labels = spikes.to(device), labels.to(device)
                outputs = model(spikes)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        val_accs.append(val_acc)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
            print(f'  ✓ New best model! Val Acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f'\nEpoch {epoch+1}: Loss={train_loss:.4f}, Train={train_acc:.2f}%, Val={val_acc:.2f}%, LR={current_lr:.6f}')
        
        if patience_counter >= CONFIG['early_stop_patience']:
            print(f'\n⚠️  Early stopping at epoch {epoch+1}')
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'\n✓ Loaded best model: {best_val_acc:.2f}%')
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(model, val_loader, device)
    
    return model, best_val_acc, train_losses, train_accs, val_accs, optimal_threshold

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # Load data
    print("\n📂 Loading data...")
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = torch.load('data/y_train.pt', weights_only=False)
    y_test = torch.load('data/y_test.pt', weights_only=False)
    
    print(f"✓ Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")
    
    # Encode spikes
    print("\n🔄 Encoding spikes...")
    encoder = RobustSpikeEncoder()
    X_train_spikes = encoder.rate_encode(X_train, CONFIG['time_steps'])
    X_test_spikes = encoder.rate_encode(X_test, CONFIG['time_steps'])
    
    print(f"✓ Spike shape: {X_train_spikes.shape}")
    print(f"✓ Spike density: {X_train_spikes.mean().item()*100:.2f}%")
    
    # Split for validation
    print("\n📊 Splitting data...")
    indices = list(range(len(X_train_spikes)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, 
                                          stratify=y_train, random_state=SEED)
    
    X_train_fold = X_train_spikes[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train_spikes[val_idx]
    y_val_fold = y_train[val_idx]
    
    print(f"✓ Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(X_test_spikes)}")
    
    # Create datasets
    train_dataset = AugmentedDataset(X_train_fold, y_train_fold, 
                                     noise_prob=CONFIG['noise_prob'], 
                                     training_mode=True)
    val_dataset = TensorDataset(X_val_fold, y_val_fold)
    test_dataset = TensorDataset(X_test_spikes, y_test)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {DEVICE}")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = ImprovedSNN(
        input_size=X_train_spikes.shape[1],
        hidden_size_1=CONFIG['hidden_size_1'],
        hidden_size_2=CONFIG['hidden_size_2'],
        hidden_size_3=CONFIG['hidden_size_3'],
        output_size=CONFIG['output_size'],
        dropout=CONFIG['dropout']
    )
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {total_params:,}")
    
    # Train
    model, best_val_acc, train_losses, train_accs, val_accs, optimal_threshold = train_model(
        model, train_loader, val_loader, test_loader, DEVICE, train_dataset
    )
    
    # Test with optimal threshold
    print("\n📊 Testing with optimal threshold...")
    model.eval()
    test_correct, test_total = 0, 0
    all_probs = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for spikes, labels in tqdm(test_loader, desc="Testing"):
            spikes, labels = spikes.to(DEVICE), labels.to(DEVICE)
            outputs = model(spikes)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            # Use optimal threshold
            predicted = (probs >= optimal_threshold).long()
            
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_accuracy = 100 * test_correct / test_total
    
    # Final results
    print(f"\n{'='*60}")
    print("🏆 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Optimal Threshold: {optimal_threshold:.2f}")
    print(f"Improvement over baseline (75%): +{test_accuracy - 75:.2f}%")
    
    # Detailed metrics
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_predictions, 
                               target_names=['Normal', 'Anomaly']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\n📊 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Anomaly")
    print(f"Actual Normal    {cm[0,0]:>3}     {cm[0,1]:>3}")
    print(f"       Anomaly   {cm[1,0]:>3}     {cm[1,1]:>3}")
    
    # Anomaly-specific metrics
    anomaly_recall = cm[1,1] / (cm[1,0] + cm[1,1]) * 100 if (cm[1,0] + cm[1,1]) > 0 else 0
    anomaly_precision = cm[1,1] / (cm[0,1] + cm[1,1]) * 100 if (cm[0,1] + cm[1,1]) > 0 else 0
    print(f"\n🎯 Anomaly Detection Performance:")
    print(f"  - Recall (caught): {anomaly_recall:.1f}%")
    print(f"  - Precision (accurate): {anomaly_precision:.1f}%")
    
    # ROC-AUC
    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f"  - ROC-AUC: {roc_auc:.4f}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimal_threshold': optimal_threshold,
        'config': CONFIG
    }, 'data/snn_model_optimized.pth')
    print(f"\n✓ Model saved: data/snn_model_optimized.pth")
    
    # Plot
    print("\n📈 Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_losses, 'b-o', linewidth=2)
    axes[0].set_title('Training Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, 'g-o', label='Train', linewidth=2)
    axes[1].plot(val_accs, 'r-s', label='Val', linewidth=2)
    axes[1].axhline(y=test_accuracy, color='purple', linestyle='--', 
                   label=f'Test: {test_accuracy:.1f}%')
    axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('data/training_progress_optimized.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data/training_progress_optimized.png")
    plt.show()
    
    print(f"\n{'='*60}")
    print("✅ OPTIMIZED MODEL READY!")
    print(f"{'='*60}")
    print(f"\n📊 Summary:")
    print(f"  - Baseline Accuracy: 75.00%")
    print(f"  - Previous Version:  82.50%")
    print(f"  - Optimized Accuracy: {test_accuracy:.2f}%")
    print(f"  - Total Improvement: +{test_accuracy - 75:.2f}%")
    print(f"\n🎯 Key Improvements Applied:")
    print(f"  1. ✓ Focal Loss for imbalanced data")
    print(f"  2. ✓ Class weights properly applied")
    print(f"  3. ✓ Optimal threshold tuning")
    print(f"  4. ✓ CosineAnnealing LR scheduler")
    print(f"  5. ✓ Less aggressive early stopping (patience=12)")
    print(f"  6. ✓ AdamW optimizer with better regularization")
    print(f"  7. ✓ Fixed XOR bug (arithmetic flip)")
    print(f"{'='*60}\n")