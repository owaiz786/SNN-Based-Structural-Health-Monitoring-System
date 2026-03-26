import torch
import torch.nn as nn
import norse.torch as norse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("EVALUATION & SIMULATION")
print("=" * 60)

# ============================================================
# STEP 1: Load Model and Test Data
# ============================================================
print("\n📂 Loading model and test data...")

# Load test data
X_test_spikes = torch.load('data/X_test_spikes.pt', weights_only=False)
y_test = torch.load('data/y_test.pt', weights_only=False)

# Load trained model
class SpikingNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size_1=64, hidden_size_2=32, output_size=2):
        super(SpikingNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.lif1 = norse.LIFCell()
        self.lif2 = norse.LIFCell()
        self.lif3 = norse.LIFCell()
        self.output_size = output_size
    
    def forward(self, spike_input):
        batch_size = spike_input.size(0)
        state1, state2, state3 = None, None, None
        output_accumulator = torch.zeros(batch_size, self.output_size, device=spike_input.device)
        
        for t in range(spike_input.size(2)):
            x_t = spike_input[:, :, t]
            spikes1, state1 = self.lif1(self.fc1(x_t), state1)
            spikes2, state2 = self.lif2(self.fc2(spikes1), state2)
            spikes3, state3 = self.lif3(self.fc3(spikes2), state3)
            output_accumulator += spikes3
        return output_accumulator

num_features = X_test_spikes.shape[1]
model = SpikingNeuralNetwork(input_size=num_features)
model.load_state_dict(torch.load('data/snn_model.pth', weights_only=False))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE).eval()

print(f"✓ Model loaded on {DEVICE}")
print(f"✓ Test samples: {len(X_test_spikes)}")

# ============================================================
# STEP 2: Generate Predictions
# ============================================================
print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

model.eval()
predictions = []
probabilities = []

with torch.no_grad():
    for i in range(len(X_test_spikes)):
        sample = X_test_spikes[i:i+1].to(DEVICE)  # Add batch dimension
        output = model(sample)
        
        # Get predicted class
        _, predicted = torch.max(output, 1)
        predictions.append(predicted.item())
        
        # Get probability (softmax)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        probabilities.append(probs)

predictions = np.array(predictions)
probabilities = np.array(probabilities)
y_test_np = y_test.numpy()

print(f"✓ Predictions generated for {len(predictions)} samples")

# ============================================================
# STEP 3: Calculate Metrics
# ============================================================
print("\n" + "=" * 60)
print("PERFORMANCE METRICS")
print("=" * 60)

accuracy = accuracy_score(y_test_np, predictions)
precision = precision_score(y_test_np, predictions)
recall = recall_score(y_test_np, predictions)
f1 = f1_score(y_test_np, predictions)

print(f"\n📊 Classification Metrics:")
print(f"  - Accuracy:  {accuracy * 100:.2f}%")
print(f"  - Precision: {precision * 100:.2f}%  (Of predicted anomalies, how many were real?)")
print(f"  - Recall:    {recall * 100:.2f}%  (Of real anomalies, how many did we catch?)")
print(f"  - F1-Score:  {f1 * 100:.2f}%  (Harmonic mean of precision & recall)")

print(f"\n📋 Classification Report:")
print(classification_report(y_test_np, predictions, 
                           target_names=['Normal (Healthy)', 'Anomaly (Damaged)']))

# ============================================================
# STEP 4: Confusion Matrix
# ============================================================
print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test_np, predictions)
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"              Normal  Anomaly")
print(f"Actual Normal    {cm[0,0]:>3}     {cm[0,1]:>3}")
print(f"       Anomaly   {cm[1,0]:>3}     {cm[1,1]:>3}")

# Visualize confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label', fontsize=11)
plt.xlabel('Predicted Label', fontsize=11)
plt.tight_layout()
plt.savefig('data/confusion_matrix.png', dpi=300)
print("\n✓ Saved: data/confusion_matrix.png")
plt.show()

# ============================================================
# STEP 5: ROC Curve & AUC (Optional but valuable)
# ============================================================
print("\n" + "=" * 60)
print("ROC ANALYSIS")
print("=" * 60)

from sklearn.metrics import roc_curve, auc

# Get probability of class 1 (anomaly)
y_scores = probabilities[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test_np, y_scores)
roc_auc = auc(fpr, tpr)

print(f"\n📊 ROC-AUC Score: {roc_auc:.4f}")
print(f"  - 0.5 = random guessing")
print(f"  - 1.0 = perfect classifier")
print(f"  - Your model: {roc_auc:.4f} ({'Good' if roc_auc > 0.7 else 'Fair'})")

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/roc_curve.png', dpi=300)
print("✓ Saved: data/roc_curve.png")
plt.show()

# ============================================================
# STEP 6: Simulation - Visualize Individual Predictions
# ============================================================
print("\n" + "=" * 60)
print("SAMPLE SIMULATIONS")
print("=" * 60)

feature_names = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Strain', 'Temp']

def simulate_sample(sample_idx, title_suffix=""):
    """Visualize spike input and model prediction for one sample."""
    
    sample_spikes = X_test_spikes[sample_idx:sample_idx+1].to(DEVICE)
    actual_label = y_test_np[sample_idx]
    actual_name = "Anomaly" if actual_label == 1 else "Normal"
    
    with torch.no_grad():
        output = model(sample_spikes)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        _, predicted = torch.max(output, 1)
        predicted_label = predicted.item()
        predicted_name = "Anomaly" if predicted_label == 1 else "Normal"
    
    # Print prediction details
    print(f"\n🔍 Sample #{sample_idx} {title_suffix}:")
    print(f"  Actual:     {actual_name} (Label: {actual_label})")
    print(f"  Predicted:  {predicted_name} (Label: {predicted_label})")
    print(f"  Confidence: Normal={probs[0]*100:.1f}%, Anomaly={probs[1]*100:.1f}%")
    print(f"  Result:     {'✅ Correct' if actual_label == predicted_label else '❌ Wrong'}")
    
    # Plot 1: Input spike trains
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Input spikes
    ax1 = axes[0]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    for i in range(sample_spikes.shape[1]):
        spikes = sample_spikes[0, i, :].cpu().numpy()
        spike_times = np.where(spikes == 1)[0]
        ax1.eventplot(spike_times, orientation='horizontal', 
                     colors=[colors[i]], linelengths=0.8, linewidths=0.5)
    ax1.set_xlim(0, sample_spikes.shape[2])
    ax1.set_ylabel('Sensor', fontsize=10)
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(feature_names, fontsize=9)
    ax1.set_title(f'Input Spike Trains ({actual_name} Sample)', fontsize=11, fontweight='bold')
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Output probabilities bar chart
    ax2 = axes[1]
    bars = ax2.bar(['Normal', 'Anomaly'], probs * 100, 
                   color=['#2ecc71' if predicted_label == 0 else '#95a5a6',
                          '#e74c3c' if predicted_label == 1 else '#95a5a6'],
                   edgecolor='black')
    ax2.set_ylabel('Confidence (%)', fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_title('Model Output Probabilities', fontsize=11, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    filename = f'data/simulation_sample_{sample_idx}_{actual_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()
    
    return actual_label == predicted_label

# Run simulations on 4 representative samples:
# 1. True Positive (correctly detected anomaly)
# 2. True Negative (correctly detected normal)
# 3. False Positive (normal misclassified as anomaly)
# 4. False Negative (anomaly missed)

print("\n🎬 Running simulations on representative samples...")

# Find indices for each case
tp_idx = None  # True Positive: actual=1, predicted=1
tn_idx = None  # True Negative: actual=0, predicted=0
fp_idx = None  # False Positive: actual=0, predicted=1
fn_idx = None  # False Negative: actual=1, predicted=0

for i in range(len(y_test_np)):
    if y_test_np[i] == 1 and predictions[i] == 1 and tp_idx is None:
        tp_idx = i
    elif y_test_np[i] == 0 and predictions[i] == 0 and tn_idx is None:
        tn_idx = i
    elif y_test_np[i] == 0 and predictions[i] == 1 and fp_idx is None:
        fp_idx = i
    elif y_test_np[i] == 1 and predictions[i] == 0 and fn_idx is None:
        fn_idx = i
    if all(x is not None for x in [tp_idx, tn_idx, fp_idx, fn_idx]):
        break

# Run and display simulations
results = {}
if tp_idx is not None:
    results['True Positive'] = simulate_sample(tp_idx, "(TP)")
if tn_idx is not None:
    results['True Negative'] = simulate_sample(tn_idx, "(TN)")
if fp_idx is not None:
    results['False Positive'] = simulate_sample(fp_idx, "(FP)")
if fn_idx is not None:
    results['False Negative'] = simulate_sample(fn_idx, "(FN)")

# ============================================================
# STEP 7: Save Evaluation Summary
# ============================================================
print("\n" + "=" * 60)
print("SAVING EVALUATION RESULTS")
print("=" * 60)

evaluation_summary = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'confusion_matrix': cm.tolist(),
    'total_samples': len(y_test_np),
    'correct_predictions': int((predictions == y_test_np).sum()),
    'false_positives': int(cm[0, 1]),
    'false_negatives': int(cm[1, 0])
}

import json
with open('data/evaluation_summary.json', 'w') as f:
    json.dump(evaluation_summary, f, indent=2)

print("\n✓ Saved: data/evaluation_summary.json")

# ============================================================
# FINAL REPORT SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("📋 PROJECT EVALUATION SUMMARY")
print("=" * 60)

print(f"""
🎯 Model Performance:
   • Test Accuracy:  {accuracy * 100:.2f}%
   • Precision:      {precision * 100:.2f}%
   • Recall:         {recall * 100:.2f}%
   • F1-Score:       {f1 * 100:.2f}%
   • ROC-AUC:        {roc_auc:.4f}

📊 Confusion Matrix:
   • True Negatives:  {cm[0, 0]} (Healthy correctly identified)
   • False Positives: {cm[0, 1]} (Healthy misclassified as damaged)
   • False Negatives: {cm[1, 0]} (Damaged misclassified as healthy)
   • True Positives:  {cm[1, 1]} (Damaged correctly identified)

📁 Generated Files:
   • data/confusion_matrix.png
   • data/roc_curve.png
   • data/simulation_sample_*.png
   • data/evaluation_summary.json

✅ Your SNN project is complete and ready for submission!
""")

print("=" * 60)
print("🎉 CONGRATULATIONS!")
print("=" * 60)
print("""
Your Spiking Neural Network for Structural Health Monitoring:
✓ Loads and preprocesses sensor data
✓ Converts data to spike trains using rate encoding
✓ Trains an SNN with LIF neurons using Norse
✓ Evaluates performance with comprehensive metrics
✓ Generates visualizations for your report

Next steps for your report:
1. Include the training progress plot
2. Add the confusion matrix and ROC curve
3. Show 1-2 simulation visualizations
4. Discuss the 75% accuracy and potential improvements
""")
print("=" * 60 + "\n")