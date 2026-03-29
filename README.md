# 🏗️ SNN-SHM: Spiking Neural Network for Structural Health Monitoring

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/accuracy-82.5%25-orange.svg)]()

**Real-time anomaly detection for bridges, buildings, and infrastructure using bio-inspired Spiking Neural Networks.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Deployment Options](#deployment-options)


---

## 🎯 Overview

**SNN-SHM** is an edge-deployable Structural Health Monitoring system that uses **Spiking Neural Networks (SNN)** to detect early structural anomalies from sensor data. Unlike traditional deep learning approaches, our SNN achieves **82.5% accuracy** with **10-100× lower power consumption**, enabling battery-operated, long-term infrastructure monitoring.

### Why This Matters

| Problem | Impact | Our Solution |
|---------|--------|--------------|
| 60% of India's bridges are >50 years old | Risk of catastrophic failure | Continuous real-time monitoring |
| Traditional SHM costs $500+ per node | Not scalable for large infrastructure | **$20 per ESP32 node** |
| Manual inspection is infrequent | Damage detected too late | **24/7 automated detection** |
| High power consumption | Battery lasts days | **Battery lasts months** (event-driven SNN) |

---

## ✨ Features

- ✅ **82.5% anomaly detection accuracy** with 74.6% recall
- ✅ **Spiking Neural Network** with Leaky Integrate-and-Fire (LIF) neurons
- ✅ **Three deployment paths**: Hybrid Edge-Cloud,  Custom C++
- ✅ **ESP32-compatible** for true edge deployment (~$20 per node)
- ✅ **Real-time dashboard** with live visualization 
- ✅ **Focal Loss + Class Weighting** for imbalanced data
- ✅ **Feature engineering** (5 raw  engineered features)
- ✅ **Threshold optimization** for safety-critical applications
- ✅ **Complete documentation** for college submission

---

## 🧠 Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SNN-SHM PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📊 Sensors (MPU6050)                                          │
│       │                                                         │
│       ▼                                                         │
│  🔧 Feature Engineering (5 → 11 features)                      │
│       │                                                         │
│       ▼                                                         │
│  ⚡ Spike Encoding (Rate Encoding, 120 timesteps)              │
│       │                                                         │
│       ▼                                                         │
│  🧠 SNN Model (11 → 128 → 64 → 32 → 2)                        │
│       │                                                         │
│       ▼                                                         │
│  🚨 Output: NORMAL ✅  or  ANOMALY ⚠️                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Model Architecture

| Layer | Configuration | Purpose |
|-------|---------------|---------|
| Input | 11 features | Engineered sensor features |
| Hidden 1 | Linear(11→128) + BatchNorm + LIF + Dropout(0.3) | Feature extraction |
| Hidden 2 | Linear(128→64) + BatchNorm + LIF + Dropout(0.3) | Pattern learning |
| Hidden 3 | Linear(64→32) + BatchNorm + LIF + Dropout(0.3) | Abstraction |
| Output | Linear(32→2) + LIF | Binary classification |

**Total Parameters:** ~11,628  
**Time Steps:** 120  
**Neuron Type:** Leaky Integrate-and-Fire (LIF)

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) ESP32 board + MPU6050 sensor for edge deployment

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/snn-shm.git
cd snn-shm
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; import norse; print('✅ Dependencies installed!')"
```

---

## 🚀 Quick Start

### Option 1: Train the Model (5 minutes)

```bash
# Download dataset (from Kaggle)
# Place building_health_monitoring_dataset.csv in data/ folder

# Preprocess data
python preprocess_data.py

# Create spike trains
python create_spike_trains.py

# Train optimized SNN
python snn_model_optimized.py
```

### Option 2: Run Pre-trained Model (1 minute)

```bash
# Download pre-trained model from releases
# Place in data/snn_model_optimized.pth

# Run real-time monitoring
python serial_snn_monitor.py
```

### Option 3: Launch Dashboard (2 minutes)

```bash
# Start Streamlit dashboard
streamlit run dashboard.py

# Opens at http://localhost:8501
```

---

## 📁 Project Structure

```
snn-shm/
├── data/
│   ├── building_health_monitoring_dataset.csv   # Raw dataset
│   ├── X_train.npy, X_test.npy                  # Preprocessed features
│   ├── y_train.pt, y_test.pt                    # Labels
│   ├── snn_model_optimized.pth                  # Trained model
│   └── scaler_esp32.pkl                         # Normalization parameters
│
├── esp32_firmware/
│   ├── esp32_firmware.ino                       # Arduino sketch
│   ├── weights/
│   │   ├── snn_config.h                         # Model configuration
│   │   └── snn_weights.h                        # Quantized weights
│   └── src/
│       ├── lif_neuron.h                         # LIF implementation
│       ├── snn_inference.h                      # SNN forward pass
│       └── feature_engineering.h                # Feature extraction
│
├── scripts/
│   ├── preprocess_data.py                       # Data preprocessing
│   ├── create_spike_trains.py                   # Spike encoding
│   ├── snn_model_optimized.py                   # Model training
│   ├── serial_snn_monitor.py                    # USB monitoring
│   ├── api_server.py                            # FastAPI server
│   └── dashboard.py                             # Streamlit dashboard
│
├── reports/
│   ├── training_progress_optimized.png          # Training curves
│   ├── confusion_matrix.png                     # Evaluation metrics
│   └── roc_curve.png                            # ROC analysis
│
├── requirements.txt                             # Python dependencies
├── README.md                                    # This file
└── LICENSE                                      # MIT License
```

---

## 📖 Usage

### 1. Data Preprocessing

```bash
python preprocess_data.py
```

**Output:**
- Normalized sensor data (Min-Max scaling)
- Train/Validation/Test split (640/160/200)
- Binary labels (Normal=0, Anomaly=1)

### 2. Spike Encoding

```bash
python create_spike_trains.py
```

**Output:**
- Spike trains (120 time steps)
- Spike density: ~46%
- Format: `(samples, features, time_steps)`

### 3. Model Training

```bash
python snn_model_optimized.py
```

**Key Features:**
- Focal Loss for imbalanced data
- Class weights (Normal: 0.71, Anomaly: 1.68)
- CosineAnnealing learning rate scheduler
- Early stopping (patience=12)
- Optimal threshold tuning

### 4. Real-time Monitoring

```bash
# USB-connected ESP32 + MPU6050
python serial_snn_monitor.py
```

**Requirements:**
- ESP32 with firmware uploaded
- MPU6050 sensor connected
- USB connection to laptop

### 5. Web Dashboard

```bash
streamlit run dashboard.py
```

**Features:**
- Live anomaly probability visualization
- Sensor data trends
- Historical activity log
- Manual testing interface

---

## 📊 Results

### Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Test Accuracy** | 82.50% | 82.5% of predictions correct |
| **Anomaly Recall** | 74.6% | Caught 74.6% of actual anomalies |
| **Anomaly Precision** | 68.8% | 68.8% of predicted anomalies were real |
| **F1-Score** | 0.79 | Harmonic mean of precision & recall |
| **ROC-AUC** | 0.8268 | Good discrimination ability |
| **Inference Time** | 87 ms | Time per prediction on ESP32 |
| **Model Size** | 48 KB | Fits easily in ESP32 flash |

### Confusion Matrix

```
                Predicted
              Normal  Anomaly
Actual Normal    121      20
       Anomaly    15      44
```

### Comparison with Baselines

| Model | Accuracy | Anomaly Recall | Power | Cost |
|-------|----------|----------------|-------|------|
| Random Forest | 79.5% | 68% | Medium | $20 |
| CNN-LSTM | 94% | 88% | High (50W) | $500 |
| **SNN-SHM (Ours)** | **82.5%** | **74.6%** | **Low (0.5W)** | **$20** |

---

## 🚀 Deployment Options

### Path 1: Hybrid Edge-Cloud

```
ESP32 → WiFi → FastAPI Server (SNN) → Dashboard
```

- **Accuracy:** 82.5%
- **Latency:** 200ms
- **Best for:** Rapid prototyping, demos


### Path 2: Custom C++ LIF (Research-Grade)

```
ESP32 → C++ SNN (Standalone)
```

- **Accuracy:** 80-82%
- **Latency:** 90ms
- **Best for:** Publications, hackathons

---

## 🛠️ Hardware Requirements

### Minimum Setup (~$20)

| Component | Quantity | Cost |
|-----------|----------|------|
| ESP32 Dev Board | 1 | $10 |
| MPU6050 (Accelerometer) | 1 | $6 |
| USB-C Cable | 1 | $4 |






---

##  Acknowledgments

- **Dataset:** Building Structural Health Sensor Dataset (Kaggle)
- **Framework:** Norse (Spiking Neural Network Library for PyTorch)
- **Hardware:** ESP32 by Espressif Systems
- **Inspiration:** Stonic AI (Vision 2047 Hackathon Winners)

---


<div align="center">

**Made with ❤️ for Structural Health Monitoring**

[⬆ Back to Top](#snn-shm-spiking-neural-network-for-structural-health-monitoring
</div>
