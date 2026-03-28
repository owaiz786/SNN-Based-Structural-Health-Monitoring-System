// snn_config.h
// Auto-generated configuration for SNN-SHM ESP32 Firmware
// Generated from PyTorch model: snn_model_optimized.pth

#ifndef SNN_CONFIG_H
#define SNN_CONFIG_H

// Model Architecture
#define INPUT_SIZE 5
#define HIDDEN_1_SIZE 128
#define HIDDEN_2_SIZE 64
#define HIDDEN_3_SIZE 32
#define OUTPUT_SIZE 2
#define TIME_STEPS 120

// LIF Neuron Parameters (Q15 fixed-point)
#define LIF_THRESHOLD 16384      // 1.0 in Q15
#define LIF_RESET 0              // 0.0 in Q15
#define LIF_DECAY 32678          // 0.995 in Q15
#define LIF_SCALE 127            // Weight quantization scale

// Classification Threshold (Q15 format)
#define ANOMALY_THRESHOLD 26214  // 0.8000 in Q15

// Hardware Pins
#define MPU_SDA 21
#define MPU_SCL 22
#define LED_PIN 2
#define BUZZER_PIN 4

// Sampling
#define SAMPLE_INTERVAL_MS 500

#endif // SNN_CONFIG_H
