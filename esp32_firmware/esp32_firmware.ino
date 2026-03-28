/*
 * SNN-SHM ESP32 Firmware
 * Custom C++ LIF Implementation for Edge Anomaly Detection
 * 
 * Hardware: ESP32 + MPU6050
 * No WiFi, No Server - True Edge AI!
 */

#include <Wire.h>
#include <MPU6050.h>

// Custom SNN headers
#include "weights/snn_config.h"
#include "src/lif_neuron.h"
#include "src/snn_inference.h"
#include "src/feature_engineering.h"

// ==================== GLOBAL OBJECTS ====================
MPU6050 mpu;
SNNInference snn;

// ==================== SETUP ====================
void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    digitalWrite(BUZZER_PIN, LOW);
    
    // Initialize I2C
    Wire.begin(MPU_SDA, MPU_SCL);
    
    // Initialize MPU6050
    Serial.println("🔧 Initializing MPU6050...");
    mpu.initialize();
    
    if (mpu.testConnection()) {
        Serial.println("✅ MPU6050 connected");
    } else {
        Serial.println("❌ MPU6050 connection failed!");
        while (1) {
            digitalWrite(LED_PIN, HIGH);
            delay(500);
            digitalWrite(LED_PIN, LOW);
            delay(500);
        }
    }
    
    // Configure MPU6050
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
    
    Serial.println("🚀 SNN-SHM Edge Node Ready");
    Serial.println("==================================================");
    Serial.println("Model: Custom C++ LIF SNN");
    Serial.print("Architecture: ");
    Serial.print(INPUT_SIZE);
    Serial.print(" -> ");
    Serial.print(HIDDEN_1_SIZE);
    Serial.print(" -> ");
    Serial.print(HIDDEN_2_SIZE);
    Serial.print(" -> ");
    Serial.print(HIDDEN_3_SIZE);
    Serial.print(" -> ");
    Serial.println(OUTPUT_SIZE);
    Serial.print("Time Steps: ");
    Serial.println(TIME_STEPS);
    Serial.print("Threshold: ");
    Serial.println(ANOMALY_THRESHOLD / 32768.0f);
    Serial.println("==================================================");
    
    delay(2000);
}

// ==================== MAIN LOOP ====================
void loop() {
    // Read accelerometer
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getAcceleration(&ax, &ay, &az);
    mpu.getRotation(&gx, &gy, &gz);
    
    // Convert to physical units (m/s²)
    RawSensorData raw;
    raw.accel_x = ax / 16384.0f * 9.81f;
    raw.accel_y = ay / 16384.0f * 9.81f;
    raw.accel_z = az / 16384.0f * 9.81f;
    raw.strain = 85.0f;
    raw.temp = 25.0f;
    
    // Compute engineered features
    int16_t features[INPUT_SIZE];
    compute_features(raw, features);
    
    // Run SNN inference
    unsigned long start_time = micros();
    int prediction = snn.predict(features);
    unsigned long inference_time = micros() - start_time;
    
    // Get output scores for debugging
    int32_t scores[OUTPUT_SIZE];
    snn.get_output_spikes(scores);
    
    // Display results
    Serial.print("[");
    Serial.print(millis());
    Serial.print("ms] ");
    
    if (prediction == 1) {
        Serial.println("⚠️  ANOMALY DETECTED!");
        digitalWrite(LED_PIN, HIGH);
        digitalWrite(BUZZER_PIN, HIGH);
        delay(200);
        digitalWrite(BUZZER_PIN, LOW);
    } else {
        Serial.println("✅ Normal");
        digitalWrite(LED_PIN, LOW);
    }
    
    Serial.print("  Accel: [");
    Serial.print(raw.accel_x, 3);
    Serial.print(", ");
    Serial.print(raw.accel_y, 3);
    Serial.print(", ");
    Serial.print(raw.accel_z, 3);
    Serial.println("] m/s²");
    
    Serial.print("  Inference Time: ");
    Serial.print(inference_time);
    Serial.println(" µs");
    
    Serial.print("  Output Spikes: [");
    Serial.print(scores[0]);
    Serial.print(", ");
    Serial.print(scores[1]);
    Serial.println("]");
    
    Serial.println();
    
    // Wait for next sample
    delay(SAMPLE_INTERVAL_MS);
}