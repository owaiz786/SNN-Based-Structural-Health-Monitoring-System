// feature_engineering.h
// Convert raw sensor data to 5 engineered features

#ifndef FEATURE_ENGINEERING_H
#define FEATURE_ENGINEERING_H

#include <Arduino.h>
#include <math.h>
#include "lif_neuron.h"  // For float_to_q15
#include "../weights/snn_config.h"

struct RawSensorData {
    float accel_x, accel_y, accel_z;
    float strain;
    float temp;
};

/**
 * Convert 3-axis acceleration to 5 engineered features
 * Features: [ax, ay, az, magnitude, sum_abs]
 */
void compute_features(const RawSensorData& raw, int16_t* features) {
    // Compute derived features
    float magnitude = sqrt(raw.accel_x * raw.accel_x +
                           raw.accel_y * raw.accel_y +
                           raw.accel_z * raw.accel_z);
    
    float sum_abs = fabs(raw.accel_x) + fabs(raw.accel_y) + fabs(raw.accel_z);
    
    // 5 engineered features (matching Python training)
    float features_float[INPUT_SIZE] = {
        raw.accel_x,   // 0: Accel_X
        raw.accel_y,   // 1: Accel_Y
        raw.accel_z,   // 2: Accel_Z
        magnitude,     // 3: Acceleration magnitude
        sum_abs        // 4: Total absolute acceleration
    };
    
    // Normalize features to [-1, 1] range
    // These ranges should match your training data
    float min_vals[INPUT_SIZE] = {-15.0, -15.0, -15.0, 0.0, 0.0};
    float max_vals[INPUT_SIZE] = {15.0, 15.0, 15.0, 25.0, 30.0};
    
    for (int i = 0; i < INPUT_SIZE; i++) {
        // Min-max normalization to [0, 1]
        float normalized = (features_float[i] - min_vals[i]) / 
                           (max_vals[i] - min_vals[i] + 0.000001f);
        
        // Clamp to [0, 1]
        if (normalized < 0.0f) normalized = 0.0f;
        if (normalized > 1.0f) normalized = 1.0f;
        
        // Scale to [-1, 1] for Q15 representation
        float scaled = normalized * 2.0f - 1.0f;
        
        // Convert to Q15
        features[i] = float_to_q15(scaled);
    }
}

#endif // FEATURE_ENGINEERING_H