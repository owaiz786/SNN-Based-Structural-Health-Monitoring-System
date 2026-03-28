// lif_neuron.h
// Leaky Integrate-and-Fire Neuron with Q15 Fixed-Point Arithmetic

#ifndef LIF_NEURON_H
#define LIF_NEURON_H

#include <Arduino.h>
#include "weights/snn_config.h"

/**
 * Q15 Fixed-Point Arithmetic
 * Range: -1.0 to 0.99997
 * Resolution: 1/32768 ≈ 0.00003
 * 
 * Conversion:
 *   float → Q15: q15 = (int16_t)(float * 32768)
 *   Q15 → float: float = q15 / 32768.0
 */

// Global Q15 arithmetic functions (inline for performance)

/**
 * Q15 multiplication with proper scaling
 * (a * b) / 32768
 */
inline int16_t q15_multiply(int16_t a, int16_t b) {
    int32_t result = (int32_t)a * (int32_t)b;
    // Shift right by 15 bits (divide by 32768)
    result = result >> 15;
    // Saturate to Q15 range
    if (result > 32767) result = 32767;
    if (result < -32768) result = -32768;
    return (int16_t)result;
}

/**
 * Q15 addition with saturation
 */
inline int16_t q15_add(int16_t a, int16_t b) {
    int32_t result = (int32_t)a + (int32_t)b;
    // Saturate to Q15 range
    if (result > 32767) return 32767;
    if (result < -32768) return -32768;
    return (int16_t)result;
}

/**
 * Q15 subtraction with saturation
 */
inline int16_t q15_subtract(int16_t a, int16_t b) {
    int32_t result = (int32_t)a - (int32_t)b;
    // Saturate to Q15 range
    if (result > 32767) return 32767;
    if (result < -32768) return -32768;
    return (int16_t)result;
}

/**
 * Convert float to Q15 fixed-point
 */
inline int16_t float_to_q15(float value) {
    // Clamp to valid range
    if (value > 0.99997f) value = 0.99997f;
    if (value < -1.0f) value = -1.0f;
    return (int16_t)(value * 32768.0f);
}

/**
 * Convert Q15 fixed-point to float
 */
inline float q15_to_float(int16_t value) {
    return (float)value / 32768.0f;
}

/**
 * LIF Neuron Class
 */
class LIFNeuron {
private:
    int16_t membrane_potential;  // Q15 fixed-point
    int16_t threshold;           // Q15 spike threshold
    int16_t reset_value;         // Q15 post-spike reset
    int16_t decay;               // Q15 leak factor
    
public:
    /**
     * Constructor with default LIF parameters
     * threshold: 1.0 (16384 in Q15)
     * reset: 0.0 (0 in Q15)
     * decay: 0.995 (32678 in Q15)
     */
    LIFNeuron(
        int16_t thresh = LIF_THRESHOLD,
        int16_t reset = LIF_RESET,
        int16_t decay = LIF_DECAY
    ) : membrane_potential(0), threshold(thresh), 
        reset_value(reset), decay(decay) {}
    
    /**
     * Single timestep forward pass
     * @param input_current Q15 input current
     * @return true if spike occurred
     */
    bool step(int16_t input_current) {
        // Leak: V[t] = decay * V[t-1] (Q15 multiplication)
        membrane_potential = q15_multiply(membrane_potential, decay);
        
        // Integrate: V[t] += input
        membrane_potential = q15_add(membrane_potential, input_current);
        
        // Spike check
        if (membrane_potential >= threshold) {
            membrane_potential = reset_value;  // Reset after spike
            return true;  // Spike!
        }
        return false;  // No spike
    }
    
    /**
     * Get current membrane potential (for debugging)
     * @return Q15 membrane potential
     */
    int16_t get_potential() const {
        return membrane_potential;
    }
    
    /**
     * Reset neuron state
     */
    void reset() {
        membrane_potential = 0;
    }
};

#endif // LIF_NEURON_H