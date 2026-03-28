// snn_inference.h
// Complete SNN Forward Pass for ESP32
// FIXED: (1) threshold comparison, (2) accumulate-then-divide,
//        (3) pgm_read_byte signed cast, (4) bias scaling

#ifndef SNN_INFERENCE_H
#define SNN_INFERENCE_H

#include <Arduino.h>
#include <pgmspace.h>
#include "../weights/snn_config.h"
#include "../weights/snn_weights.h"
#include "lif_neuron.h"

class SNNInference {
private:
    LIFNeuron layer1[HIDDEN_1_SIZE];
    LIFNeuron layer2[HIDDEN_2_SIZE];
    LIFNeuron layer3[HIDDEN_3_SIZE];
    LIFNeuron output_neurons[OUTPUT_SIZE];

    int32_t output_spikes[OUTPUT_SIZE];

public:
    SNNInference() {
        reset();
    }

    void reset() {
        for (int i = 0; i < HIDDEN_1_SIZE; i++) layer1[i].reset();
        for (int i = 0; i < HIDDEN_2_SIZE; i++) layer2[i].reset();
        for (int i = 0; i < HIDDEN_3_SIZE; i++) layer3[i].reset();
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output_neurons[i].reset();
            output_spikes[i] = 0;
        }
    }

    int predict(const int16_t* input_features) {
        reset();

        int16_t current1[HIDDEN_1_SIZE];
        int16_t current2[HIDDEN_2_SIZE];
        int16_t current3[HIDDEN_3_SIZE];
        int16_t current_out[OUTPUT_SIZE];

        for (int t = 0; t < TIME_STEPS; t++) {

            // ── Layer 1: Input (INPUT_SIZE) → Hidden1 (HIDDEN_1_SIZE) ──────────
            for (int n = 0; n < HIDDEN_1_SIZE; n++) {
                int32_t sum = 0;
                for (int i = 0; i < INPUT_SIZE; i++) {
                    // FIX #3: cast to int8_t so negative weights stay negative
                    int8_t w = (int8_t)pgm_read_byte(&fc1_weight[n][i]);
                    // FIX #2: accumulate full products first
                    sum += (int32_t)input_features[i] * (int32_t)w;
                }
                // FIX #2: single divide after full accumulation
                sum /= LIF_SCALE;
                // FIX #4: bias was quantised at the same scale; add directly
                int8_t b = (int8_t)pgm_read_byte(&fc1_bias[n]);
                sum += (int32_t)b;

                if (sum >  32767) sum =  32767;
                if (sum < -32768) sum = -32768;
                current1[n] = (int16_t)sum;

                bool spiked = layer1[n].step(current1[n]);
                current1[n] = spiked ? 32767 : 0;
            }

            // ── Layer 2: Hidden1 (HIDDEN_1_SIZE) → Hidden2 (HIDDEN_2_SIZE) ────
            for (int n = 0; n < HIDDEN_2_SIZE; n++) {
                int32_t sum = 0;
                for (int i = 0; i < HIDDEN_1_SIZE; i++) {
                    int8_t w = (int8_t)pgm_read_byte(&fc2_weight[n][i]);
                    sum += (int32_t)current1[i] * (int32_t)w;
                }
                sum /= LIF_SCALE;
                int8_t b = (int8_t)pgm_read_byte(&fc2_bias[n]);
                sum += (int32_t)b;

                if (sum >  32767) sum =  32767;
                if (sum < -32768) sum = -32768;
                current2[n] = (int16_t)sum;

                bool spiked = layer2[n].step(current2[n]);
                current2[n] = spiked ? 32767 : 0;
            }

            // ── Layer 3: Hidden2 (HIDDEN_2_SIZE) → Hidden3 (HIDDEN_3_SIZE) ────
            for (int n = 0; n < HIDDEN_3_SIZE; n++) {
                int32_t sum = 0;
                for (int i = 0; i < HIDDEN_2_SIZE; i++) {
                    int8_t w = (int8_t)pgm_read_byte(&fc3_weight[n][i]);
                    sum += (int32_t)current2[i] * (int32_t)w;
                }
                sum /= LIF_SCALE;
                int8_t b = (int8_t)pgm_read_byte(&fc3_bias[n]);
                sum += (int32_t)b;

                if (sum >  32767) sum =  32767;
                if (sum < -32768) sum = -32768;
                current3[n] = (int16_t)sum;

                bool spiked = layer3[n].step(current3[n]);
                current3[n] = spiked ? 32767 : 0;
            }

            // ── Output layer: Hidden3 (HIDDEN_3_SIZE) → Output (OUTPUT_SIZE) ──
            for (int n = 0; n < OUTPUT_SIZE; n++) {
                int32_t sum = 0;
                for (int i = 0; i < HIDDEN_3_SIZE; i++) {
                    int8_t w = (int8_t)pgm_read_byte(&fc_out_weight[n][i]);
                    sum += (int32_t)current3[i] * (int32_t)w;
                }
                sum /= LIF_SCALE;
                int8_t b = (int8_t)pgm_read_byte(&fc_out_bias[n]);
                sum += (int32_t)b;

                if (sum >  32767) sum =  32767;
                if (sum < -32768) sum = -32768;
                current_out[n] = (int16_t)sum;

                bool spiked = output_neurons[n].step(current_out[n]);
                if (spiked) output_spikes[n]++;
            }
        }

        // FIX #1: compare spike *count* (0-TIME_STEPS) against a spike-rate
        // threshold, NOT a Q15 value.
        // ANOMALY_THRESHOLD in snn_config.h is 0.8 in Q15 (= 26214).
        // Here we convert that ratio to a spike count: 0.8 * 120 = 96.
        int spike_threshold = (int)(((float)ANOMALY_THRESHOLD / 32768.0f) * TIME_STEPS);
        return (output_spikes[1] > spike_threshold) ? 1 : 0;
    }

    /**
     * Return raw spike counts for both output neurons.
     * Useful for serial debugging – print both values to verify
     * they are no longer always saturated at 120.
     */
    void get_output_spikes(int32_t* spikes) {
        spikes[0] = output_spikes[0];
        spikes[1] = output_spikes[1];
    }

    /**
     * Return spike *rates* in [0, 1] range (spikes / TIME_STEPS).
     * Mimics the softmax probability your PyTorch model outputs.
     */
    void get_output_rates(float* rates) {
        rates[0] = (float)output_spikes[0] / (float)TIME_STEPS;
        rates[1] = (float)output_spikes[1] / (float)TIME_STEPS;
    }
};

#endif // SNN_INFERENCE_H