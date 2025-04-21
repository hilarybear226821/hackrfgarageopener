
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdint.h>

// Advanced signal processing functions in C for better performance
typedef struct {
    double sample_rate;
    double center_freq;
    double bandwidth;
    uint32_t buffer_size;
} SignalConfig;

typedef struct {
    double complex* data;
    size_t length;
} SignalBuffer;

// Initialize signal processing configuration
SignalConfig* init_signal_config(double sample_rate, double center_freq, 
                               double bandwidth, uint32_t buffer_size) {
    SignalConfig* config = (SignalConfig*)malloc(sizeof(SignalConfig));
    if (config) {
        config->sample_rate = sample_rate;
        config->center_freq = center_freq;
        config->bandwidth = bandwidth;
        config->buffer_size = buffer_size;
    }
    return config;
}

// Create signal buffer
SignalBuffer* create_signal_buffer(size_t length) {
    SignalBuffer* buffer = (SignalBuffer*)malloc(sizeof(SignalBuffer));
    if (buffer) {
        buffer->data = (double complex*)malloc(length * sizeof(double complex));
        buffer->length = length;
    }
    return buffer;
}

// Fast Fourier Transform implementation
void fft(double complex* signal, size_t length) {
    if (length <= 1) return;

    size_t half = length / 2;
    double complex* even = (double complex*)malloc(half * sizeof(double complex));
    double complex* odd = (double complex*)malloc(half * sizeof(double complex));

    for (size_t i = 0; i < half; i++) {
        even[i] = signal[2*i];
        odd[i] = signal[2*i+1];
    }

    fft(even, half);
    fft(odd, half);

    for (size_t k = 0; k < half; k++) {
        double complex t = cexp(-2.0 * I * M_PI * k / length) * odd[k];
        signal[k] = even[k] + t;
        signal[k + half] = even[k] - t;
    }

    free(even);
    free(odd);
}

// High-performance signal demodulation with error checking
int demodulate_signal(SignalBuffer* buffer, SignalConfig* config) {
    if (!buffer || !config || !buffer->data) {
        return -1;  // Invalid parameters
    }

    double freq_step = config->sample_rate / buffer->length;
    double complex* demod = (double complex*)malloc(buffer->length * sizeof(double complex));
    
    if (!demod) {
        return -2;  // Memory allocation failed
    }

    // Use SIMD-friendly loop structure
    #pragma omp parallel for
    for (size_t i = 0; i < buffer->length; i++) {
        double t = i / config->sample_rate;
        double phase = -2.0 * M_PI * config->center_freq * t;
        demod[i] = buffer->data[i] * (cos(phase) + I * sin(phase));
    }

    fft(demod, buffer->length);

    // Optimize memory copy
    memcpy(buffer->data, demod, buffer->length * sizeof(double complex));
    
    free(demod);
    return 0;  // Success
}

// Extract rolling code patterns
uint32_t* extract_patterns(SignalBuffer* buffer, size_t* num_patterns) {
    uint32_t threshold = 0.7 * config->sample_rate;
    uint32_t* patterns = (uint32_t*)malloc(sizeof(uint32_t) * 100);
    *num_patterns = 0;

    for (size_t i = 0; i < buffer->length - 32; i++) {
        if (cabs(buffer->data[i]) > threshold) {
            uint32_t pattern = 0;
            for (int j = 0; j < 32; j++) {
                pattern = (pattern << 1) | (cabs(buffer->data[i+j]) > threshold);
            }
            patterns[(*num_patterns)++] = pattern;
        }
    }

    return patterns;
}

// Cleanup functions
void free_signal_config(SignalConfig* config) {
    free(config);
}

void free_signal_buffer(SignalBuffer* buffer) {
    if (buffer) {
        free(buffer->data);
        free(buffer);
    }
}
