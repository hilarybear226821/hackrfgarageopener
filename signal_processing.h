
#ifndef SIGNAL_PROCESSING_H
#define SIGNAL_PROCESSING_H

#include <complex.h>
#include <stdint.h>

typedef struct SignalConfig SignalConfig;
typedef struct SignalBuffer SignalBuffer;

SignalConfig* init_signal_config(double sample_rate, double center_freq, 
                               double bandwidth, uint32_t buffer_size);
SignalBuffer* create_signal_buffer(size_t length);
void fft(double complex* signal, size_t length);
void demodulate_signal(SignalBuffer* buffer, SignalConfig* config);
uint32_t* extract_patterns(SignalBuffer* buffer, size_t* num_patterns);
void free_signal_config(SignalConfig* config);
void free_signal_buffer(SignalBuffer* buffer);

#endif
