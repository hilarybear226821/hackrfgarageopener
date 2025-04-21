#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hackrf.h>
#include <unistd.h> // For sleep()

// Define constants
#define SAMPLE_RATE 2e6 // 2 MHz sample rate
#define TX_GAIN 40 // Transmit gain (adjust as needed)
#define IF_GAIN 20 // Intermediate frequency gain (adjust as needed)
#define BB_GAIN 20 // Baseband gain (adjust as needed)
#define MODULATION_TYPE "OOK" // On-Off Keying (OOK) modulation
#define CODE_BIT_DURATION 0.001 // 1ms bit duration (adjust as needed)

// Global HackRF device handle
hackrf_device* device = NULL;

// Function to initialize the HackRF device
int init_hackrf() {
    int result = hackrf_init();
    if (result != HACKRF_SUCCESS) {
        fprintf(stderr, "hackrf_init() failed: %s\n", hackrf_error_name(result));
        return 1;
    }

    result = hackrf_open(&device);
    if (result != HACKRF_SUCCESS) {
        fprintf(stderr, "hackrf_open() failed: %s\n", hackrf_error_name(result));
        hackrf_exit();
        return 1;
    }

    result = hackrf_set_sample_rate(device, SAMPLE_RATE);
    if (result != HACKRF_SUCCESS) {
        fprintf(stderr, "hackrf_set_sample_rate() failed: %s\n", hackrf_error_name(result));
        hackrf_close(device);
        hackrf_exit();
        return 1;
    }

    result = hackrf_set_txvga_gain(device, TX_GAIN);
    if (result != HACKRF_SUCCESS) {
        fprintf(stderr, "hackrf_set_txvga_gain() failed: %s\n", hackrf_error_name(result));
        hackrf_close(device);
        hackrf_exit();
        return 1;
    }

    result = hackrf_set_amp_enable(device, 1); // Enable amplifier
    if (result != HACKRF_SUCCESS) {
        fprintf(stderr, "hackrf_set_amp_enable() failed: %s\n", hackrf_error_name(result));
        hackrf_close(device);
        hackrf_exit();
        return 1;
    }

    result = hackrf_set_lna_gain(device, IF_GAIN);
    if (result != HACKRF_SUCCESS) {
        fprintf(stderr, "hackrf_set_lna_gain() failed: %s\n", hackrf_error_name(result));
        hackrf_close(device);
        hackrf_exit();
        return 1;
    }

    result = hackrf_set_baseband_filter_bandwidth(device, SAMPLE_RATE / 2); // Set filter bandwidth
    if (result != HACKRF_SUCCESS) {
        fprintf(stderr, "hackrf_set_baseband_filter_bandwidth() failed: %s\n", hackrf_error_name(result));
        hackrf_close(device);
        hackrf_exit();
        return 1;
    }

    return 0;
}

// Function to transmit a single bit using OOK modulation
int transmit_bit(double frequency, int bit) {
    int result;
    if (bit == 1) {
        // Transmit carrier for bit duration
        result = hackrf_start_tx(device, NULL, NULL); // Start transmission
        if (result != HACKRF_SUCCESS) {
            fprintf(stderr, "hackrf_start_tx() failed: %s\n", hackrf_error_name(result));
            return 1;
        }
        usleep((int)(CODE_BIT_DURATION * 1e6)); // Sleep for bit duration (microseconds)
        result = hackrf_stop_tx(device); // Stop transmission
        if (result != HACKRF_SUCCESS) {
            fprintf(stderr, "hackrf_stop_tx() failed: %s\n", hackrf_error_name(result));
            return 1;
        }
    } else {
        // No transmission for bit duration
        usleep((int)(CODE_BIT_DURATION * 1e6)); // Sleep for bit duration (microseconds)
    }
    return 0;
}

// Function to transmit the code
int transmit_code(double frequency, char *code) {
    int result;
    result = hackrf_set_freq(device, (uint64_t)(frequency * 1e6));
    if (result != HACKRF_SUCCESS) {
        fprintf(stderr, "hackrf_set_freq() failed: %s\n", hackrf_error_name(result));
        return 1;
    }

    printf("Transmitting code '%s' at %.6f MHz using %s modulation.\n", code, frequency, MODULATION_TYPE);

    for (int i = 0; i < strlen(code); i++) {
        if (code[i] == '0') {
            if (transmit_bit(frequency, 0) != 0) return 1;
        } else if (code[i] == '1') {
            if (transmit_bit(frequency, 1) != 0) return 1;
        } else {
            fprintf(stderr, "Invalid character in code: %c\n", code[i]);
            return 1;
        }
    }

    return 0;
}

// Function to cleanup and exit
int cleanup_hackrf() {
    if (device != NULL) {
        hackrf_stop_tx(device); // Ensure transmission is stopped
        hackrf_close(device);
    }
    hackrf_exit();
    return 0;
}

// Extended transmission with frequency oscillation
int transmit_with_oscillation(double frequency, char *code, double bandwidth, double rate) {
    int result;
    double start_freq = frequency - (bandwidth / 2);
    double end_freq = frequency + (bandwidth / 2);
    int code_len = strlen(code);
    
    printf("Transmitting code '%s' with frequency oscillation.\n", code);
    printf("Center: %.6f MHz, Bandwidth: %.3f MHz, Rate: %.2f Hz\n", 
           frequency, bandwidth / 1e6, rate);
    
    // Calculate total transmission time based on code length
    double total_duration = code_len * CODE_BIT_DURATION;
    
    // Calculate number of oscillation cycles
    double oscillation_cycles = total_duration * rate;
    
    // Transmit the code multiple times with frequency oscillation
    for (int cycle = 0; cycle < (int)(oscillation_cycles) + 1; cycle++) {
        // Calculate current frequency based on sinusoidal oscillation
        double phase = (double)cycle / oscillation_cycles * 2 * M_PI;
        double current_freq = frequency + (bandwidth / 2) * sin(phase);
        
        result = hackrf_set_freq(device, (uint64_t)(current_freq * 1e6));
        if (result != HACKRF_SUCCESS) {
            fprintf(stderr, "hackrf_set_freq() failed: %s\n", hackrf_error_name(result));
            return 1;
        }
        
        printf("Oscillation cycle %d/%d at %.6f MHz\n", 
               cycle + 1, (int)(oscillation_cycles) + 1, current_freq);
        
        // Transmit the code once at this frequency
        for (int i = 0; i < code_len; i++) {
            if (code[i] == '0') {
                if (transmit_bit(current_freq, 0) != 0) return 1;
            } else if (code[i] == '1') {
                if (transmit_bit(current_freq, 1) != 0) return 1;
            } else {
                fprintf(stderr, "Invalid character in code: %c\n", code[i]);
                return 1;
            }
        }
        
        // Short pause between cycles
        usleep(10000); // 10ms pause
    }
    
    return 0;
}

int main(int argc, char *argv[]) {
    double frequency = 0.0;
    char *code = NULL;
    double bandwidth = 0.0;
    double rate = 0.0;
    int oscillation_mode = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            frequency = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            code = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            bandwidth = atof(argv[i + 1]) * 1e6; // Convert MHz to Hz
            oscillation_mode = 1;
            i++;
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            rate = atof(argv[i + 1]);
            oscillation_mode = 1;
            i++;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s -f <frequency_MHz> -c <code> [-b <bandwidth_MHz> -r <rate_Hz>]\n", argv[0]);
            printf("  -f <frequency>   : Center frequency in MHz\n");
            printf("  -c <code>        : Binary code to transmit (0s and 1s)\n");
            printf("  -b <bandwidth>   : Oscillation bandwidth in MHz (optional)\n");
            printf("  -r <rate>        : Oscillation rate in Hz (optional)\n");
            printf("  -h, --help       : Show this help message\n");
            return 0;
        } else {
            fprintf(stderr, "Invalid argument: %s\n", argv[i]);
            fprintf(stderr, "Use --help for usage information\n");
            return 1;
        }
    }

    // Validate required parameters
    if (frequency == 0.0) {
        fprintf(stderr, "Error: Frequency must be provided and non-zero.\n");
        return 1;
    }
    
    if (code == NULL) {
        fprintf(stderr, "Error: Code must be provided.\n");
        return 1;
    }
    
    // For oscillation mode, ensure both bandwidth and rate are provided
    if (oscillation_mode && (bandwidth == 0.0 || rate == 0.0)) {
        fprintf(stderr, "Error: Both bandwidth and rate must be provided for oscillation mode.\n");
        return 1;
    }

    // Initialize HackRF
    if (init_hackrf() != 0) {
        cleanup_hackrf();
        return 1;
    }

    int result;
    if (oscillation_mode) {
        // Transmit with frequency oscillation
        result = transmit_with_oscillation(frequency, code, bandwidth, rate);
    } else {
        // Standard transmission
        result = transmit_code(frequency, code);
    }

    // Cleanup and exit
    cleanup_hackrf();
    
    if (result == 0) {
        printf("Transmission complete.\n");
        return 0;
    } else {
        fprintf(stderr, "Transmission failed.\n");
        return 1;
    }
}