#!/bin/bash
# Compile script for transmitter.c

set -e  # Exit on error

echo "Compiling transmitter.c..."

# Check if hackrf development files are installed
if ! pkg-config --exists libhackrf; then
    echo "Error: libhackrf development files not found."
    echo "Please install the required dependencies:"
    echo "  sudo apt-get install libhackrf-dev"
    exit 1
fi

# Get compile flags from pkg-config
HACKRF_CFLAGS=$(pkg-config --cflags libhackrf)
HACKRF_LIBS=$(pkg-config --libs libhackrf)

# Compile with proper flags
gcc -Wall -Wextra -o transmitter transmitter.c -lm $HACKRF_CFLAGS $HACKRF_LIBS

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful! Executable: ./transmitter"
    chmod +x ./transmitter
    echo ""
    echo "Usage examples:"
    echo "  ./transmitter -f 315.0 -c 101010101010"
    echo "  ./transmitter -f 390.0 -c 101010101010 -b 0.2 -r 10.0"
    echo ""
    echo "For more information:"
    echo "  ./transmitter --help"
else
    echo "Compilation failed."
    exit 1
fi