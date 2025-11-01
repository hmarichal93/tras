#!/bin/bash
# Compile Devernay edge detector for CS-TRD

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEVERNAY_DIR="$SCRIPT_DIR/devernay"

cd "$DEVERNAY_DIR"

# Compile the Devernay edge detector
echo "Compiling Devernay edge detector..."
make clean 2>/dev/null || true
make

if [ $? -eq 0 ]; then
    echo "Devernay edge detector compiled successfully!"
    echo "Executable created at: $DEVERNAY_DIR/devernay_cmd"
else
    echo "Error: Failed to compile Devernay edge detector"
    echo "Make sure you have a C compiler (gcc) installed"
    exit 1
fi

