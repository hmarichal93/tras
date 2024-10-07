#!/bin/bash

# Ensure the script stops if any command fails
set -e

pip install -r requirements.txt


# Install repository dependencies
echo "Installing repository dependencies..."
if git submodule update --init; then
    echo "Repository dependencies installed successfully."
else
    echo "Error installing repository dependencies."
    exit 1
fi

# Install INBD project dependencies
echo "Installing INBD dependencies..."
cd "./automatic_methods/tree_ring_delineation/mlbrief_inbd" || { echo "Directory not found: ./automatic_methods/tree_ring_delineation/mlbrief_inbd"; exit 1; }
if git submodule update --init; then
    echo "INBD dependencies installed successfully."
else
    echo "Error installing INBD dependencies."
    exit 1
fi

# Install CS-TRD project dependencies
echo "Installing CS-TRD dependencies..."
cd "../cstrd_ipol" || { echo "Directory not found: ../cstrd_ipol"; exit 1; }
if git submodule update --init && git checkout early_late_wood; then
    echo "CS-TRD dependencies installed successfully."
else
    echo "Error installing CS-TRD dependencies."
    exit 1
fi

# Compile the CS-TRD project
echo "Compiling devernay_1.0..."
cd ./externas/devernay_1.0 || { echo "Directory not found: ./externas/devernay_1.0"; exit 1; }
if make clean && make; then
    echo "devernay_1.0 compiled successfully."
else
    echo "Error compiling devernay_1.0."
    exit 1
fi

echo "Installation complete and environment ready."
