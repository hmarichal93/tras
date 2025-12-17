#!/bin/bash
# Download INBD pretrained models
# This script downloads models from the INBD repository

set -e

echo "Downloading INBD pretrained models..."

# Check if src directory exists
if [ ! -d "src" ]; then
    echo "Error: src directory not found. Please clone INBD repository first:"
    echo "  git clone https://github.com/hmarichal93/INBD.git src"
    exit 1
fi

cd src

# Use INBD's fetch script if available
if [ -f "fetch_pretrained_models.py" ]; then
    echo "Using INBD's fetch_pretrained_models.py script..."
    python fetch_pretrained_models.py
else
    echo "Error: fetch_pretrained_models.py not found in INBD repository"
    exit 1
fi

echo "✓ INBD models downloaded successfully!"
echo ""
echo "Available models:"
echo "  - INBD_EH: Empetrum hermaphroditum (shrub)"
echo "  - INBD_DO: Dryas octopetala (shrub)"
echo "  - INBD_VM: Vaccinium myrtillus (shrub)"
echo "  - INBD_UruDendro: Pinus taeda (tree)"
echo ""
echo "Models are stored in: $(pwd)/checkpoints/"

