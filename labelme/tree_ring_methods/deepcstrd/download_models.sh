#!/bin/bash
# Download DeepCS-TRD pre-trained models

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="$SCRIPT_DIR/models/deep_cstrd"

echo "This script would download DeepCS-TRD models from the original repository."
echo ""
echo "Model files are large (~550MB total) and should be downloaded separately."
echo ""
echo "To use DeepCS-TRD, please:"
echo "1. Clone the original repository:"
echo "   git clone https://github.com/hmarichal93/deepcstrd.git /tmp/deepcstrd"
echo ""
echo "2. Copy the model files:"
echo "   mkdir -p $MODELS_DIR"
echo "   cp -r /tmp/deepcstrd/models/deep_cstrd/*.pth $MODELS_DIR/"
echo ""
echo "Available models:"
echo "  - 0_all_1504.pth (generic model)"
echo "  - 0_pinus_v1_1504.pth"
echo "  - 0_pinus_v2_1504.pth"
echo "  - 0_gleditsia_1504.pth"
echo "  - 0_salix_1504.pth"
echo "  - 256_* (tiled versions)"
echo ""
echo "Or download directly from:"
echo "https://github.com/hmarichal93/deepcstrd/tree/main/models/deep_cstrd"

