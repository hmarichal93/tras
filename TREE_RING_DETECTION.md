# Labelme for Tree Ring Detection

This specialized branch of labelme includes integrated methods for automatic tree ring detection and pith detection in wood cross-section images, based on methods from the [TRAS repository](https://github.com/hmarichal93/tras).

## Features

### 🎯 Three Detection Methods

1. **APD (Automatic Pith Detection)** - Automatically detect the pith (center) using structural tensor analysis
2. **CS-TRD (Classical Tree Ring Detection)** - Edge-based method using Canny detection and polar coordinates
3. **DeepCS-TRD (Deep Learning Tree Ring Detection)** - Neural network-based detection with pre-trained models

### 🌲 Pre-trained Models for DeepCS-TRD

- **Generic** - Works with most wood species
- **Pinus V1 & V2** - Optimized for Pinus species
- **Gleditsia** - Optimized for Gleditsia species
- **Salix Glauca** - Optimized for Salix species

## Installation

### 1. Clone the repository
```bash
git clone -b tras https://github.com/yourusername/labelme.git
cd labelme
```

### 2. Install dependencies
```bash
pip install -e .
```

This will install all required dependencies including:
- PyTorch (for deep learning models)
- OpenCV (for image processing)
- Ultralytics (for YOLO-based pith detection)
- Shapely (for geometric operations)
- And more...

### 3. Download DeepCS-TRD models (for deep learning detection)
```bash
cd labelme/tree_ring_methods/deepcstrd
./download_models.sh
# Follow the instructions to download model files (~550MB)
```

**Note:** Model files are not included in the repository due to their size. They must be downloaded separately from the original DeepCS-TRD repository.

### 4. Compile Devernay edge detector (for CS-TRD)
```bash
cd labelme/tree_ring_methods/cstrd
./compile_devernay.sh
```

## Quick Start

### Using the GUI

1. **Launch labelme:**
   ```bash
   labelme
   ```

2. **Open a wood cross-section image**

3. **Access tree ring detection:**
   - Menu: `Tools > Tree Ring Detection`
   - Or use keyboard shortcut (if configured)

4. **Auto-detect pith (center):**
   - Click "Auto-detect pith" button
   - The center coordinates will be automatically set

5. **Choose detection method:**
   - **Recommended:** Click "Detect with CS-TRD" for edge-based detection (CPU, 73 sec)
   - **Recommended:** Click "Detect with DeepCS-TRD (AI)" for AI-based detection (GPU, 101 sec)
   - **Alternative:** Adjust parameters and click "OK" for legacy polar-based detection

6. **Refine results:**
   - Manually edit detected rings
   - Add or remove ring boundaries
   - Save annotations as JSON

### Using the CLI

**CS-TRD detection (Recommended for CPU):**
```bash
# Run CS-TRD via Python API
python -c "
from labelme.utils.cstrd_helper import detect_rings_cstrd
from PIL import Image
import numpy as np

img = np.array(Image.open('path/to/image.jpg'))
rings = detect_rings_cstrd(img, center_xy=(600, 600))
print(f'Detected {len(rings)} rings')
"
```

**Legacy polar-based detection:**
```bash
labelme_ring_detect path/to/image.jpg \
  --out output_dir \
  --center-x 600 \
  --center-y 600 \
  --angular-steps 720 \
  --min-radius 10 \
  --relative-threshold 0.35 \
  --min-peak-distance 4 \
  --min-coverage 0.6
```

**Parameters:**
- `--center-x, --center-y`: Pith coordinates (required)
- `--angular-steps`: Number of angular samples (default: 720)
- `--min-radius`: Minimum radius from pith (default: 5.0)
- `--relative-threshold`: Peak detection threshold (default: 0.3)
- `--min-peak-distance`: Minimum pixels between rings (default: 3)
- `--min-coverage`: Minimum angular coverage for valid ring (default: 0.6)
- `--max-rings`: Maximum number of rings to detect (0 = no limit)

## Method Descriptions

### APD (Automatic Pith Detection)

Uses structural tensor analysis to automatically locate the pith center:
- Computes local orientation at each pixel
- Samples high-coherence orientations
- Optimizes to find convergence point (pith)
- No manual input required

**Files:** `labelme/tree_ring_methods/apd/`

### CS-TRD (Classical Tree Ring Detection)

Classical computer vision approach:
1. Converts image to polar coordinates centered at pith
2. Applies Canny edge detection
3. Chains edges along angular direction
4. Filters and merges chains to form ring boundaries
5. Converts back to Cartesian coordinates

**Files:** `labelme/tree_ring_methods/cstrd/`

**Note:** Requires compilation of Devernay edge detector (see installation)

### DeepCS-TRD (Deep Learning Tree Ring Detection)

Deep learning approach using U-Net architecture:
1. Pre-trained on labeled wood cross-section datasets
2. Processes image in polar coordinates
3. Predicts probability map of ring boundaries
4. Post-processes to extract ring contours
5. Test-time augmentation for improved accuracy

**Files:** `labelme/tree_ring_methods/deepcstrd/`

**Models:** Pre-trained weights are included in `labelme/tree_ring_methods/deepcstrd/models/`

## Directory Structure

```
labelme/
├── tree_ring_methods/          # Tree ring detection implementations
│   ├── apd/                    # Automatic Pith Detection
│   │   ├── automatic_wood_pith_detector.py
│   │   ├── structural_tensor.py
│   │   ├── optimization.py
│   │   └── ...
│   ├── cstrd/                  # Classical Tree Ring Detection
│   │   ├── cross_section_tree_ring_detection/
│   │   ├── devernay/          # Edge detector (requires compilation)
│   │   └── compile_devernay.sh
│   └── deepcstrd/             # Deep Learning Tree Ring Detection
│       ├── deep_cstrd/
│       ├── models/            # Pre-trained model weights
│       └── ...
├── utils/
│   ├── apd_helper.py          # APD integration
│   ├── cstrd_helper.py        # CS-TRD integration
│   └── deepcstrd_helper.py    # DeepCS-TRD integration
└── widgets/
    └── tree_ring_dialog.py    # GUI dialog for tree ring detection
```

## Parameter Tuning

### For Low-Quality Samples
- Increase `min-peak-distance` (5-8) - reduces false positives
- Increase `relative-threshold` (0.4-0.5) - only keeps strong peaks
- Decrease `min-coverage` (0.4-0.5) - allows partial rings

### For High-Quality Samples
- Use default parameters
- Prefer DeepCS-TRD for best results

### For Species-Specific Detection
- Use appropriate DeepCS-TRD model (pinus_v1, pinus_v2, gleditsia, salix)
- Fall back to `generic` model if species not available

## Examples

See `examples/tree_rings/` for:
- Sample wood cross-section images
- Example JSON annotations
- Detailed usage instructions

## Troubleshooting

### "Devernay edge detector not found" error
- Make sure to compile the Devernay detector: `cd labelme/tree_ring_methods/cstrd && ./compile_devernay.sh`
- Requires gcc or compatible C compiler

### "Model file not found" error
- Ensure model files are present in `labelme/tree_ring_methods/deepcstrd/models/deep_cstrd/`
- Model files should have been copied during installation

### Poor detection results
- Verify pith center is correctly placed
- Try adjusting parameters (see Parameter Tuning section)
- Try different detection method (classical vs deep learning)
- For deep learning, try species-specific model if available

### Import errors
- Ensure all dependencies are installed: `pip install -e .`
- For PyTorch, you may need to install manually: `pip install torch torchvision`

## Credits

This implementation integrates methods from:
- **APD**: https://github.com/hmarichal93/apd
- **CS-TRD**: https://github.com/hmarichal93/cstrd_ipol
- **DeepCS-TRD**: https://github.com/hmarichal93/deepcstrd
- **TRAS**: https://github.com/hmarichal93/tras

## Citation

If you use these tree ring detection methods in your research, please cite:

```bibtex
@article{marichal2023tras,
  title={TRAS: An Interactive Software for Tracing Tree Ring Cross Sections},
  author={Marichal, Henry and others},
  journal={Software X},
  year={2023}
}
```

## License

This software maintains the GPL-3.0 license from the original labelme project. The integrated tree ring detection methods maintain their respective licenses (see individual method repositories).

## Support

For issues specific to tree ring detection:
- Check the examples in `examples/tree_rings/`
- Review parameter tuning guidelines above
- Open an issue on GitHub with sample images and parameters used

For general labelme issues:
- See original labelme documentation
- Check https://github.com/wkentaro/labelme

