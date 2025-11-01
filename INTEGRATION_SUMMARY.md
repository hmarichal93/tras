# Tree Ring Detection Integration Summary

## Overview

Successfully converted the labelme repository into a specialized tool for tree ring detection in wood cross-sections by integrating methods from the TRAS repository (https://github.com/hmarichal93/tras).

## Branch: `tras`

All work has been completed in the new git branch named `tras`.

## What Was Integrated

### 1. APD (Automatic Pith Detection)
**Source:** https://github.com/hmarichal93/apd  
**Location:** `labelme/tree_ring_methods/apd/`  
**Purpose:** Automatically detect the pith (center) of tree cross-sections using structural tensor analysis

**Files:**
- `automatic_wood_pith_detector.py` - Main detection algorithm
- `structural_tensor.py` - Structural tensor computation
- `optimization.py` - Optimization for pith localization
- `pclines_parallel_coordinates.py` - Parallel coordinates filtering
- Supporting utilities (geometry, image, io, logger)

### 2. CS-TRD (Classical Tree Ring Detection)
**Source:** https://github.com/hmarichal93/cstrd_ipol  
**Location:** `labelme/tree_ring_methods/cstrd/`  
**Purpose:** Classical edge-based ring detection using polar coordinates

**Files:**
- `cross_section_tree_ring_detection/` - Complete detection pipeline
- `devernay/` - Devernay edge detector (C code, requires compilation)
- `compile_devernay.sh` - Compilation script

**Key Components:**
- Polar coordinate transformation
- Canny-Devernay edge detection
- Chain formation and merging
- Ring boundary extraction

### 3. DeepCS-TRD (Deep Learning Tree Ring Detection)
**Source:** https://github.com/hmarichal93/deepcstrd  
**Location:** `labelme/tree_ring_methods/deepcstrd/`  
**Purpose:** Neural network-based detection using U-Net architecture

**Files:**
- `deep_cstrd/` - Deep learning implementation
  - `deep_tree_ring_detection.py` - Main detection class
  - `model.py` - U-Net model architecture
  - `dataset.py` - Data loading and preprocessing
  - `preprocessing.py`, `sampling.py` - Image processing
  - `losses.py`, `training.py` - Training components
- `models/deep_cstrd/` - Pre-trained model weights (download separately)
- `download_models.sh` - Model download helper

**Available Models:**
- `0_all_1504.pth` - Generic model (~55MB each)
- `0_pinus_v1_1504.pth` - Pinus V1
- `0_pinus_v2_1504.pth` - Pinus V2
- `0_gleditsia_1504.pth` - Gleditsia
- `0_salix_1504.pth` - Salix Glauca
- `256_*.pth` - Tiled versions for large images

## Integration Points

### Helper Modules
Created clean integration interfaces:

1. **`labelme/utils/apd_helper.py`**
   - `detect_pith_apd(image, method="apd")` → (x, y) pith coordinates
   - Handles image format conversion
   - Wraps APD algorithm with sensible defaults

2. **`labelme/utils/cstrd_helper.py`**
   - `detect_rings_cstrd(image, center_xy, sigma, th_low, th_high, ...)` → list of ring polylines
   - Classical edge-based detection
   - Configurable parameters

3. **`labelme/utils/deepcstrd_helper.py`**
   - `detect_rings_deepcstrd(image, center_xy, model_id, ...)` → list of ring polylines
   - Deep learning detection
   - Model selection and parameter tuning

### GUI Integration

**File:** `labelme/widgets/tree_ring_dialog.py`
- Dialog for tree ring detection with parameter controls
- "Auto-detect pith" button using APD
- "Detect with DeepCS-TRD (AI)" button for neural detection
- Manual parameter adjustment for classical detection
- Real-time parameter validation

### CLI Integration

**File:** `labelme/cli/ring_detect.py`
- Command-line tool: `labelme_ring_detect`
- Batch processing support
- Configurable parameters
- JSON output compatible with labelme

### Existing Method

**File:** `labelme/_automation/tree_rings.py`
- Polar-based ring detection (already existed)
- Enhanced with better documentation
- Used as fallback method

## Dependencies Added

Updated `pyproject.toml` with:
```python
"opencv-python>=4.5.0",
"scipy>=1.7.0", 
"shapely>=1.7.0",
"torch>=2.0.0",
"torchvision>=0.15.0",
"ultralytics>=8.0.0",  # For YOLO-based pith detection
"pandas>=1.3.0",
```

## Documentation

### Main Documentation
- **`TREE_RING_DETECTION.md`** - Comprehensive guide
  - Installation instructions
  - Method descriptions
  - Usage examples (GUI and CLI)
  - Parameter tuning guide
  - Troubleshooting

### Example Documentation
- **`examples/tree_rings/README.md`** - Detailed usage guide
  - Workflow examples
  - Parameter explanations
  - Tips for different sample qualities
  - CLI examples

### README Updates
- **`README.md`** - Added tree ring detection section
- Brief overview and CLI example

## File Statistics

- **65 files changed**
- **11,667 lines added**
- **47 lines removed**

## Code Organization

```
labelme/
├── tree_ring_methods/          # New directory for detection methods
│   ├── __init__.py
│   ├── apd/                    # APD implementation (8 files)
│   ├── cstrd/                  # CS-TRD implementation (15 files)
│   └── deepcstrd/              # DeepCS-TRD implementation (12 files)
├── utils/                      # Integration helpers
│   ├── apd_helper.py          # New
│   ├── cstrd_helper.py        # New
│   └── deepcstrd_helper.py    # New
├── widgets/
│   └── tree_ring_dialog.py    # New GUI dialog
├── cli/
│   └── ring_detect.py         # New CLI tool
└── _automation/
    └── tree_rings.py          # Enhanced existing method
```

## Key Design Decisions

### 1. Minimal Code Approach
- Copied only essential source files from TRAS methods
- Created thin wrapper functions in helper modules
- Reused existing labelme infrastructure

### 2. Model Handling
- Large model files (~550MB) excluded from git
- Added to `.gitignore`
- Created download script and documentation
- Users download models separately

### 3. Compilation Requirements
- Devernay edge detector (C code) requires compilation
- Created `compile_devernay.sh` script
- Clear instructions in documentation

### 4. Method Independence
- Each method in separate directory
- Clean interfaces in helper modules
- Can use methods independently or together

## Usage Examples

### GUI Usage
```bash
# Launch labelme
labelme path/to/wood_image.jpg

# In GUI:
# 1. Tools > Tree Ring Detection
# 2. Click "Auto-detect pith"
# 3. Click "Detect with DeepCS-TRD (AI)"
# 4. Edit results manually if needed
# 5. Save JSON
```

### CLI Usage
```bash
# Classical detection
labelme_ring_detect samples/*.jpg \
  --out output_dir \
  --center-x 600 \
  --center-y 600 \
  --angular-steps 720 \
  --min-radius 10 \
  --relative-threshold 0.35

# Batch processing
labelme_ring_detect images/**/*.png \
  --out results \
  --center-x 500 \
  --center-y 500
```

## Testing

Created test file: `tests/labelme_tests/utils_tests/test_tree_rings.py`
- Test structure in place
- TODO: Add comprehensive unit tests

## Next Steps (Optional)

1. **Download Models:** Run `labelme/tree_ring_methods/deepcstrd/download_models.sh`
2. **Compile Devernay:** Run `labelme/tree_ring_methods/cstrd/compile_devernay.sh`
3. **Install Dependencies:** `pip install -e .`
4. **Test Installation:** Try example workflows in `examples/tree_rings/`

## Credits

This integration is based on work by Henry Marichal and collaborators:
- TRAS: https://github.com/hmarichal93/tras
- APD: https://github.com/hmarichal93/apd
- CS-TRD: https://github.com/hmarichal93/cstrd_ipol
- DeepCS-TRD: https://github.com/hmarichal93/deepcstrd

## License

Maintains GPL-3.0 from original labelme. Integrated methods maintain their respective licenses.

## Git Details

- **Branch:** `tras`
- **Commit:** `3203a1a`
- **Commit Message:** "feat: Add tree ring detection methods for wood cross-section analysis"
- **Date:** November 1, 2025

## Summary

✅ Successfully integrated three tree ring detection methods  
✅ Created clean interfaces and helper modules  
✅ Added comprehensive documentation  
✅ Updated dependencies  
✅ Created example data and workflows  
✅ Minimal code approach - only essential files copied  
✅ All work committed to `tras` branch

The labelme repository is now specialized for tree ring detection in wood cross-sections!

