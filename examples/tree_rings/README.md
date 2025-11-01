# Tree Ring Detection in Wood Cross Sections

This specialized version of labelme includes integrated methods for automatic tree ring detection in wood cross-section images.

## Available Methods

### 1. APD (Automatic Pith Detection)
Automatically detects the pith (center) of a tree cross-section using structural tensor analysis.

**Usage in GUI:**
1. Open a wood cross-section image in labelme
2. Click on "Tree Ring Detection" or press the appropriate shortcut
3. Click "Auto-detect pith" button
4. The center coordinates will be automatically updated

**CLI Usage:**
```bash
# The pith detection is integrated into the ring detection workflow
```

### 2. CS-TRD (Classical Tree Ring Detection)
Classical method using Canny edge detection and polar coordinate transformation.

**Parameters:**
- `sigma`: Gaussian smoothing sigma (default: 3.0)
- `th_low`: Low threshold for Canny edge detection (default: 5.0)
- `th_high`: High threshold for Canny edge detection (default: 20.0)
- `alpha`: Alpha parameter for angular sampling (default: 30)
- `nr`: Number of radial samples (default: 360)

**CLI Usage:**
```bash
labelme_ring_detect examples/tree_rings/sample.png \
  --out examples/tree_rings \
  --center-x 600 \
  --center-y 600 \
  --angular-steps 360 \
  --min-radius 10 \
  --relative-threshold 0.35 \
  --min-peak-distance 4 \
  --min-coverage 0.6
```

### 3. DeepCS-TRD (Deep Learning Tree Ring Detection)
Deep learning-based method using U-Net architecture for automatic tree ring delineation.

**Available Models:**
- `generic`: General purpose model (default)
- `pinus_v1`: Specialized for Pinus species (version 1)
- `pinus_v2`: Specialized for Pinus species (version 2)
- `gleditsia`: Specialized for Gleditsia species
- `salix`: Specialized for Salix Glauca species

**Usage in GUI:**
1. Open a wood cross-section image in labelme
2. Click on "Tree Ring Detection"
3. Optionally click "Auto-detect pith" to find the center
4. Click "Detect with DeepCS-TRD (AI)" button
5. The rings will be automatically detected and added to the image

**Parameters:**
- `model_id`: Model to use (see above)
- `tile_size`: Tile size for processing (0 for no tiling, 256 for tiling)
- `alpha`: Alpha parameter for angular sampling (default: 45)
- `nr`: Number of radial samples (default: 360)
- `total_rotations`: Number of rotations for test-time augmentation (default: 5)
- `prediction_map_threshold`: Threshold for binary prediction map (default: 0.5)

## Example Workflow

1. **Load Image:**
   ```bash
   labelme examples/tree_rings/sample.png
   ```

2. **Automatic Detection:**
   - Go to menu: `Tools > Tree Ring Detection` or use shortcut
   - Click "Auto-detect pith" to find the center
   - Click "Detect with DeepCS-TRD (AI)" for automatic ring detection
   - Or adjust parameters and click "OK" for classical polar-based detection

3. **Manual Refinement:**
   - Edit detected rings by selecting and dragging points
   - Add missing rings manually using polygon tool
   - Delete incorrect detections

4. **Save Results:**
   - Save as JSON file containing ring annotations
   - Export to other formats as needed

## Parameters Tuning Guide

### For low-quality or damaged samples:
- Increase `min-peak-distance` (e.g., 5-8)
- Increase `relative-threshold` (e.g., 0.4-0.5)
- Decrease `min-coverage` (e.g., 0.4-0.5)

### For high-quality samples:
- Use default parameters
- Consider using DeepCS-TRD for best results

### For species-specific detection:
- Use the appropriate DeepCS-TRD model for your species
- If no specific model is available, use the `generic` model

## References

This implementation is based on methods from the TRAS repository:
- APD: https://github.com/hmarichal93/apd
- CS-TRD: https://github.com/hmarichal93/cstrd_ipol
- DeepCS-TRD: https://github.com/hmarichal93/deepcstrd
- TRAS: https://github.com/hmarichal93/tras

## Citation

If you use these methods in your research, please cite the original papers:

```
@article{marichal2023apd,
  title={Automatic Pith Detection in Wood Cross-Section Images},
  author={Marichal, Henry and others},
  journal={Image Processing On Line},
  year={2023}
}

@article{marichal2023cstrd,
  title={Tree Ring Delineation in Wood Cross-Section Images},
  author={Marichal, Henry and others},
  journal={Image Processing On Line},
  year={2023}
}

@article{marichal2024deepcstrd,
  title={Deep Learning for Tree Ring Delineation},
  author={Marichal, Henry and others},
  journal={arXiv preprint},
  year={2024}
}
```

## Notes

- The center (pith) detection is crucial for accurate ring detection
- Manual refinement may be needed for challenging samples
- For best results with DeepCS-TRD, ensure the pith is correctly detected
- CS-TRD requires compilation of the Devernay edge detector (C code)
