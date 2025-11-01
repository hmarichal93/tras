# TRAS - Tree Ring Analyzer Suite

Professional tool for automatic tree ring detection and measurement in wood cross-section images. Integrates state-of-the-art methods for dendrochronology research.

## Installation

### 1. Clone Repository
```bash
git clone -b tras https://github.com/hmarichal93/tras.git
cd labelme
```

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

### 3. Compile Devernay Edge Detector (for CS-TRD)
```bash
cd tras/tree_ring_methods/cstrd/devernay
make
```

### 4. Download DeepCS-TRD Models (Optional)
```bash
cd tras/tree_ring_methods/deepcstrd
./download_models.sh
```

## Features

### 🔬 Automatic Detection Methods
- **APD (Automatic Pith Detection)**: Finds tree center using structural tensor analysis (~1 second, CPU)
- **CS-TRD (Classical Tree Ring Detection)**: Edge-based method using Canny detection (~73s on 2400×2400px, CPU)
- **DeepCS-TRD**: Deep learning U-Net architecture with pre-trained models (~101s on 2400×2400px, GPU)

### 🖼️ Image Preprocessing
- Manual crop with margin warnings for edge detection
- Resize (10-100% scaling)
- Background removal using U2Net model
- All preprocessing parameters stored for traceability

### 📏 Scale Calibration
- Draw line segment of known length OR direct input
- Units: mm, cm, μm per pixel
- Auto-adjusts when resizing images
- Physical measurements in all exports

### 📊 Ring Analysis
- Area, perimeter, cumulative area
- Radial width measurement along user-defined transect
- Year-based ring labeling (outermost = harvested year)
- Sample metadata (code, harvested year, observations)

### 💾 Export Options
- **CSV**: Ring properties with metadata (area, perimeter, widths)
- **.POS**: CooRecorder-compatible format for radial measurements
- Physical units automatically converted from pixel measurements


## Usage

### Quick Start
```bash
tras                              # Launch GUI
tras /path/to/image.jpg          # Open with image
tras --version                    # Check version
```

### Workflow

1. **Load Image**: `File > Open` or drag-and-drop
2. **Set Scale** (optional): `Tools > Set Image Scale`
3. **Preprocess** (optional): `Tools > Preprocess Image`
   - Crop to focus on cross-section
   - Resize for faster processing
   - Remove background
4. **Detect Rings**: `Tools > Tree Ring Detection`
   - Choose pith method (APD or manual click)
   - Select detection method (CS-TRD or DeepCS-TRD)
5. **Add Metadata**: `Tools > Sample Metadata`
6. **Measure Radial Width** (optional): `Tools > Measure Ring Width`
7. **View Properties**: `Tools > Ring Properties`
8. **Export**: CSV or .POS format

## Citations

If you use TRAS in your research, please cite the following papers:

### APD (Automatic Pith Detection)
Marichal, H., Passarella, D., & Randall, G. (2024). *Automatic Pith Detection in Cross Section Tree Ring Images*. In: Proceedings of the 27th International Conference on Pattern Recognition (ICPR 2024). Lecture Notes in Computer Science, vol 15317. Springer.

**DOI:** [10.1007/978-3-031-78447-7_1](https://doi.org/10.1007/978-3-031-78447-7_1)

### CS-TRD (Classical Tree Ring Detection)
```bibtex
@misc{marichal2024cstrdcrosssectionstree,
  title={CS-TRD: a Cross Sections Tree Ring Detection method}, 
  author={Henry Marichal and Diego Passarella and Gregory Randall},
  year={2024},
  eprint={2305.10809},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2305.10809}
}
```

### DeepCS-TRD (Deep Learning Tree Ring Detection)
```bibtex
@misc{marichal2025deepcstrddeeplearningbasedcrosssection,
  title={DeepCS-TRD, a Deep Learning-based Cross-Section Tree Ring Detector}, 
  author={Henry Marichal and Verónica Casaravilla and Candice Power and Karolain Mello and Joaquín Mazarino and Christine Lucas and Ludmila Profumo and Diego Passarella and Gregory Randall},
  year={2025},
  eprint={2504.16242},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2504.16242}
}
```

## Requirements

- Python ≥ 3.9
- PyQt5
- OpenCV
- PyTorch (for DeepCS-TRD)
- Shapely 1.7.0
- See `pyproject.toml` for complete list

## Configuration

TRAS stores its configuration in `~/.trasrc`. You can customize:
- Default shape colors (wood theme)
- Keyboard shortcuts
- Detection method defaults
- UI preferences

## License

GPL-3.0-only

## Acknowledgments

Based on the original [LabelMe](https://github.com/wkentaro/labelme) project by Kentaro Wada.

Specialized for dendrochronology research by integrating methods from the [TRAS repository](https://github.com/hmarichal93/tras).

---

**TRAS v1.0.0** - Tree Ring Analyzer Suite  
Professional dendrochronology tool
