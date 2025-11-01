# LabelMe - Tree Ring Detection

Specialized tool for automatic tree ring detection and measurement in wood cross-section images.

## Installation

### 1. Clone Repository
```bash
git clone -b tras https://github.com/yourusername/labelme.git
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
cd labelme/tree_ring_methods/cstrd/devernay
./compile.sh
```

### 4. Download DeepCS-TRD Models (Optional)
```bash
cd labelme/tree_ring_methods/deepcstrd
./download_models.sh
```

## Features

### Automatic Detection Methods
- **APD (Automatic Pith Detection)**: Finds tree center using structural tensor analysis (~1 second, CPU)
- **CS-TRD (Classical Tree Ring Detection)**: Edge-based method using Canny detection (~73s on 2400×2400px, CPU)
- **DeepCS-TRD**: Deep learning U-Net architecture with pre-trained models (~101s on 2400×2400px, GPU)

### Image Preprocessing
- Manual crop with margin warnings for edge detection
- Resize (10-100% scaling)
- Background removal using U2Net model
- All preprocessing parameters stored for traceability

### Scale Calibration
- Draw line segment of known length OR direct input
- Units: mm, cm, μm per pixel
- Auto-adjusts when resizing images
- Physical measurements in all exports

### Radial Width Measurement
- Set measurement direction from pith
- Visual cyan line shows transect
- Choose detected or custom pith origin
- Export to .POS format for CooRecorder
- Measures actual ring boundaries (standard dendrochronology method)

### Ring Properties & Export
- Area, perimeter, cumulative area
- Radial width along transect line
- Both pixel and physical units (if scaled)
- CSV export with metadata
- .POS export for CooRecorder software

### Sample Metadata
- Harvested year, sample code, observations
- Automatic year-based ring labeling
- Metadata included in all exports

## Usage

```bash
labelme path/to/image.jpg
```

**Workflow:**
1. **Tools > Sample Metadata** - Set harvested year and sample code
2. **Tools > Set Scale** - Calibrate physical units
3. **Tools > Preprocess Image** - Crop, resize, remove background
4. **Tools > Tree Ring Detection** - APD + CS-TRD or DeepCS-TRD
5. **Tools > Measure Ring Width Along Line** - Set direction and export to .POS
6. **Tools > Ring Properties** - View and export measurements to CSV

## Citations

### APD (Automatic Pith Detection)
Marichal, H., Passarella, D., Randall, G. (2025). Automatic Wood Pith Detector: Local Orientation Estimation and Robust Accumulation. In: Pattern Recognition. ICPR 2024. Lecture Notes in Computer Science, vol 15317. Springer, Cham.

**Link:** https://link.springer.com/chapter/10.1007/978-3-031-78447-7_1

```bibtex
@inproceedings{marichal2025apd,
  title={Automatic Wood Pith Detector: Local Orientation Estimation and Robust Accumulation},
  author={Marichal, Henry and Passarella, Diego and Randall, Gregory},
  booktitle={Pattern Recognition. ICPR 2024},
  series={Lecture Notes in Computer Science},
  volume={15317},
  pages={1--15},
  year={2025},
  publisher={Springer, Cham},
  doi={10.1007/978-3-031-78447-7_1}
}
```

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

- Python 3.8+
- PyTorch (for DeepCS-TRD)
- OpenCV
- Shapely 1.7.0 (required for CS-TRD/DeepCS-TRD)
- Full dependencies in `pyproject.toml`

## License

GPLv3
