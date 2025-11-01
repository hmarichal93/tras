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

### CS-TRD
```bibtex
@article{gonzalez2018automatic,
  title={Automatic tree-ring detection and delineation in microscopic images of wood samples},
  author={Gonzalez-Jorge, H and Sanchez, A and Solla, M and Laguela, S and Diaz-Vilari{\~n}o, L and Riveiro, B},
  journal={Computers and Electronics in Agriculture},
  volume={150},
  pages={173--182},
  year={2018}
}
```

### APD
```bibtex
@article{norell2019automatic,
  title={Automatic pith detection in CT images of wood logs},
  author={Norell, K and Borgefors, G},
  journal={Computers and Electronics in Agriculture},
  volume={157},
  pages={435--443},
  year={2019}
}
```

### DeepCS-TRD
```bibtex
@article{marichal2021deepcstrd,
  title={DeepCS-TRD: Deep learning for tree ring detection},
  author={Marichal, H and Morel, J and Silveira, M and others},
  journal={Dendrochronologia},
  volume={68},
  pages={125847},
  year={2021}
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
