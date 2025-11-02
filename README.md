<div align="center">

# 🌲 TRAS - Tree Ring Analyzer Suite

<p align="center">
  <img src="assets/tras-logo.png" alt="TRAS Logo" width="200"/>
</p>

**Professional dendrochronology software for automatic tree ring detection and measurement**

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org)
[![Version](https://img.shields.io/badge/Version-2.0.0-green)](https://github.com/hmarichal93/tras)
[![arXiv](https://img.shields.io/badge/arXiv-2305.10809-b31b1b.svg)]()

</div>

---

## 🎯 Overview

TRAS integrates **state-of-the-art computer vision** and **deep learning methods** for dendrochronology research. Automatically detect tree rings, measure ring widths, and export data in standard formats—all through an intuitive graphical interface.

<p align="center">
  <img src="assets/screenshot-main.png" alt="TRAS Main Interface" width="800"/>
</p>

## Installation

### 1. Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install -y libgeos-dev

# macOS
brew install geos

# Windows: install via conda
conda install -c conda-forge geos
```

### 2. Clone Repository
```bash
git clone -b tras https://github.com/hmarichal93/tras.git
cd tras
```

### 3. Install Python Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

### 4. Compile Devernay Edge Detector (for CS-TRD)
```bash
cd tras/tree_ring_methods/cstrd/devernay
make
```

### 5. Download DeepCS-TRD Models (Optional)
```bash
cd tras/tree_ring_methods/deepcstrd
./download_models.sh
```

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🔬 **Automatic Detection**
- **APD**: Automatic Pith Detection (~1s, CPU)
- **CS-TRD**: Classical edge detection (~73s, CPU)
- **DeepCS-TRD**: Deep learning U-Net (~101s, GPU)

<img src="assets/detection-methods.png" alt="Detection Methods" width="100%"/>

</td>
<td width="50%">

### 🖼️ **Preprocessing**
- Smart crop with edge warnings
- Resize (10-100% scaling)
- U2Net background removal
- Full parameter tracking

<img src="assets/preprocessing.png" alt="Preprocessing" width="100%"/>

</td>
</tr>
<tr>
<td width="50%">

### 📏 **Scale Calibration**
- Draw known-length segment
- Direct μm/cm/mm input
- Auto-adjust on resize
- Physical unit exports

<img src="assets/scale-calibration.png" alt="Scale Calibration" width="100%"/>

</td>
<td width="50%">

### 📊 **Analysis & Export**
- Ring properties (area, perimeter)
- Radial width measurements
- Year-based labeling
- CSV & .POS formats

<img src="assets/analysis-export.png" alt="Analysis" width="100%"/>

</td>
</tr>
</table>


## 🚀 Quick Start

```bash
tras                              # Launch GUI
tras /path/to/image.jpg          # Open with image
tras --version                    # Check version
```

## 📖 Workflow

<p align="center">
  <img src="assets/workflow-diagram.png" alt="TRAS Workflow" width="700"/>
</p>

<details>
<summary><b>📋 Step-by-step guide</b></summary>

1. **📁 Load Image**
   - `File > Open` or drag-and-drop wood cross-section image

2. **📏 Set Scale** *(optional)*
   - `Tools > Set Image Scale`
   - Draw known-length line or enter directly

3. **🖼️ Preprocess** *(optional)*
   - `Tools > Preprocess Image`
   - Crop, resize, or remove background

4. **🎯 Detect Rings**
   - `Tools > Tree Ring Detection`
   - Choose pith: APD (auto) or manual click
   - Select method: CS-TRD or DeepCS-TRD

5. **📝 Add Metadata**
   - `Tools > Sample Metadata`
   - Harvested year, sample code, notes

6. **📐 Measure Width** *(optional)*
   - `Tools > Measure Ring Width`
   - Define radial transect

7. **📊 View Properties**
   - `Tools > Ring Properties`
   - Review measurements

8. **💾 Export**
   - CSV (all data) or .POS (CooRecorder format)

</details>

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

- **Author**: Henry Marichal ([@hmarichal93](https://github.com/hmarichal93))
- **Email**: hmarichal93@gmail.com
- **Website**: [https://hmarichal93.github.io/tras/](https://hmarichal93.github.io/tras/)

## 🙏 Acknowledgments

- Based on the original [LabelMe](https://github.com/wkentaro/labelme) annotation tool by Kentaro Wada
- Integrates methods from the [TRAS research repository](https://github.com/hmarichal93/tras)
- U2Net model for background removal from [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)

---

<div align="center">

**TRAS v2.0.0** - Tree Ring Analyzer Suite

*Advancing dendrochronology research through intelligent automation*

[📖 Documentation](https://hmarichal93.github.io/tras/) • [🐛 Report Bug](https://github.com/hmarichal93/tras/issues) • [💡 Request Feature](https://github.com/hmarichal93/tras/issues)

Made with 🌲 for the dendrochronology community

</div>
