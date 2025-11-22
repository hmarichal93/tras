<div align="center">

# 🌲 TRAS - Tree Ring Analyzer Suite

<p align="center">
  <img src="assets/tras-logo.png" alt="TRAS Logo" width="200"/>
</p>

**Professional dendrochronology software for automatic tree ring detection and measurement**

[![Release](https://img.shields.io/github/v/release/hmarichal93/tras?color=green&label=Release)](https://github.com/hmarichal93/tras/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/hmarichal93/tras/total?color=blue&label=Downloads)](https://github.com/hmarichal93/tras/releases)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org)
[![arXiv](https://img.shields.io/badge/arXiv-2305.10809-b31b1b.svg)](https://arxiv.org/abs/2305.10809)

</div>

---

## 🎯 Overview

TRAS integrates **state-of-the-art computer vision** and **deep learning methods** for dendrochronology research. Automatically detect tree rings, measure ring widths, and export data in standard formats—all through an intuitive graphical interface.

<p align="center">
  <img src="assets/screenshot-main.png" alt="TRAS Main Interface" width="800"/>
</p>

## Installation
### 1. Python
#### Conda
##### 1. Download Anaconda
> **Windows users**: [Video tutorial for installing Anaconda on Windows](https://youtu.be/4DQGBQMvwZo)

##### 2. Download TRAS

**Linux/macOS:**
```bash
export TRAS_VERSION=2.0.2  # Change this value when updating to a newer release
wget https://github.com/hmarichal93/tras/archive/refs/tags/v${TRAS_VERSION}.tar.gz
tar -xzf v${TRAS_VERSION}.tar.gz
cd tras-${TRAS_VERSION}
```

> **Alternative**: Download manually from [Releases](https://github.com/hmarichal93/tras/releases/latest) and extract

> **🪟 Important for Windows Users:**
> After downloading and extracting TRAS:
> 1. Open **Anaconda Prompt** (not regular Command Prompt - search for "Anaconda Prompt" in the Start menu)
> 2. Navigate to the extracted directory: `cd path\to\tras-$version`
> 3. Then follow the installation commands below

##### 3. Install
```bash
conda env create -f environment.yml
conda activate tras
pip install -e . 
python tools/download_release_assets.py 
```

### 2. Compile Devernay Edge Detector (for CS-TRD, Linux/macOS only)
```bash
# From the tras-${TRAS_VERSION} directory
cd tras/tree_ring_methods/cstrd/devernay
make
cd ../../../..  # Return to tras-${TRAS_VERSION} directory
```
> **Note**: CS-TRD is not available on Windows. Windows users should use DeepCS-TRD instead.

## ✨ Key Features

<table>
<tr>
<td width="50%" valign="top">

### 🔬 **Automatic Detection**
- **APD**: Automatic Pith Detection (~1s, CPU)
- **CS-TRD**: Classical edge detection (~73s, CPU) *[Linux/macOS only]*
- **DeepCS-TRD**: Deep learning U-Net (~101s, GPU)

<img src="assets/detection-methods.png" alt="Detection Methods" width="100%"/>

</td>
<td width="50%" valign="top">

### 🖼️ **Preprocessing**
- Smart crop with edge warnings
- Resize (10-100% scaling)
- U2Net background removal
- Full parameter tracking

<img src="assets/preprocessing.png" alt="Preprocessing" width="100%"/>

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📏 **Scale Calibration**
- Draw known-length segment
- Direct μm/cm/mm input
- Auto-adjust on resize
- Physical unit exports

<img src="assets/scale-calibration.png" alt="Scale Calibration" width="100%"/>

</td>
<td width="50%" valign="top">

### 📊 **Analysis & Export**
- Ring properties (area, perimeter)
- Radial width measurements
- Year-based labeling
- CSV & .POS formats

<img src="assets/analysis-export.png" alt="Analysis" width="100%"/>

</td>
</tr>
</table>

### 📄 **Professional PDF Reports**

Generate comprehensive reports with a single click, including:
- Sample metadata and summary statistics
- Ring overlay with detected boundaries
- Multi-panel analysis plots (area, growth rate, radial width)
- Ready for publication or archival

<p align="center">
<table>
<tr>
<td width="33%" align="center">
<img src="assets/report_page1.png" alt="Report Page 1" width="100%"/>
<br/><b>Cover Page</b>
</td>
<td width="33%" align="center">
<img src="assets/report_page2.png" alt="Report Page 2" width="100%"/>
<br/><b>Ring Overlay</b>
</td>
<td width="33%" align="center">
<img src="assets/report_page3.png" alt="Report Page 3" width="100%"/>
<br/><b>Analysis Plots</b>
</td>
</tr>
</table>
</p>


## 🚀 Quick Start

```bash
tras                              # Launch GUI
tras /path/to/image.jpg          # Open with image
tras --version                    # Check version
```

## 📖 Workflow


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
   - Select method: CS-TRD (Linux/macOS) or DeepCS-TRD (all platforms)

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
```bibtex
@inproceedings{apd,
  isbn = {978-3-031-78447-7},
  year = {2025},
  pages = {1--15},
  title = {Automatic Wood Pith Detector: Local Orientation Estimation and Robust Accumulation},
  author = {Marichal, Henry and Passarella, Diego and Randall, Gregory},
  booktitle = {International Conference on Pattern Recognition (ICPR)}
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
- U2Net model for background removal from [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)

---

<div align="center">

**TRAS v2.0.2** - Tree Ring Analyzer Suite

*Advancing dendrochronology research through intelligent automation*

[📖 Documentation](https://hmarichal93.github.io/tras/) • [🐛 Report Bug](https://github.com/hmarichal93/tras/issues) • [💡 Request Feature](https://github.com/hmarichal93/tras/issues)

Made with 🌲 for the dendrochronology community

</div>
