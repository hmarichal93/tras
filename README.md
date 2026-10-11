<div align="center">

# 🌲 TRAS — Tree Ring Analyzer Suite

<p align="center">
  <img src="assets/tras-logo.png" alt="TRAS Logo" width="200"/>
</p>

**Open-source dendrochronology software for automatic tree ring detection and measurement**

*Published in [Forestry: An International Journal of Forest Research](https://doi.org/10.1093/forestry/cpag046) 99(4), cpag046 (2026)*

[![Release](https://img.shields.io/github/v/release/hmarichal93/tras?color=green&label=Release)](https://github.com/hmarichal93/tras/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/hmarichal93/tras/total?color=blue&label=Downloads)](https://github.com/hmarichal93/tras/releases)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org)
[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fforestry%2Fcpag046-blue)](https://doi.org/10.1093/forestry/cpag046)
[![arXiv](https://img.shields.io/badge/arXiv-2605.08025-b31b1b.svg)](https://arxiv.org/abs/2605.08025)

[Overview](#-overview) • [Features](#-key-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Increment Cores](#-increment-cores) • [Accuracy](#-accuracy) • [Workflow](#-workflow) • [Citations](#-citations)

</div>

---

## 🎯 Overview

TRAS integrates published **computer vision** and **deep learning** methods (APD, CS-TRD, DeepCS-TRD, INBD; see [Citations](#-citations)) for dendrochronology research. Automatically detect tree rings on **wood cross-sections** and **increment cores**, measure ring widths, check the dating, and export data in standard formats (CooRecorder `.pos`, Tucson `.rwl`, CSV, PDF) — through a graphical interface, a Python API, or command-line tools for batch jobs.

<p align="center">
  <img src="assets/screenshot-main.png" alt="TRAS Main Interface" width="800"/>
</p>

**Ways to use TRAS:**

| Interface | Command | Best for |
|-----------|---------|----------|
| 🖥️ **GUI** | `tras` | Interactive annotation, single samples, batch folders |
| ⌨️ **CLI — cross-sections** | `tras_detect` | Scripting, reproducible batch pipelines |
| ⌨️ **CLI — increment cores** | `tras_core` | CooRecorder-style automatic core measurement |
| 📈 **Benchmarks** | `tras_benchmark` | Reproducing accuracy numbers on annotated data |
| 🐍 **Python API** | `from tras.api import detect`, `from tras.cores import analyze_core_image` | Custom workflows and integration |

---

## ✨ Key Features

<table>
<tr>
<td width="50%" valign="top">

### 🔬 Automatic Detection
- **APD** — Automatic Pith Detection (~1 s, CPU, estimate); methods `apd`, `apd_pcl`, `apd_dl` (YOLO)
- **CS-TRD** — Classical edge detection (131 s per disc, CPU) *[Linux/macOS only]*
- **DeepCS-TRD** — Deep learning U-Net (45 s per disc, GTX 1650 GPU)
- **INBD** — Iterative Next Boundary Detection (CVPR 2023, GPU)
- Timings: [docs/benchmarks.md](docs/benchmarks.md) (2026-10-10)

<img src="assets/detection-methods.png" alt="Detection Methods" width="100%"/>

</td>
<td width="50%" valign="top">

### 🖼️ Preprocessing
- Smart crop with edge warnings
- Resize (10–100 % scaling)
- U2Net background removal
- Full parameter tracking

<img src="assets/preprocessing.png" alt="Preprocessing" width="100%"/>

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📏 Scale Calibration
- Draw a known-length segment
- Direct μm/cm/mm input
- Auto-adjust on resize
- Physical-unit exports

<img src="assets/scale-calibration.png" alt="Scale Calibration" width="100%"/>

</td>
<td width="50%" valign="top">

### 📊 Analysis & Export
- Ring properties: area, perimeter, mean/min/max width over 360 directions, eccentricity
- Exclusion zones (cracks, knots) subtracted from areas
- Quality-control flags for rings worth reviewing
- Crossdating statistics: t-value, Gleichläufigkeit, COFECHA-style segment checks
- JSON, CSV, PDF, CooRecorder `.pos` and Tucson `.rwl`

<img src="assets/analysis-export.png" alt="Analysis" width="100%"/>

</td>
</tr>
</table>

### 📄 PDF Reports

Generate comprehensive reports with a single click:

- Sample metadata and summary statistics
- Ring overlay with detected boundaries
- Multi-panel analysis plots (area, growth rate, radial width)
- One file per sample for sharing or archiving

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

---

## 📦 Installation

> **Windows users:** run every command below from the **Anaconda Prompt** (search "Anaconda Prompt" in the Start menu), **not** the regular Command Prompt. CS-TRD is unavailable on Windows — use DeepCS-TRD instead.

### Prerequisites

- [Anaconda / Miniconda](https://www.anaconda.com/download) — [Windows video tutorial](https://youtu.be/4DQGBQMvwZo)
- Python ≥ 3.9 (provided by the conda environment)

### 1. Download TRAS

```bash
git clone https://github.com/hmarichal93/tras.git
cd tras
```

> **Note:** increment cores (`tras_core`), `tras_benchmark` and Tucson `.rwl` export are not in a
> release yet (latest release: v2.2.0). Until the next release, install from the repository as above.
>
> **Alternative after the next release:** download the source archive from
> [Releases](https://github.com/hmarichal93/tras/releases/latest), extract it, then `cd` into the
> extracted folder.

### 2. Create the Environment

```bash
conda env create -f environment.yml
conda activate tras
pip install -e .
```

### 3. Download Model Weights

```bash
python tools/download_release_assets.py
```

> Fetches all required model weights, including the APD YOLO weights (`downloaded_assets/apd/yolo/all_best_yolov8.pt`) used by the `apd_dl` method.

### 4. Compile the Devernay Edge Detector *(CS-TRD — Linux/macOS only, optional)*

```bash
cd tras/tree_ring_methods/cstrd/devernay
make
cd ../../../..                      # back to the project root
```

### 5. Install the INBD Method *(optional)*

```bash
mkdir -p tras/tree_ring_methods/inbd && cd tras/tree_ring_methods/inbd
git clone https://github.com/hmarichal93/INBD.git src
./download_models.sh
cd ../../..                         # back to the project root
```

---

## 🚀 Quick Start

### 🖥️ GUI

```bash
tras                               # launch the GUI
tras /path/to/image.jpg            # launch with an image open
tras --version                     # print the version
```

The **Tools** menu has three entries:

- **Cross-section** — the step-by-step workflow for an open disc image (set scale → preprocess → detect → measure → export).
- **Increment Core** — measure cores (set scale and metadata → optional core path → Core Analysis → correct boundaries → Export Core Measurements to `.pos`/`.rwl`/`.csv`).
- **Batch Processing (Cross-sections)…** — run detection over an entire folder (see below).

### 📁 GUI Batch Processing

**`Tools → Batch Processing (Cross-sections)…`** opens a dialog where you pick an input folder
and configure the detection settings inline — optional physical scale, preprocessing, pith method, and ring method
with its parameters. Detection runs in the background with a live progress bar.

For every image, TRAS writes a `.json` label file. When the batch finishes it also produces:

| Output | Description |
|--------|-------------|
| `summary.pdf` | One page per image — detected rings overlaid on the image, with the filename as the page title |
| `batch_config.yml` | The exact settings used, re-loadable by the CLI (`tras_detect --config`) |
| `radii.rwl` | Tucson ring widths along four compass radii (N/E/S/W) per image; only when you set *Outer year* and a scale |
| `radii_ids.csv` | Maps series IDs back to image names; only when two image names share their first 7 characters |

> 💡 All outputs are written **into the input folder itself**, alongside the images. Images that
> already have a `.json` are skipped, so an interrupted batch resumes simply by running it again.

### ⌨️ CLI

`tras_detect` automatically detects whether the input is a single image or a folder.

**Single image:**
```bash
# Auto-detect pith and rings
tras_detect image.jpg -o output.json

# With custom parameters
tras_detect image.jpg --ring-method cstrd --pith-method apd
```

**Batch folder:**
```bash
# With a YAML config file (recommended)
tras_detect /path/to/images --config config.yml

# With CLI flags only
tras_detect /path/to/images --scale-value 0.0213 --scale-unit mm --ring-method deepcstrd

# Config file overridden by CLI flags
tras_detect /path/to/images --config config.yml --scale-value 0.025 --ring-method cstrd
```

**Outputs:**
- **Single image** → `<stem>_detected.json` (or your custom `-o` path)
- **Batch folder** → `tras_out/<stem>.json`, `<stem>.csv`, and `<stem>.pdf` per image, plus a
  combined `summary.pdf` and the `batch_config.yml` actually used. With `--outer-year` (or
  `dating.outer_year`) and a scale it also writes `radii.rwl` (and `radii_ids.csv` when image
  names collide)

> A physical scale (config file or `--scale-value`/`--scale-unit`) gives CSV and PDF values in
> physical units; without it they are in pixels. Defaults for every parameter come from
> [`tras/config/detection_defaults.yaml`](tras/config/detection_defaults.yaml), shared with the GUI.

See [`examples/cli/process_config.yml`](examples/cli/process_config.yml) for a configuration template
(every key and its default: [`tras/config/detection_defaults.yaml`](tras/config/detection_defaults.yaml)).

---

## 🪵 Increment Cores

TRAS measures increment cores the way CooRecorder does, but automatically: it finds the cores
in a scan, detects every ring boundary along each core, measures widths perpendicular to the
rings, dates them from the outermost year and writes CooRecorder `.pos` and Tucson `.rwl` files.

```bash
tras_core scan.tif --outer-year 2023              # find cores automatically
tras_core disc.jpg --path "120,40;3900,85"         # measure along a given path
```

In the GUI (all under **Tools → Increment Core**):

1. Open the image, then **Set Image Scale** and **Sample Metadata**.
2. Optional: **Draw Core Path** from pith to bark.
3. **Core Analysis…** → set *Outermost complete ring year* → **Detect rings** → **Add to image**.
4. Correct: drag or delete boundary lines; add missed ones with **Draw Core Boundary** (Ctrl+B).
5. **Export Core Measurements…** writes `.pos`/`.rwl`/`.csv` from the corrected image.

Full guide: **[docs/cores.md](docs/cores.md)**.

---

## 🎯 Accuracy

Measured with `tras_benchmark` on annotated data held out from parameter tuning. Full protocol,
datasets and caveats: **[docs/benchmarks.md](docs/benchmarks.md)**.

| Task | Data | Precision | Recall | F1 |
|------|------|-----------|--------|----|
| Increment cores (deep evidence) | 16 expert CooRecorder radii | 0.965 | 0.896 | 0.929 |
| Increment cores (deep evidence) | 256 virtual cores (UruDendro test split) | 0.958 | 0.849 | 0.900 |
| Cross-sections (CS-TRD, CPU only) | 32 UruDendro test discs | 0.938 | 0.872 | 0.904 |
| Cross-sections (DeepCS-TRD) | 32 UruDendro test discs | 0.993 | 0.905 | 0.947 |

Core ring widths differ from the CooRecorder expert by a median of 3.3 %.

The released DeepCS-TRD weights were trained on the DeepCS-TRD train/validation discs, not
its test set. Our UruDendro test split is a different partition, so some of its discs may be
in that training data and the deep-model UruDendro rows may be optimistic. The CooRecorder
radii are the independent check ([caveats](docs/benchmarks.md#datasets)).

The TRAS article ([Marichal et al. 2026, *Forestry*](https://doi.org/10.1093/forestry/cpag046))
reports DeepCS-TRD F = 81.0 % on a different set (18 *P. taeda* discs, automatic pith,
stricter matching rule); it is a separate result, not comparable with this table. Why the numbers differ:
[docs/benchmarks.md](docs/benchmarks.md#tras-article-results-and-why-they-differ).

---

## 📖 Workflow

<details>
<summary><b>📋 Step-by-step guide (single image)</b></summary>

<br/>

> These per-image tools live under the **`Tools → Cross-section`** submenu. To process a whole folder at
> once, use **`Tools → Batch Processing (Cross-sections)…`** instead (see [GUI Batch Processing](#-gui-batch-processing)).

1. **📁 Load Image** — `File → Open` or drag-and-drop a wood cross-section image
2. **📏 Set Scale** *(optional)* — `Tools → Cross-section → Set Image Scale`; draw a known-length line or enter it directly
3. **🖼️ Preprocess** *(optional)* — `Tools → Cross-section → Preprocess Image`; crop, resize, or remove background
4. **🎯 Detect Rings** — `Tools → Cross-section → Detect Tree Rings`
   - Pith: APD auto (`apd`, `apd_pcl`, `apd_dl`) or manual click
   - Method: CS-TRD (Linux/macOS), DeepCS-TRD, or INBD
5. **📝 Add Metadata** — `Tools → Cross-section → Sample Metadata`; harvested year, sample code, notes
6. **📐 Measure Width** *(optional)* — `Tools → Cross-section → Measure Ring Width`; define a radial transect
7. **📊 View Properties** — `Tools → Cross-section → View Ring Properties`; review measurements
8. **💾 Export** — `Tools → Cross-section → Export Data`: JSON, CSV (all data), .POS (CooRecorder), Tucson .rwl (N/E/S/W radii), PDF report

</details>

---

## 🧠 Training Custom Models

TRAS supports loading user-trained models for **INBD** and **DeepCS-TRD** to improve detection on
your specific datasets. See the **[Training Custom Models guide](docs/training_custom_models.md)**
for step-by-step instructions.

---

## ⚙️ Configuration

TRAS stores its configuration in `~/.trasrc`. You can customize:

- Default shape colors (wood theme)
- Keyboard shortcuts
- UI preferences

Detection defaults live in [`tras/config/detection_defaults.yaml`](tras/config/detection_defaults.yaml)
(shared by the GUI, CLI and API), not in `~/.trasrc`.

---

## 📋 Requirements

- Python ≥ 3.9
- PyQt5
- OpenCV
- PyTorch (for DeepCS-TRD)
- Shapely 1.7.0

See [`pyproject.toml`](pyproject.toml) for the complete dependency list.

---

## 📚 Citations

If you use TRAS in your research, please cite the TRAS article and the papers of the methods
you used. The same metadata is in [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this
repository" button).

**Cite TRAS**
```bibtex
@article{marichal2026tras,
  author  = {Marichal, Henry and Passarella, Diego and Randall, Gregory},
  title   = {{TRAS}: an interactive software for tracing tree ring cross sections},
  journal = {Forestry: An International Journal of Forest Research},
  volume  = {99},
  number  = {4},
  pages   = {cpag046},
  year    = {2026},
  doi     = {10.1093/forestry/cpag046},
  url     = {https://doi.org/10.1093/forestry/cpag046}
}
```

The preprint is [arXiv:2605.08025](https://arxiv.org/abs/2605.08025); cite the journal
version above.

<details>
<summary><b>📖 BibTeX entries for the methods, dataset and tools</b></summary>

<br/>

**UruDendro** (dataset used for benchmarks and model training)
```bibtex
@article{UruDendro,
  author    = {Henry Marichal and Diego Passarella and Christine Lucas and Ludmila Profumo and Verónica Casaravilla and María Noel Rocha Galli and Serrana Ambite and Gregory Randall},
  title     = {UruDendro, a public dataset of 64 cross-section images and manual annual ring delineations of \textit{Pinus taeda L.}},
  journal   = {Annals of Forest Science},
  volume    = {82},
  number    = {1},
  pages     = {25},
  year      = {2025},
  issn      = {1297-966X},
  doi       = {10.1186/s13595-025-01296-5},
  url       = {https://doi.org/10.1186/s13595-025-01296-5}
}
```

**APD — Automatic Pith Detection**
```bibtex
@inproceedings{apd,
  author        = {Marichal, Henry and Passarella, Diego and Randall, Gregory},
  title         = {Automatic Wood Pith Detector: Local Orientation Estimation and Robust Accumulation},
  booktitle     = {Pattern Recognition (International Conference on Pattern Recognition, ICPR)},
  series        = {Lecture Notes in Computer Science},
  publisher     = {Springer Nature Switzerland},
  address       = {Cham},
  pages         = {1--15},
  year          = {2025},
  isbn          = {978-3-031-78447-7},
  doi           = {10.1007/978-3-031-78447-7_1},
  eprint        = {2404.01952},
  archivePrefix = {arXiv}
}
```

**CS-TRD — Cross-Section Tree Ring Detection**
```bibtex
@article{ipol.2025.485,
  author  = {Marichal, Henry and Passarella, Diego and Randall, Gregory},
  title   = {{CS-TRD}: a Cross-Section Tree Ring Detection Method},
  journal = {Image Processing On Line},
  volume  = {15},
  pages   = {78--107},
  year    = {2025},
  doi     = {10.5201/ipol.2025.485},
  url     = {https://doi.org/10.5201/ipol.2025.485}
}
```

**DeepCS-TRD — Deep Learning Tree Ring Detection**
```bibtex
@inproceedings{deepcstrd,
  author        = {Marichal, Henry and Casaravilla, Ver{\'o}nica and Power, Candice and Mello, Karolain and Mazarino, Joaqu{\'i}n and Lucas, Christine and Profumo, Ludmila and Passarella, Diego and Randall, Gregory},
  editor        = {Rodol{\`a}, Emanuele and Galasso, Fabio and Masi, Iacopo},
  title         = {{DeepCS-TRD}, a Deep Learning-Based Cross-Section Tree Ring Detector},
  booktitle     = {Image Analysis and Processing -- ICIAP 2025},
  series        = {Lecture Notes in Computer Science},
  publisher     = {Springer Nature Switzerland},
  address       = {Cham},
  pages         = {29--41},
  year          = {2026},
  isbn          = {978-3-032-10185-3},
  doi           = {10.1007/978-3-032-10185-3_3},
  eprint        = {2504.16242},
  archivePrefix = {arXiv}
}
```

**INBD — Iterative Next Boundary Detection**
```bibtex
@inproceedings{Gillert_2023_CVPR,
  author    = {Gillert, Alexander and Resente, Giulia and Anadon-Rosell, Alba and Wilmking, Martin and von Lukas, Uwe Freiherr},
  title     = {Iterative Next Boundary Detection for Instance Segmentation of Tree Rings in Microscopy Images of Shrub Cross Sections},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
  pages     = {14540--14548},
  doi       = {10.1109/CVPR52729.2023.01397}
}
```

**U²-Net — background removal** (Preprocess Image → Remove background)
```bibtex
@article{qin2020u2net,
  author  = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar R. and Jagersand, Martin},
  title   = {{U}$^2$-{Net}: Going deeper with nested {U}-structure for salient object detection},
  journal = {Pattern Recognition},
  volume  = {106},
  pages   = {107404},
  year    = {2020},
  doi     = {10.1016/j.patcog.2020.107404}
}
```

**CooRecorder** (`.pos` format; the expert reference in the core benchmark)
```bibtex
@article{maxwell2021coorecorder,
  author  = {Maxwell, R. Stockton and Larsson, Lars-{\AA}ke},
  title   = {Measuring tree-ring widths using the {CooRecorder} software application},
  journal = {Dendrochronologia},
  volume  = {67},
  pages   = {125841},
  year    = {2021},
  doi     = {10.1016/j.dendro.2021.125841}
}
```

Crossdating statistics (COFECHA-style segments, Gleichläufigkeit) follow Holmes (1983) and
Eckstein & Bauch (1969); references in
[docs/research/core-detection-sota.md](docs/research/core-detection-sota.md#references).

</details>

---

## 🤝 Contributing

Contributions are welcome! Feel free to submit a Pull Request. For major changes, please open an
issue first to discuss what you would like to change.

- Code map and design rules: [docs/architecture.md](docs/architecture.md)
- Before a PR: `make test` and `make lint_strict` (both run in CI)

## 📄 License

[GPL-3.0-only](LICENSE)

## 📧 Contact

- **Author:** Henry Marichal ([@hmarichal93](https://github.com/hmarichal93))
- **Email:** hmarichal93@gmail.com
- **Website:** [https://hmarichal93.github.io/tras/](https://hmarichal93.github.io/tras/)

## 🙏 Acknowledgments

- Based on the original [LabelMe](https://github.com/wkentaro/labelme) annotation tool by Kentaro Wada
- U2Net model for background removal from [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)

---

<div align="center">

**TRAS** — Tree Ring Analyzer Suite

*Automatic tree ring detection, manual correction and measurement*

[📖 Documentation](https://hmarichal93.github.io/tras/) • [🐛 Report Bug](https://github.com/hmarichal93/tras/issues) • [💡 Request Feature](https://github.com/hmarichal93/tras/issues)

Made with 🌲 for the dendrochronology community

</div>
