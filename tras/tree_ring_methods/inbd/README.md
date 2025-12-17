# INBD Integration

This directory contains the integration of INBD (Iterative Next Boundary Detection) for tree ring detection.

## Setup

To use INBD, you need to:

1. Clone the INBD repository into the `src` directory:
```bash
cd tras/tree_ring_methods/inbd
git clone https://github.com/hmarichal93/INBD.git src
```

2. Download pre-trained models:
```bash
cd src
python fetch_pretrained_models.py
```

3. Install INBD dependencies (if not already installed):
```bash
pip install -r src/requirements.txt
```

## Models

INBD provides pre-trained models for:
- **EH**: Empetrum hermaphroditum (shrub species)
- **DO**: Dryas octopetala (shrub species)
- **VM**: Vaccinium myrtillus (shrub species)
- **UruDendro**: Pinus taeda (tree species)

Models are stored in `src/checkpoints/` after download.

## Usage

INBD is integrated into TRAS through the Tree Ring Detection dialog:
- Tools > Tree Ring Detection
- Expand "INBD Model & Parameters"
- Click "Detect with INBD"

## Reference

Gillert, A., Resente, G., Anadon-Rosell, A., Wilmking, M., & von Lukas, U. (2023).
Iterative Next Boundary Detection for Instance Segmentation of Tree Rings in Microscopy Images of Shrub Cross Sections.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 14540-14548).

