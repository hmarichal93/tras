# DeepCS-TRD Model Files

This directory should contain pre-trained model files for DeepCS-TRD.

## Download Models

Run the download script:
```bash
cd labelme/tree_ring_methods/deepcstrd
./download_models.sh
```

Or manually download from:
https://github.com/hmarichal93/deepcstrd/tree/main/models/deep_cstrd

## Available Models

Place the following `.pth` files in this directory:

- `0_all_1504.pth` - Generic model (recommended for most species)
- `0_pinus_v1_1504.pth` - Pinus species (version 1)
- `0_pinus_v2_1504.pth` - Pinus species (version 2)
- `0_gleditsia_1504.pth` - Gleditsia species
- `0_salix_1504.pth` - Salix Glauca species
- `256_*.pth` - Tiled versions for large images

Total size: ~550MB

## Model Architecture

All models use a U-Net architecture trained on labeled wood cross-section datasets.
- Input: Polar-transformed wood cross-section images
- Output: Binary segmentation map of ring boundaries
- Training details: See https://github.com/hmarichal93/deepcstrd

