# Training Custom Models for TRAS

TRAS supports loading user-trained models for both the **INBD** and **DeepCS-TRD** detection methods.
This is useful when the pre-trained models underperform on your specific image acquisition conditions
(species, preparation method, scanner resolution, etc.).

The general workflow is:

1. Annotate and correct tree rings in TRAS.
2. Export the corrected annotations.
3. Train a new model using the exported annotations.
4. Load the custom model in TRAS.

---

## Step 1 — Export corrected annotations from TRAS

After detecting and manually correcting rings in TRAS, export each image's annotation as a
**LabelMe JSON** file (`File > Save` or `File > Save As`). Each `.json` file embeds the image
and all ring polygons. Collect your images and their corresponding `.json` files into a single
directory, for example:

```
my_dataset/
  image_001.jpg
  image_001.json
  image_002.jpg
  image_002.json
  ...
```

---

## Step 2a — Fine-tuning INBD

INBD training requires the INBD submodule to be installed (see [INBD setup](../tras/tree_ring_methods/inbd/README.md)).

Training is a **two-stage** process run from inside the INBD source directory:

```bash
cd tras/tree_ring_methods/inbd/src
```

### Create image and annotation list files

INBD training expects plain-text files, each containing one file path per line.

```bash
# List all images
ls /path/to/my_dataset/*.jpg > train_images.txt

# List the corresponding annotation JSONs (same order)
ls /path/to/my_dataset/*.json > train_annotations.txt
```

Optionally, create equivalent files for a validation split.

### Stage 1 — Train the segmentation model

```bash
python main.py train segmentation \
    train_images.txt train_annotations.txt \
    --validation_images val_images.txt \
    --validation_annotations val_annotations.txt \
    --epochs 30 \
    --output checkpoints/ \
    --suffix my_dataset
```

The trained model is saved as `checkpoints/segmentation_my_dataset/model.pt.zip`.

### Stage 2 — Train the INBD model

```bash
python main.py train INBD \
    train_images.txt train_annotations.txt \
    --segmentationmodel checkpoints/segmentation_my_dataset/model.pt.zip \
    --epochs 100 \
    --output checkpoints/ \
    --suffix my_dataset
```

The final model is saved as `checkpoints/INBD_my_dataset/model.pt.zip`.

### Load the custom INBD model in TRAS

In the **Tree Ring Detection** dialog, under the **INBD** tab:
- Click **Upload Model** and select the `model.pt.zip` file produced above,
  or type the full path directly in the model field.

---

## Step 2b — Fine-tuning DeepCS-TRD

The DeepCS-TRD U-Net model is trained using the
[DeepCS-TRD training scripts](https://github.com/hmarichal93/DeepCS-TRD).
Follow the instructions in that repository to prepare your dataset and run training.

The training pipeline expects:
- **Images**: pre-segmented (background removed) cross-section images.
- **Annotations**: LabelMe JSON files as exported from TRAS (placed in
  `dataset_dir/annotations/labelme/images/`).

After training, a `.pth` weights file is produced.

### Load the custom DeepCS-TRD model in TRAS

In the **Tree Ring Detection** dialog, under the **DeepCS-TRD** tab:
- Click **Upload Model** and select the `.pth` weights file,
  or type the full path directly in the model field.

---

## Tips

- **Data volume**: 20–50 corrected images are usually sufficient for fine-tuning.
  More images improve robustness, especially across samples with variable contrast.
- **GPU**: Both methods require a CUDA-capable GPU for practical training times.
- **Iterative refinement**: After training, run detection on new samples in TRAS,
  correct residual errors, and re-train to improve performance further.
