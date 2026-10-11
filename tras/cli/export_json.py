import argparse
import os
import os.path as osp

import imgviz
import numpy as np
import numpy.typing as npt
import PIL.Image
from loguru import logger

from tras import utils
from tras._label_file import LabelFile

# Minimum point count per shape type for rasterization (see tras.utils.shape).
_MIN_POINTS = {"polygon": 3, None: 3, "point": 1, "line": 2, "linestrip": 2}
UNLABELED = "unlabeled"


def _rasterizable_shapes(shapes: list[dict]) -> list[dict]:
    """Drop shapes that cannot be drawn and give unlabeled shapes a placeholder label.

    One degenerate shape (e.g. a 2-point polygon) must not abort the whole export.
    """
    kept = []
    for shape in shapes:
        min_points = _MIN_POINTS.get(shape.get("shape_type"), 1)
        if len(shape.get("points", [])) < min_points:
            logger.warning(
                f"Skipping shape '{shape.get('label')}': {len(shape.get('points', []))} "
                f"point(s), needs >= {min_points}"
            )
            continue
        kept.append({**shape, "label": shape.get("label") or UNLABELED})
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    json_file = args.json_file

    if args.out is None:
        out_dir = osp.splitext(osp.basename(json_file))[0]
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = args.out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    label_file: LabelFile = LabelFile(filename=json_file)

    image: npt.NDArray[np.uint8] = utils.img_data_to_arr(label_file.imageData)

    shapes = _rasterizable_shapes(label_file.shapes)
    label_name_to_value = {"_background_": 0}
    for shape in sorted(shapes, key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(image.shape, shapes, label_name_to_value)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name  # type: ignore[call-overload]

    lbl_viz = imgviz.label2rgb(lbl, imgviz.asgray(image), label_names=label_names, loc="rb")

    PIL.Image.fromarray(image).save(osp.join(out_dir, "img.png"))
    utils.lblsave(osp.join(out_dir, "label.png"), lbl)
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, "label_viz.png"))

    with open(osp.join(out_dir, "label_names.txt"), "w") as f:
        for lbl_name in label_names:
            f.write(f"{lbl_name}\n")

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
