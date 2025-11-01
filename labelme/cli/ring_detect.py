from __future__ import annotations

import argparse
import glob
import os
import os.path as osp

import numpy as np
from loguru import logger

from labelme._automation.tree_rings import RingDetectParams, detect_tree_rings
from labelme._label_file import LabelFile
from labelme import utils


def process_image(image_path: str, out_json: str, args) -> None:
    image = utils.img_data_to_arr(LabelFile.load_image_file(image_path))
    if image.ndim == 2:
        image = np.dstack([image] * 3)
    params = RingDetectParams(
        angular_steps=args.angular_steps,
        min_radius=args.min_radius,
        relative_threshold=args.relative_threshold,
        min_peak_distance=args.min_peak_distance,
        min_coverage=args.min_coverage,
        max_rings=None if args.max_rings <= 0 else args.max_rings,
    )
    rings = detect_tree_rings(image=image, center_xy=(args.center_x, args.center_y), params=params)
    shapes = []
    for i, ring in enumerate(rings, start=1):
        shapes.append(
            dict(
                label=f"ring_{i}",
                points=[[float(x), float(y)] for x, y in ring],
                group_id=None,
                description="",
                shape_type="polygon",
                flags={},
                mask=None,
            )
        )
    lf = LabelFile()
    if not osp.exists(osp.dirname(out_json)):
        os.makedirs(osp.dirname(out_json), exist_ok=True)
    lf.save(
        filename=out_json,
        shapes=shapes,
        imagePath=osp.relpath(image_path, osp.dirname(out_json)),
        imageData=None,
        imageHeight=image.shape[0],
        imageWidth=image.shape[1],
        otherData={},
        flags={},
    )
    logger.info("Saved %s", out_json)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect tree rings and write LabelMe JSON")
    parser.add_argument("inputs", nargs="+", help="Input image files or globs")
    parser.add_argument("--out", required=True, help="Output directory for JSON files")
    parser.add_argument("--center-x", type=float, required=True)
    parser.add_argument("--center-y", type=float, required=True)
    parser.add_argument("--angular-steps", type=int, default=720)
    parser.add_argument("--min-radius", type=float, default=5.0)
    parser.add_argument("--relative-threshold", type=float, default=0.3)
    parser.add_argument("--min-peak-distance", type=int, default=3)
    parser.add_argument("--min-coverage", type=float, default=0.6)
    parser.add_argument("--max-rings", type=int, default=0)
    args = parser.parse_args(argv)

    paths: list[str] = []
    for pat in args.inputs:
        matches = glob.glob(pat)
        if matches:
            paths.extend(matches)
        else:
            paths.append(pat)
    paths = sorted(set(paths))
    os.makedirs(args.out, exist_ok=True)

    for path in paths:
        base = osp.splitext(osp.basename(path))[0]
        out_json = osp.join(args.out, f"{base}.json")
        process_image(path, out_json, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
