"""Loaders for annotated datasets used by the benchmarks.

* **UruDendro** (Marichal et al.): cross-section images, labelme ring annotations and a pith
  CSV. Note: the CSV header says ``Image,cy,cx`` but the columns hold ``x`` then ``y``
  (checked: only this reading puts the pith inside the innermost ring for all 64 images).
* **CooRecorder pairs**: an image and ``.pos`` files measured on it (pith first).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tras.analysis.rings import order_rings_by_area
from tras.io.pos import RING
from tras.io.pos import read_pos

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass(frozen=True)
class AnnotatedSection:
    """A cross-section image with ground-truth rings (innermost first) and pith."""

    name: str
    image_path: Path
    rings: tuple[npt.NDArray[np.float64], ...]
    pith_xy: tuple[float, float]


@dataclass(frozen=True)
class AnnotatedPath:
    """A measurement path on an image and the true boundary positions along it."""

    name: str
    image_path: Path
    path_xy: npt.NDArray[np.float64]  # (N, 2) px, pith end first
    boundaries_px: tuple[float, ...]  # arc lengths from the path start
    mm_per_px: float | None = None


def load_urudendro(
    root: Path | str,
    images_dir: str = "images_no_background/images",
    annotations_dir: str = "ring_annotations/ring_annotations",
    pith_csv: str = "pith_location.csv",
) -> list[AnnotatedSection]:
    """Load every UruDendro sample that has an image, an annotation and a pith."""
    root = Path(root)
    sections = []
    with open(root / pith_csv, newline="") as f:
        for row in csv.DictReader(f):
            name = row["Image"]
            image = _find_image(root / images_dir, name)
            annotation = root / annotations_dir / f"{name}.json"
            if image is None or not annotation.exists():
                continue
            sections.append(
                AnnotatedSection(
                    name=name,
                    image_path=image,
                    rings=_load_rings(annotation),
                    pith_xy=(float(row["cy"]), float(row["cx"])),  # columns are x, y
                )
            )
    return sorted(sections, key=lambda s: s.name)


def load_coorecorder_pairs(pos_dir: Path | str, images_dir: Path | str) -> list[AnnotatedPath]:
    """Pair each ``<disc>_<direction>.pos`` with the clean image ``<disc>.<ext>``.

    Points are assumed pith-first (as in the TRAS ForestScience comparison data): the
    first point is the pith, the others are ring boundaries.
    """
    pairs = []
    for pos_path in sorted(Path(pos_dir).glob("*.pos")):
        disc = pos_path.stem.split("_")[0]
        image = _find_image(Path(images_dir), disc)
        pos = read_pos(pos_path)
        if image is None or not pos.dpi:
            continue
        px_per_mm = pos.dpi / 25.4
        points = np.array([(p.x, p.y) for p in pos.points if p.kind == RING]) * px_per_mm
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        pairs.append(
            AnnotatedPath(
                name=pos_path.stem,
                image_path=image,
                path_xy=points,
                boundaries_px=tuple(float(a) for a in arc[1:]),
                mm_per_px=1.0 / px_per_mm,
            )
        )
    return pairs


def _find_image(folder: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _load_rings(annotation: Path) -> tuple[npt.NDArray[np.float64], ...]:
    shapes = json.loads(annotation.read_text())["shapes"]
    rings = [np.asarray(s["points"], dtype=np.float64) for s in shapes if len(s["points"]) >= 3]
    return tuple(rings[i] for i in order_rings_by_area(rings))
