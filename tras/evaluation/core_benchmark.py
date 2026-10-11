"""Benchmark the core detector on paths with known boundary positions.

Two sources of cases:

* **Virtual cores**: rays from the pith of annotated cross-sections (UruDendro). Ground
  truth = where the annotated rings cross the ray. No public annotated core dataset
  exists (2026-10), so this is the main quantitative test.
* **CooRecorder paths**: real expert measurements (``.pos``) on real images.

Each case is run twice: with the pith end given (as when the user draws the path from
the pith) and with ``pith_side="auto"`` to measure orientation accuracy.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image

from tras.analysis.rings import radial_profiles
from tras.cores.detect import CoreParams
from tras.cores.detect import detect_core_rings
from tras.evaluation.datasets import AnnotatedPath
from tras.evaluation.datasets import AnnotatedSection
from tras.evaluation.path import match_boundaries
from tras.evaluation.path import score_boundaries

DEFAULT_TOLERANCE_PX = 4.0
# Path extends this far past the outermost ring (px), like a core reaching the bark.
VIRTUAL_CORE_MARGIN_PX = 15.0


@dataclass(frozen=True)
class CaseResult:
    name: str
    n_true: int
    n_detected: int
    tp: int
    mae_px: float
    orientation_ok: bool
    ring_count_error: int  # detected − true
    width_errors: tuple[float, ...] = ()  # |Δwidth| / true width, matched consecutive rings


@dataclass(frozen=True)
class BenchmarkSummary:
    cases: int
    precision: float
    recall: float
    f1: float
    mae_px: float
    orientation_accuracy: float
    mean_abs_ring_count_error: float
    width_mdape: float  # median absolute % error of matched ring widths

    def format(self) -> str:
        return (
            f"cases={self.cases}  P={self.precision:.3f}  R={self.recall:.3f}  "
            f"F1={self.f1:.3f}  MAE={self.mae_px:.2f}px  width-MdAPE={self.width_mdape:.1%}  "
            f"orientation={self.orientation_accuracy:.1%}  "
            f"|Δrings|={self.mean_abs_ring_count_error:.2f}"
        )


def virtual_cores(section: AnnotatedSection, n_directions: int) -> list[AnnotatedPath]:
    """Straight pith→bark paths at ``n_directions`` evenly spaced angles."""
    profiles = radial_profiles(section.rings, section.pith_xy, 360)
    cases = []
    for k in range(n_directions):
        ray = int(round(k * 360 / n_directions)) % 360
        radii = profiles[:, ray]
        length = float(radii.max()) + VIRTUAL_CORE_MARGIN_PX
        theta = np.deg2rad(ray)
        start = np.asarray(section.pith_xy, dtype=np.float64)
        end = start + length * np.array([np.cos(theta), np.sin(theta)])
        cases.append(
            AnnotatedPath(
                name=f"{section.name}@{ray:03d}",
                image_path=section.image_path,
                path_xy=np.stack([start, end]),
                boundaries_px=tuple(float(r) for r in np.sort(radii)),
            )
        )
    return cases


def evaluate_path(
    image: np.ndarray, case: AnnotatedPath, params: CoreParams, tolerance_px: float
) -> CaseResult:
    """Detect with the pith side given, then check automatic orientation."""
    given = detect_core_rings(image, case.path_xy, dataclasses.replace(params, pith_side="start"))
    detected = [b.s for b in given.boundaries]
    score = score_boundaries(detected, list(case.boundaries_px), tolerance_px)
    pairs = dict(match_boundaries(detected, list(case.boundaries_px), tolerance_px))

    auto = detect_core_rings(image, case.path_xy, dataclasses.replace(params, pith_side="auto"))
    orientation_ok = bool(np.allclose(auto.path.xy[0], given.path.xy[0]))
    return CaseResult(
        name=case.name,
        n_true=len(case.boundaries_px),
        n_detected=len(detected),
        tp=score.tp,
        mae_px=score.mae,
        orientation_ok=orientation_ok,
        ring_count_error=len(detected) - len(case.boundaries_px),
        width_errors=_width_errors(given.widths_px, pairs, case.boundaries_px),
    )


def _width_errors(
    widths_px: np.ndarray, pairs: dict[int, int], truth: tuple[float, ...]
) -> tuple[float, ...]:
    """Relative width error for rings whose both boundaries matched consecutive truths."""
    errors = []
    for k, width in enumerate(widths_px):
        inner, outer = pairs.get(k), pairs.get(k + 1)
        if inner is not None and outer == inner + 1:
            true_width = truth[outer] - truth[inner]
            if true_width > 0:
                errors.append(abs(width - true_width) / true_width)
    return tuple(errors)


def summarize(results: Iterable[CaseResult]) -> BenchmarkSummary:
    """Micro-averaged scores over all cases."""
    rows = list(results)
    tp = sum(r.tp for r in rows)
    n_det = sum(r.n_detected for r in rows)
    n_true = sum(r.n_true for r in rows)
    precision = tp / n_det if n_det else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    weighted_mae = [r.mae_px * r.tp for r in rows if r.tp]
    return BenchmarkSummary(
        cases=len(rows),
        precision=precision,
        recall=recall,
        f1=f1,
        mae_px=sum(weighted_mae) / tp if tp else float("nan"),
        orientation_accuracy=float(np.mean([r.orientation_ok for r in rows])) if rows else 0.0,
        mean_abs_ring_count_error=float(np.mean([abs(r.ring_count_error) for r in rows]))
        if rows
        else 0.0,
        width_mdape=float(np.median([e for r in rows for e in r.width_errors]))
        if any(r.width_errors for r in rows)
        else float("nan"),
    )


def run_cases(
    cases: list[AnnotatedPath],
    params: CoreParams = CoreParams(),
    tolerance_px: float = DEFAULT_TOLERANCE_PX,
    workers: int | None = None,
) -> list[CaseResult]:
    """Evaluate every case, grouping cases by image so each image loads once."""
    by_image: dict[str, list[AnnotatedPath]] = {}
    for case in cases:
        by_image.setdefault(str(case.image_path), []).append(case)
    jobs = [(path, group, params, tolerance_px) for path, group in by_image.items()]
    if workers == 1:
        batches = [_evaluate_image(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            batches = list(pool.map(_evaluate_image, *zip(*jobs)))
    return [result for batch in batches for result in batch]


def _evaluate_image(
    image_path: str, cases: list[AnnotatedPath], params: CoreParams, tolerance_px: float
) -> list[CaseResult]:
    image = np.array(Image.open(image_path).convert("RGB"))
    return [evaluate_path(image, case, params, tolerance_px) for case in cases]


@dataclass(frozen=True)
class RenderedCore:
    """A synthetic core image cut from a cross-section, with known boundary positions."""

    image: np.ndarray
    axis_start: np.ndarray  # image xy of the pith end of the core axis
    axis_direction: np.ndarray  # unit vector pith → bark
    boundaries_px: tuple[float, ...]  # distances from ``axis_start`` along the axis


def render_core_image(
    image: np.ndarray,
    case: AnnotatedPath,
    diameter_px: int = 60,
    rotation_deg: float = 0.0,
    background: int = 40,
    padding_px: int = 80,
) -> RenderedCore:
    """Cut a ``diameter_px``-wide strip along ``case.path_xy`` and lay it on a background.

    The strip is rotated by ``rotation_deg`` about the canvas centre, imitating a core
    scanned at an angle.
    """
    import cv2

    from tras.cores.ribbon import extract_ribbon

    ribbon = extract_ribbon(image, case.path_xy, diameter_px // 2)
    strip = np.clip(ribbon.values, 0, 255).astype(np.uint8)
    h, w = strip.shape[:2]
    canvas = np.full((h + 2 * padding_px, w + 2 * padding_px, 3), background, dtype=np.uint8)
    canvas[padding_px : padding_px + h, padding_px : padding_px + w] = strip

    centre = (canvas.shape[1] / 2, canvas.shape[0] / 2)
    rotation = cv2.getRotationMatrix2D(centre, rotation_deg, 1.0)
    # Enlarge the output so the rotated strip is not clipped.
    cos, sin = abs(rotation[0, 0]), abs(rotation[0, 1])
    out_w = int(canvas.shape[0] * sin + canvas.shape[1] * cos)
    out_h = int(canvas.shape[0] * cos + canvas.shape[1] * sin)
    rotation[0, 2] += out_w / 2 - centre[0]
    rotation[1, 2] += out_h / 2 - centre[1]
    rotated = cv2.warpAffine(
        canvas,
        rotation,
        (out_w, out_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(background,) * 3,
    )
    start = rotation @ np.array([padding_px, padding_px + h // 2, 1.0])
    end = rotation @ np.array([padding_px + w - 1, padding_px + h // 2, 1.0])
    direction = (end - start) / np.linalg.norm(end - start)
    return RenderedCore(rotated, start, direction, case.boundaries_px)
