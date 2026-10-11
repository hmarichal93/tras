"""``tras_benchmark``: reproducible accuracy numbers for TRAS detectors.

Subcommands:

* ``cores``: core detector on virtual cores cut from UruDendro cross-sections;
* ``coorecorder``: core detector along real CooRecorder measurement paths;
* ``sections``: cross-section ring detection (CS-TRD / DeepCS-TRD / INBD) on UruDendro.

Results are printed and, with ``--json``, saved for ``docs/benchmarks.md``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from tras.cores.pipeline import default_core_params
from tras.evaluation.core_benchmark import DEFAULT_TOLERANCE_PX
from tras.evaluation.core_benchmark import run_cases
from tras.evaluation.core_benchmark import summarize
from tras.evaluation.core_benchmark import virtual_cores
from tras.evaluation.datasets import load_coorecorder_pairs
from tras.evaluation.datasets import load_urudendro

app = typer.Typer(name="benchmark", help="Measure TRAS detection accuracy on annotated data")

SPLITS = ("all", "tune", "test")


@app.command()
def cores(
    urudendro_root: Path = typer.Argument(
        ..., help="UruDendro folder (images, annotations, pith CSV)"
    ),
    directions: int = typer.Option(8, help="Virtual cores per cross-section"),
    tolerance: float = typer.Option(DEFAULT_TOLERANCE_PX, help="Match tolerance (px)"),
    evidence: Optional[str] = typer.Option(None, help="classical | deep | hybrid (default: best)"),
    split: str = typer.Option("all", help="all | tune (even samples) | test (odd samples)"),
    workers: Optional[int] = typer.Option(None, help="Parallel processes (deep: use 1)"),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Save summary as JSON"),
) -> None:
    """Core detector on virtual cores (pith → bark rays of annotated cross-sections)."""
    sections = _split(load_urudendro(urudendro_root), split)
    cases = [case for section in sections for case in virtual_cores(section, directions)]
    params = default_core_params(**({"evidence": evidence} if evidence else {}))
    _run("virtual cores", cases, params, tolerance, workers, json_out)


@app.command()
def coorecorder(
    pos_dir: Path = typer.Argument(..., help="Folder of <disc>_<direction>.pos files"),
    images_dir: Path = typer.Argument(..., help="Folder with the clean <disc> images"),
    tolerance: float = typer.Option(6.0, help="Match tolerance (px)"),
    evidence: Optional[str] = typer.Option(None, help="classical | deep | hybrid (default: best)"),
    workers: Optional[int] = typer.Option(None, help="Parallel processes (deep: use 1)"),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Save summary as JSON"),
) -> None:
    """Core detector along real CooRecorder paths, scored against the expert's points."""
    cases = load_coorecorder_pairs(pos_dir, images_dir)
    params = default_core_params(**({"evidence": evidence} if evidence else {}))
    _run("CooRecorder paths", cases, params, tolerance, workers, json_out)


@app.command()
def sections(
    urudendro_root: Path = typer.Argument(
        ..., help="UruDendro folder (images, annotations, pith CSV)"
    ),
    method: str = typer.Option("deepcstrd", help="cstrd | deepcstrd | inbd"),
    split: str = typer.Option("all", help="all | tune | test"),
    limit: Optional[int] = typer.Option(None, help="Only the first N samples"),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Save per-image results as JSON"),
) -> None:
    """Cross-section ring detection with the ground-truth pith, UruDendro matching."""
    from tras.api import detect
    from tras.evaluation.cross_section import score_rings

    samples = _split(load_urudendro(urudendro_root), split)[:limit]
    rows = []
    for k, sample in enumerate(samples, 1):
        start = time.time()
        result = detect(
            sample.image_path,
            ring_method=method,
            pith_x=sample.pith_xy[0],
            pith_y=sample.pith_xy[1],
        )
        score = score_rings(result.rings, sample.rings, sample.pith_xy)
        rows.append(
            {
                "name": sample.name,
                "tp": score.tp,
                "fp": score.fp,
                "fn": score.fn,
                "rmse_px": score.rmse_px,
                "seconds": round(time.time() - start, 1),
            }
        )
        logger.info(f"[{k}/{len(samples)}] {sample.name}: F1={score.f1:.3f}")
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"{method}: samples={len(rows)} P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
    if json_out:
        summary = {"method": method, "precision": precision, "recall": recall, "f1": f1}
        json_out.write_text(json.dumps({"summary": summary, "samples": rows}, indent=2))


def _split(items: list, split: str) -> list:
    if split not in SPLITS:
        raise typer.BadParameter(f"split must be one of {SPLITS}")
    return {"all": items, "tune": items[0::2], "test": items[1::2]}[split]


def _run(label, cases, params, tolerance, workers, json_out) -> None:
    start = time.time()
    if params.evidence != "classical" and workers is None:
        workers = 1  # one GPU model per process would exhaust GPU memory
    summary = summarize(run_cases(cases, params, tolerance_px=tolerance, workers=workers))
    print(f"{label} [{params.evidence}]: {summary.format()}  ({time.time() - start:.0f}s)")
    if json_out:
        payload = {"summary": dataclasses.asdict(summary), "params": dataclasses.asdict(params)}
        json_out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    app()
