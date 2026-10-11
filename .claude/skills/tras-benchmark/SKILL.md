---
name: tras-benchmark
description: Measure and compare TRAS detection accuracy with tras_benchmark (virtual cores, CooRecorder paths, cross-sections on UruDendro). Use when changing or tuning a ring/core detector, before claiming an accuracy improvement, when updating docs/benchmarks.md, or when the user asks "is it better", "benchmark", "F1", "evaluate".
---

# Benchmark a TRAS detector

Accuracy claims in TRAS come from `tras_benchmark`, never from eyeballing one image.

## Data on this machine

| Set | Path | Use |
|-----|------|-----|
| UruDendro (64 discs, rings + pith) | `/home/henry/cstrd_benchmark/data` | Virtual cores and cross-sections |
| CooRecorder expert paths (16) | `/home/henry/tras/experiments/coorecorder/data` (.pos) + clean images in `/home/henry/Desktop/TRAS_ForestScience` | Real held-out core test |

UruDendro's `pith_location.csv` header says `cy,cx` but the columns are x, y
(`tras/evaluation/datasets.py` handles it).

## Protocol (do not skip)

1. **Tune only on `--split tune`** (even-indexed discs).
2. **Report on `--split test`** (odd discs) **and** the CooRecorder set, which is never tuned on.
3. Report P, R, F1, MAE (px), width MdAPE and orientation. A change that raises F1 but
   lowers orientation or width accuracy needs a sentence explaining the trade-off.
4. The DeepCS-TRD "generic" model was likely trained on UruDendro: deep numbers on
   UruDendro may be optimistic. Use CooRecorder results as the generalization check.

## Commands

```bash
U=/home/henry/cstrd_benchmark/data
uv run tras_benchmark cores $U --split tune --evidence classical          # fast, multi-process
uv run tras_benchmark cores $U --split test --evidence deep --json out.json # GPU, 1 worker
uv run tras_benchmark coorecorder /home/henry/tras/experiments/coorecorder/data \
    /home/henry/Desktop/TRAS_ForestScience --evidence deep
uv run tras_benchmark sections $U --split test --method deepcstrd --json sections.json
```

Quick parameter sweeps: call `tras.evaluation.core_benchmark.run_cases(cases, CoreParams(...))`
from a scratch script (scratchpad dir, not the repo), print `summarize(...).format()`.

## Diagnose before tuning

Plot ribbon + evidence + GT/detections for the worst cases (sort `CaseResult` by
`tp / n_true`), and tally errors by position along the path (pith end / middle / bark end).
In 2026-10 most core errors were at the two ends, not in the middle.

## Finish

Update `docs/benchmarks.md` (table + date + commit) and the README accuracy table.
