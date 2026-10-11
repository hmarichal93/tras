# TRAS accuracy benchmarks

Last run: 2026-10-10 on branch `claude-full-autonomus`; GTX 1650 (4 GB) GPU, 8-core CPU.
Every number below is reproducible with `tras_benchmark` (commands at the end).

**Summary.**

- **Increment cores:** on 16 real radii measured by an expert in CooRecorder, the automatic
  core detector finds boundaries with **precision 0.97, recall 0.87 (F1 0.92)**. Its
  boundaries are within **1.4 px** of the expert's clicks, and ring widths differ by a
  median **3.2 %**.
- **Cross-sections:** DeepCS-TRD reaches **F1 0.947** on held-out UruDendro discs.

## Datasets

| Set | Content | Role |
|-----|---------|------|
| UruDendro | 64 *Pinus taeda* discs with manual ring polygons and pith ([Marichal et al. 2025](https://doi.org/10.1186/s13595-025-01296-5)) | Cross-sections; virtual cores |
| UruDendro tune split | 32 discs (even index) | Only split used to choose parameters |
| UruDendro test split | 32 discs (odd index) | Reported numbers |
| CooRecorder paths | 4 *Pinus* discs × 4 radii (N/E/S/W), measured by an expert in CooRecorder 9.8 | Real-world core test; never used for tuning |

**Virtual cores.** No public increment-core dataset with ring annotations exists
(2026-10; see [research notes](research/core-detection-sota.md)). A virtual core is the
straight path from the pith to past the bark along one ray of an annotated disc. The
ground truth is where the annotated rings cross that ray. Each disc gives 8 cores (every 45°).

> **Caveat:** the DeepCS-TRD "generic" weights were trained on data that likely includes
> UruDendro. Deep-model numbers on UruDendro may be optimistic; the CooRecorder set is the
> independent check.

## Increment cores

A detected boundary is correct if it lies within the tolerance of a true boundary, matched
one-to-one (4 px for virtual cores, 6 px for CooRecorder clicks).

| Data | Evidence | Precision | Recall | F1 | Position error (px) | Width error (MdAPE) | Pith end correct |
|------|----------|-----------|--------|----|---------------------|---------------------|------------------|
| Virtual cores, test split (256) | classical | 0.843 | 0.808 | 0.825 | 1.02 | 2.1 % | 96.1 % |
| Virtual cores, test split (256) | deep | **0.961** | **0.849** | **0.901** | 0.99 | 2.5 % | 96.1 % |
| CooRecorder paths (16) | classical | 0.823 | 0.855 | 0.839 | 1.37 | 2.5 % | 100 % |
| CooRecorder paths (16) | deep | **0.967** | **0.874** | **0.918** | 1.42 | 3.2 % | 100 % |

- *Width error* is the median absolute percentage difference of ring widths, over rings
  whose two boundaries were both found. The expert's own click scatter (~1.4 px on
  ~40 px rings) is of the same order.
- *Pith end correct* is the share of paths where `pith_side="auto"` picked the right end.

### What limits recall

On virtual cores, most misses sit at the two ends of the path. These are the small ring
around the pith (missed in 50 of 64 cases) and the outermost boundary at the bark (36 of 64).
The middle 80 % of the path is detected at precision and recall above 0.9.

### How the defaults were chosen

Tuned on the tune split only, then confirmed on the test split and the CooRecorder set:

| Change | Virtual cores F1 (tune) | CooRecorder F1 |
|--------|------------------------|----------------|
| Classical baseline (`half_width=15`) | 0.811 | 0.803 |
| Classical, `half_width=25`, contrast floor | 0.821 | 0.833 |
| U-Net on the straightened ribbon | 0.723 | — |
| U-Net on the natural image crop, `half_width=10` | 0.880 | 0.902 |
| U-Net on the crop, `half_width=5`, threshold 0.15 (default) | 0.918 | 0.918 |

The U-Net is pre-trained for cross-sections. Running it on the natural image around the
path, not on a straightened strip, raised F1 from 0.72 to 0.88: the network sees what it
was trained on.

## Cross-sections

Ring detection with the ground-truth pith, UruDendro test split (32 discs); matching per
the UruDendro protocol (a detection must fall in one ring's influence zone on ≥ 60 % of
360 rays).

| Method | Precision | Recall | F1 | Time per disc |
|--------|-----------|--------|----|---------------|
| DeepCS-TRD (generic) | 0.993 | 0.905 | **0.947** | ~54 s (GTX 1650) |
| CS-TRD | — | — | — | — |

## Quality-control flags

On the 1,221 ground-truth rings of all 64 UruDendro discs, `tras.analysis.qc` flags
**2 rings** (0.16 %). The QC page in reports therefore rarely raises false alarms on
correct rings.

## Reproduce

```bash
U=/path/to/UruDendro          # images_no_background/images, ring_annotations/..., pith_location.csv
uv run tras_benchmark cores "$U" --split test --evidence classical
uv run tras_benchmark cores "$U" --split test --evidence deep
uv run tras_benchmark coorecorder <pos_dir> <images_dir> --evidence deep
uv run tras_benchmark sections "$U" --split test --method deepcstrd
uv run tras_benchmark sections "$U" --split test --method cstrd
```

Add `--json results.json` to keep the numbers. When changing a detector, tune with
`--split tune` and report `--split test` plus the CooRecorder set.
