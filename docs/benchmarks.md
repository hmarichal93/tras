# TRAS accuracy benchmarks

Last run: 2026-10-10 on branch `claude-full-autonomus`; GTX 1650 (4 GB) GPU, 8-core CPU.
Every number below is reproducible with `tras_benchmark` (commands at the end).

**Summary.**

- **Increment cores:** on 16 real radii measured by an expert in CooRecorder, the automatic
  core detector finds boundaries with **precision 0.97, recall 0.90 (F1 0.93)**. Its
  boundaries are within **1.4 px** of the expert's clicks, and ring widths differ by a
  median **3.3 %**.
- **Cross-sections:** DeepCS-TRD reaches **F1 0.947** on the UruDendro test split. Its
  weights were trained on a different partition of UruDendro, so some test discs may have been
  in their training data; read this as a regression check (see the caveat below).

## Datasets

| Set | Content | Role |
|-----|---------|------|
| UruDendro | 64 *Pinus taeda* discs with manual ring polygons and pith ([Marichal et al. 2025](https://doi.org/10.1186/s13595-025-01296-5)) | Cross-sections; virtual cores |
| UruDendro tune split | 32 discs (even index) | Only split used to choose parameters |
| UruDendro test split | 32 discs (odd index) | Reported numbers |
| CooRecorder paths | 4 *Pinus* discs × 4 radii (N/E/S/W), measured by an expert in CooRecorder 9.8 | Real-world core test; never used for tuning |

**Virtual cores.** A virtual core is the straight path from the pith to past the bark along
one ray of an annotated disc. The ground truth is where the annotated rings cross that ray.
Each disc gives 8 cores (every 45°).

**Public core datasets.** Two increment-core datasets with ring annotations are public
under CC BY 4.0. TRAS is not benchmarked on them; they are listed as related work (details in
the [competitive landscape](research/competitive-landscape.md)).

| Dataset | Content | Licence |
|---------|---------|---------|
| Poláček et al. 2023, "Automation of tree-ring detection and measurements using deep learning" ([Zenodo 7981462](https://zenodo.org/records/7981462)) | *Picea abies* core patches with hand-annotated boundaries; full cores of 4 conifers with measurement tables | CC BY 4.0 |
| Zambrano-Suárez et al. 2026, "Tree ring segmentation performance in highly disturbed trees using deep learning" ([Zenodo 22915111](https://zenodo.org/records/22915111)) | *Pinus sylvestris* cores with binary ring masks | CC BY 4.0 |

> **Caveat (training overlap):** the DeepCS-TRD "generic" weights that TRAS ships
> (`0_all_1504.pth`) were trained only on the DeepCS-TRD training and validation discs, not
> on its test set (H. Marichal, author, 2026-10-11). The DeepCS-TRD paper splits each dataset
> 60 % train, 20 % validation, 20 % test
> ([arXiv:2504.16242](https://arxiv.org/abs/2504.16242), §3.2). Our even/odd tune/test split
> is a different partition, so some of our test discs may be in DeepCS-TRD's train/validation
> set. Deep-model numbers on UruDendro (cross-sections and virtual cores) may therefore be
> optimistic. The CooRecorder set is the independent check.

## Increment cores

A detected boundary is correct if it lies within the tolerance of a true boundary, matched
one-to-one (4 px for virtual cores, 6 px for CooRecorder clicks).

| Data | Evidence | Precision | Recall | F1 | Position error (px) | Width error (MdAPE) | Pith end correct |
|------|----------|-----------|--------|----|---------------------|---------------------|------------------|
| Virtual cores, test split (256) | classical | 0.843 | 0.808 | 0.825 | 1.02 | 2.1 % | 96.1 % |
| Virtual cores, test split (256) | deep | **0.958** | **0.849** | **0.900** | 0.99 | 2.5 % | 96.9 % |
| CooRecorder paths (16) | classical | 0.823 | 0.855 | 0.839 | 1.37 | 2.5 % | 100 % |
| CooRecorder paths (16) | deep | **0.965** | **0.896** | **0.929** | 1.43 | 3.3 % | 100 % |

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

Two later changes, A/B-tested the same way:

| Change (deep evidence) | Virtual F1 (tune) | CooRecorder F1 | CooRecorder ring-count error |
|------------------------|-------------------|----------------|------------------------------|
| Without end margin | 0.904 | 0.918 | 2.19 |
| **Analyse 2 min-ring-widths past both path ends (default)** | **0.906** | **0.929** | 1.75 |
| + gap filling (`--fill-gaps`, ratio 2.2) | 0.901 | 0.923 | **1.31** |

Gap filling recovers rings the U-Net missed (it is blind within ~100 px of the pith) but
adds about as many misplaced ones, so it is opt-in: use it when a complete ring count
matters more than boundary precision. The end margin hurts classical evidence (F1 0.839 →
0.827 on CooRecorder), so it is a deep-only default.

The U-Net is pre-trained for cross-sections. Running it on the natural image around the
path, not on a straightened strip, raised F1 from 0.72 to 0.88: the network sees what it
was trained on.

## Cross-sections

Ring detection with the ground-truth pith, UruDendro test split (32 discs); matching per
the UruDendro protocol (a detection must fall in one ring's influence zone on ≥ 60 % of
360 rays; threshold from [CS-TRD](https://arxiv.org/abs/2305.10809) §6.3.3). Precision
and recall pool the true/false positives of all discs (micro-average).

| Method | Precision | Recall | F1 | Time per disc |
|--------|-----------|--------|----|---------------|
| DeepCS-TRD (generic) | 0.993 | 0.905 | **0.947** | 45 s (GTX 1650) |
| CS-TRD | 0.938 | 0.872 | 0.904 | 131 s (CPU) |

## TRAS article results and why they differ

The TRAS article ([Marichal et al. 2026, *Forestry* 99(4) cpag046](https://doi.org/10.1093/forestry/cpag046);
§3.1 of the [arXiv preprint](https://arxiv.org/abs/2605.08025)) reports, for fully automatic detection with no user correction:

| Method | Precision | Recall | F | Data |
|--------|-----------|--------|---|------|
| DeepCS-TRD | 86.4 % | 76.5 % | 81.0 % | 18 *P. taeda* discs, 414 rings |
| CS-TRD | 85.2 % | 75.4 % | 79.7 % | same |
| INBD | 81.2 % | 77.5 % | 79.2 % | same |

It also reports that TRAS and CooRecorder ring widths agree with Pearson r > 0.99
(336 widths, 4 discs × 4 paths). That compares the two tools' **measurement** of the same
corrected rings, not automatic detection.

These are valid results for a different question. **Do not compare them with the
cross-section table above**: the protocols differ on every axis that moves the score.

| Factor | TRAS article | `tras_benchmark sections` | Expected effect |
|--------|--------------|---------------------------|-----------------|
| Images | 18 discs (sample IDs A–C), iPhone 6S, mixed surface finish | 32 UruDendro discs (odd index) | Different difficulty |
| Pith | Automatic (APD) | Ground-truth pith | GT pith raises F (F falls as pith error grows, [CS-TRD](https://arxiv.org/abs/2305.10809) §6.3.2) |
| Matching rule | ≥ 90 % of a detection's nodes in the ring's influence zone | ≥ 60 % of 360 rays | 90 % is stricter, so lower F |
| Preprocessing | Resize factor 2, automatic background removal, web interface | Benchmark defaults | Unknown |
| Averaging | Mean of per-disc scores | Pooled over all rings | Small |
| Training overlap | Models trained on UruDendro; overlap with the 18 discs not stated | Weights trained on DeepCS-TRD train/val; some of our test discs may be in it (see caveat above) | Our F1 may be optimistic |

The article's F = 81.0 % is closer to what a user gets out of the box on new images with
automatic pith. The F1 0.947 here is a regression check: it tells whether a code
change made detection better or worse under fixed conditions.

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
