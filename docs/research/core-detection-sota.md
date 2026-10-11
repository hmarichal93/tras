# Increment-core ring detection: formats, prior art, and design choices

Last updated: 2026-10-10. Literature and format survey behind `tras/cores/` and
`tras/io/`. Claims link to their source. Items marked *unverified* come from parsers
and real files, not from vendor documentation.

**Bottom line.**

- CooRecorder `.pos` files list ring boundaries from **bark to pith**, in millimetres.
  TRAS writes one *double point* per boundary so each width is an exact perpendicular segment.
- No public increment-core dataset with ring annotations exists (2026-10). TRAS benchmarks its
  core detector on *virtual cores*: radial strips cut from annotated cross-sections.
- The best published core detector is Mask R-CNN (Poláček et al. 2023): precision 0.95–0.97,
  recall 0.99 on Norway spruce. Cracks caused 27 % of its misses.

## CooRecorder `.pos` format

Sources: [modendro `read_pos.R`](https://github.com/ctmaher/modendro/blob/main/R/read_pos.R),
[Cybis label help](https://www.cdendro.se/forfun/dendro/helpcoorecorder7/lblcomments/index.htm),
[crack help](https://www.cdendro.se/forfun/dendro/helpcoorecorder7/meacrack/index.htm),
[pith help](https://www.cdendro.se/forfun/dendro/helpcoorecorder7/distanceToPith/index.htm).

### Header

```text
#DENDRO (Cybis Dendro program compatible format) Coordinate file written as <path> <YYYY-MM-DD HH:MM:SS>
#Imagefile <name>
#DPI 1200.0
#All coordinates in millimeters (mm)
SCALE 1
#C DATED 2012
#C PithCoordinates=113.5486,11.8745; DistanceToPith=0.5; YearsToPith=1;
#C CalcRadius=Yes; Radius=51.5514; Written=2024-12-17 14:41:00;
#C CooRecorder=9.8.1 March 23 2022;
```

- `DATED <year>` is the year of the **outermost** ring. `DATED` with no year means undated.
- Coordinates are millimetres in image space: `mm = px / DPI × 25.4`. The DPI may use a
  comma decimal (`2400,0`); readers must accept both.
- `DistanceToPith` is in mm; `YearsToPith` is an integer (one real file has `2-4`, *unverified*).
- `#C` lines hold `key=value;` metadata or free-text comments.

### Point lines

| Line | Meaning |
|------|---------|
| `x,y` | Ring boundary |
| `x1,y1  x2,y2` | Double point: same boundary, two places. Width ends at point 1; the next ring starts at point 2 |
| `x,y #text` | Labelled point |
| `x,y #ZERO<n>` | `n` missing rings after this point (*unverified*) |
| `Dx,y #W#` | Earlywood/latewood border inside a ring |
| `Dx,y #%gap` | Crack marker; the gap is not counted in the width |

Ring width for year `DATED − k` is the path length between boundary `k` and `k+1`, skipping
double-point jumps and crack gaps.

## Tucson `.rwl` format

Sources: [NOAA treeinfo.txt](https://www.ncei.noaa.gov/pub/data/paleo/treering/treeinfo.txt),
[dplR `write.tucson.R`](https://github.com/OpenDendro/dplR/blob/master/R/write.tucson.R).

- Data line: series ID (cols 1–8, max 8 chars), year right-aligned in cols 9–12, then up to
  ten `%6d` values.
- Units: 0.01 mm (end marker `999`) or 0.001 mm (end marker `-9999`). The end marker sits in
  the slot of year last+1. Decade lines start at years ending in 0, except the first line.
- At 0.01 mm a true value of 999 must be written as 998 or 1000 so it is not read as the end.
- `0` is a locally absent ring (a real value), not missing data.

## Published core methods

| Method | Approach | Reported accuracy | Code / data |
|--------|----------|-------------------|-------------|
| [Poláček et al. 2023, MEE](https://github.com/Gregor-Mendel-Institute/TRG-ImageProcessing) | Mask R-CNN on sliding windows; width = min distance between ring centrelines | P 0.95–0.97, R 0.99 (34 spruce cores) | Code + weights MIT; data not released |
| [Zambrano-Suárez et al. 2025, PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0321841) | U-Net variants (VGG16 FTU-Net) | Dice ≈ 0.77; poor below 200 µm rings | Zenodo access restricted |
| DeepDendro (Fabijańska & Danek 2018) | U-Net boundary map, ring-porous species | P 0.97 / R 0.96 | Not released |
| [MtreeRing](https://github.com/ropensci/MtreeRing) | Morphology + watershed/Canny along a path; two-path tilt correction | — | GPL-3 |
| CooRecorder [AutoPlace](https://www.cdendro.se/forfun/dendro/helpcoorecorder7/autoplace/index.htm) | Intensity pattern along a user path band; "lowest earlywood width" suppresses false borders | — | Proprietary |

## What leading tools offer beyond detection

- **Crossdating:** Pearson *r* and Student's *t* per lag, Gleichläufigkeit (GLK, the share of
  year-to-year changes with the same sign), COFECHA-style segment checks
  ([dplR](https://rdrr.io/cran/dplR/man/glk.html)).
- **Pith offset for cores:** distance to pith (DTP) and years to pith (RTP = DTP ÷ mean width
  of the innermost rings, [modendro](https://github.com/ctmaher/modendro/blob/main/R/yrs_to_pith.R)).
- **QC flags:** width outliers, low-contrast boundaries, boundaries at a steep angle to the path.

## Design choices for TRAS

1. Straighten the image along the core path into a ribbon (a rectified strip).
2. Boundary evidence: signed derivative of a robust (median) across-band profile, or the
   DeepCS-TRD U-Net boundary probability map run on the ribbon.
3. Peak picking with a minimum ring width (CooRecorder's "lowest earlywood width").
4. Width measured perpendicular to each boundary, using its local orientation.
5. Export `.pos` with double points, Tucson `.rwl`, CSV, and a TRAS label file.
