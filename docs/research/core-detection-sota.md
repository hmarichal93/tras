# Increment-core ring detection: formats, prior art, and design choices

Last updated: 2026-10-11. Literature and format survey behind `tras/cores/` and
`tras/io/`. Claims link to their source. Items marked *unverified* come from parsers
and real files, not from vendor documentation.

**Bottom line.**

- CooRecorder `.pos` files list ring boundaries from **bark to pith**, in millimetres.
  TRAS writes one *double point* per boundary so each width is an exact perpendicular segment.
- No public increment-core dataset with ring annotations exists (2026-10). TRAS benchmarks its
  core detector on *virtual cores*: radial strips cut from annotated cross-sections.
- The highest reported core-detection scores come from a Mask R-CNN (Poláček et al. 2023):
  recall > 0.98 and precision 0.96 on microscope-imaged Norway spruce cores. These are on the
  authors' own data, not a shared benchmark, so they rank nothing. (A "cracks caused 27 % of
  misses" figure in earlier versions of this note is *unverified*.)

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
| [Poláček et al. 2023, MEE](https://doi.org/10.1111/2041-210X.14183) ([code](https://github.com/Gregor-Mendel-Institute/TRG-ImageProcessing)) | Mask R-CNN on sliding windows; width = min distance between ring centrelines | P 0.96, R > 0.98 on real-world spruce cores (abstract) | Code + weights MIT; data not released |
| [Zambrano-Suárez et al. 2025, PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0321841) | U-Net variants (VGG16 FTU-Net) | Dice ≈ 0.77; poor below 200 µm rings | Zenodo access restricted |
| DeepDendro (Fabijańska & Danek 2018) | U-Net boundary map, ring-porous species | P 0.97 / R 0.96 | Not released |
| [MtreeRing](https://github.com/ropensci/MtreeRing) | Morphology + watershed/Canny along a path; two-path tilt correction | — | GPL-3 |
| CooRecorder [AutoPlace](https://www.cdendro.se/forfun/dendro/helpcoorecorder7/autoplace/index.htm) | Intensity pattern along a user path band; "lowest earlywood width" suppresses false borders | — | Proprietary |

## What established tools offer beyond detection

- **Crossdating:** Pearson *r* and Student's *t* per lag, Gleichläufigkeit (GLK, the share of
  year-to-year changes with the same sign), COFECHA-style segment checks
  (Holmes 1983; implemented in dplR, Bunn 2008; [dplR `glk`](https://rdrr.io/cran/dplR/man/glk.html)).
  See [References](#references).
- **Pith offset for cores:** distance to pith (DTP) and years to pith (RTP = DTP ÷ mean width
  of the innermost rings, [modendro](https://github.com/ctmaher/modendro/blob/main/R/yrs_to_pith.R)).
- **QC flags:** width outliers, low-contrast boundaries, boundaries at a steep angle to the path.

## Design choices for TRAS

1. Straighten the image along the core path into a ribbon (a rectified strip).
2. Boundary evidence: signed derivative of a robust (median) across-band profile, or the
   DeepCS-TRD U-Net boundary probability map, run on the natural image crop around the path
   and then straightened into the ribbon (`tras/cores/deep.py::boundary_probability_along_path`).
3. Peak picking with a minimum ring width (CooRecorder's "lowest earlywood width").
4. Width measured perpendicular to each boundary, using its local orientation.
5. Export `.pos` with double points, Tucson `.rwl`, CSV, and a TRAS label file.

## References

Checked 2026-10-11 against Crossref, the publisher or arXiv. BibTeX for the detection
methods, UruDendro, U²-Net and CooRecorder is in the [README](../../README.md#-citations).

| Topic | Reference | Used in TRAS |
|-------|-----------|--------------|
| CooRecorder | Maxwell, R.S. & Larsson, L.-Å. (2021). Measuring tree-ring widths using the CooRecorder software application. *Dendrochronologia* 67, 125841. [doi:10.1016/j.dendro.2021.125841](https://doi.org/10.1016/j.dendro.2021.125841) | `.pos` I/O, core benchmark reference |
| COFECHA | Holmes, R.L. (1983). Computer-assisted quality control in tree-ring dating and measurement. *Tree-Ring Bulletin* 43, 69–78. [repository.arizona.edu](https://repository.arizona.edu/handle/10150/261223) (no DOI) | Segment check in `tras/analysis/dating.py` (50-year segments, 25-year lag) |
| dplR | Bunn, A.G. (2008). A dendrochronology program library in R (dplR). *Dendrochronologia* 26(2), 115–124. [doi:10.1016/j.dendro.2008.01.002](https://doi.org/10.1016/j.dendro.2008.01.002) | Tucson `.rwl` conventions; reference implementation of crossdating statistics |
| Gleichläufigkeit | Eckstein, D. & Bauch, J. (1969). Beitrag zur Rationalisierung eines dendrochronologischen Verfahrens und zur Analyse seiner Aussagesicherheit. *Forstwissenschaftliches Centralblatt* 88, 230–250. [doi:10.1007/BF02741777](https://doi.org/10.1007/BF02741777) | GLK in `tras/analysis/dating.py` |
| GLK ties | Buras, A. & Wilmking, M. (2015). Correcting the calculation of Gleichläufigkeit. *Dendrochronologia* 34, 29–30. [doi:10.1016/j.dendro.2015.03.003](https://doi.org/10.1016/j.dendro.2015.03.003) | Ties count half |
| Mask R-CNN cores | Poláček, M., Arizpe, A., Hüther, P., Weidlich, L., Steindl, S. & Swarts, K. (2023). Automation of tree-ring detection and measurements using deep learning. *Methods in Ecology and Evolution* 14, 2233–2242. [doi:10.1111/2041-210X.14183](https://doi.org/10.1111/2041-210X.14183) | Prior art only |
