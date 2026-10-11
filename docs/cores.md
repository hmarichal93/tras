# Measure increment cores automatically

TRAS detects annual ring boundaries along increment cores (or along any path you draw on a
cross-section) and exports ring widths to CooRecorder `.pos`, Tucson `.rwl` and CSV.
On 16 held-out radii measured by an expert in CooRecorder, 97 % of the boundaries it reports
are correct and it finds 90 % of the expert's boundaries; ring widths differ from the expert
by a median of 3.3 %. See [benchmarks](benchmarks.md).

## Quick start

```bash
# Find every core in a scan, date the outermost ring as 2023, write results next to it
tras_core scan.tif --outer-year 2023
```

Outputs in `tras_cores/` next to the image:

| File | Content |
|------|---------|
| `<image>_core<k>.csv` | One row per ring: year, width (mm and px), boundary position, tilt, strength |
| `<image>_core<k>.pos` | CooRecorder/CDendro coordinate file (needs a scale) |
| `<image>.rwl` | Tucson ring-width file with every core of the image (needs a scale and `--outer-year`) |
| `<image>.json` | TRAS label file: open it in the GUI to review and edit boundaries |

The scale comes from the image's DPI metadata, which flatbed scanners write. Otherwise pass
`--dpi 2400` or `--mm-per-px 0.0106`.

## Choose where to measure

| Situation | Command |
|-----------|---------|
| Scan with one or more cores on a plain background | `tras_core scan.tif` (cores are found automatically) |
| You know the path (pixel coordinates) | `tras_core scan.tif --path "120,40;3900,85"` |
| Re-measure an existing CooRecorder path | `tras_core disc.jpg --path-from-pos disc_north.pos` |
| A radius of a cross-section | Draw a linestrip from pith to bark in the GUI (below) |
| A whole folder | `tras_core cores/ --dpi 2400 --outer-year 2023` |

## In the GUI

1. Open the image. Set the scale (**Tools → Increment Core → Set Image Scale**) and, for
   dating, the harvest year (**Sample Metadata**).
2. Optional: draw one path per core with **Tools → Increment Core → Draw Core Path**, from
   pith to bark (Ctrl+click ends the path).
3. **Tools → Increment Core → Core Analysis…** → choose *Find cores automatically* or *Use drawn
   core paths*, set *Outermost complete ring year* → **Detect rings** → **Add to image**. The
   path and one line per boundary appear on the canvas.
4. Correct the boundaries: drag a line to move it, delete it to merge two rings, or add a
   missed one with **Tools → Increment Core → Draw Core Boundary** (Ctrl+B): click two points
   across the boundary.
5. **Tools → Increment Core → Export Core Measurements…** writes `.csv` per core, `.pos`
   (needs the scale) and `.rwl` (needs the scale and a year) from the lines as they are on
   the image, and relabels the lines with their years. Lines that cross no core path are
   ignored and selected.

**Outermost complete ring year.** It defaults to the harvest year. Use the previous year for
southern-hemisphere samples cut before the growing season ends, or when the outer ring is
incomplete. TRAS saves it with the sample metadata in the label file.

## How it works

1. **Ribbon.** The image is resampled along the path into a straight strip, so ring
   boundaries become near-vertical lines.
2. **Boundary evidence.** Two sources:
   - *deep* (default when the DeepCS-TRD weights are installed): the U-Net ring-boundary
     probability, computed on the image around the path;
   - *classical*: the latewood → earlywood brightness step. Lanes across the core are
     combined with a median, so a crack or resin streak in a few lanes cannot fake a ring.
3. **Tilt search.** Each position tries boundary angles of ±45° and keeps the best. The
   angle is later used to measure widths perpendicular to the rings, as CooRecorder does.
4. **Peaks.** Boundaries are evidence peaks at least *minimum ring width* apart.
5. **Orientation.** The pith end is found from ring curvature (rings bend around the pith)
   when that is measurable, otherwise from the sharp side of each ring.
6. **Dating.** With `--outer-year`, the outermost ring gets that year and years count back
   towards the pith.

## Tune the detector

| Symptom | Change |
|---------|--------|
| Extra boundaries inside latewood bands | Raise `--min-ring-width` (samples: 1 px, or 0.02 mm on scans finer than ~1270 dpi) |
| Faint rings missed (classical) | Lower `--sensitivity` (default 2.5) |
| Faint rings missed (deep) | Lower `--deep-threshold` (default 0.15) |
| Ring count must be complete (dating) | `--fill-gaps`: recovers rings from width continuity (slightly lower precision) |
| Pith detected at the wrong end | `--pith-side start` or `end` (start = first path point) |
| Ring-porous wood (dark earlywood vessels) | `--evidence classical` and, in Python, `CoreParams(boundary_step="light_to_dark")` |
| Very narrow rings (< 1 mm) | `--evidence classical`: the U-Net works at ~0.3 mm per model pixel |
| No boundaries found with the default (deep) evidence | Try `--evidence classical` (in the GUI: *Boundary evidence*) |

## Python API

```python
import numpy as np
from PIL import Image

from tras.cores import analyze_core_image, export_results

image = np.array(Image.open("scan.tif").convert("RGB"))
results = analyze_core_image(image, mm_per_px=25.4 / 2400, outer_year=2023)
for result in results:
    print(result.years[:3], result.widths_mm[:3])
export_results(results, "out", stem="scan")
```

## Check the dating

A missed or false boundary shifts every older year. Compare cores of one tree (or radii of
one disc) with `tras.analysis.dating`:

```python
from tras.analysis.dating import as_year_series, scan_lags
from tras.io.tucson import read_rwl

core_a, core_b = (as_year_series(s) for s in read_rwl("out/scan.rwl")[:2])
best = scan_lags(core_a, core_b)[0]  # best lag first; 0 = dated consistently
print(best.lag, round(best.t, 1), round(best.glk, 2))
```

`check_segments` flags 50-year segments that match the master better at another lag
(flag `B`, COFECHA's convention), which points to the missing or false ring.

## File formats

- **`.pos`**: boundaries from bark to pith in millimetres, `#C DATED <year>` = outermost
  ring. Each inner boundary is written as a double point so CooRecorder recomputes exactly
  the perpendicular widths TRAS measured. Spec notes:
  [core-detection-sota.md](research/core-detection-sota.md).
- **`.rwl`**: Tucson decadal format, 0.01 mm units (0.001 mm available via
  `tras.io.tucson.write_rwl(..., precision=0.001)`).
