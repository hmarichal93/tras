---
name: dendro-formats
description: Read, write, validate or debug dendrochronology exchange files in TRAS — CooRecorder/CDendro .pos coordinate files and Tucson .rwl ring-width files — and convert between TRAS results and them. Use when a .pos/.rwl file is involved, when users report CooRecorder/dplR/COFECHA cannot read TRAS output, or when adding an export format.
---

# Dendro file formats in TRAS

Code: `tras/io/pos.py`, `tras/io/tucson.py`. Spec notes with sources:
`docs/research/core-detection-sota.md`. Tests: `tests/io_tests/test_dendro_formats.py`.

## .pos (CooRecorder / CDendro)

- Header: `#DENDRO ...`, `#Imagefile`, `#DPI <float>` (may use a comma decimal),
  `#All coordinates in millimeters (mm)`, `SCALE 1`, `#C DATED <outer year>`,
  `#C key=value; ...` (PithCoordinates, DistanceToPith, YearsToPith).
- Points in mm: `x,y`; double point `x1,y1  x2,y2` (ring ends at 1, next starts at 2);
  `Dx,y #W#` seasonwood; `Dx,y #%gap` crack markers (pair = skip between).
- Direction: convention is bark first (`DATED` = first ring). Real files measured from the
  pith outward exist; `PosFile.pith_first()` decides from `PithCoordinates`, otherwise
  callers must say (`chronological_widths_mm(pith_first=...)`).
- TRAS writes one double point per inner boundary (perpendicular foot + path point) so
  CooRecorder recomputes TRAS's perpendicular widths exactly.

## .rwl (Tucson)

- `ID` (8 chars) + year (4) + up to ten `%6d` values; lines end at years ending in 9.
- 0.01 mm → stop marker `999`; 0.001 mm → `-9999`. A true 999 at 0.01 mm is written 998.
- `0` = locally absent ring (a value), not missing data. IDs sanitized to `[A-Za-z0-9_.-]{1,8}`.

## Validate a change

1. Round-trip tests (`write` → `read` → equal) for both precisions.
2. Parse a real CooRecorder file: `/home/henry/tras/experiments/coorecorder/data/*.pos`.
3. If R is available: `dplR::read.rwl("x.rwl")` must load without warnings.
