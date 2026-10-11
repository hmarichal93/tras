"""Automatic ring detection on increment cores and measurement paths (CooRecorder-style).

Typical use::

    from tras.cores import analyze_core_image, to_pos, to_series
    results = analyze_core_image(image, mm_per_px=0.0106, outer_year=2023)

See ``docs/cores.md``.
"""

from tras.cores.axis import DetectedCore
from tras.cores.axis import find_cores
from tras.cores.detect import CoreBoundary
from tras.cores.detect import CoreParams
from tras.cores.detect import CoreResult
from tras.cores.detect import detect_core_rings
from tras.cores.export import to_label_shapes
from tras.cores.export import to_pos
from tras.cores.export import to_series
from tras.cores.export import write_csv
from tras.cores.pipeline import analyze_core_image
from tras.cores.pipeline import default_core_params
