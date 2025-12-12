"""

# :material-heart-pulse: ECG API

Electrocardiogram processing utilities for cleaning, peak detection, segmentation, metrics, and synthesis.

## Available Tools

- **[clean](./clean)**: ECG filtering utilities (`clean`, `square_filter_mask`).
- **[peaks](./peaks)**: QRS detection and RR interval helpers (`find_peaks`, `compute_rr_intervals`, `filter_peaks`, `filter_rr_intervals`).
- **[metrics](./metrics)**: Heart rate, RR-derived respiration, and pulse arrival time (`compute_heart_rate`, `compute_heart_rate_from_peaks`, `derive_respiratory_rate`).
- **[segment](./segment)**: Wave segmentation helpers (`locate_qrs`, `locate_pwave_from_qrs_anchor`, `locate_twave_from_qrs_anchor`, `apply_segmentation`).
- **[synthesize](./synthesize)**: Synthetic ECG generation (`synthesize`, `simulate_daubechies`, `simulate_ecgsyn`, `simulate_brisk`).
- **[defines](./defines)**: Enumerations for fiducials and segments (`EcgFiducial`, `EcgSegment`).
- **[synthetic](./synthetic)**: Preset definitions and WaSP-based synthesis (`EcgPreset`, `EcgPresetParameters`, `generate_preset_parameters`).

Copyright 2025 Ambiq. All Rights Reserved.

"""

from .clean import clean, square_filter_mask
from .defines import EcgFiducial, EcgSegment
from .metrics import (
    compute_heart_rate,
    compute_heart_rate_from_peaks,
    derive_respiratory_rate,
)
from .peaks import compute_rr_intervals, filter_peaks, filter_rr_intervals, find_peaks
from .segment import (
    apply_segmentation,
    locate_pwave_from_qrs_anchor,
    locate_qrs,
    locate_twave_from_qrs_anchor,
)
from .synthesize import simulate_daubechies, simulate_ecgsyn, synthesize
from .synthetic import (
    EcgPreset,
    EcgPresetParameters,
    generate_preset_parameters,
    simulate_brisk,
)
