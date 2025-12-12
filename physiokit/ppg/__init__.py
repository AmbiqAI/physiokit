"""

# :material-waveform: PPG API

Photoplethysmography tools for cleaning, peak detection, heart/respiratory metrics, SpO2, and synthesis.

## Available Tools

- **[clean](./clean)**: PPG filtering (`clean`).
- **[peaks](./peaks)**: Peak finding and RR helpers (`find_peaks`, `compute_rr_intervals`, `filter_peaks`, `filter_rr_intervals`).
- **[metrics](./metrics)**: Heart rate (FFT/peaks), respiratory rate, and SpO2 (`compute_heart_rate`, `compute_heart_rate_from_fft`, `compute_heart_rate_from_peaks`, `derive_respiratory_rate`, `compute_spo2_from_perfusion`, `compute_spo2_in_time`, `compute_spo2_in_frequency`).
- **[synthesize](./synthesize)**: Synthetic PPG generation (`synthesize`).
- **[defines](./defines)**: PPG fiducial and segment enums (`PpgFiducial`, `PpgSegment`).

Copyright 2025 Ambiq. All Rights Reserved.

"""

from .clean import clean
from .defines import PpgFiducial, PpgSegment
from .metrics import (
    compute_heart_rate,
    compute_heart_rate_from_fft,
    compute_heart_rate_from_peaks,
    compute_spo2_from_perfusion,
    compute_spo2_in_frequency,
    compute_spo2_in_time,
    derive_respiratory_rate,
)
from .peaks import compute_rr_intervals, filter_peaks, filter_rr_intervals, find_peaks
from .synthesize import synthesize
