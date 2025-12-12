"""

# :material-lungs: RSP API

Respiratory signal utilities for cleaning, peak detection, rates, and dual-band metrics.

## Available Tools

- **[clean](./clean)**: Respiratory filtering (`clean`).
- **[peaks](./peaks)**: Peak finding and RR helpers (`find_peaks`, `compute_rr_intervals`, `filter_peaks`, `filter_rr_intervals`).
- **[metrics](./metrics)**: Respiratory rate (FFT/peaks) and dual-band metrics (`compute_respiratory_rate`, `compute_respiratory_rate_from_fft`, `compute_respiratory_rate_from_peaks`, `compute_dual_band_metrics`).
- **[synthesize](./synthesize)**: Synthetic respiratory signal generation (`synthesize`).
- **[defines](./defines)**: RSP fiducial and segment enums (`RspFiducial`, `RspSegment`).

Copyright 2025 Ambiq. All Rights Reserved.

"""

from .clean import clean
from .defines import RspFiducial, RspSegment
from .metrics import (
    compute_dual_band_metrics,
    compute_respiratory_rate,
    compute_respiratory_rate_from_fft,
    compute_respiratory_rate_from_peaks,
)
from .peaks import compute_rr_intervals, filter_peaks, filter_rr_intervals, find_peaks
from .synthesize import synthesize
