"""

# :material-heart-pulse: physioKIT API

The top-level package exposes core physiology processing modules.

## Available Modules

- **[ecg](./ecg)**: ECG cleaning, peak detection, metrics, and synthesis helpers.
- **[ppg](./ppg)**: PPG cleaning, peaks, heart/respiratory rate, SpO2, and synthesis.
- **[rsp](./rsp)**: Respiratory signal cleaning, peaks, rates, and dual-band metrics.
- **[imu](./imu)**: IMU-derived metrics such as ENMO, tilt, and counts.
- **[hrv](./hrv)**: Heart-rate variability frequency/time/nonlinear metrics.
- **[signal](./signal)**: Shared signal filtering, noise, smoothing, distortion, and transforms.

Copyright 2025 Ambiq. All Rights Reserved.

"""
from importlib.metadata import version

__version__ = version(__name__)

from . import ecg, hrv, imu, ppg, rsp, signal
