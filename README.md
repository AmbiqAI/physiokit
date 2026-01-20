# physioKIT

> An AI Development Kit (ADK) for physiological signal processing in Python.

[![PyPI](https://img.shields.io/pypi/v/physiokit?color=%2334D058&label=PyPI)](https://pypi.org/project/physiokit/)
[![Python](https://img.shields.io/badge/python-3.12%2B-34D058)](https://pypi.org/project/physiokit/)
[![Downloads](https://img.shields.io/pypi/dm/physiokit.svg?color=%2334D058)](https://pypi.org/project/physiokit/)
[![GitHub Stars](https://img.shields.io/github/stars/AmbiqAI/physiokit.svg?color=%2334D058)](https://github.com/AmbiqAI/physiokit)
[![License](https://img.shields.io/pypi/l/physiokit)](https://github.com/AmbiqAI/physiokit/blob/main/LICENSE)


physioKIT is an open-source Python package designed for processing and analyzing physiological signals commonly acquired from wearable sensors. It provides a comprehensive suite of tools for signal processing, feature extraction, and synthetic signal generation, making it ideal for researchers and developers working in the field of ambulatory health monitoring and wearable technology.

**Key Features:**

* Handles a variety of physiological signals including ECG, PPG, RSP, and IMU.
* Geared towards real-time, noisy wearable sensor data.
* Provide advanced signal processing and feature extraction methods.
* Create synthetic signals for testing and benchmarking.

## Requirements

* [Python 3.12+](https://www.python.org)

## Installation

Installing physioKIT can be done using `uv` or `pip`.

```console
pip install physiokit
```

```console
uv add physiokit
```

## Example

In this example, we will generate a synthetic ECG signal, clean it, and compute heart rate and HRV metrics.


```python

import numpy as np
import physiokit as pk

sample_rate = 1000 # Hz
heart_rate = 64 # BPM
signal_length = 8*sample_rate # 8 seconds

# Generate synthetic ECG signal
ecg, segs, fids = pk.ecg.synthesize(
    signal_length=signal_length,
    sample_rate=sample_rate,
    heart_rate=heart_rate,
    leads=1
)
ecg = ecg.squeeze()

# Clean ECG signal
ecg_clean = pk.ecg.clean(ecg, sample_rate=sample_rate)

# Compute heart rate
hr_bpm, _ = pk.ecg.compute_heart_rate(ecg_clean, sample_rate=sample_rate)

# Extract R-peaks and RR-intervals
peaks = pk.ecg.find_peaks(ecg_clean, sample_rate=sample_rate)
rri = pk.ecg.compute_rr_intervals(peaks)
mask = pk.ecg.filter_rr_intervals(rri, sample_rate=sample_rate)

# Re-compute heart rate
hr_bpm = 60 / (np.nanmean(rri[mask == 0]) / sample_rate)

# Compute HRV metrics
hrv_td = pk.hrv.compute_hrv_time(rri[mask == 0], sample_rate=sample_rate)

bands = [(0.04, 0.15), (0.15, 0.4), (0.4, 0.5)]
hrv_fd = pk.hrv.compute_hrv_frequency(
    peaks[mask == 0],
    rri[mask == 0],
    bands=bands,
    sample_rate=sample_rate
)

```

## License

This project is licensed under the terms of BSD 3-Clause.
