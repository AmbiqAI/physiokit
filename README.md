<p align="center">
  <a href="https://github.com/AmbiqAI/physiokit"><img src="./docs/assets/physiokit-banner.png" alt="PhysioKit"></a>
</p>

<p align="center">
    <em>A Python toolkit to process raw ambulatory bio-signals. </em>
</p>

<p align="center">
<a href="https://pypi.org/project/physiokit" target="_blank">
    <img src="https://img.shields.io/pypi/v/physiokit?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/physiokit" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/physiokit.svg?color=%2334D058" alt="Supported Python versions">
</a>
<a href="https://pypi.python.org/pypi/physiokit" target="_blank">
    <img src="https://img.shields.io/pypi/dm/physiokit.svg?color=%2334D058" alt="Package downloads">
</a>
<a href="https://github.com/AmbiqAI/physiokit" target="_blank">
    <img src="https://img.shields.io/github/stars/AmbiqAI/physiokit.svg?color=%2334D058" alt="Package downloads">
</a>
<a href="https://github.com/AmbiqAI/physiokit/LICENSE" target="_blank">
    <img src="https://img.shields.io/pypi/l/physiokit" alt="License">
</a>
</p>

<p style="color:rgb(201,48,198); font-size: 1.2em;">
🚧 PhysioKit is under active development
</p>

---

**Documentation**: <a href="https://ambiqai.github.io/physiokit" target="_blank">https://ambiqai.github.io/physiokit</a>

**Source Code**: <a href="https://github.com/AmbiqAI/physiokit" target="_blank">https://github.com/AmbiqAI/physiokit</a>

---

**Key Features:**

* Handles a variety of physiological signals including ECG, PPG, RSP, and IMU.
* Geared towards real-time, noisy wearable sensor data.
* Provide advanced signal processing and feature extraction methods.
* Create synthetic signals for testing and benchmarking.

## Requirements

* [Python 3.11+](https://www.python.org)

## Installation

Installing PhysioKit can be done using `Poetry` or `pip`.

```console
pip install physiokit
```

```console
poetry add physiokit
```

## Example

In this example, we will generate a synthetic ECG signal, clean it, and compute heart rate and HRV metrics.


```python

import physiokit as pk

sample_rate = 1000 # Hz
heart_rate = 64 # BPM
signal_length = 8*sample_rate # 8 seconds

# Generate synthetic ECG signal
ecg, segs, fids = pk.ecg.synthesize(
    duration=10,
    sample_rate=sample_rate,
    heart_rate=heart_rate,
    leads=1
)

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
