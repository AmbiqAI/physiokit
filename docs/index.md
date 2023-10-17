---
title:
---
#


<p align="center">
  <a href="https://github.com/AmbiqAI/physiokit"><img src="./assets/physiokit-banner.png" alt="PhysioKit"></a>
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
</p>

---


**Documentation**: <a href="https://ambiqai.github.io/physiokit" target="_blank">https://ambiqai.github.io/physiokit</a>

**Source Code**: <a href="https://github.com/AmbiqAI/physiokit" target="_blank">https://github.com/AmbiqAI/physiokit</a>

---


**Key Features:**

* Feature 1
* Feature 2
* Feature 3


## Requirements

* [Python 3.11+](https://www.python.org)


## Installation

<div class="termy">

```console
$ pip install physiokit

---> 100%
```
</div>

Or if using `Poetry`


<div class="termy">

```console
$ poetry add physiokit

---> 100%
```
</div>


## Example

In this example, we will generate a synthetic ECG signal, clean it, and compute heart rate and HRV metrics.


```python

import physiokit as pk

fs = 1000 # Hz
tgt_hr = 64 # BPM


# Generate synthetic ECG signal
ecg = pk.ecg.synthesize(duration=10, sample_rate=fs, heart_rate=tgt_hr, leads=1)

# Clean ECG signal
ecg_clean = pk.ecg.clean(ecg, sample_rate=fs)

# Compute heart rate
hr_bpm, _ = pk.ecg.compute_heart_rate(ecg_clean, sample_rate=fs)

# Extract R-peaks and RR-intervals
peaks = pk.ecg.find_peaks(ecg_clean, sample_rate=fs)
rri = pk.ecg.compute_rr_intervals(peaks)
mask = pk.ecg.filter_rr_intervals(rri, sample_rate=fs)

# Re-compute heart rate
hr_bpm = 60 / (np.nanmean(rri[mask == 0]) / fs)

# Compute HRV metrics
hrv_td = pk.hrv.compute_hrv_time(rri[mask == 0], sample_rate=fs)

bands = [(0.04, 0.15), (0.15, 0.4), (0.4, 0.5)]
hrv_fd = pk.hrv.compute_hrv_frequency(peaks[mask == 0], rri[mask == 0], bands=bands, sample_rate=fs)

```


## License

This project is licensed under the terms of BSD 3-Clause.
