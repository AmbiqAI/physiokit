---
title:
---
#

<p align="center">
  <a href="https://github.com/AmbiqAI/physiokit"><img src="./assets/physiokit-banner.png" alt="PhysioKit"></a>
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


**Documentation**: <a href="https://ambiqai.github.io/physiokit" target="_blank">https://ambiqai.github.io/physiokit</a>

**Source Code**: <a href="https://github.com/AmbiqAI/physiokit" target="_blank">https://github.com/AmbiqAI/physiokit</a>

---

**Key Features**

- Multi-signal coverage: ECG, PPG, RSP, IMU, and HRV.
- Wearable-first: robust to noisy, real-time data.
- Batteries included: cleaning, peak detection, metrics, and synthesis.
- Extensible utilities: shared filters, smoothing, FFT, and noise/distortion.

PhysioKit is built to be modular: each signal family has a consistent API for cleaning, peak finding, intervals, and downstream metrics. Shared `pk.signal` primitives keep filtering and resampling uniform so you can mix modalities without rewriting glue code. Synthetic generators let you prototype quickly or build regression data for your own pipelines.

## Requirements

* [Python 3.10+](https://www.python.org)

## Installation

Installing PhysioKit can be done using `uv` or `pip`.

=== "via uv"

    <div class="termy">

    ```console
    $ uv add physiokit

    ---> 100%
    ```
    </div>

=== "via pip"

    <div class="termy">

    ```console
    $ pip install physiokit

    ---> 100%
    ```
    </div>

## Quickstart

=== "ECG: synth → clean → HR/HRV"

    ```python
    import numpy as np
    import physiokit as pk

    fs = 500
    ecg, segs, fids = pk.ecg.synthesize(signal_length=5*fs, sample_rate=fs, heart_rate=70, leads=1)
    ecg = ecg.squeeze()

    ecg_clean = pk.ecg.clean(ecg, sample_rate=fs)
    peaks = pk.ecg.find_peaks(ecg_clean, sample_rate=fs)
    rri = pk.ecg.compute_rr_intervals(peaks)
    mask = pk.ecg.filter_rr_intervals(rri, sample_rate=fs)

    hr_bpm, _ = pk.ecg.compute_heart_rate(ecg_clean, sample_rate=fs)
    hrv_td = pk.hrv.compute_hrv_time(rri[mask == 0], sample_rate=fs)
    ```

=== "PPG: HR + SpO₂"

    ```python
    import numpy as np
    import physiokit as pk

    fs = 100
    t = np.arange(0, 10, 1/fs)
    ppg = np.sin(2*np.pi*1.1*t)

    hr_bpm, qos = pk.ppg.compute_heart_rate(ppg, sample_rate=fs, method="peak")
    spo2 = pk.ppg.compute_spo2_in_time(ppg, ppg, sample_rate=fs)
    ```

=== "RSP: rate from peaks"

    ```python
    import numpy as np
    import physiokit as pk

    fs = 25
    t = np.arange(0, 40, 1/fs)
    rsp = np.sin(2*np.pi*0.2*t)

    bpm, qos = pk.rsp.compute_respiratory_rate(rsp, sample_rate=fs, method="peak")
    ```

=== "Signal helpers"

    ```python
    import numpy as np
    import physiokit as pk

    fs = 100
    t = np.arange(0, 5, 1/fs)
    sig = np.sin(2*np.pi*2*t) + 0.2*np.random.randn(t.size)

    clean = pk.signal.filter_signal(sig, lowcut=0.5, highcut=20, sample_rate=fs)
    freqs, sp = pk.signal.compute_fft(clean, sample_rate=fs)
    ```


## License

This project is licensed under the terms of BSD 3-Clause.
