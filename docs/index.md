#

[![](./assets/physiokit-logo-light.png#only-light)](https://ambiqai.github.io/physiokit/)
[![](./assets/physiokit-logo-dark.png#only-dark)](https://ambiqai.github.io/physiokit/)

*An AI Development Kit (ADK) for physiological signal processing in ambulatory health monitoring and wearable technology.*

[![PyPI](https://img.shields.io/pypi/v/physiokit?color=%2334D058&label=PyPI)](https://pypi.org/project/physiokit/)
[![Python](https://img.shields.io/badge/python-3.12%2B-34D058)](https://pypi.org/project/physiokit/)
[![Downloads](https://img.shields.io/pypi/dm/physiokit.svg?color=%2334D058)](https://pypi.org/project/physiokit/)
[![GitHub Stars](https://img.shields.io/github/stars/AmbiqAI/physiokit.svg?color=%2334D058)](https://github.com/AmbiqAI/physiokit)
[![License](https://img.shields.io/pypi/l/physiokit)](https://github.com/AmbiqAI/physiokit/blob/main/LICENSE)

## Overview

physioKIT is an AI Development Kit (ADK) for physiological signal processing in ambulatory health monitoring and wearable technology. It offers tools to process, analyze, and synthesize signals commonly captured by wearable sensors, including ECG, PPG, RSP, and IMU data. Its modular design and consistent APIs help researchers and developers integrate advanced signal processing and feature extraction into applications, speeding up development and deployment of health monitoring solutions.

**Key Features**

- **Multi-signal coverage**: ECG, PPG, RSP, IMU, and HRV out of the box.
- **End-to-end pipeline**: Cleaning, peak detection, intervals, metrics, and synthesis.
- **Shared utilities**: Filtering, smoothing, FFT, resampling, distortion/noise.
- **Synthetic data**: Generate realistic signals for testing and benchmarking.
- **Realtime-friendly**: Functions designed for streaming and incremental use.

## Getting Started

- **Install** physioKIT with pip and get running in minutes. &nbsp; [:material-clock-fast: Install physioKIT](./tutorial/quickstart.md#installation){ .md-button }
- **Check out** quickstart examples for ECG, PPG, RSP, and IMU. &nbsp; [:material-rocket-launch: Quickstart Examples](./reference/index.md){ .md-button }
- **Explore** the [API Reference](api/physiokit) for detailed docs. &nbsp; [:material-book-open-page-variant: API Reference](api/physiokit){ .md-button }


## Installation

Install physioKIT using `uv` or `pip`.

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

## Quick links

- **Docs**: https://ambiqai.github.io/physiokit
- **Reference**: [reference/index.md](reference/index.md)
- **Source**: https://github.com/AmbiqAI/physiokit
