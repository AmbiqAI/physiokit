# Signals Overview

physioKIT bundles end-to-end utilities for common wearable signals—ECG, PPG, respiratory (RSP), inertial (IMU), and derived HRV. Each module ships cleaning, peak detection, metrics, and synthetic data helpers built on a consistent API.

## Modules at a Glance

- **[ECG](./ecg.md)**: Cleaning, R-peak detection, RR intervals, HR/respiration, segmentation, and synthetic ECG.
- **[PPG](./ppg.md)**: Peak detection, HR/respiration from peaks or FFT, SpO₂ estimation, and PPG synthesis.
- **[RSP](./rsp.md)**: Respiratory peak detection, rate estimation (FFT/peaks), and dual-band ribcage/abdomen metrics.
- **[IMU](./imu.md)**: ENMO, tilt angles, and activity counts for accelerometer data.
- **[HRV](./hrv.md)**: Time and frequency-domain HRV metrics and supporting dataclasses.
- **[Signal](./signal.md)**: Shared filters, resampling, smoothing, noise/distortion helpers, and FFT utilities.

## Quickstart Examples

??? example "Compute heart rate from ECG peaks"

    ```python
    import numpy as np
    import physiokit as pk

    fs = 250
    t = np.arange(0, 10, 1 / fs)
    ecg = np.sin(2 * np.pi * 1.2 * t)  # toy signal

    bpm, qos = pk.ecg.compute_heart_rate(ecg, sample_rate=fs, method="peak")
    print(bpm, qos)
    ```

??? example "Find PPG peaks and estimate SpO₂"

    ```python
    import numpy as np
    import physiokit as pk

    fs = 100
    t = np.arange(0, 8, 1 / fs)
    ppg = np.sin(2 * np.pi * 1.1 * t)

    peaks = pk.ppg.find_peaks(ppg, sample_rate=fs)
    spo2 = pk.ppg.compute_spo2_in_time(ppg, ppg, sample_rate=fs)
    print(len(peaks), spo2)
    ```

??? example "Respiratory rate from RSP using FFT"

    ```python
    import numpy as np
    import physiokit as pk

    fs = 25
    t = np.arange(0, 40, 1 / fs)
    rsp = np.sin(2 * np.pi * 0.2 * t)

    bpm, qos = pk.rsp.compute_respiratory_rate(rsp, sample_rate=fs, method="fft")
    print(bpm, qos)
    ```

??? example "IMU ENMO and tilt"

    ```python
    import numpy as np
    import physiokit as pk

    x = np.array([0, 0.1, 0.2])
    y = np.array([0, 0.1, 0.2])
    z = np.array([1, 1.0, 1.1])

    enmo = pk.imu.compute_enmo(x, y, z)
    angles = pk.imu.compute_tilt_angles(x, y, z, in_radians=False)
    print(enmo, angles)
    ```

??? example "Signal helpers: filter and FFT"

    ```python
    import numpy as np
    import physiokit as pk

    fs = 100
    t = np.arange(0, 5, 1 / fs)
    sig = np.sin(2 * np.pi * 2 * t) + 0.2 * np.random.randn(t.size)

    clean = pk.signal.filter_signal(sig, lowcut=0.5, highcut=20, sample_rate=fs)
    freqs, sp = pk.signal.compute_fft(clean, sample_rate=fs)
    print(freqs[np.argmax(np.abs(sp))])
    ```
