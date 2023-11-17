# Electrocardiography (ECG)

Electrocardiography (ECG) is a non-invasive technique used to measure the electrical activity of the heart. ECG signals are often used to measure heart rate, heart rate variability (HRV), and respiratory rate. In PhysioKit, we provide a variety of routines for processing ECG signals.

## Synthetic ECG

PhysioKit provides a simple way to generate synthetic ECG signals using the `ecg.synthesize` function. By supplying only a handful of parameters, we can generate a fairly authentic ECG signal. PhysioKit also provides a more customizable way to generate synthetic ECG signals using either `ecg.generate_nsr` or `ecg.generate_afib`. In addition to returning the ECG signal, these advanced funtions also return the PQRST segmentations and fiducials.

???+ example
    In the following snippet, we generate a synthetic ECG signal with a heart rate of 64 BPM sampled at 1000 Hz.

    ```python
    import physiokit as pk

    fs = 1000 # Hz
    tgt_hr = 64 # BPM

    ecg = pk.ecg.synthesize(
        duration=10, # in seconds
        sample_rate=fs,
        heart_rate=tgt_hr,
        leads=1 # number of ECG leads
    )
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk-synthetic-ecg-raw.html"
    </div>

---

## Noise Injection

Often real world ECG signals are corrupted by a variaty of noise sources. To generate a more realistic ECG signal, PhysioKit provides a number of noise injection routines such as baseline wander, powerline noise, and custom noise sources.

???+ example

    Given the previous synthetic ECG signal, we can add baseline wander, powerline noise, and custom noise sources.

    ```python
    # Add baseline wander
    ecg = pk.signal.add_baseline_wander(
        data=ecg,
        amplitude=2,
        frequency=1,
        sample_rate=fs
    )

    # Add powerline noise
    ecg = pk.signal.add_powerline_noise(
        data=ecg,
        amplitude=0.05,
        frequency=60,
        sample_rate=fs
    )

    # Add additional noise sources
    ecg = pk.signal.add_noise_sources(
        data=ecg,
        amplitudes=[0.05, 0.05],
        frequencies=[60, 80],
        sample_rate=fs
    )
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk-synthetic-ecg-noise.html"
    </div>

---

## Sanitize ECG

As mentioned above, ECG signals are often corrupted by noise and artifacts. The `ecg.clean` function provides a simple way to remove noise from ECG signals. By default, the routine implements a bandpass filter with cutoff frequencies of 0.5 and 40 Hz.

???+ example

    In the following snippet, we apply a bandpass filter from 2 to 30 Hz to the synthetic ECG signal.

    ```python
    ...

    # Using ECG signal from above ^
    ecg_clean = pk.ecg.clean(
        data=ecg,
        lowcut=2,
        highcut=30,
        order=5,
        sample_rate=fs
    )
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk-synthetic-ecg-clean.html"
    </div>

---

## Extract R-Peaks and RR-Intervals

A common task in ECG processing is to extract R-peaks and RR-intervals. The `ecg.find_peaks` function implements a gradient-based peak detection algorithm to extract R-peaks from a given ECG signal. The `ecg.compute_rr_intervals` function can then be used to compute the corresponding RR-intervals from the R-peaks. Lastly, the `ecg.filter_rr_intervals` function removes RR-intervals that are too short or too long. This can often be due to artifacts such as noise or ectopic beats.

???+ example

    Continuing from the previous example, we can extract R-peaks and RR-intervals from the cleaned ECG signal. We further create a mask to identify abnormal RR-intervals.

    ```python
    ...

    # Extract R-peaks
    peaks = pk.ecg.find_peaks(data=ecg_clean, sample_rate=fs)

    # Compute RR-intervals
    rri = pk.ecg.compute_rr_intervals(peaks=peaks)

    # Identify abnormal RR-intervals (e.g., ectopic beats)
    # Mask is a boolean array where 0 indicates a normal RR-interval
    mask = pk.ecg.filter_rr_intervals(rr_ints=rri, sample_rate=fs)

    # Keep normal RR-intervals
    peaks_clean = peaks[mask == 0]
    rri_clean = rri[mask == 0]
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk-synthetic-ecg-rpeak.html"
    </div>

---

## Compute Heart Rate

The `ecg.compute_heart_rate` function computes the heart rate based on the selected `method`. Currently this is done based on the RR-intervals (`method="peak"`). The function returns the heart rate in beats per minute (BPM) along with a "quality of signal" (QoS) metric. The QoS metric is a value between 0 and 1 where the higher the value, the higher the quality of the signal.

???+ example

    ```python
    # Compute heart rate
    hr_bpm, hr_qos = pk.ecg.compute_heart_rate(
        ecg_clean,
        method="peak",
        sample_rate=fs
    )
    ```

    __OUTPUT__: <br>
    :heart:{ .heart } __Heart Rate: 64 BPM__

---

## Compute HRV Metrics

Refer to [HRV Documentation](./hrv.md) for computing HRV metrics based on RR-intervals.

---

## Derive Respiratory Rate

Respiratory sinus arrhythmia (RSA) is a phenomenon where the heart rate varies with respiration. Specifically, during inspiration, the heart rate increases and during expiration, the heart rate decreases. This is due to the parasympathetic nervous system which is responsible for slowing the heart rate. The `ecg.derive_respiratory_rate` function derives the respiratory rate based on the selected method. The default method (`rifv`) looks at frequency modulation of the RR intervals. The function returns the respiratory rate in breaths per minute (BPM) along with a "quality of signal" (QoS) metric.

???+ example

    Once again, continuing from the previous example, we derive the respiratory rate from the ECG signal.

    ```python
    ...

    # Compute respiratory rate using RIFV method
    rr_bpm, rr_qos = pk.ecg.derive_respiratory_rate(
        peaks=peaks[mask == 0],
        rri=rri[mask == 0],
        method="rifv",
        sample_rate=fs
    )
    ```
    __OUTPUT__: <br>
    :lungs:{ .lungs } __Respiratory Rate: 16 BPM__

!!! note
    In certain conditions such as noisy environments, certain subjects, ECG lead position, etc., the respiratory rate may not be accurately estimated. In these cases, the QoS metric will be low.

## ECG Segmentation

Coming soon...

---

## API

[Refer to ECG API for more details](../api/ecg.md)
