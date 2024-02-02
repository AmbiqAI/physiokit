# Photoplethysmography (PPG)

Photoplethysmography (PPG) is a non-invasive optical technique used to measure blood volume changes in the microvascular bed of tissue. PPG signals are often used to measure heart rate, heart rate variability (HRV), respiratory rate, and oxygen saturation (SpO2). In PhysioKit, we provide a variety of routines for processing PPG signals.

## Synthetic PPG

We can generate a synthetic PPG signal using the `ppg.synthesize` function. The function returns a numpy array with the PPG signal. The `duration` parameter specifies the length of the signal in seconds. The `sample_rate` parameter specifies the sampling rate in Hz. The `heart_rate` parameter specifies the heart rate in beats per minute (BPM).

???+ example

    In the following snippet, we generate a synthetic PPG signal with a heart rate of 64 BPM sampled at 1000 Hz.

    ```python
    import physiokit as pk

    sample_rate = 1000  # Hz
    heart_rate = 64  # BPM
    signal_length = 10*sample_rate # 10 seconds

    ppg, segs, fids = pk.ppg.synthesize(
        signal_length=signal_length,
        sample_rate=sample_rate,
        heart_rate=heart_rate,
        frequency_modulation=0.3,
        ibi_randomness=0.1
    )
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk-synthetic-ppg-raw.html"
    </div>

---

## Noise Injection

We can additionally add noise to generate a more realistic PPG signal.

???+ example

    Given the previous synthetic PPG signal, we can add baseline wander, powerline noise, and custom noise sources.

    ```python
    # Add baseline wander
    ppg_noise = pk.signal.add_baseline_wander(
        data=ppg,
        amplitude=1,
        frequency=0.5,
        sample_rate=sample_rate
    )

    # Add powerline noise
    ppg_noise = pk.signal.add_powerline_noise(
        data=ppg_noise,
        amplitude=0.05,
        frequency=60,
        sample_rate=sample_rate
    )

    # Add additional noise sources
    ppg_noise = pk.signal.add_noise_sources(
        data=ppg_noise,
        amplitudes=[0.05, 0.05],
        frequencies=[10, 20],
        noise_shapes=["laplace", "laplace"],
        sample_rate=sample_rate
    )
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk-synthetic-ppg-noise.html"
    </div>

---

## Sanitize PPG

PPG signals are often corrupted by noise. The `ppg.clean` function provides a simple way to remove noise from PPG signals. By default, the routine implements a bandpass filter with cutoff frequencies of 0.5 Hz and 4 Hz. The `lowcut` and `highcut` parameters can be used to specify the cutoff frequencies. The `order` parameter specifies the order of the filter. The `sample_rate` parameter specifies the sampling rate in Hz. The function returns a numpy array with the cleaned PPG signal.

???+ example

    In the following snippet, we clean the noisy PPG signal using a bandpass filter with cutoff frequencies of 0.5 Hz and 4 Hz.

    ```python
    ...

    # Clean PPG signal
    ppg_clean = pk.ppg.clean(
        data=ppg_noise,
        lowcut=0.5,
        highcut=4,
        order=3,
        sample_rate=sample_rate
    )
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk-synthetic-ppg-clean.html"
    </div>

---

## Extract R-Peaks and RR-Intervals

A common task in PPG processing is to extract systolic peaks and peak-to-peak intervals. This has a 1:1 correspondance with r-peaks and therefore can be used to compute heart rate and HRV.

???+ example

    In the following snippet, we extract systolic peaks and peak-to-peak intervals from the cleaned PPG signal.

    ```python
    ...

    # Extract s-peaks
    peaks = pk.ppg.find_peaks(data=ppg_clean, sample_rate=sample_rate)

    # Compute peak-to-peak intervals
    rri = pk.ppg.compute_rr_intervals(peaks=peaks)

    # Identify abnormal RR-intervals (e.g., ectopic beats)
    # Mask is a boolean array where 0 indicates a normal RR-interval
    mask = pk.ppg.filter_rr_intervals(rr_ints=rri, sample_rate=sample_rate)

    # Keep normal RR-intervals
    peaks_clean = peaks[mask == 0]
    rri_clean = rri[mask == 0]
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk-synthetic-ppg-rpeak.html"
    </div>

---

## Compute Heart Rate

The `ppg.compute_heart_rate` function computes the heart rate based on the selected `method`. The `peak` method computes heart rate based on identified systolic peaks whereas `fft` method uses FFT to compute the heart rate. The function returns the heart rate in beats per minute (BPM) along with a "quality of signal" (QoS) metric. The QoS metric is a value between 0 and 1 where 1 indicates a high quality signal and 0 indicates a low quality signal. The QoS metric is based on the selected `method`.

???+ example

    Continuing from the previous example, we can compute the heart rate using the `peak` method.

    ```python
    # Compute heart rate using FFT
    hr_bpm, hr_qos = pk.ppg.compute_heart_rate(
        ppg_clean,
        method="fft",
        sample_rate=sample_rate
    )
    ```

    __OUTPUT__: <br>
    :heart:{ .heart } __Heart Rate: 64 BPM__

---

## Compute HRV Metrics

Refer to [HRV Documentation](./hrv.md) for computing HRV metrics based on systolic peaks.

---

## Derive Respiratory Rate

Respiratory sinus arrhythmia (RSA) is a phenomenon where the heart rate varies with respiration. Specifically, during inspiration, the heart rate increases and during expiration, the heart rate decreases. This is due to the parasympathetic nervous system which is responsible for slowing the heart rate. The `ppg.compute_respiratory_rate` function computes the respiratory rate based on the RR-intervals. The function returns the respiratory rate in breaths per minute (BPM) along with a "quality of signal" (QoS) metric. The QoS metric is a value between 0 and 1 where 1 indicates a high quality signal and 0 indicates a low quality signal. The QoS metric is based on the selected `method`. Beyond modulating the peak-to-peak intervals, respiration also modulates the amplitude of the PPG signal.


???+ example

    In the following snippet, we derive the respiratory rate from the PPG signal.

    ```python
    ...

    # Compute respiratory rate using RIFV method
    rr_bpm, rr_qos = pk.ppg.derive_respiratory_rate(
        ppg=peaks[mask == 0],
        peaks=peaks[mask == 0],
        rri=rri[mask == 0],
        method="rifv",
        sample_rate=sample_rate
    )
    ```
    __OUTPUT__: <br>
    :lungs:{ .lungs } __Respiratory Rate: 18 BPM__

???+ note
    In certain conditions such as noisy environments, certain subjects, PPG position, etc., the respiratory rate may not be accurately estimated. In these cases, the QoS metric will be low.

---

## Compute SpO2

Using two PPG signals with different wavelengths, we can compute the oxygen saturation (SpO2). In PhysioKit, we can compute SpO2 in time domain using `ppg.compute_spo2_in_time` or frequency domain using `ppg.compute_spo2_in_frequency`.

???+ example

    In the following snippet, we compute SpO2 in time and frequency domain from PPG signals captured by the MAX8614x sensor.

    ```python
    max8614x_coefs = [-16.666666, 8.333333, 100]

    # Load PPG signals
    ppg_red = ...
    ppg_ir = ...

    # NOTE: Pass the raw PPG signals as routines need to extract DC component

    # Compute SpO2 in time domain
    spo2_td = pk.ppg.compute_spo2_in_time(
        ppg1=ppg_red,
        ppg2=ppg_ir,
        coefs=max8614x_coefs,
        lowcut=0.5,
        highcut=4,
        sample_rate=sample_rate
    )

    # Compute SpO2 in frequency domain
    spo2_fd = pk.ppg.compute_spo2_in_frequency(
        ppg1=ppg_red,
        ppg2=ppg_ir,
        coefs=max8614x_coefs,
        lowcut=0.5,
        highcut=4,
        sample_rate=sample_rate
    )
    ```
    __OUTPUT__: <br>
    :drop_of_blood:{ .heart } __SpO2: 98%__

!!! note
    When calling `ppg.compute_spo2_in_time` or `ppg.compute_spo2_in_frequency`, the `coefs` parameter must be specified. The coefficients are used to convert the PPG signals to SpO2 via a 2nd order polynomial. The coefficients are specific to the PPG sensor and should be provided by the manufacturer.

---

## API

[Refer to PPG API for more details](../api/ppg.md)
