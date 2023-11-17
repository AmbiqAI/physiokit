# Respiratory (RSP)

Respiratory rate is often measured on the chest using a respiration belt or a respiratory inductance plethysmography (RIP) sensor. PhysioKit provides a set of functions to process RSP signals. The functions can be used to generate synthetic RSP signals, clean noisy RSP signals, extract respiratory peaks, compute respiratory rate, and compute dual band metrics.

## Synthetic RSP

We can generate a synthetic RSP signal using the `rsp.synthesize` function. The function returns a numpy array with the RSP signal. The `duration` parameter specifies the length of the signal in seconds. The `sample_rate` parameter specifies the sampling rate in Hz. The `respiratory_rate` parameter specifies the respiratory rate in breaths per minute (BPM). The function returns a numpy array with the RSP signal.

???+ example
    In the following snippet, we generate a synthetic RSP inductance band signal with a respiratory rate of 12 BPM sampled at 1000 Hz.

    ```python
    import physiokit as pk

    fs = 1000 # Hz
    tgt_rr = 12 # BPM

    rsp = pk.rsp.synthesize(
        duration=10,
        sample_rate=fs,
        respiratory_rate=tgt_rr,
        amplitude=1
    )
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk-synthetic-rsp-raw.html"
    </div>

---

## Noise Injection

We can additionally add noise to generate a more realistic RSP signal.

???+ example

    Below adds baseline wander, powerline noise, and custom noise sources to the synthetic RSP signal.

    ```python
    # Add baseline wander
    rsp_noise = pk.signal.add_baseline_wander(
        data=rsp,
        amplitude=2,
        frequency=.05,
        sample_rate=fs
    )

    # Add powerline noise
    rsp_noise = pk.signal.add_powerline_noise(
        data=rsp_noise,
        amplitude=0.05,
        frequency=60,
        sample_rate=fs
    )

    # Add additional noise sources
    rsp_noise = pk.signal.add_noise_sources(
        data=rsp_noise,
        amplitudes=[0.05, 0.05],
        frequencies=[10, 20],
        sample_rate=fs
    )
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk-synthetic-rsp-noise.html"
    </div>

---

## Sanitize RSP

We can clean the RSP signal using the `rsp.clean` function. By default, the routine implements a bandpass filter with cutoff frequencies of 0.05 Hz and 3 Hz. The `lowcut` and `highcut` parameters can be used to specify the cutoff frequencies. The `order` parameter specifies the order of the filter. The `sample_rate` parameter specifies the sampling rate in Hz. The function returns a numpy array with the cleaned RSP signal.

???+ example

    In the following snippet, we clean the noisy RSP signal using a bandpass filter with cutoff frequencies of 0.05 Hz and 3 Hz.

    ```python
    ...

    # Clean RSP signal
    rsp_clean = pk.rsp.clean(
        data=rsp_noise,
        lowcut=0.05,
        highcut=3,
        order=3,
        sample_rate=fs
    )
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk-synthetic-rsp-clean.html"
    </div>

## Extract Respiratory Peaks

A common task in RSP processing is to extract respiratory peaks. The `rsp.find_peaks` function implements a simple peak detection algorithm. The function returns a numpy array with the indices of the peaks.

???+ example

    In the following snippet, we extract respiratory peaks from the cleaned RSP signal.

    ```python
    ...

    # Extract respiratory cycles
    peaks = pk.rsp.find_peaks(data=rsp_clean, sample_rate=fs)

    # Compute RR-intervals
    rri = pk.rsp.compute_rr_intervals(peaks=peaks)

    # Filter RR-intervals
    mask = pk.rsp.filter_rr_intervals(rr_ints=rri, sample_rate=fs)

    # Keep normal RR-intervals
    peaks_clean = peaks[mask == 0]
    rri_clean = rri[mask == 0]
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk-synthetic-rsp-rpeak.html"
    </div>

---

## Compute Respiratory Rate

The `rsp.compute_respiratory_rate` function computes the respiratory rate in breaths per minute (BPM) based on the selected `method`. The `peak` method computes respiratory rate based on identified respiratory peaks whereas `fft` method uses FFT to compute the respiratory rate. The function returns the respiratory rate in breaths per minute (BPM) along with a "quality of signal" (QoS) metric. The QoS metric is based on the selected `method`.

???+ example

    In the following snippet, we compute the respiratory rate based on the identified respiratory peaks.

    ```python
    # Compute respiratory rate
    rr_bpm, rr_qos = pk.rsp.compute_respiratory_rate(
        data=rsp_clean,
        method="fft",
        sample_rate=fs,
        lowcut=0.05,
        highcut=1
    )
    ```
    __OUTPUT:__ <br>
    :lungs:{ .lungs } __Respiratory Rate: 12 BPM__

---

## Compute Dual Band Metrics

Using dual RIP bands, a ribcage (RC) band and a abdominal (AB) band, we can compute additional respiratory metrics. The `rsp.compute_dual_band_metrics` function computes the following metrics:

| Metric | Description |
| --- | --- |
| **rc_rr** | RC respiratory rate (BPM) |
| **ab_rr** | AB respiratory rate (BPM) |
| **vt_rr** | VT respiratory rate (BPM) |
| **phase** | Phase angle (degrees) |
| **lbi** | Labored breathing index |
| **rc_lead** | RC leads AB |
| **rc_percent** | Percent RC contribution |
| **qos** | Quality of signal (0-1) |
| **rc_pk_freq** | RC peak frequency (Hz) |
| **rc_pk_pwr** | RC peak power |
| **ab_pk_freq** | AB peak frequency (Hz) |
| **ab_pk_pwr** | AB peak power |
| **vt_pk_freq** | VT peak frequency (Hz) |
| **vt_pk_pwr** | VT peak power |

???+ example

    In the following example, we generate synthetic RC and AB band data, compute the dual band metrics, and plot the results. We compute the metrics over a sliding window with a length of 10 seconds and an overlap of 1 second.

    ```python
    fs = 1000
    tgt_rr = 12.2
    rc_amp = 1.5
    ab_amp = 1.0
    dur_sec = 60
    win_len = 10*fs
    ovl_len = 1*fs

    # Synthesize RC and AB band data
    rc = rc_amp*pk.rsp.synthesize(
        duration=dur_sec,
        sample_rate=fs,
        respiratory_rate=tgt_rr
    )
    ab = ab_amp*pk.rsp.synthesize(
        duration=dur_sec,
        sample_rate=fs,
        respiratory_rate=tgt_rr
    )

    # Compute metrics over sliding window
    ts_metrics, dual_metrics = [], []
    for i in range(0, rc.size - win_len, ovl_len):
        rc_win = rc[i:i+win_len]
        ab_win = ab[i:i+win_len]
        ts_metrics.append((i+win_len/2)/fs)
        dual_metrics.append(pk.rsp.compute_dual_band_metrics(
            rc=rc_win,
            ab=ab_win,
            sample_rate=fs,
            pwr_threshold=0.9
        ))

    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk_rsp_dual_metrics.html"
    </div>

---

## API

[Refer to RSP API for more details](../api/rsp.md)
