# Heart Rate Variability (HRV)

Heart rate variability (HRV) is the variation in the time interval between consecutive heartbeats. HRV is a measure of the autonomic nervous system (ANS) and is often used as a proxy for stress. HRV is also used to assess the risk of cardiovascular disease and sudden cardiac death.

## Compute Time-Domain HRV

The `hrv.compute_hrv_time` function computes time-domain HRV metrics based on the supplied RR-intervals. The RR-intervals can be computed from ECG or PPG signals.

| Metric | Description |
| --- | --- |
| **mean_nn** | Mean of NN intervals |
| **sd_nn** | Standard deviation of NN intervals |
| **rms_sd** | Root mean square of successive differences |
| **sd_sd** | Standard deviation of successive differences |
| **cv_nn** | Coefficient of variation of NN intervals |
| **cv_sd** | Coefficient of variation of successive differences |
| **median_nn** | Median of NN intervals |
| **mad_nn** | Median absolute deviation of NN intervals |
| **mcv_nn** | Median coefficient of variation of NN intervals |
| **iqr_nn** | Interquartile range of NN intervals |
| **prc20_nn** | 20th percentile of NN intervals |
| **prc80_nn** | 80th percentile of NN intervals |
| **nn50** | Number of pairs of successive NN intervals that differ by more than 50 ms |
| **nn20** | Number of pairs of successive NN intervals that differ by more than 20 ms |
| **pnn50** | Proportion of NN50 divided by total number of NN intervals |
| **pnn20** | Proportion of NN20 divided by total number of NN intervals |
| **min_nn** | Minimum of NN intervals |
| **max_nn** | Maximum of NN intervals |

???+ example

    In the following snippet, we generate a synthetic ECG signal, extract its r peaks and RR intervals, and finally compute time-domain HRV metrics.

    ```python
    tgt_hr = 64 # BPM

    # Generate synthetic ECG signal
    ecg = pk.ecg.synthesize(
        duration=8,
        sample_rate=fs,
        heart_rate=tgt_hr,
        leads=1
    )

    # Create timestamps
    ts = np.arange(0, ecg.size) / fs

    # Clean ECG signal
    ecg_clean = pk.ecg.clean(
        data=ecg_noise,
        lowcut=2,
        highcut=30,
        order=5,
        sample_rate=fs
    )

    # Extract R-peaks and RR-intervals
    peaks = pk.ecg.find_peaks(ecg_clean, sample_rate=fs)
    rri = pk.ecg.compute_rr_intervals(peaks)
    mask = pk.ecg.filter_rr_intervals(rri, sample_rate=fs)

    # Compute HRV metrics
    hrv_td = pk.hrv.compute_hrv_time(
        rr_intervals=rri[mask == 0],
        sample_rate=fs
    )
    ```

---

## Compute Frequency-Domain HRV

The `hrv.compute_hrv_frequency` function computes frequency-domain HRV metrics based on the supplied RR-intervals. The RR-intervals can be computed from ECG or PPG signals.

???+ example

    Continuing from the previous example, we can also compute frequency-domain HRV metrics.

    ```python
    ...

    bands = [(0.04, 0.15), (0.15, 0.4), (0.4, 0.5)]
    hrv_fd = pk.hrv.compute_hrv_frequency(
        peaks[mask == 0], rri[mask == 0],
        bands=bands,
        sample_rate=fs
    )
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/hrv_frequency_power.html"
    </div>

---

## API

[Refer to HRV API for more details](../api/hrv.md)
