# Quick Start

## Installation

Currently, the package is available on PyPI as a universal wheel for Python 3.11+. Installing PhysioKit can be done using `Poetry` or `pip`.

=== "via Poetry"

    <div class="termy">

    ```console
    $ poetry add physiokit

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

???+ example Example

    In this example, we will generate a synthetic ECG signal, clean it, and compute heart rate and HRV metrics.

    ```python

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

<div class="sk-plotly-graph-div">
--8<-- "assets/pk-synthetic-ecg-clean.html"
</div>
