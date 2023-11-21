# Signal Processing

The `signal` submodule contains lots of underlying signal processing functionality used by the higher-level modules. This includes filtering routines, noise injection, and transformations.

## Filtering

`signal.filter` contains a number of filtering routines such as a generic second order section biquad filter, resampling, normalizing, smoothing, and more. Below we highlight a few of these routines.

| Function | Description |
| --- | --- |
| pk.signal.resample_signal | Resamples a signal to a new sampling rate. |
| pk.signal.resample_categorical | Resamples a categorical signal to a new sampling rate. |
| pk.signal.normalize_signal | Normalizes a signal to a new range. |
| pk.signal.filter_signal | Filters a signal using a second order section biquad filter. |
| pk.signal.remove_baseline_wander | Removes baseline wander from a signal. |
| pk.signal.smooth_signal | Smooths a signal using a Savitzky-Golay filter. |
| pk.signal.quotient_filter_mask | Applies a quotient filter to identify outliers from list. |

???+ example

    In the following snippet, we generate a synthetic PPG signal with a heart rate of 64 BPM sampled at 1000 Hz. We then filter the signal using a second order section biquad filter.

    ```python
    import physiokit as pk

    fs = 1000  # Hz
    tgt_hr = 64  # BPM

    ppg = pk.ppg.synthesize(
        duration=10,  # in seconds
        sample_rate=fs,
        heart_rate=tgt_hr,
    )

    ppg_clean = pk.signal.filter_signal(
        data=ppg_noise,
        lowcut=0.5,
        highcut=4,
        order=3,
        sample_rate=fs
        forward_backward=True
    )
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk-synthetic-ppg-raw.html"
    </div>

## Noise Injection

`signal.noise` contains a number of noise injection routines such as Gaussian noise, uniform noise, and more. Below we highlight a few of these routines.

| Function | Description |
| --- | --- |
| pk.signal.add_baseline_wander | Adds baseline wander to a signal. |
| pk.signal.add_motion_noise | Adds motion noise to a signal. |
| pk.signal.add_burst_noise | Adds burst noise to a signal. |
| pk.signal.add_powerline_noise | Adds powerline noise to a signal. |
| pk.signal.add_noise_sources | Adds additional noise sources to a signal. |

???+ example

    In the following snippet, we generate a synthetic PPG signal and inject the following noises: baseline wander, powerline noise, and custom noise sources.

    ```python
    import physiokit as pk

    fs = 1000  # Hz
    tgt_hr = 64  # BPM

    ppg = pk.ppg.synthesize(
        duration=10,  # in seconds
        sample_rate=fs,
        heart_rate=tgt_hr,
    )

    # Add baseline wander
    ppg_noise = pk.signal.add_baseline_wander(
        data=ppg,
        amplitude=2,
        frequency=1,
        sample_rate=fs
    )

    # Add powerline noise
    ppg_noise = pk.signal.add_powerline_noise(
        data=ppg_noise,
        amplitude=0.05,
        frequency=60,
        sample_rate=fs
    )

    # Add additional noise sources
    ppg_noise = pk.signal.add_noise_sources(
        data=ppg_noise,
        amplitudes=[0.05, 0.05],
        frequencies=[10, 20],
        sample_rate=fs
    )
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk-synthetic-ppg-noise.html"
    </div>

---

## Transformations

`signal.transform` contains a number of transformation routines such as the fast Fourier transform, the inverse fast Fourier transform, and more. Below we highlight a few of these routines.

---

## API

[Refer to signal API for more details](../api/signal.md)
