import numpy as np
import numpy.typing as npt
import scipy.interpolate


def synthesize(
    duration: float = 10,
    sample_rate: float = 1000,
    heart_rate: int = 60,
    frequency_modulation: float = 0,
    ibi_randomness: float = 0,
) -> npt.NDArray:
    """Generate synthetic PPG signal. Utilize pk.signal.noise methods to make more realistic.

    Args:
        duration (float, optional): Duration in seconds. Defaults to 10.
        sample_rate (float, optional): Sample rate in Hz. Defaults to 1000 Hz.
        heart_rate (int, optional): Heart rate in BPM. Defaults to 60 BPM.
        frequency_modulation (float, optional): Frequency modulation in Hz. Defaults to 0 Hz.
        ibi_randomness (float, optional): IBI randomness in Hz. Defaults to 0 Hz.

    Returns:
        npt.NDArray: Synthetic PPG signal
    """

    period = 60 / heart_rate  # in seconds
    n_period = int(np.floor(duration / period))
    periods = np.ones(n_period) * period

    # Seconds at which waves begin.
    x_onset = np.cumsum(periods)
    x_onset -= x_onset[0]  # make sure seconds start at zero
    # Add respiratory sinus arrythmia (frequency modulation).
    periods, x_onset = _frequency_modulation(
        periods,
        x_onset,
        modulation_frequency=0.05,
        modulation_strength=frequency_modulation,
    )
    # Randomly modulate duration of waves by subracting a random value between
    # 0 and ibi_randomness% of the wave duration (see function definition).
    x_onset = _random_x_offset(x_onset, ibi_randomness)
    # Corresponding signal amplitudes.
    y_onset = np.random.normal(0, 0.1, n_period)

    # Seconds at which the systolic peaks occur within the waves.
    x_sys = x_onset + np.random.normal(0.175, 0.01, n_period) * periods
    # Corresponding signal amplitudes.
    y_sys = y_onset + np.random.normal(1.5, 0.15, n_period)

    # Seconds at which the dicrotic notches occur within the waves.
    x_notch = x_onset + np.random.normal(0.4, 0.001, n_period) * periods
    # Corresponding signal amplitudes (percentage of systolic peak height).
    y_notch = y_sys * np.random.normal(0.49, 0.01, n_period)

    # Seconds at which the diastolic peaks occur within the waves.
    x_dia = x_onset + np.random.normal(0.45, 0.001, n_period) * periods
    # Corresponding signal amplitudes (percentage of systolic peak height).
    y_dia = y_sys * np.random.normal(0.51, 0.01, n_period)

    x_all = np.concatenate((x_onset, x_sys, x_notch, x_dia))
    x_all.sort(kind="mergesort")
    x_all = np.ceil(x_all * sample_rate).astype(int)  # convert seconds to samples

    y_all = np.zeros(n_period * 4)
    y_all[0::4] = y_onset
    y_all[1::4] = y_sys
    y_all[2::4] = y_notch
    y_all[3::4] = y_dia

    # Interpolate a continuous signal between the landmarks (i.e., Cartesian coordinates).
    samples = np.arange(int(np.ceil(duration * sample_rate)))
    interp_function = scipy.interpolate.Akima1DInterpolator(x_all, y_all)
    ppg = interp_function(samples)

    # Remove NAN (values outside interpolation range, i.e., after last sample).
    ppg[np.isnan(ppg)] = np.nanmean(ppg)
    return ppg


def _frequency_modulation(periods, seconds, modulation_frequency, modulation_strength):
    """modulator_frequency determines the frequency at which respiratory sinus arrhythmia occurs (in Hz).

    modulator_strength must be between 0 and 1.

    """
    modulation_mean = 1
    # Enforce minimum inter-beat-interval of 300 milliseconds.
    if (modulation_mean - modulation_strength) * periods[
        0
    ] < 0.3:  # elements in periods all have the same value at this point
        print(
            "Skipping frequency modulation, since the modulation_strength"
            f" {modulation_strength} leads to physiologically implausible"
            f" wave durations of {((modulation_mean - modulation_strength) * periods[0]) * 1000}"
            f" milliseconds."
        )
        return periods, seconds

    # Apply a very conservative Nyquist criterion.
    nyquist = (1 / periods[0]) * 0.1
    if modulation_frequency > nyquist:
        print(f"Please choose a modulation frequency lower than {nyquist}.")

    # Generate a sine with mean 1 and amplitude 0.5 * modulation_strength, that is,
    # ranging from 1 - 0.5 * modulation_strength to 1 + 0.5 * modulation_strength.
    # For example, at a heart rate of 100 and modulation_strenght=1, the heart rate will
    # fluctuate between 150 and 50. At the default modulatiom_strenght=.2, it will
    # fluctuate between 110 and 90.
    modulator = 0.5 * modulation_strength * np.sin(2 * np.pi * modulation_frequency * seconds) + modulation_mean
    periods_modulated = periods * modulator
    seconds_modulated = np.cumsum(periods_modulated)
    seconds_modulated -= seconds_modulated[0]  # make sure seconds start at zero

    return periods_modulated, seconds_modulated


def _random_x_offset(x, offset_weight):
    """From each wave onset xi subtract offset_weight * (xi - xi-1) where xi-1 is
    the wave onset preceding xi. offset_weight must be between 0 and 1.
    """
    # Sanitize offset to min 0 and max .99
    offset_weight = min(offset_weight, 0.99)
    offset_weight = max(offset_weight, 0)

    x_diff = np.diff(x)
    # Enforce minimum inter-beat-interval of 300 milliseconds.
    min_x_diff = min(x_diff)
    if (min_x_diff - (min_x_diff * offset_weight)) < 0.3:
        print(
            "Skipping random IBI modulation, since the offset_weight"
            f" {offset_weight} leads to physiologically implausible wave"
            f" durations of {(min_x_diff - (min_x_diff * offset_weight)) * 1000}"
            f" milliseconds."
        )
        return x

    max_offsets = offset_weight * x_diff
    offsets = [np.random.uniform(0, i) for i in max_offsets]

    x_offset = x.copy()
    x_offset[1:] -= offsets

    return x_offset


def _amplitude_modulation():
    pass
