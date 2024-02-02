import numpy as np
import numpy.typing as npt
import scipy.interpolate

from .defines import PpgFiducial, PpgSegment


def synthesize(
    signal_length: int = 10000,
    sample_rate: float = 1000,
    heart_rate: float = 60,
    frequency_modulation: float = 0.3,
    ibi_randomness: float = 0.1,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Generate synthetic PPG signal. Utilize pk.signal.noise methods to make more realistic.

    Args:
        signal_length (int, optional): Length of signal in samples. Defaults to 10000.
        sample_rate (float, optional): Sample rate in Hz. Defaults to 1000 Hz.
        heart_rate (float, optional): Heart rate in BPM. Defaults to 60 BPM.
        frequency_modulation (float, optional): Frequency modulation strength [0,1]. Defaults to 0.3.
        ibi_randomness (float, optional): IBI randomness in range [0,1]. Defaults to 0.1.

    Returns:
        npt.NDArray: Synthetic PPG, segmentation mask, fiducial mask
    """
    duration = signal_length / sample_rate
    period = 60 / heart_rate  # in seconds
    n_period = int(np.rint(duration / period) + 1)
    periods = np.ones(n_period) * period

    # Mark onset of each wave in seconds
    x_onset = np.cumsum(periods)
    x_onset -= x_onset[0]  # make sure seconds start at zero

    # Add respiratory sinus arrythmia (RSA)
    periods, x_onset = _frequency_modulation(
        periods,
        x_onset,
        modulation_frequency=0.05,
        modulation_strength=frequency_modulation,
    )

    # Modulate onset of each wave randomly ~[0, ibi_randomness]
    x_onset = _random_x_offset(x_onset, ibi_randomness)
    y_onset = np.random.normal(0, 0.1, n_period)

    # Create systolic peaks within the waves in seconds
    x_sys = x_onset + np.random.normal(0.175, 0.01, n_period) * periods
    y_sys = y_onset + np.random.normal(1.5, 0.15, n_period)

    # Create dicrotic notches within the waves in seconds
    x_notch = x_onset + np.random.normal(0.4, 0.001, n_period) * periods
    y_notch = y_sys * np.random.normal(0.49, 0.01, n_period)

    # Create diastolic peaks within the waves in seconds
    x_dia = x_onset + np.random.normal(0.45, 0.001, n_period) * periods
    y_dia = y_sys * np.random.normal(0.51, 0.01, n_period)

    # Convert seconds to sample
    x_onset_n = np.ceil(x_onset * sample_rate).astype(int)
    x_sys_n = np.ceil(x_sys * sample_rate).astype(int)
    x_notch_n = np.ceil(x_notch * sample_rate).astype(int)
    x_dia_n = np.ceil(x_dia * sample_rate).astype(int)

    # Concatenate all landmarks and sort them
    x_all = np.concatenate((x_onset_n, x_sys_n, x_notch_n, x_dia_n))
    x_all.sort(kind="mergesort")

    y_all = np.zeros(n_period * 4)
    y_all[0::4] = y_onset
    y_all[1::4] = y_sys
    y_all[2::4] = y_notch
    y_all[3::4] = y_dia

    # Interpolate a continuous signal between the landmarks (i.e., Cartesian coordinates).
    samples = np.arange(int(np.ceil(duration * sample_rate)))

    # Create fiducial mask
    fids = np.zeros(len(samples), dtype=np.int32)
    fids[x_sys_n[x_sys_n < fids.size]] = PpgFiducial.systolic_peak
    fids[x_notch_n[x_notch_n < fids.size]] = PpgFiducial.dicrotic_notch
    fids[x_dia_n[x_dia_n < fids.size]] = PpgFiducial.diastolic_peak

    # Create segmentation mask
    x_sys_seg = np.concatenate((x_onset_n, x_dia_n - 1))
    x_sys_seg.sort(kind="mergesort")
    segs = np.full(len(samples), fill_value=PpgSegment.diastolic, dtype=np.int32)
    for i in range(len(x_sys_seg) // 2):
        segs[x_sys_seg[2 * i] : x_sys_seg[2 * i + 1]] = PpgSegment.systolic

    # Interpolate
    interp_function = scipy.interpolate.Akima1DInterpolator(x_all, y_all)
    ppg = interp_function(samples)

    ppg = ppg[:signal_length]
    segs = segs[:signal_length]
    fids = fids[:signal_length]

    return ppg, segs, fids


def _frequency_modulation(
    periods: npt.NDArray, seconds: npt.NDArray, modulation_frequency: float, modulation_strength: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Modulate signal via sine wave modulation.

    Args:
        periods (npt.NDArray): Periods in seconds.
        seconds (npt.NDArray): Onset of each wave in seconds.
        modulation_frequency (float): Frequency of modulation in Hz.
        modulation_strength (float): Strength of modulation in range [0, 1].

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Modulated periods and onset of each wave in seconds.


    """
    modulation_mean = 1
    modulation_duration = (modulation_mean - modulation_strength) * periods[0]
    # Enforce minimum inter-beat-interval of 300 milliseconds.
    if modulation_duration < 0.3:
        print(
            "Skipping frequency modulation, since the modulation_strength"
            f" {modulation_strength} leads to physiologically implausible"
            f" wave durations of {modulation_duration * 1000}"
            f" milliseconds."
        )
        return periods, seconds

    # Apply a very conservative Nyquist criterion.
    nyquist = (1 / periods[0]) * 0.5
    if modulation_frequency > nyquist:
        print(f"Please choose a modulation frequency lower than {nyquist}.")

    # Generate a sine with mean 1 and amplitude 0.5 * modulation_strength
    # At 100 bpm and modulation_strenght=1.0, the heart rate will fluctuate between 50 to 150.
    # At 100 bpm and modulation_strenght=0.2, the heart rate will fluctuate between 90 to 110.
    modulator = 0.5 * modulation_strength * np.sin(2 * np.pi * modulation_frequency * seconds) + modulation_mean
    periods_modulated = periods * modulator
    seconds_modulated = np.cumsum(periods_modulated)
    seconds_modulated -= seconds_modulated[0]  # make sure seconds start at zero

    return periods_modulated, seconds_modulated


# def _amplitude_modulation():
#     pass


def _random_x_offset(x: npt.NDArray, offset_weight: float) -> npt.NDArray:
    """Offset each wave/beat by subtracting offset_weight * (xi - xi-1)
    where xi-1 is the wave onset preceding xi.

    Args:
        x (npt.NDArray): Onset of each wave in seconds.
        offset_weight (float): Weight of offset in range [0, 1].

    Returns:
        npt.NDArray: Offset onset of each wave in seconds.

    """
    # Clip to [0, 0.99]
    offset_weight = max(min(offset_weight, 0.99), 0)

    x_diff = np.diff(x)
    # Enforce minimum inter-beat-interval of 300 milliseconds.
    min_x_diff = min(x_diff)
    offset_duration = min_x_diff - (min_x_diff * offset_weight)
    if offset_duration < 0.3:
        print(
            "Skipping random IBI modulation, since the offset_weight"
            f" {offset_weight} leads to physiologically implausible wave"
            f" durations of {offset_duration * 1000}"
            f" milliseconds."
        )
        return x

    max_offsets = offset_weight * x_diff
    offsets = [np.random.uniform(0, i) for i in max_offsets]

    x_offset = x.copy()
    x_offset[1:] -= offsets

    return x_offset
