import numpy as np
import numpy.typing as npt
import scipy.ndimage as spn

from ..signal import quotient_filter_mask


def find_peaks(
    data: npt.NDArray,
    sample_rate: float = 1000,
    peak_window: float = 0.111,
    beat_window: float = 0.667,
    beat_offset: float = 0.02,
    peak_delay: float = 0.3,
) -> npt.NDArray:
    """Find systolic peaks in PPG signal.

    Implementation based on Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection in
    Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions. PLoS ONE 8(10): e76585.
    doi:10.1371/journal.pone.0076585.
    Assumes input data is bandpass filtered with a lowcut of .5 Hz and a highcut of 8 Hz.

    Args:
        data (array): PPG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        peak_window (float, optional): Peak window in seconds. Defaults to 0.111 s.
        beat_window (float, optional): Beat window in seconds. Defaults to 0.667 s.
        beat_offset (float, optional): Beat offset in seconds. Defaults to 0.02 s.
        peak_delay (float, optional): Peak delay in seconds. Defaults to 0.3 s.

    Returns:
        npt.NDArray: Peak locations.
    """

    # Clip negative values and square the signal
    sqrd = np.where(data > 0, data**2, 0)

    # Apply 1st moving average filter
    ma_peak_kernel = int(np.rint(peak_window * sample_rate))
    ma_peak = spn.uniform_filter1d(sqrd, ma_peak_kernel, mode="nearest")

    # Apply 2nd moving average filter
    ma_beat_kernel = int(np.rint(beat_window * sample_rate))
    ma_beat = spn.uniform_filter1d(sqrd, ma_beat_kernel, mode="nearest")

    # Thresholds
    min_height = ma_beat + beat_offset * np.mean(sqrd)
    min_width = int(np.rint(peak_window * sample_rate))
    min_delay = int(np.rint(peak_delay * sample_rate))

    # Identify wave boundaries
    waves = ma_peak > min_height
    beg_waves = np.where(np.logical_and(np.logical_not(waves[0:-1]), waves[1:]))[0]
    end_waves = np.where(np.logical_and(waves[0:-1], np.logical_not(waves[1:])))[0]
    end_waves = end_waves[end_waves > beg_waves[0]]

    # Identify systolic peaks
    peaks = []
    for i in range(min(beg_waves.size, end_waves.size)):
        beg, end = beg_waves[i], end_waves[i]
        peak = beg + np.argmax(data[beg:end])
        peak_width = end - beg
        peak_delay = peak - peaks[-1] if peaks else min_delay

        # Enforce minimum length and delay between peaks
        if (peak_width < min_width) or (peak_delay < min_delay):
            continue
        peaks.append(peak)
    # END FOR

    return np.array(peaks, dtype=int)


def filter_peaks(
    peaks: npt.NDArray,
    sample_rate: float = 1000,
    min_rr: float = 0.3,
    max_rr: float = 2.0,
    min_delta: float | None = 0.3,
) -> npt.NDArray:
    """Filter out peaks with RR intervals outside of normal range.

    Args:
        peaks (array): Systolic peaks.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        min_rr (float, optional): Minimum RR interval in seconds. Defaults to 0.3 s.
        max_rr (float, optional): Maximum RR interval in seconds. Defaults to 2.0 s.
        min_delta (float, optional): Minimum RR interval delta. Defaults to 0.3.

    Returns:
        npt.NDArray: Filtered peaks.
    """
    if peaks.size <= 1:
        return peaks
    # Capture RR intervals
    rr_ints = np.diff(peaks)
    rr_ints = np.hstack((rr_ints[0], rr_ints))

    # Filter out peaks with RR intervals outside of normal range
    rr_mask = np.where((rr_ints < min_rr * sample_rate) | (rr_ints > max_rr * sample_rate), 1, 0)

    # Filter out peaks that deviate more than delta
    if min_delta is not None:
        rr_mask = quotient_filter_mask(rr_ints, mask=rr_mask, lowcut=1 - min_delta, highcut=1 + min_delta)
    filt_peaks = peaks[np.where(rr_mask == 0)[0]]
    return filt_peaks


def compute_rr_intervals(
    peaks: npt.NDArray,
) -> npt.NDArray:
    """Compute RR intervals from R peaks.

    Args:
        peaks (array): R peaks.

    Returns:
        npt.NDArray: RR intervals.
    """

    rr_ints = np.diff(peaks)
    if rr_ints.size == 0:
        return rr_ints
    rr_ints = np.hstack((rr_ints[0], rr_ints))
    return rr_ints


def filter_rr_intervals(
    rr_ints: npt.NDArray, sample_rate: float = 1000, min_rr: float = 0.3, max_rr: float = 2.0, min_delta: float = 0.3
) -> npt.NDArray:
    """Filter out peaks with RR intervals outside of normal range.

    Args:
        rr_ints (array): RR intervals.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        min_rr (float, optional): Minimum RR interval in seconds. Defaults to 0.3 s.
        max_rr (float, optional): Maximum RR interval in seconds. Defaults to 2.0 s.
        min_delta (float, optional): Minimum RR interval delta. Defaults to 0.3.

    Returns:
        npt.NDArray: Filtered RR intervals.
    """
    if rr_ints.size == 0:
        return np.array([])

    # Filter out peaks with RR intervals outside of normal range
    rr_mask = np.where((rr_ints < min_rr * sample_rate) | (rr_ints > max_rr * sample_rate), 1, 0)

    # Filter out peaks that deviate more than delta
    rr_mask = quotient_filter_mask(rr_ints, mask=rr_mask, lowcut=1 - min_delta, highcut=1 + min_delta)

    return rr_mask
