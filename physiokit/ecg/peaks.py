import numpy as np
import numpy.typing as npt

from ..signal import moving_gradient_filter, quotient_filter_mask


def find_peaks(
    data: npt.NDArray,
    sample_rate: float = 1000,
    qrs_window: float = 0.1,
    avg_window: float = 1.0,
    qrs_prom_weight: float = 1.5,
    qrs_min_len_weight: float = 0.4,
    qrs_min_delay: float = 0.3,
) -> npt.NDArray:
    """Find R peaks in ECG signal using QRS gradient method.

    Args:
        data (array): ECG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        qrs_window (float, optional): Window size in seconds to compute QRS gradient. Defaults to 0.1 s.
        avg_window (float, optional): Window size in seconds to compute average gradient. Defaults to 1.0 s.
        qrs_prom_weight (float, optional): Weight to compute minimum QRS height. Defaults to 1.5.
        qrs_min_len_weight (float, optional): Weight to compute minimum QRS length. Defaults to 0.4.
        qrs_min_delay (float, optional): Minimum delay between QRS complexes. Defaults to 0.3 s.

    Returns:
        npt.NDArray: R peaks.
    """

    # Identify start and end of QRS complexes.
    qrs = (
        moving_gradient_filter(
            data, sample_rate=sample_rate, sig_window=qrs_window, avg_window=avg_window, sig_prom_weight=qrs_prom_weight
        )
        > 0
    )

    beg_qrs = np.where(np.logical_and(np.logical_not(qrs[0:-1]), qrs[1:]))[0]
    end_qrs = np.where(np.logical_and(qrs[0:-1], np.logical_not(qrs[1:])))[0]
    end_qrs = end_qrs[end_qrs > beg_qrs[0]]

    num_qrs = min(beg_qrs.size, end_qrs.size)
    min_qrs_len = np.mean(end_qrs[:num_qrs] - beg_qrs[:num_qrs]) * qrs_min_len_weight
    min_qrs_delay = int(np.rint(qrs_min_delay * sample_rate))

    peaks = []
    for i in range(num_qrs):
        beg, end = beg_qrs[i], end_qrs[i]
        peak = beg + np.argmax(data[beg:end])
        qrs_len = end - beg
        qrs_delay = peak - peaks[-1] if peaks else min_qrs_delay

        # Enforce minimum delay between peaks
        if qrs_delay < min_qrs_delay or qrs_len < min_qrs_len:
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
        peaks (array): R peaks.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        min_rr (float, optional): Minimum RR interval in seconds. Defaults to 0.3 s.
        max_rr (float, optional): Maximum RR interval in seconds. Defaults to 2.0 s.
        min_delta (float, optional): Minimum RR interval delta. Defaults to 0.3.

    Returns:
        npt.NDArray: Filtered peaks.
    """

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
    rr_ints: npt.NDArray,
    sample_rate: float = 1000,
    min_rr: float = 0.3,
    max_rr: float = 2.0,
    min_delta: float | None = 0.3,
) -> npt.NDArray:
    """Filter out peaks with RR intervals outside of normal range.

    Args:
        rr_ints (array): RR intervals.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        min_rr (float, optional): Minimum RR interval in seconds. Defaults to 0.3 s.
        max_rr (float, optional): Maximum RR interval in seconds. Defaults to 2.0 s.
        min_delta (float, optional): Minimum RR interval delta. Defaults to 0.3.

    Returns:
        npt.NDArray: RR interval mask.
    """
    if rr_ints.size == 0:
        return np.array([])

    # Filter out peaks with RR intervals outside of normal range
    rr_mask = np.where((rr_ints < min_rr * sample_rate) | (rr_ints > max_rr * sample_rate), 1, 0)

    # Filter out peaks that deviate more than delta
    if min_delta is not None:
        rr_mask = quotient_filter_mask(rr_ints, mask=rr_mask, lowcut=1 - min_delta, highcut=1 + min_delta)

    return rr_mask
