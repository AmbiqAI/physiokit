import numpy as np
import numpy.typing as npt
import scipy.ndimage as spn

from ..signal import quotient_filter_mask


def find_peaks(
    data: npt.NDArray,
    sample_rate: float = 1000,
    qrs_window: float = 0.1,
    avg_window: float = 1.0,
    qrs_prom_weight: float = 1.5,
    qrs_min_len_weight: float = 0.4,
    qrs_min_delay: float = 0.3,
):
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

    # Compute gradient of signal for both QRS and average.
    abs_grad = np.abs(np.gradient(data))
    qrs_kernel = int(np.rint(qrs_window * sample_rate))
    avg_kernel = int(np.rint(avg_window * sample_rate))

    # Smooth gradients
    qrs_grad = spn.uniform_filter1d(abs_grad, qrs_kernel, mode="nearest")
    avg_grad = spn.uniform_filter1d(qrs_grad, avg_kernel, mode="nearest")

    min_qrs_height = qrs_prom_weight * avg_grad

    # Identify start and end of QRS complexes.
    qrs = qrs_grad > min_qrs_height
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
        qrs_delay = peak - peaks[-1] if len(peaks) else min_qrs_delay

        # Enforce minimum delay between peaks
        if qrs_delay < min_qrs_delay or qrs_len < min_qrs_len:
            continue
        peaks.append(peak)
    # END FOR

    return np.array(peaks, dtype=int)


def filter_peaks(
        peaks: npt.NDArray,
        sample_rate: float = 1000,
    ) -> npt.NDArray:
    """Filter out peaks with RR intervals outside of normal range.
    Args:
        peaks (array): R peaks.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
    Returns:
        npt.NDArray: Filtered peaks.
    """
    lowcut = 0.3*sample_rate
    highcut = 2*sample_rate

    # Capture RR intervals
    rr_ints = np.diff(peaks)
    rr_ints = np.hstack((rr_ints[0], rr_ints))

    # Filter out peaks with RR intervals outside of normal range
    rr_mask = np.where((rr_ints < lowcut) | (rr_ints > highcut), 1, 0)

    # Filter out peaks that deviate more than 30%
    rr_mask = quotient_filter_mask(rr_ints, mask=rr_mask, lowcut=0.7, highcut=1.3)
    filt_peaks = peaks[np.where(rr_mask == 0)[0]]
    return filt_peaks