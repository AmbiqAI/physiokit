import numpy as np
import numpy.typing as npt
import scipy.ndimage as spn

from .clean import clean as clean_ecg


def locate_qrs(
    data: npt.NDArray,
    sample_rate: float = 1000,
    qrs_window: float = 0.1,
    avg_window: float = 1.0,
    qrs_prom_weight: float = 1.5,
    qrs_min_len_weight: float = 0.4,
    qrs_min_delay: float = 0.3,
):
    """Find QRS segments in ECG signal using QRS gradient method.
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

    peaks: list[tuple(int, int, int)] = []
    for i in range(num_qrs):
        beg, end = beg_qrs[i], end_qrs[i]
        peak = beg + np.argmax(data[beg:end])
        qrs_len = end - beg
        qrs_delay = peak - peaks[-1] if peaks else min_qrs_delay

        # Enforce minimum delay between peaks
        if qrs_delay < min_qrs_delay or qrs_len < min_qrs_len:
            continue
        peaks.append((beg, peak, end))
    # END FOR

    return np.array(peaks, dtype=int), beg_qrs, end_qrs


def apply_segmentation(
    data: npt.NDArray,
    sample_rate: float = 1000,
    # lead: int|None = None
):
    """Apply segmentation to ECG signal."""

    qrs = clean_ecg(data, sample_rate=sample_rate, lowcut=10.0, highcut=30.0)
    qrs_segs = locate_qrs(qrs, sample_rate=sample_rate)
    q_waves = []
    r_waves = []
    s_waves = []
    for beg, peak, end in qrs_segs:
        q_waves.append(np.argmin(qrs[beg:peak]) + beg)
        r_waves.append(peak)
        s_waves.append(np.argmin(qrs[peak:end]) + peak)
    # Given signal and lead name
    # Identify R peaks
    # Extract nominal RR interval, filter out R peaks (mark as noise)
    # For each R peak, extract beat segment
    # Identify QRS segment -> delineate Q wave, R wave, S wave
    # Identify P wave
    # Identify T wave
    # Optionally, identify U wave
    # Derive PR interval, PR segment, QRS complex, ST segment, QT segment


def find_pwave():
    """Find P wave in ECG signal"""
    raise NotImplementedError()


def find_twave():
    """Find T wave in ECG signal"""
    raise NotImplementedError()


def find_qrs():
    """Find QRS complex in ECG signal"""
    raise NotImplementedError()
