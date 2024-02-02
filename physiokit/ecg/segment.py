import numpy as np
import numpy.typing as npt

from ..signal import moving_gradient_filter
from .defines import EcgSegment

# from .clean import clean as clean_ecg


def locate_qrs(
    data: npt.NDArray,
    sample_rate: float = 1000,
    qrs_window: float = 0.1,
    avg_window: float = 1.0,
    qrs_prom_weight: float = 1.5,
    qrs_min_len_weight: float = 0.4,
    qrs_min_delay: float = 0.3,
) -> tuple[npt.NDArray]:
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
        tuple[npt.NDArray]: QRS segments
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
    min_qrs_len = int(np.mean(end_qrs[:num_qrs] - beg_qrs[:num_qrs]) * qrs_min_len_weight)
    min_qrs_delay = int(np.rint(qrs_min_delay * sample_rate))

    peaks: list[tuple(int, int, int)] = []
    for i in range(num_qrs):
        beg, end = beg_qrs[i], end_qrs[i]
        peak = beg + np.argmax(data[beg:end])
        qrs_len = int(end - beg)
        qrs_delay = peak - peaks[-1][1] if peaks else min_qrs_delay

        # Enforce minimum delay between peaks
        if qrs_delay < min_qrs_delay or qrs_len < min_qrs_len:
            continue
        peaks.append((beg, peak, end))
    # END FOR

    return np.array(peaks, dtype=int)


def _locate_wave_in_region(
    roi: npt.NDArray,
    sample_rate: float = 1000,
    wave_window: float = 0.1,
    avg_window: float = 0.3,
    wave_prom_weight: float = 1.0,
    wave_min_window: float = 0.1,
    reverse: bool = False,
) -> tuple[int, int, int] | None:
    """Locate wave (e.g. p-wave) in region of interest.

    Args:
        roi (array): Region of interest.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        wave_window (float, optional): Window size in seconds to compute wave gradient. Defaults to 0.1 s.
        avg_window (float, optional): Window size in seconds to compute average gradient. Defaults to 0.3 s.
        wave_prom_weight (float, optional): Weight to compute minimum wave height. Defaults to 1.0.
        wave_min_window (float, optional): Minimum wave length in seconds. Defaults to 0.1 s.

    Returns:
        tuple[int, int, int] | None: Wave onset, peak, and offset.
    """

    # Compute wave gradient
    wave = (
        moving_gradient_filter(
            roi,
            sample_rate=sample_rate,
            sig_window=wave_window,
            avg_window=avg_window,
            sig_prom_weight=wave_prom_weight,
            mode="constant",
            fval=0,
        )
        > 0
    )

    # Find wave onset and offset
    on_waves = np.where(np.logical_and(np.logical_not(wave[0:-1]), wave[1:]))[0]
    off_waves = np.where(np.logical_and(wave[0:-1], np.logical_not(wave[1:])))[0]
    print("wave", on_waves, off_waves)
    if on_waves.size == 0 or off_waves.size == 0:
        return None
    off_waves = off_waves[off_waves > on_waves[0]]
    if off_waves.size == 0:
        return None
    on_waves = on_waves[on_waves < off_waves[-1]]
    num_waves = min(on_waves.size, off_waves.size)
    min_wave_len = int(np.rint(wave_min_window * sample_rate))

    # Grab first wave satisfying constraints
    for i in range(num_waves):
        if reverse:
            beg, end = on_waves[num_waves - i - 1], off_waves[num_waves - i - 1]
        else:
            beg, end = on_waves[i], off_waves[i]
        peak = beg + np.argmax(np.abs(roi))
        wave_len = int(end - beg)
        # Enforce minimum wave duration
        if wave_len < min_wave_len:
            continue
        return (beg, peak, end)
    # END FOR
    return None


def locate_pwave_from_qrs_anchor(
    data: npt.NDArray,
    qrs_seg: tuple[int, int, int],
    sample_rate: float = 1000,
    wave_window: float = 0.1,
    avg_window: float = 0.3,
    wave_prom_weight: float = 1.0,
    wave_min_window: float = 0.01,
) -> tuple[int, int, int] | None:
    """Locate P wave in ECG signal using QRS anchor method.

    Args:
        data (array): ECG signal.
        qrs_seg (tuple[int, int, int]): QRS segment.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        wave_window (float, optional): Window size in seconds to compute wave gradient. Defaults to 0.1 s.
        avg_window (float, optional): Window size in seconds to compute average gradient. Defaults to 0.3 s.
        wave_prom_weight (float, optional): Weight to compute minimum wave height. Defaults to 1.0.
        wave_min_window (float, optional): Minimum wave length in seconds. Defaults to 0.05 s.

    Returns:
        tuple[int, int, int] | None: Wave onset, peak, and offset.
    """
    # Grab window from end of QRS to 300 ms before (PQ interval)
    pq_window = int(np.rint(0.3 * sample_rate))
    # roi_offset = qrs_seg[2]
    # roi_onset = max(0, roi_offset - pq_window)
    roi_offset = qrs_seg[0]
    roi_onset = max(0, roi_offset - pq_window)

    roi = data[roi_onset:roi_offset].copy()

    # Zero out QRS region
    # qrs_offset = qrs_seg[0] - roi_onset
    # roi[qrs_offset:] = roi[qrs_offset]

    wave = _locate_wave_in_region(
        roi,
        sample_rate=sample_rate,
        wave_window=wave_window,
        avg_window=avg_window,
        wave_prom_weight=wave_prom_weight,
        wave_min_window=wave_min_window,
        reverse=True,
    )
    if wave:
        return (roi_onset + wave[0], roi_onset + wave[1], roi_onset + wave[2])
    return None


def locate_twave_from_qrs_anchor(
    data: npt.NDArray,
    qrs_seg: tuple[int, int, int],
    sample_rate: float = 1000,
    wave_window: float = 0.3,
    avg_window: float = 0.4,
    wave_prom_weight: float = 1.0,
    wave_min_window: float = 0.1,
) -> tuple[int, int, int] | None:
    """Locate T wave in ECG signal using QRS anchor method.

    Args:
        data (array): ECG signal.
        qrs_seg (tuple[int, int, int]): QRS segment.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        wave_window (float, optional): Window size in seconds to compute wave gradient. Defaults to 0.3 s.
        avg_window (float, optional): Window size in seconds to compute average gradient. Defaults to 0.6 s.
        wave_prom_weight (float, optional): Weight to compute minimum wave height. Defaults to 1.0.
        wave_min_window (float, optional): Minimum wave length in seconds. Defaults to 0.1 s.

    Returns:
        tuple[int, int, int] | None: Wave onset, peak, and offset.
    """

    # Grab window from end of QRS to 400 ms after (ST interval)
    st_window = int(np.rint(0.4 * sample_rate))
    roi_onset = qrs_seg[2]
    roi_offset = min(roi_onset + st_window, data.size)
    # roi_onset = qrs_seg[0]
    # roi_offset = min(roi_onset + qt_window, data.size)
    roi = data[roi_onset:roi_offset].copy()

    # Zero out QRS region
    # qrs_offset = qrs_seg[2] - roi_onset
    # roi[:qrs_offset] = roi[qrs_offset]

    wave = _locate_wave_in_region(
        roi,
        sample_rate=sample_rate,
        wave_window=wave_window,
        avg_window=avg_window,
        wave_prom_weight=wave_prom_weight,
        wave_min_window=wave_min_window,
    )
    if wave:
        return (roi_onset + wave[0], roi_onset + wave[1], roi_onset + wave[2])
    return None


def apply_segmentation(
    data: npt.NDArray,
    sample_rate: float = 1000,
    # lead: int|None = None
) -> npt.NDArray:
    """Apply segmentation to ECG signal."""
    segs = np.zeros(data.size, dtype=int)
    segs[:] = EcgSegment.background

    qrs_segs: list[tuple[int, int, int]] = []
    p_segs: list[tuple[int, int, int]] = []
    t_segs: list[tuple[int, int, int]] = []

    # Identify QRS segments and peaks
    # qrs = clean_ecg(data, sample_rate=sample_rate, lowcut=10.0, highcut=30.0)
    qrs_segs = locate_qrs(data, sample_rate=sample_rate)
    # Extract nominal RR interval, filter out R peaks (mark as noise)
    # For each QRS segment, extract P wave and T wave
    for qrs_seg in qrs_segs:
        print("QRS", qrs_seg)
        segs[qrs_seg[0] : qrs_seg[2]] = EcgSegment.qrs_complex
        # Extract P wave and T wave
        print("P-wave")
        p_seg = locate_pwave_from_qrs_anchor(data, qrs_seg, sample_rate=sample_rate)
        if p_seg:
            segs[p_seg[0] : p_seg[2]] = EcgSegment.p_wave
            p_segs.append(p_seg)
        # END IF
        print("T-wave")
        t_seg = locate_twave_from_qrs_anchor(data, qrs_seg, sample_rate=sample_rate)
        if t_seg:
            segs[t_seg[0] : t_seg[2]] = EcgSegment.t_wave
            t_segs.append(t_seg)
        # END IF
    # END FOR
    return segs


def find_pwave():
    """Find P wave in ECG signal"""
    raise NotImplementedError()


def find_twave():
    """Find T wave in ECG signal"""
    raise NotImplementedError()


def find_qrs():
    """Find QRS complex in ECG signal"""
    raise NotImplementedError()
