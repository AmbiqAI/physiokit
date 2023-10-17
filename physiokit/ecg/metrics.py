from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.interpolate

from ..signal import compute_fft, filter_signal
from .peaks import compute_rr_intervals, filter_rr_intervals, find_peaks


def compute_heart_rate(
    data: npt.NDArray, sample_rate: float = 1000, method: str = "peak", **kwargs: dict
) -> tuple[float, float]:
    """Compute heart rate from ECG signal.

    Args:
        data (array): ECG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        method (str, optional): Method to compute heart rate. Defaults to 'peak'.
        **kwargs (dict): Keyword arguments to pass to method.

    Returns:
        tuple[float, float]: Heart rate in BPM and qos metric.
    """
    match method:
        case "peak":
            bpm, qos = compute_heart_rate_from_peaks(data=data, sample_rate=sample_rate, **kwargs)
        case _:
            raise NotImplementedError(f"Heart rate computation method {method} not implemented.")
    # END MATH
    return bpm, qos


def compute_heart_rate_from_peaks(
    data: npt.NDArray,
    sample_rate: float = 1000,
    min_rr: float = 0.3,
    max_rr: float = 2.0,
    min_delta: float | None = 0.3,
) -> tuple[float, float]:
    """Compute heart rate from peaks of ECG signal.

    Args:
        data (array): ECG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.

    Returns:
        tuple[float, float]: Heart rate in BPM and qos metric.
    """
    peaks = find_peaks(data=data, sample_rate=sample_rate)
    rri = compute_rr_intervals(peaks=peaks)
    rmask = filter_rr_intervals(rr_ints=rri, sample_rate=sample_rate, min_rr=min_rr, max_rr=max_rr, min_delta=min_delta)
    bpm = 60 / (np.nanmean(rri[rmask == 0]) / sample_rate)
    qos = rmask[rmask == 0].size / rmask.size
    return bpm, qos


def derive_respiratory_rate(
    peaks: npt.NDArray,
    rri: npt.NDArray | None = None,
    sample_rate: float = 1000,
    method: Literal["rifv"] = "rifv",
    lowcut: float = 0.1,
    highcut: float = 1.0,
    order: int = 3,
    threshold: float | None = 0.85,
    interpolate_method: str = "linear",
) -> tuple[float, float]:
    """Derive respiratory rate from ECG signal using given method.

    Args:
        peaks (array): QRS peaks of ECG signal.
        rri (array, optional): RR intervals. Defaults to None.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        method (str, optional): Method to compute respiratory rate. Defaults to 'riav'.
        lowcut (float, optional): Lowcut frequency in Hz. Defaults to 0.1 Hz.
        highcut (float, optional): Highcut frequency in Hz. Defaults to 1.0 Hz.
        order (int, optional): Order of filter. Defaults to 3.
        threshold (float, optional): Threshold for peak detection. Defaults to 0.85.

    Returns:
        tuple[float, float]: Respiratory rate in BPM and qos metric.
    """
    if peaks.size < 4:
        raise ValueError("At least 4 peaks are required to compute respiratory rate")

    ts = np.arange(peaks[0], peaks[-1], 1)
    match method:
        case "rifv":
            rsp = rri
        case _:
            raise ValueError(f"Method {method} not implemented")
    rsp = scipy.interpolate.interp1d(peaks, rsp, kind=interpolate_method, fill_value="extrapolate")(ts)
    rsp = filter_signal(rsp, lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=order)

    freqs, rsp_sp = compute_fft(rsp, sample_rate=sample_rate)
    l_idx = np.where(freqs >= lowcut)[0][0]
    r_idx = np.where(freqs >= highcut)[0][0]
    rsp_ps = 2 * np.abs(rsp_sp)
    freqs = freqs[l_idx:r_idx]
    rsp_ps = rsp_ps[l_idx:r_idx]

    fft_pk_idx = np.argmax(rsp_ps)
    if threshold is not None:
        fft_pk_indices = np.where(rsp_ps > threshold * rsp_ps[fft_pk_idx])[0]
    else:
        fft_pk_indices = [fft_pk_idx]

    rsp_bpm_weights = rsp_ps[fft_pk_indices]
    tgt_pwr = np.sum(rsp_bpm_weights)
    rsp_bpm = 60 * np.sum(rsp_bpm_weights * freqs[fft_pk_indices]) / tgt_pwr
    qos = tgt_pwr / np.mean(rsp_ps)

    return rsp_bpm, qos
