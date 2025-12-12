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
        data (npt.NDArray): ECG signal (1-D).
        sample_rate (float): Sampling rate in Hz.
        method (str): Heart-rate method, currently supports ``"peak"``.
        **kwargs: Additional keyword arguments for the selected method.

    Returns:
        tuple[float, float]: (bpm, quality score in [0, 1]).

    Example:
        >>> import numpy as np
        >>> ecg = np.sin(2 * np.pi * 1.2 * np.arange(0, 2, 1/1000))
        >>> bpm, _ = compute_heart_rate(ecg, sample_rate=1000)
        >>> int(round(bpm))
        72
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
        data (npt.NDArray): ECG signal (1-D).
        sample_rate (float): Sampling rate in Hz.
        min_rr (float): Minimum RR interval (s) allowed.
        max_rr (float): Maximum RR interval (s) allowed.
        min_delta (float | None): Allowed fractional RR deviation; ``None`` to skip quotient filtering.

    Returns:
        tuple[float, float]: (bpm, quality score in [0, 1]).

    Example:
        >>> import numpy as np
        >>> ecg = np.sin(2 * np.pi * 1.1 * np.arange(0, 2, 1/250))
        >>> bpm, qos = compute_heart_rate_from_peaks(ecg, sample_rate=250)
        >>> bpm > 60, qos > 0
        (True, True)
    """
    peaks = find_peaks(data=data, sample_rate=sample_rate)
    rri = compute_rr_intervals(peaks=peaks)
    rmask = filter_rr_intervals(rr_ints=rri, sample_rate=sample_rate, min_rr=min_rr, max_rr=max_rr, min_delta=min_delta)
    bpm = float(60 / (np.nanmean(rri[rmask == 0]) / sample_rate))
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
        peaks (npt.NDArray): QRS peaks (indices).
        rri (npt.NDArray | None): RR intervals; required for ``"rifv"``.
        sample_rate (float): Sampling rate in Hz.
        method (Literal["rifv"]): Respiratory method; only ``"rifv"`` supported.
        lowcut (float): Lowcut frequency in Hz for respiration band.
        highcut (float): Highcut frequency in Hz for respiration band.
        order (int): Filter order.
        threshold (float | None): Threshold for FFT peak selection; ``None`` to keep max only.
        interpolate_method (str): Interpolation method for resampling the RR-derived signal.

    Returns:
        tuple[float, float]: (respiratory bpm, quality score).

    Example:
        >>> import numpy as np
        >>> peaks = np.arange(0, 500, 50)
        >>> rri = np.full(peaks.size, 50)
        >>> derive_respiratory_rate(peaks=peaks, rri=rri, sample_rate=1000)
        (72.0, np.float64(1.0))
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


def compute_pulse_arrival_time(ecg: npt.NDArray, ppg: npt.NDArray, sample_rate: float, min_delay: float = 0.1) -> float:
    """Compute pulse arrival time from ECG and PPG signals.

    Args:
        ecg (npt.NDArray): ECG signal (1-D).
        ppg (npt.NDArray): PPG signal (aligned, 1-D).
        sample_rate (float): Sampling rate in Hz.
        min_delay (float): Minimum delay between ECG and PPG peaks in seconds.

    Returns:
        float: Mean pulse arrival time in samples.

    Example:
        >>> import numpy as np
        >>> ecg = np.zeros(200); ppg = np.zeros(200)
        >>> ecg[[50, 150]] = 1; ppg[[55, 155]] = 1
        >>> round(compute_pulse_arrival_time(ecg, ppg, sample_rate=100))
        5
    """
    ecg_peaks = find_peaks(ecg, sample_rate=sample_rate)
    ecg_rri = compute_rr_intervals(ecg_peaks)
    ecg_mask = filter_rr_intervals(ecg_rri, min_rr=0.3, max_rr=2.0, sample_rate=sample_rate)
    mean_lag = 0
    num_lags = 0
    for i in range(0, len(ecg_mask) - 1):
        if ecg_mask[i] == 0 and ecg_mask[i + 1] == 0:
            # Enforce a minimum delay between ECG and PPG peaks
            start = ecg_peaks[i] + round(sample_rate * min_delay)
            ppg_peak = np.argmax(ppg[start : ecg_peaks[i + 1] + 1]) + start
            mean_lag += ppg_peak - ecg_peaks[i]
            num_lags += 1
        # END IF
    # END FOR
    mean_lag /= num_lags
    return mean_lag
