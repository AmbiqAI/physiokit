from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.interpolate

from ..signal import compute_fft, filter_signal
from .peaks import compute_rr_intervals, filter_rr_intervals, find_peaks


def compute_heart_rate(
    data: npt.NDArray, sample_rate: float = 1000, method: str = "fft", **kwargs: dict
) -> tuple[float, float]:
    """Compute heart rate in BPM from PPG signal.

    Args:
        data (array): PPG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        method (str, optional): Method to compute heart rate. Defaults to 'fft'.
        **kwargs (dict): Keyword arguments to pass to method.

    Returns:
        float: Heart rate in BPM.
    """
    match method:
        case "fft":
            bpm, qos = compute_heart_rate_from_fft(data=data, sample_rate=sample_rate, **kwargs)
        case "peak":
            bpm, qos = compute_heart_rate_from_peaks(data=data, sample_rate=sample_rate, **kwargs)
        case _:
            raise NotImplementedError(f"Heart rate computation method {method} not implemented.")
    # END MATCH
    return bpm, qos


def compute_heart_rate_from_peaks(
    data: npt.NDArray, sample_rate: float = 1000, min_rr: float = 0.3, max_rr: float = 2.0, min_delta: float = 0.3
) -> tuple[float, float]:
    """Compute heart rate from peaks of PPG signal.

    Args:
        data (array): PPG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        min_rr (float, optional): Minimum RR interval in seconds. Defaults to 0.3 s.
        max_rr (float, optional): Maximum RR interval in seconds. Defaults to 2.0 s.
        min_delta (float, optional): Minimum RR interval delta. Defaults to 0.3.

    Returns:
        tuple[float, float]: Heart rate (BPM) and qos metric.
    """
    peaks = find_peaks(data=data, sample_rate=sample_rate)
    rri = compute_rr_intervals(peaks=peaks)
    rmask = filter_rr_intervals(rr_ints=rri, sample_rate=sample_rate, min_rr=min_rr, max_rr=max_rr, min_delta=min_delta)
    bpm = 60 / (np.nanmean(rri[rmask == 0]) / sample_rate)
    qos = rmask[rmask == 0].size / rmask.size
    return bpm, qos


def compute_heart_rate_from_fft(
    data: npt.NDArray, sample_rate: float = 1000, lowcut: float = 0.5, highcut: float = 4.0
) -> tuple[float, float]:
    """Compute heart rate from FFT of PPG signal.

    Args:
        data (array): PPG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        lowcut (float, optional): Lowcut frequency in Hz. Defaults to 0.5 Hz.
        highcut (float, optional): Highcut frequency in Hz. Defaults to 4.0 Hz.

    Returns:
        tuple[float, float]: Heart rate (BPM) and qos metric.
    """
    freqs, sp = compute_fft(data, sample_rate)
    l_idx = np.where(freqs >= lowcut)[0][0]
    r_idx = np.where(freqs >= highcut)[0][0]
    freqs = freqs[l_idx:r_idx]
    ps = 2 * np.abs(sp[l_idx:r_idx])
    fft_pk_idx = np.argmax(ps)
    bpm = 60 * freqs[fft_pk_idx]
    qos = ps[fft_pk_idx] / np.sum(ps)
    return bpm, qos


def derive_respiratory_rate(
    ppg: npt.NDArray,
    peaks: npt.NDArray,
    troughs: npt.NDArray | None = None,
    rri: npt.NDArray | None = None,
    sample_rate: float = 1000,
    method: Literal["riav", "riiv", "rifv"] = "rifv",
    lowcut: float = 0.1,
    highcut: float = 1.0,
    order: int = 3,
    threshold: float | None = 0.85,
    interpolate_method: str = "linear",
) -> tuple[float, float]:
    """Derive respiratory rate from PPG signal using given method.

    Args:
        ppg (array): PPG signal.
        peaks (array): Peaks of PPG signal.
        troughs (array, optional): Troughs of PPG signal. Defaults to None.
        rri (array, optional): Respiratory interval. Defaults to None.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        method (str, optional): Method to compute respiratory rate. Defaults to 'riav'.
        lowcut (float, optional): Lowcut frequency in Hz. Defaults to 0.1 Hz.
        highcut (float, optional): Highcut frequency in Hz. Defaults to 1.0 Hz.
        order (int, optional): Order of filter. Defaults to 3.
        threshold (float, optional): Threshold for peak detection. Defaults to 0.85.
        interpolate_method (str, optional): Interpolation method. Defaults to 'linear'.

    Returns:
        tuple[float, float]: Respiratory rate (BPM) and qos metric.
    """
    if peaks.size < 4:
        raise ValueError("At least 4 peaks are required to compute respiratory rate")

    ts = np.arange(peaks[0], peaks[-1], 1)
    match method:
        case "riav":
            rsp = ppg[peaks] - ppg[troughs]
        case "riiv":
            rsp = ppg[peaks]
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
    qos = tgt_pwr / np.sum(rsp_ps)
    rsp_bpm = 60 * np.sum(rsp_bpm_weights * freqs[fft_pk_indices]) / tgt_pwr
    return rsp_bpm, qos


def compute_spo2_from_perfusion(
    dc1: float, ac1: float, dc2: float, ac2: float, coefs: tuple[float, float, float] = (1, 0, 0)
) -> float:
    """Compute SpO2 from ratio of perfusion indexes (AC/DC).

    Device Coefficients:
        * MAX30101: [1.5958422, -34.6596622, 112.6898759]
        * MAX8614X: [-16.666666, 8.333333, 100]

    Args:
        dc1 (float): DC component of 1st PPG signal (e.g RED).
        ac1 (float): AC component of 1st PPG signal (e.g RED).
        dc2 (float): DC component of 2nd PPG signal (e.g. IR).
        ac2 (float): AC component of 2nd PPG signal (e.g. IR).
        coefs (tuple[float, float, float], optional): Calibration coefficients. Defaults to (1, 0, 0).

    Returns:
        float: SpO2 value clipped to [50, 100].
    """
    r = (ac1 / dc1) / (ac2 / dc2)
    spo2 = coefs[0] * r**2 + coefs[1] * r + coefs[2]
    return max(min(spo2, 100), 50)


def compute_spo2_in_time(
    ppg1: npt.NDArray,
    ppg2: npt.NDArray,
    coefs: tuple[float, float, float] = (1, 0, 0),
    sample_rate: float = 1000,
    lowcut: float = 0.5,
    highcut: float = 4,
    order: int = 3,
) -> float:
    """Compute SpO2 from PPG signals in time domain.

    Args:
        ppg1 (array): 1st PPG signal (e.g RED).
        ppg2 (array): 2nd PPG signal (e.g. IR).
        coefs (tuple[float, float, float], optional): Calibration coefficients. Defaults to (1, 0, 0).
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        lowcut (float, optional): Lowcut frequency in Hz. Defaults to 0.5 Hz.
        highcut (float, optional): Highcut frequency in Hz. Defaults to 4.0 Hz.
        order (int, optional): Order of filter. Defaults to 3.

    Returns:
        float: SpO2 value
    """

    # Compute DC
    ppg1_dc = np.mean(ppg1)
    ppg2_dc = np.mean(ppg2)

    # Bandpass filter
    ppg1_clean = filter_signal(
        data=ppg1, lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=order, forward_backward=True
    )

    ppg2_clean = filter_signal(
        data=ppg2, lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=order, forward_backward=True
    )

    # Compute AC via RMS
    ppg1_ac = np.sqrt(np.mean(ppg1_clean**2))
    ppg2_ac = np.sqrt(np.mean(ppg2_clean**2))

    spo2 = compute_spo2_from_perfusion(dc1=ppg1_dc, ac1=ppg1_ac, dc2=ppg2_dc, ac2=ppg2_ac, coefs=coefs)
    return spo2


def compute_spo2_in_frequency(
    ppg1: npt.NDArray,
    ppg2: npt.NDArray,
    coefs: tuple[float, float, float] = (1, 0, 0),
    sample_rate: float = 1000,
    lowcut: float = 0.5,
    highcut: float = 4.0,
    order: int = 3,
) -> float:
    """Compute SpO2 from PPG signals in frequency domain.

    Args:
        ppg1 (array): 1st PPG signal (e.g RED).
        ppg2 (array): 2nd PPG signal (e.g. IR).
        coefs (tuple[float, float, float], optional): Calibration coefficients. Defaults to (1, 0, 0).
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        lowcut (float, optional): Lowcut frequency in Hz. Defaults to 0.5 Hz.
        highcut (float, optional): Highcut frequency in Hz. Defaults to 4.0 Hz.
        order (int, optional): Order of filter. Defaults to 3.

    Returns:
        float: SpO2 value
    """

    # Compute DC
    ppg1_dc = np.mean(ppg1)
    ppg2_dc = np.mean(ppg2)

    # Bandpass filter
    ppg1_clean = filter_signal(
        data=ppg1, lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=order, forward_backward=True
    )
    ppg2_clean = filter_signal(
        data=ppg2, lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=order, forward_backward=True
    )

    # Compute AC via FFT
    freqs, ppg1_fft = compute_fft(ppg1_clean, sample_rate=sample_rate)
    freqs, ppg2_fft = compute_fft(ppg2_clean, sample_rate=sample_rate)

    l_idx = np.where(freqs >= lowcut)[0][0]
    r_idx = np.where(freqs >= highcut)[0][0]

    freqs = freqs[l_idx:r_idx]
    ppg1_ps = 2 * np.abs(ppg1_fft[l_idx:r_idx])
    ppg2_ps = 2 * np.abs(ppg2_fft[l_idx:r_idx])

    # Find peak
    fft_pk_idx = np.argmax(ppg1_ps + ppg2_ps)

    # Compute AC
    ppg1_ac = ppg1_ps[fft_pk_idx]
    ppg2_ac = ppg2_ps[fft_pk_idx]

    spo2 = compute_spo2_from_perfusion(dc1=ppg1_dc, ac1=ppg1_ac, dc2=ppg2_dc, ac2=ppg2_ac, coefs=coefs)

    return spo2
