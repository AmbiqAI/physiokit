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
        data (npt.NDArray): PPG signal (1-D).
        sample_rate (float): Sampling rate in Hz.
        method (str): Method to compute heart rate, ``"fft"`` or ``"peak"``.
        **kwargs: Keyword arguments to pass to the selected method.

    Returns:
        tuple[float, float]: (heart rate in BPM, quality score).

    Example:
        >>> import numpy as np
        >>> ppg = np.sin(2 * np.pi * 1.2 * np.arange(0, 5, 1/100))
        >>> bpm, _ = compute_heart_rate(ppg, sample_rate=100)
        >>> int(round(bpm))
        72
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
        data (npt.NDArray): PPG signal (1-D).
        sample_rate (float): Sampling rate in Hz.
        min_rr (float): Minimum RR interval (s).
        max_rr (float): Maximum RR interval (s).
        min_delta (float): Allowed fractional RR deviation.

    Returns:
        tuple[float, float]: (BPM, quality score).

    Example:
        >>> import numpy as np
        >>> ppg = np.sin(2 * np.pi * 1.0 * np.arange(0, 5, 1/50))
        >>> bpm, qos = compute_heart_rate_from_peaks(ppg, sample_rate=50)
        >>> bpm > 50 and qos > 0
        True
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
        data (npt.NDArray): PPG signal (1-D).
        sample_rate (float): Sampling rate in Hz.
        lowcut (float): Lowcut frequency in Hz.
        highcut (float): Highcut frequency in Hz.

    Returns:
        tuple[float, float]: (BPM, quality score).

    Example:
        >>> import numpy as np
        >>> ppg = np.sin(2 * np.pi * 1.3 * np.arange(0, 5, 1/100))
        >>> bpm, _ = compute_heart_rate_from_fft(ppg, sample_rate=100)
        >>> int(round(bpm))
        78
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
        ppg (npt.NDArray): PPG signal.
        peaks (npt.NDArray): Peak indices of PPG signal.
        troughs (npt.NDArray | None): Trough indices (required for ``"riav"``).
        rri (npt.NDArray | None): RR intervals (required for ``"rifv"``).
        sample_rate (float): Sampling rate in Hz.
        method (Literal["riav", "riiv", "rifv"]): Respiratory method.
        lowcut (float): Lowcut frequency in Hz.
        highcut (float): Highcut frequency in Hz.
        order (int): Filter order.
        threshold (float | None): Threshold for FFT peak selection.
        interpolate_method (str): Interpolation method for resampling the derived respiratory signal.

    Returns:
        tuple[float, float]: (respiratory BPM, quality score).

    Example:
        >>> import numpy as np
        >>> t = np.arange(0, 10, 0.1)
        >>> ppg = np.sin(2 * np.pi * 0.2 * t)
        >>> peaks = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
        >>> troughs = peaks + 2
        >>> derive_respiratory_rate(ppg=ppg, peaks=peaks, troughs=troughs[troughs < ppg.size], sample_rate=10, method="riiv")[0]
        12.0
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

    Example:
        >>> compute_spo2_from_perfusion(dc1=1.0, ac1=0.2, dc2=1.0, ac2=0.2, coefs=(1, 0, 90))
        90.0
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
        ppg1 (npt.NDArray): 1st PPG signal (e.g RED).
        ppg2 (npt.NDArray): 2nd PPG signal (e.g. IR).
        coefs (tuple[float, float, float]): Calibration coefficients.
        sample_rate (float): Sampling rate in Hz.
        lowcut (float): Lowcut frequency in Hz.
        highcut (float): Highcut frequency in Hz.
        order (int): Order of filter.

    Returns:
        float: SpO2 value.

    Example:
        >>> import numpy as np
        >>> t = np.arange(0, 2, 0.01)
        >>> ppg1 = 1.0 + 0.1 * np.sin(2 * np.pi * t)
        >>> ppg2 = 1.0 + 0.1 * np.sin(2 * np.pi * t + 0.1)
        >>> 50 <= compute_spo2_in_time(ppg1, ppg2, sample_rate=100) <= 100
        True
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
        ppg1 (npt.NDArray): 1st PPG signal (e.g RED).
        ppg2 (npt.NDArray): 2nd PPG signal (e.g. IR).
        coefs (tuple[float, float, float]): Calibration coefficients.
        sample_rate (float): Sampling rate in Hz.
        lowcut (float): Lowcut frequency in Hz.
        highcut (float): Highcut frequency in Hz.
        order (int): Order of filter.

    Returns:
        float: SpO2 value.

    Example:
        >>> import numpy as np
        >>> t = np.arange(0, 2, 0.01)
        >>> ppg1 = 1.0 + 0.1 * np.sin(2 * np.pi * t)
        >>> ppg2 = 1.0 + 0.1 * np.sin(2 * np.pi * t + 0.2)
        >>> 50 <= compute_spo2_in_frequency(ppg1, ppg2, sample_rate=100) <= 100
        True
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
