import numpy as np
import numpy.typing as npt

from ..signal import compute_fft
from .peaks import compute_rr_intervals, filter_rr_intervals, find_peaks


def compute_respiratory_rate(
    data: npt.NDArray, sample_rate: float = 1000, method: str = "fft", **kwargs: dict
) -> tuple[float, float]:
    """Compute respiratory rate in BPM from signal.

    Args:
        data (array): RSP signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        method (str, optional): Method to compute respiratory rate. Defaults to 'fft'.
        **kwargs (dict): Keyword arguments to pass to method.

    Returns:
        tuple[float, float]: Respiratory rate (BPM) and qos metric.
    """
    match method:
        case "fft":
            bpm, qos = compute_respiratory_rate_from_fft(data=data, sample_rate=sample_rate, **kwargs)
        case "peak":
            bpm, qos = compute_respiratory_rate_from_peaks(data=data, sample_rate=sample_rate, **kwargs)
        case _:
            raise NotImplementedError(f"Respiratory rate computation method {method} not implemented.")
    # END MATCH
    return bpm, qos


def compute_respiratory_rate_from_peaks(
    data: npt.NDArray, sample_rate: float = 1000, min_rr: float = 0.5, max_rr: float = 20, min_delta: float = 0.5
) -> tuple[float, float]:
    """Compute respiratory rate in BPM from peaks of PPG signal.

    Args:
        data (array): RSP signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        min_rr (float, optional): Minimum RR interval in seconds. Defaults to 0.5 s.
        max_rr (float, optional): Maximum RR interval in seconds. Defaults to 20 s.
        min_delta (float, optional): Minimum delta between RR intervals in seconds. Defaults to 0.5 s.

    Returns:
        float: Respiratory rate (BPM).
    """
    peaks = find_peaks(data=data, sample_rate=sample_rate)
    rri = compute_rr_intervals(peaks=peaks)
    rmask = filter_rr_intervals(rr_ints=rri, sample_rate=sample_rate, min_rr=min_rr, max_rr=max_rr, min_delta=min_delta)

    bpm = 60 / (np.nanmean(rri[rmask == 0]) / sample_rate)
    qos = rmask[rmask == 0].size / rmask.size
    return bpm, qos


def compute_respiratory_rate_from_fft(
    data: npt.NDArray,
    sample_rate: float = 1000,
    lowcut: float = 0.05,
    highcut: float = 3.0,
) -> tuple[float, float]:
    """Compute respiratory rate in BPM from FFT of respiratory signal.

    Args:
        data (array): RSP signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        lowcut (float, optional): Lowcut frequency in Hz. Defaults to 0.05 Hz.
        highcut (float, optional): Highcut frequency in Hz. Defaults to 3.0 Hz.

    Returns:
        float: Respiratory rate (BPM).
    """
    freqs, sp = compute_fft(data, sample_rate=sample_rate, window="blackman")
    l_idx = np.where(freqs >= lowcut)[0][0]
    r_idx = np.where(freqs >= highcut)[0][0]
    ps = 2 * np.abs(sp)
    fft_pk_idx = np.argmax(ps[l_idx:r_idx]) + l_idx
    bpm = freqs[fft_pk_idx] * 60
    qos = ps[fft_pk_idx] / np.sum(ps)
    return bpm, qos
