import numpy as np
import numpy.typing as npt
import scipy.interpolate as spi
from typing import Literal

from ..signal import filter_signal
from .peaks import filter_peaks, find_peaks


def compute_respiratory_rate(data: npt.NDArray, sample_rate: float = 1000, method: str = "fft"):
    """Compute respiratory rate from signal.
    Args:
        data (array): RSP signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        method (str, optional): Method to compute respiratory rate. Defaults to 'fft'.
    Returns:
        float: Respiratory rate (BPM).
    """

    if method == "fft":
        return compute_respiratory_rate_from_fft(data=data, sample_rate=sample_rate)

    if method == "peak":
        return compute_respiratory_rate_from_peaks(data=data, sample_rate=sample_rate)

    raise NotImplementedError(f"Respiratory rate computation method {method} not implemented.")


def compute_respiratory_rate_from_peaks(
    data: npt.NDArray,
    sample_rate: float = 1000,
) -> float:
    """Compute respiratory rate from peaks of PPG signal.
    Args:
        data (array): PPG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
    Returns:
        float: Respiratory rate (BPM).
    """
    peaks = find_peaks(
        data=data,
        sample_rate=sample_rate,
    )
    peaks = filter_peaks(
        peaks=peaks,
        sample_rate=sample_rate,
    )
    return 60 / (np.diff(peaks).mean() / sample_rate)


def compute_respiratory_rate_from_fft(
    data: npt.NDArray,
    sample_rate: float = 1000,
    lowcut: float = 0.05,
    highcut: float = 3.0,
) -> float:
    """Compute respiratory rate from FFT of respiratory signal.
    Args:
        data (array): RSP signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
    Returns:
        float: Respiratory rate (BPM).
    """
    freqs, sp = compute_fft(data, sample_rate)
    l_idx = np.where(freqs >= lowcut)[0][0]
    r_idx = np.where(freqs >= highcut)[0][0]
    ps = 2 * np.abs(sp)
    fft_pk_idx = np.argmax(ps[l_idx:r_idx]) + l_idx
    bpm = freqs[fft_pk_idx] * 60
    return bpm


def compute_respiratory_rate_from_ppg(
    ppg: npt.NDArray,
    peaks: npt.NDArray,
    troughs: npt.NDArray|None = None,
    rri: npt.NDArray|None = None,
    sample_rate: float = 1000,
    method: Literal["riav", "riiv", "rifv"] = "rifv",
    lowcut: float = 0.1,
    highcut: float = 1.0,
    order: int = 3,
    threshold: float|None = 0.85,
) -> tuple[float, float]:
    """Compute respiratory rate from PPG signal using given method.
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
    Returns:
        float: Respiratory rate (BPM).
    """
    if peaks.size < 4:
        raise ValueError("At least 4 peaks are required to compute respiratory rate")

    ts = np.arange(peaks[0], peaks[-1], 1)

    if method == "riav":
        rsp = ppg[peaks] - ppg[troughs]
    elif method == "riiv":
        rsp = ppg[peaks]
    else:
        rsp = rri

    rsp = spi.interp1d(peaks, rsp, kind="linear", fill_value="extrapolate")(ts)
    rsp = filter_signal(rsp, lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=order)

    freqs, rsp_sp = compute_fft(ppg, sample_rate=sample_rate)
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

    rsp_bpm_weights = rsp_sp[fft_pk_indices]
    tgt_pwr = np.sum(rsp_bpm_weights)
    background_pwr = np.sum(rsp_ps - tgt_pwr)
    qnr = tgt_pwr / background_pwr
    rsp_bpm = 60*np.sum(rsp_bpm_weights*freqs[fft_pk_indices])/tgt_pwr
    return rsp_bpm, qnr


def compute_fft(
    data: npt.NDArray,
    sample_rate: float = 1000,
    axis: int = -1,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute FFT of RSP signal.
    Args:
        data (array): RSP signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        axis (int, optional): Axis to compute FFT. Defaults to -1.
    Returns:
        tuple[array, array]: Frequencies and FFT of RSP signal.
    """
    data_len = data.shape[axis]
    fft_len = int(2 ** np.ceil(np.log2(data_len)))
    fft_win = np.blackman(data_len)
    amp_corr = 1.93

    freqs = np.fft.fftfreq(fft_len, 1 / sample_rate)
    sp = amp_corr * np.fft.fft(fft_win * data, fft_len, axis=axis) / data_len
    return freqs, sp
