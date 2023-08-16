import numpy as np
import numpy.typing as npt

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
