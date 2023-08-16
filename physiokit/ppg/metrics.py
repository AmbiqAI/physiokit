import numpy as np
import numpy.typing as npt

from ..signal import filter_signal
from .peaks import filter_peaks, find_peaks


def compute_heart_rate(data: npt.NDArray, sample_rate: float = 1000, method: str = "fft"):
    """Compute heart rate from ECG signal.
    Args:
        data (array): ECG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        method (str, optional): Method to compute heart rate. Defaults to 'fft'.
    Returns:
        float: Heart rate in BPM.
    """

    if method == "fft":
        return compute_heart_rate_from_fft(data=data, sample_rate=sample_rate)

    if method == "peak":
        return compute_heart_rate_from_peaks(data=data, sample_rate=sample_rate)

    raise NotImplementedError(f"Heart rate computation method {method} not implemented.")


def compute_heart_rate_from_peaks(
    data: npt.NDArray,
    sample_rate: float = 1000,
) -> float:
    """Compute heart rate from peaks of ECG signal.
    Args:
        data (array): ECG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
    Returns:
        float: Heart rate in BPM.
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


def compute_heart_rate_from_fft(
    data: npt.NDArray, sample_rate: float = 1000, lowcut: float = 0.7, highcut: float = 4.0
) -> float:
    """Compute heart rate from FFT of PPG signal.
    Args:
        data (array): PPG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
    Returns:
        float: Heart rate in BPM.
    """
    freqs, sp = compute_fft(data, sample_rate)
    l_idx = np.where(freqs >= lowcut)[0][0]
    r_idx = np.where(freqs >= highcut)[0][0]
    ps = 2 * np.abs(sp)
    fft_pk_idx = np.argmax(ps[l_idx:r_idx]) + l_idx
    hr = 60 * freqs[fft_pk_idx]
    return hr


def compute_fft(
    data: npt.NDArray,
    sample_rate: float = 1000,
    axis: int = -1,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute FFT of PPG signal.
    Args:
        data (array): PPG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        axis (int, optional): Axis to compute FFT. Defaults to -1.
    Returns:
        tuple[array, array]: Frequencies and FFT of PPG signal.
    """
    data_len = data.shape[axis]
    fft_len = int(2 ** np.ceil(np.log2(data_len)))
    fft_win = np.blackman(data_len)
    amp_corr = 1.93

    freqs = np.fft.fftfreq(fft_len, 1 / sample_rate)
    sp = amp_corr * np.fft.fft(fft_win * data, fft_len, axis=axis) / data_len
    return freqs, sp


def compute_spo2_from_perfusion(
    dc1: float, ac1: float, dc2: float, ac2: float, coefs: tuple[float, float, float] = (1, 0, 0)
) -> float:
    """Compute SpO2 from ratio of perfusion indexes (AC/DC).
        MAX30101: [1.5958422, -34.6596622, 112.6898759]
        MAX8614X: [-16.666666, 8.333333, 100]
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
    lowcut: float = 0.7,
    highcut: float = 4,
) -> float:
    """Compute SpO2 from PPG signals in time domain.

    Args:
        ppg1 (array): 1st PPG signal (e.g RED).
        ppg2 (array): 2nd PPG signal (e.g. IR).
        sampling_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        coefs (tuple[float, float, float], optional): Calibration coefficients. Defaults to (1, 0, 0).
    Returns:
        float: SpO2 value
    """

    # Compute DC
    ppg1_dc = np.mean(ppg1)
    ppg2_dc = np.mean(ppg2)

    # Bandpass filter
    ppg1_clean = filter_signal(
        data=ppg1, lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=3, forward_backward=True
    )

    ppg2_clean = filter_signal(
        data=ppg2, lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=3, forward_backward=True
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
    lowcut: float = 0.7,
    highcut: float = 4.0,
    axis: int = -1,
) -> float:
    """Compute SpO2 from PPG signals in frequency domain.
    Args:
        ppg1 (array): 1st PPG signal (e.g RED).
        ppg2 (array): 2nd PPG signal (e.g. IR).
        coefs (tuple[float, float, float], optional): Calibration coefficients. Defaults to (1, 0, 0).
        sampling_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        lowcut (float, optional): Lowcut frequency in Hz. Defaults to 0.7 Hz.
        highcut (float, optional): Highcut frequency in Hz. Defaults to 4.0 Hz.
        axis (int, optional): Axis along which to compute the FFT. Defaults to -1.
    Returns:
        float: SpO2 value
    """

    data_len = ppg1.shape[axis]
    fft_len = int(2 ** np.ceil(np.log2(data_len)))
    fft_win = np.blackman(data_len)
    amp_corr = 1.93

    freqs = np.fft.fftfreq(fft_len, 1 / sample_rate)
    l_idx = np.where(freqs >= lowcut)[0][0]
    r_idx = np.where(freqs >= highcut)[0][0]

    # Compute DC
    ppg1_dc = np.mean(ppg1)
    ppg2_dc = np.mean(ppg2)

    # Bandpass filter
    ppg1_clean = filter_signal(
        data=ppg1, lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=3, forward_backward=True
    )

    ppg2_clean = filter_signal(
        data=ppg2, lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=3, forward_backward=True
    )

    ppg1_fft = np.fft.fft(fft_win * ppg1_clean, fft_len, axis=axis) / data_len
    ppg2_fft = np.fft.fft(fft_win * ppg2_clean, fft_len, axis=axis) / data_len

    ppg1_ps = 2 * amp_corr * np.abs(ppg1_fft)
    ppg2_ps = 2 * amp_corr * np.abs(ppg2_fft)

    fft_pk_idx = np.argmax(ppg1_ps[l_idx:r_idx] + ppg2_ps[l_idx:r_idx]) + l_idx

    # Compute AC
    ppg1_ac = ppg1_ps[fft_pk_idx]
    ppg2_ac = ppg2_ps[fft_pk_idx]

    spo2 = compute_spo2_from_perfusion(dc1=ppg1_dc, ac1=ppg1_ac, dc2=ppg2_dc, ac2=ppg2_ac, coefs=coefs)

    return spo2
