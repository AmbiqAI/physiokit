import numpy as np
import numpy.typing as npt


def rescale_signal(
    x: npt.NDArray, old_min: float, old_max: float, new_min: float, new_max: float, clip: bool = True
) -> npt.NDArray:
    """Rescale signal to new range.

    Args:
        x (npt.NDArray): Signal
        old_min (float): Old minimum
        old_max (float): Old maximum
        new_min (float): New minimum
        new_max (float): New maximum
        clip (bool, optional): Clip values to range. Defaults to True.

    Returns:
        npt.NDArray: Rescaled signal
    """
    if clip:
        x = np.clip(x, old_min, old_max)
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def compute_fft(
    data: npt.NDArray,
    sample_rate: float = 1000,
    fft_len: int | None = None,
    window: str = "blackman",
    axis: int = -1,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute FFT of RSP signal.

    Args:
        data (array): RSP signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        fft_len (int | None, optional): FFT length. Defaults to None.
        window (str, optional): Window to apply. Defaults to 'blackman'.
        axis (int, optional): Axis to compute FFT. Defaults to -1.

    Returns:
        tuple[array, array]: Frequencies and FFT of signal.
    """
    data_len = data.shape[axis]
    if fft_len is None:
        fft_len = int(2 ** np.ceil(np.log2(data_len)))
    if window == "blackman":
        fft_win = np.blackman(data_len)
        amp_corr = 1.93
    else:
        fft_win = np.ones(data_len)
        amp_corr = 1.0
    freqs = np.fft.fftfreq(fft_len, 1 / sample_rate)
    sp = amp_corr * np.fft.fft(fft_win * data, fft_len, axis=axis) / data_len
    return freqs, sp
