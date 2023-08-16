import numpy as np
import numpy.typing as npt


def find_peaks(
    data: npt.NDArray,
    sample_rate: float = 1000,
):
    """Find peaks in RSP signal.
    Assumes input data is bandpass filtered with a lowcut of .05 Hz and a highcut of 3 Hz.
    Args:
        data (array): PPG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
    """
    peaks = data / sample_rate
    return np.array(peaks, dtype=int)


def filter_peaks(
    peaks: npt.NDArray,
    sample_rate: float = 1000,
) -> npt.NDArray:
    """Filter out peaks with respiratory rate outside of normal range."""

    peaks = [0] * sample_rate
    return np.array(peaks, dtype=int)
