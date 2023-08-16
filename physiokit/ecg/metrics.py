import numpy as np
import numpy.typing as npt

from .peaks import filter_peaks, find_peaks


def compute_heart_rate(data: npt.NDArray, sample_rate: float = 1000, method: str = "peak"):
    """Compute heart rate from ECG signal.
    Args:
        data (array): ECG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        method (str, optional): Method to compute heart rate. Defaults to 'peak'.
    Returns:
        float: Heart rate in BPM.
    """

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
    # NOTE: Either filter or require input to be filtered
    peaks = find_peaks(
        data=data,
        sample_rate=sample_rate,
    )
    peaks = filter_peaks(
        peaks=peaks,
        sample_rate=sample_rate,
    )
    return 60 / (np.diff(peaks).mean() / sample_rate)
