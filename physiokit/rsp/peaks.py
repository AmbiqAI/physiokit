import numpy as np
import numpy.typing as npt
import scipy.ndimage as spn

from ..signal import quotient_filter_mask

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
    peaks = []
    return np.array(peaks, dtype=int)


def filter_peaks(
        peaks: npt.NDArray,
        sample_rate: float = 1000,
    ) -> npt.NDArray:
    """Filter out peaks with intervals outside of normal range.
    Args:
        peaks (array): Beat peaks.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
    Returns:
        npt.NDArray: Filtered peaks.
    """
    lowcut = 0.05*sample_rate
    highcut = 3*sample_rate

    # Capture RR intervals
    beat_ints = np.diff(peaks)
    beat_ints = np.hstack((beat_ints[0], beat_ints))

    # Filter out peaks with RR intervals outside of normal range
    mask = np.where((beat_ints < lowcut) | (beat_ints > highcut), 1, 0)

    # Filter out peaks that deviate more than 30%
    mask = quotient_filter_mask(beat_ints, mask=mask, lowcut=0.7, highcut=1.3)
    filt_peaks = peaks[np.where(mask == 0)[0]]
    return filt_peaks
