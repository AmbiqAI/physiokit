"""## ECG cleaning methods."""
import numpy as np
import numpy.typing as npt

from ..signal import filter_signal


def clean(
    data: npt.NDArray,
    lowcut: float = 0.5,
    highcut: float = 30,
    sample_rate: float = 1000,
    order: int = 3,
    axis: int = -1,
    forward_backward: bool = True,
) -> npt.NDArray:
    """Clean ECG signal by applying biquad filter.

    By default, a 3rd order Butterworth bandpass filter from 0.5 to 30 Hz is applied.

    Args:
        data (npt.NDArray): ECG signal.
        lowcut (float, optional): Lower cutoff in Hz. Defaults to 0.5 Hz.
        highcut (float, optional): Upper cutoff in Hz. Defaults to 30 Hz.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        order (int, optional): Filter order. Defaults to 3 (3rd order Butterworth filter).
        axis (int, optional): Axis to apply against. Defaults to -1.
        forward_backward (bool, optional): Apply filter forward and backward. Defaults to True.

    Returns:
        npt.NDArray: Cleaned ECG signal.
    """

    # Bandpass filter
    ecg_clean = filter_signal(
        data=data,
        lowcut=lowcut,
        highcut=highcut,
        sample_rate=sample_rate,
        order=order,
        axis=axis,
        forward_backward=forward_backward,
    )

    return ecg_clean


def square_filter_mask(rr_ints: npt.NDArray, lowcut: float = 300, highcut: float = 900) -> npt.NDArray:
    """Mask out RR intervals that fall outside bounds.

    Args:
        rr_ints (npt.NDArray): RR-interval list in ms.
        lowcut (float, optional): Lower cutoff limit. Defaults to 300 ms.
        highcut (float, optional): Upper cutoff limit. Defaults to 900 ms.

    Returns:
        npt.NDArray: RR rejection mask 0=accept, 1=reject.
    """
    rr_mask = np.where((rr_ints < lowcut) | (rr_ints > highcut), 1, 0)
    return rr_mask
