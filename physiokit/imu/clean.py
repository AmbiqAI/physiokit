import numpy.typing as npt

from ..signal import filter_signal


def clean(
    data: npt.NDArray,
    lowcut: float = 3,
    highcut: float = 11,
    sample_rate: float = 1000,
    order: int = 3,
    axis: int = -1,
) -> npt.NDArray:
    """Clean accelerometer signal using biquad filter.

    By default applies a 3rd order Butterworth filter between 0.5 and 4 Hz.

    Args:
        data (npt.NDArray): Accelerometer signal
        lowcut (float, optional): Lower cutoff in Hz. Defaults to 3 Hz.
        highcut (float, optional): Upper cutoff in Hz. Defaults to 11 Hz.
        sample_rate (float, optional): Sample rate in Hz. Defaults to 1000 Hz.
        order (int, optional): Filter order. Defaults to 3 (3rd order Butterworth filter).
        axis (int, optional): Axis to apply against. Defaults to -1.

    Returns:
        npt.NDArray: Cleaned signal
    """

    return filter_signal(
        data=data,
        lowcut=lowcut,
        highcut=highcut,
        sample_rate=sample_rate,
        order=order,
        forward_backward=True,
        axis=axis,
    )
