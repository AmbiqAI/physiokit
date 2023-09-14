from ..signal import filter_signal


def clean(data, lowcut: float = 0.05, highcut: float = 3, sample_rate=1000, order: int = 3, axis: int = -1):
    """Clean respiratory signal using bandpass filter.

    Args:
        data (npt.NDArray): Signal
        lowcut (float, optional): Lower cutoff in Hz. Defaults to 0.05 Hz (3 bpm).
        highcut (float, optional): Upper cutoff in Hz. Defaults to 3 Hz (180 bpm).
        sample_rate (int, optional): Sample rate in Hz. Defaults to 1000 Hz.
        order (int, optional): Filter order. Defaults to 3 (3rd order Butterworth filter).
        axis (int, optional): Axis to apply against. Defaults to -1.

    Returns:
        npt.NDArray: Cleaned signal
    """

    # Bandpass filter
    return filter_signal(
        data=data,
        lowcut=lowcut,
        highcut=highcut,
        sample_rate=sample_rate,
        order=order,
        forward_backward=True,
        axis=axis,
    )
