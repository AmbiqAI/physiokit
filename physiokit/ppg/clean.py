from ..signal import filter_signal


def clean(
    data,
    lowcut: float = 0.5,
    highcut: float = 4,
    sample_rate=1000,
):
    """Clean PPG signal using bandpass filter.

    Args:
        data (npt.NDArray): Signal
        lowcut (float, optional): Lower cutoff in Hz. Defaults to 0.7 Hz.
        highcut (float, optional): Upper cutoff in Hz. Defaults to 4 Hz.
        sample_rate (int, optional): Sample rate in Hz. Defaults to 1000 Hz.

    Returns:
        npt.NDArray: Cleaned signal
    """

    # Bandpass filter
    return filter_signal(
        data=data, lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=3, forward_backward=True
    )
