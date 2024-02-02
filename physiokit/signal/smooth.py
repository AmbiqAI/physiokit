import numpy as np
import numpy.typing as npt
import scipy.ndimage
import scipy.signal


def signal_smooth_savgol(
    data: npt.NDArray,
    window_length: int | None = None,
    polyorder: int = 3,
    sample_rate: float = 1000,
    axis: int = -1,
) -> npt.NDArray:
    """Smooths signal using savitzky-golay filter

    Args:
        data (npt.NDArray): Signal
        window_length (int | None, optional): Filter window length. Defaults to None.
        polyorder (int, optional): Poly fit order. Defaults to 3.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        axis (int, optional): Axis to filter along. Defaults to -1.

    Returns:
        npt.NDArray: Smoothed signal

    """

    if window_length is None:
        window_length = sample_rate // 10

    if window_length % 2 == 0 or window_length == 0:
        window_length += 1

    return scipy.signal.savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=axis)


def signal_smooth_boxcar(data: npt.NDArray, size: int = 10) -> npt.NDArray:
    """Smooth signal using boxcar method.

    Args:
        data (npt.NDArray): Signal
        size (int, optional): Kernel size. Defaults to 10.

    Returns:
        npt.NDArray: Smoothed signal
    """

    return scipy.ndimage.uniform_filter1d(data, size, mode="nearest")


def signal_smooth_boxzen(data: npt.NDArray, size: int = 10) -> npt.NDArray:
    """Smooth signal using boxzen method.

    Args:
        data (npt.NDArray): Signal
        size (int, optional): Kernel size. Defaults to 10.

    Returns:
        npt.NDArray: Smoothed signal
    """

    # 1st pass - boxcar kernel
    smoothed = scipy.ndimage.uniform_filter1d(data, size, mode="nearest")
    # 2nd pass - parzen kernel
    smoothed = signal_smooth_conv(smoothed, kernel="parzen", size=size)
    return smoothed


def signal_smooth_median(data: npt.NDArray, size: int = 5) -> npt.NDArray:
    """Smooth signal using median filter.

    Args:
        data (npt.NDArray): Signal
        size (int, optional): Kernel size. Defaults to 5.

    Returns:
        npt.NDArray: Smoothed signal
    """
    # Enforce odd kernel size.
    if size % 2 == 0:
        size += 1
    smoothed = scipy.signal.medfilt(data, kernel_size=size)
    return smoothed


def signal_smooth_conv(data: npt.NDArray, kernel: str, size: int = 5):
    """Smooth signal using convolution.

    Args:
        data (npt.NDArray): Signal
        kernel (str): Kernel type
        size (int, optional): Kernel size. Defaults to 10.

    Returns:
        npt.NDArray: Smoothed signal
    """

    # Get window
    window = scipy.signal.get_window(kernel, size)
    w = window / window.sum()

    # Extend signal edges to avoid boundary effects
    x = np.concatenate((data[0] * np.ones(size), data, data[-1] * np.ones(size)))

    # Compute moving average
    smoothed = np.convolve(w, x, mode="same")
    smoothed = smoothed[size:-size]
    return smoothed
