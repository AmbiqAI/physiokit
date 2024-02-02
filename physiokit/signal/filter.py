import functools
import os

import numpy as np
import numpy.typing as npt
import scipy.interpolate as spi
import scipy.ndimage as spn
import scipy.signal as sps


@functools.cache
def get_butter_sos(
    lowcut: float | None = None,
    highcut: float | None = None,
    sample_rate: float = 1000,
    order: int = 3,
) -> npt.NDArray:
    """Compute biquad filter coefficients as SOS. This function caches.
    For lowpass, lowcut is required and highcut is ignored.
    For highpass, highcut is required and lowcut is ignored.
    For bandpass, both lowcut and highcut are required.

    Args:
        lowcut (float|None): Lower cutoff in Hz. Defaults to None.
        highcut (float|None): Upper cutoff in Hz. Defaults to None.
        sample_rate (float): Sampling rate in Hz. Defaults to 1000 Hz.
        order (int, optional): Filter order. Defaults to 3.

    Returns:
        npt.NDArray: SOS
    """
    nyq = sample_rate / 2
    if lowcut is not None and highcut is not None:
        freqs = [lowcut / nyq, highcut / nyq]
        btype = "bandpass"
    elif lowcut is not None:
        freqs = lowcut / nyq
        btype = "highpass"
    elif highcut is not None:
        freqs = highcut / nyq
        btype = "lowpass"
    else:
        raise ValueError("At least one of lowcut or highcut must be specified")
    sos = sps.butter(order, freqs, btype=btype, output="sos")
    return sos


def generate_arm_biquad_sos(
    lowcut: float,
    highcut: float,
    sample_rate: float,
    order: int = 3,
    var_name: str = "biquadFilter",
) -> str:
    """Generate ARM CMSIS second order section coefficients.

    Args:
        lowcut (float): Lower cutoff in Hz.
        highcut (float): Upper cutoff in Hz.
        sample_rate (float): Sampling rate in Hz.
        order (int, optional): Filter order. Defaults to 3.
        var_name (str, optional): Variable name. Defaults to 'biquadFilter'.

    Returns:
        str: ARM CMSIS second order section coefficients.
    """
    sos = get_butter_sos(lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=order)
    # Each section needs to be mapped as follows:
    #   [b0, b1, b2, a0, a1, a2] -> [b0, b1, b2, -a1, -a2]
    sec_len_name = f"{var_name.upper()}_NUM_SECS"
    arm_sos = sos[:, [0, 1, 2, 4, 5]] * [1, 1, 1, -1, -1]
    coef_str = ", ".join(f"{os.linesep:<4}{c}" if i % 5 == 0 else f"{c}" for i, c in enumerate(arm_sos.flatten()))
    arm_str = (
        f"#define {sec_len_name} ({order:0d}){os.linesep}"
        f"static float32_t {var_name}State[2 * {sec_len_name}];{os.linesep}"
        f"static float32_t {var_name}[5 * {sec_len_name}] = {{ {coef_str}\n}};{os.linesep}"
    )
    return arm_str


def resample_signal(
    data: npt.NDArray, sample_rate: float = 1000, target_rate: float = 500, axis: int = -1
) -> npt.NDArray:
    """Resample signal using scipy FFT-based resample routine.

    NOTE: For very large signals, this may be slow. Consider using resample_poly instead.

    Args:
        data (npt.NDArray): Signal
        sample_rate (float): Signal sampling rate. Defaults to 1000 Hz.
        target_rate (float): Target sampling rate. Defaults to 500 Hz.
        axis (int, optional): Axis to resample along. Defaults to -1.

    Returns:
        npt.NDArray: Resampled signal
    """
    desired_length = int(np.round(data.shape[axis] * target_rate / sample_rate))
    return sps.resample(data, desired_length, axis=axis)


def resample_categorical(data: npt.NDArray, sample_rate: float, target_rate: float, axis: int = 0) -> npt.NDArray:
    """Resample categorical data using nearest neighbor.

    Args:
        data (npt.NDArray): Signal
        sample_rate (float): Signal sampling rate
        target_rate (float): Target sampling rate
        axis (int, optional): Axis to resample along. Defaults to 0.

    Returns:
        npt.NDArray: Resampled signal
    """
    if sample_rate == target_rate:
        return data
    ratio = target_rate / sample_rate
    actual_length = data.shape[axis]
    target_length = int(np.round(data.shape[axis] * ratio))
    interp_fn = spi.interp1d(np.arange(0, actual_length), data, kind="nearest", axis=axis)
    return interp_fn(np.arange(0, target_length)).astype(data.dtype)


def normalize_signal(data: npt.NDArray, eps: float = 1e-3, axis: int = -1) -> npt.NDArray:
    """Normalize signal about its mean and std.

    Args:
        data (npt.NDArray): Signal
        eps (float, optional): Epsilon added to st. dev. Defaults to 1e-3.
        axis (int, optional): Axis to normalize along. Defaults to -1.

    Returns:
        npt.NDArray: Normalized signal
    """
    mu = np.nanmean(data, axis=axis)
    std = np.nanstd(data, axis=axis) + eps
    return (data - mu) / std


def filter_signal(
    data: npt.NDArray,
    lowcut: float | None = None,
    highcut: float | None = None,
    sample_rate: float = 1000,
    order: int = 2,
    axis: int = -1,
    forward_backward: bool = True,
) -> npt.NDArray:
    """Apply SOS filter to signal using butterworth design and cascaded filter.

    Args:
        data (npt.NDArray): Signal
        lowcut (float|None): Lower cutoff in Hz. Defaults to None.
        highcut (float|None): Upper cutoff in Hz. Defaults to None.
        sample_rate (float): Sampling rate in Hz Defaults to 1000 Hz.
        order (int, optional): Filter order. Defaults to 2.
        forward_backward (bool, optional): Apply filter forward and backwards. Defaults to True.

    Returns:
        npt.NDArray: Filtered signal
    """
    sos = get_butter_sos(lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=order)
    if forward_backward:
        return sps.sosfiltfilt(sos, data, axis=axis)
    return sps.sosfilt(sos, data, axis=axis)


def remove_baseline_wander(
    data: npt.NDArray,
    cutoff: float = 0.05,
    quality: float = 0.005,
    sample_rate: float = 1000,
    axis: int = -1,
    forward_backward: bool = True,
) -> npt.NDArray:
    """Remove baseline wander from signal using a notch filter.

    Args:
        data (npt.NDArray): Signal
        cutoff (float, optional): Cutoff frequency in Hz. Defaults to 0.05.
        quality (float, optional): Quality factor. Defaults to 0.005.
        sample_rate (float): Sampling rate in Hz. Defaults to 1000 Hz.
        axis (int, optional): Axis to filter along. Defaults to 0.
        forward_backward (bool, optional): Apply filter forward and backwards. Defaults to True.

    Returns:
        npt.NDArray: Filtered signal
    """
    b, a = sps.iirnotch(cutoff, Q=quality, fs=sample_rate)
    if forward_backward:
        return sps.filtfilt(b, a, data, axis=axis)
    return sps.lfilter(b, a, data, axis=axis)


def quotient_filter_mask(
    data: npt.NDArray, mask: npt.NDArray | None = None, iterations: int = 2, lowcut: float = 0.8, highcut: float = 1.2
) -> npt.NDArray:
    """Applies a quotient filter to identify outliers from list.

    Args:
        data (npt.NDArray): Signal
        mask (npt.NDArray | None, optional): Rejection mask. Defaults to None.
        iterations (int, optional): # iterations to apply. Defaults to 2.
        lowcut (float, optional): Lower cutoff ratio. Defaults to 0.8.
        highcut (float, optional): Upper cutoff ratio. Defaults to 1.2.

    Returns:
        npt.NDArray: Rejection mask 0=accept, 1=reject.
    """

    if mask is None:
        mask = np.zeros_like(data, dtype=int)

    for _ in range(iterations):
        # Get indices of intervals to be filtered
        filt_idxs = np.where(mask == 0)[0]
        if filt_idxs.size <= 1:
            break
        filt_ints = data[filt_idxs]
        # Compute quotient of each interval with the next
        filt_deltas = np.zeros(filt_ints.size)
        filt_deltas[1:] = filt_ints[:-1] / filt_ints[1:]
        filt_deltas[0] = filt_deltas[1]
        # Get indices of intervals that are outside the range
        delta_idxs = np.where((filt_deltas < lowcut) | (filt_deltas > highcut))[0]
        # Update mask with rejected intervals
        mask[filt_idxs[delta_idxs]] = 1
        # Break if no intervals are rejected
        if delta_idxs.size == 0:
            break
    # END FOR

    return mask


def moving_gradient_filter(
    data: npt.NDArray,
    sample_rate: float = 1000,
    sig_window: float = 0.1,
    avg_window: float = 1.0,
    sig_prom_weight: float = 1.5,
    mode: str = "nearest",
    fval=0,
) -> npt.NDArray:
    """Compute moving gradient filter to identify peaks in stream of data.

    Args:
        data (array): Data stream.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        sig_window (float, optional): Window size in seconds to compute signal gradient. Defaults to 0.1 s.
        avg_window (float, optional): Window size in seconds to compute average gradient. Defaults to 1.0 s.
        sig_prom_weight (float, optional): Weight to compute minimum signal height. Defaults to 1.5.

    Returns:
        array: Moving gradient filter.
    """
    # Compute gradient of signal and average.
    abs_grad = np.abs(np.gradient(data))
    sig_kernel = int(np.rint(sig_window * sample_rate))
    avg_kernel = int(np.rint(avg_window * sample_rate))

    # Smooth gradients
    sig_grad = spn.uniform_filter1d(abs_grad, sig_kernel, mode=mode, cval=fval)
    avg_grad = spn.uniform_filter1d(sig_grad, avg_kernel, mode=mode, cval=fval)

    # Apply prominance weight
    min_qrs_height = sig_prom_weight * avg_grad

    # Remove baseline
    rst = sig_grad - min_qrs_height
    return rst
