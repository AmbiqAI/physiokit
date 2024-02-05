"""Add various noise sources to signal."""
import random

import numpy as np
import numpy.typing as npt

from .distort import (
    create_noise_artifacts,
    create_noise_distortions,
    create_powerline_noise,
)


def add_baseline_wander(
    data: npt.NDArray,
    amplitude: float = 0.1,
    frequency: float = 0.05,
    sample_rate: float = 1000,
    signal_sd: float | None = None,
) -> npt.NDArray:
    """Add baseline wander to signal.

    Args:
        data (npt.NDArray): Signal
        amplitude (float, optional): Amplitude in st dev. Defaults to 0.1.
        frequency (float, optional): Baseline wander frequency. Defaults to 0.05 Hz.
        sample_rate (float, optional): Sample rate in Hz. Defaults to 1000 Hz.
        signal_sd (float|None, optional): Signal standard deviation. Defaults to None.

    Returns:
        npt.NDArray: Signal w/ baseline wander
    """
    if signal_sd is None:
        signal_sd = np.nanstd(data)

    return data + create_noise_distortions(
        len(data),
        signal_sd=signal_sd,
        sample_rate=sample_rate,
        frequencies=frequency,
        amplitudes=amplitude,
        noise_shapes="laplace",
    )


def add_motion_noise(
    data: npt.NDArray,
    amplitude: float = 0.2,
    frequency: float = 0.5,
    sample_rate: float = 1000,
    signal_sd: float | None = None,
) -> npt.NDArray:
    """Add motion noise to signal.

    Args:
        data (npt.NDArray): Signal
        amplitude (float, optional): Amplitude in st dev. Defaults to 0.2.
        frequency (float, optional): Motion frequency in Hz. Defaults to 0.5 Hz.
        sample_rate (float, optional): Sample rate in Hz. Defaults to 1000 Hz.
        signal_sd (float|None, optional): Signal standard deviation. Defaults to None.

    Returns:
        npt.NDArray: Signal w/ motion noise
    """
    if signal_sd is None:
        signal_sd = np.nanstd(data)

    return data + create_noise_distortions(
        len(data),
        signal_sd=signal_sd,
        sample_rate=sample_rate,
        frequencies=frequency,
        amplitudes=amplitude,
        noise_shapes="laplace",
    )


def add_burst_noise(
    data: npt.NDArray,
    amplitude: float = 1,
    frequency: float = 100,
    num_bursts: int = 1,
    sample_rate: float = 1000,
    signal_sd: float | None = None,
) -> npt.NDArray:
    """Add high frequency burst noise to signal.

    Args:
        data (npt.NDArray): Signal
        amplitude (float, optional): Amplitude in st dev. Defaults to 1.
        frequency (float, optional): High frequency burst in Hz. Defaults to 100 Hz.
        num_bursts (int, optional): # bursts to inject. Defaults to 1.
        sample_rate (float, optional): Sample rate in Hz. Defaults to 1000 Hz.
        signal_sd (float|None, optional): Signal standard deviation. Defaults to None.

    Returns:
        npt.NDArray: Signal w/ burst noise
    """

    if signal_sd is None:
        signal_sd = np.nanstd(data)
    return data + create_noise_artifacts(
        len(data),
        signal_sd=signal_sd,
        sample_rate=sample_rate,
        frequency=frequency,
        amplitude=amplitude,
        num_artifacts=num_bursts,
        min_artifact_percent=0.001,
        max_artifact_percent=0.01,
        artifacts_shape="laplace",
    )


def add_powerline_noise(
    data: npt.NDArray,
    amplitude: float = 0.01,
    frequency: float = 50,
    sample_rate: float = 1000,
    signal_sd: float | None = None,
) -> npt.NDArray:
    """Add powerline noise to signal.

    Args:
        data (npt.NDArray): Signal
        amplitude (float, optional): Amplitude in st dev. Defaults to 0.01.
        frequency (float, optional): Powerline frequency in Hz. Defaults to 50 Hz.
        sample_rate (float, optional): Sample rate in Hz. Defaults to 1000 Hz.
        signal_sd (float|None, optional): Signal standard deviation. Defaults to None.

    Returns:
        npt.NDArray: Signal w/ powerline noise
    """
    return data + create_powerline_noise(
        len(data),
        signal_sd=signal_sd,
        sample_rate=sample_rate,
        frequency=frequency,
        amplitude=amplitude,
    )


def add_noise_sources(
    data: npt.NDArray,
    amplitudes: list[float],
    frequencies: list[float],
    noise_shapes: list[str],
    sample_rate: float = 1000,
    signal_sd: float | None = None,
) -> npt.NDArray:
    """Add multiple noise sources to signal.

    Args:
        data (npt.NDArray): Signal
        amplitudes (list[float]): Amplitudes in st dev.
        frequencies (list[float]): Frequencies in Hz.
        noise_shapes (list[str]): Noise shapes.
        sample_rate (float, optional): Sample rate in Hz. Defaults to 1000 Hz.
        signal_sd (float|None, optional): Signal standard deviation. Defaults to None.

    Returns:
        npt.NDArray: Signal w/ noise
    """
    if signal_sd is None:
        signal_sd = np.nanstd(data)

    return data + create_noise_distortions(
        len(data),
        signal_sd=signal_sd,
        sample_rate=sample_rate,
        frequencies=frequencies,
        amplitudes=amplitudes,
        noise_shapes=noise_shapes,
    )


def add_emg_noise(data: npt.NDArray, scale: float = 1e-5, sample_rate: float = 1000) -> npt.NDArray:
    """Add EMG noise to signal.

    Args:
        data (npt.NDArray): Signal
        scale (float, optional): Noise scale. Defaults to 1e-5.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000.

    Returns:
        npt.NDArray: Signal with EMG noise
    """
    noise = np.tile(
        np.sin(np.linspace(-0.5 * np.pi, 1.5 * np.pi, int(sample_rate)) * 10 * sample_rate),
        int(np.ceil(data.size // sample_rate)),
    )
    return data + scale * noise[: data.size]


def add_lead_noise(data: npt.NDArray, scale: float = 1e-3) -> npt.NDArray:
    """Add lead noise to signal.

    Args:
        data (npt.NDArray): Signal
        scale (float, optional): Noise scale. Defaults to 1.

    Returns:
        npt.NDArray: Signal with lead noise
    """
    return data + np.random.normal(0, scale, size=data.shape)


def add_random_scaling(data: npt.NDArray, lower: float = 0.5, upper: float = 2.0) -> npt.NDArray:
    """Randomly scale signal.

    Args:
        data (npt.NDArray): Signal
        lower (float, optional): Lower bound. Defaults to 0.5.
        upper (float, optional): Upper bound. Defaults to 2.0.

    Returns:
        npt.NDArray: Signal with random scaling
    """
    return data * random.uniform(lower, upper)


def add_signal_attenuation():
    """Add signal attenuation to signal."""
    raise NotImplementedError()


def add_signal_cutout():
    """Add cutout augmentation to signal."""
    raise NotImplementedError()


def add_signal_shift(data: npt.NDArray, shift_amount: float = 0.1):
    """Add signal shift augmentation to signal."""
    shift_idx = random.randint(0, data.size)
    rst = data.copy()
    rst[shift_idx:] += shift_amount
    return rst
