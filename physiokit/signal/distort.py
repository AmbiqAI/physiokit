from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import scipy.ndimage


def add_distortions(
    signal: npt.NDArray,
    sample_rate: float = 1000,
    noise_shape: str = "laplace",
    noise_amplitude: float | list[float] = 0,
    noise_frequency: float | list[float] = 100,
    powerline_amplitude: float = 0,
    powerline_frequency: float = 50,
    artifacts_amplitude: float = 0,
    artifacts_frequency: float = 100,
    artifacts_number: int = 5,
    linear_drift: bool = False,
    epsilon: float = 1e-6,
):
    """Add various distortions to the signal.

    Args:
        signal (npt.NDArray): Signal.
        sample_rate (float, optional): Sampling rate. Defaults to 1000.
        noise_shape (str, optional): Noise shape. Defaults to "laplace".
        noise_amplitude (float|list[float], optional): Noise amplitude. Defaults to 0.
        noise_frequency (float|list[float], optional): Noise frequency. Defaults to 100.
        powerline_amplitude (float, optional): Powerline noise amplitude. Defaults to 0.
        powerline_frequency (float, optional): Powerline noise frequency. Defaults to 50.
        artifacts_amplitude (float, optional): Artifacts amplitude. Defaults to 0.
        artifacts_frequency (float, optional): Artifacts frequency. Defaults to 100.
        artifacts_number (int, optional): Number of artifacts. Defaults to 5.
        linear_drift (bool, optional): Add linear drift. Defaults to False.
        epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        npt.NDArray: Distorted signal.
    """

    # Make sure that noise_amplitude is a list.
    if not isinstance(noise_amplitude, Iterable):
        noise_amplitude = [noise_amplitude]

    signal_len = len(signal)
    signal_sd = np.nanstd(signal, ddof=1) + epsilon

    noise = np.zeros(signal_len)

    # Noise
    if min(noise_amplitude) > 0:
        noise += create_noise_distortions(
            signal_len=signal_len,
            signal_sd=signal_sd,
            sample_rate=sample_rate,
            amplitudes=noise_amplitude,
            frequencies=noise_frequency,
            noise_shapes=noise_shape,
        )

    # Powerline
    if powerline_amplitude > 0:
        noise += create_powerline_noise(
            signal_len=signal_len,
            signal_sd=signal_sd,
            sample_rate=sample_rate,
            frequency=powerline_frequency,
            amplitude=powerline_amplitude,
        )

    # Artifacts
    if artifacts_amplitude > 0:
        noise += create_noise_artifacts(
            signal_len=signal_len,
            signal_sd=signal_sd,
            sample_rate=sample_rate,
            frequency=artifacts_frequency,
            amplitude=artifacts_amplitude,
            num_artifacts=artifacts_number,
        )

    if linear_drift:
        noise += create_linear_drift(signal)

    distorted = signal + noise

    return distorted


def create_linear_drift(signal_len: int) -> npt.NDArray:
    """Create linear drift.

    Args:
        signal_len (int): Signal length.

    Returns:
        npt.NDArray: Linear drift.
    """
    linear_drift = np.arange(signal_len) * (1 / signal_len)
    return linear_drift


def create_noise_artifacts(
    signal_len: int,
    signal_sd: float | None = None,
    sample_rate: float = 1000,
    frequency: float = 0,
    amplitude: float = 0.1,
    num_artifacts: int = 5,
    min_artifact_percent=0.001,
    max_artifact_percent=0.01,
    artifacts_shape: str = "laplace",
):
    """Create noise artifact blips.

    Args:
        signal_len (int): Signal length.
        sample_rate (float, optional): Sampling rate. Defaults to 1000.
        frequency (float, optional): Noise frequency. Defaults to 0.
        amplitude (float, optional): Noise amplitude. Defaults to 0.1.
        num_artifacts (int, optional): Number of artifacts. Defaults to 5.
        min_artifact_percent (int, optional): Min artifact duration percentage. Defaults to 0.001.
        max_artifact_percent (int, optional): Max artifact duration percentage. Defaults to 0.01.
        artifacts_shape (str, optional): Noise shape. Defaults to "laplace".

    Returns:
        npt.NDArray: Noise artifacts.
    """
    # Generate random noise
    artifacts = create_noise_distortion(
        signal_len,
        sample_rate=sample_rate,
        frequency=frequency,
        amplitude=amplitude,
        noise_shape=artifacts_shape,
    )
    if artifacts.sum() == 0:
        return artifacts

    # Generate artifact regions
    min_duration = int(np.rint(len(artifacts) * min_artifact_percent))
    max_duration = int(np.rint(len(artifacts) * max_artifact_percent))
    artifact_durations = np.random.choice(range(min_duration, max_duration), size=num_artifacts)

    artifact_onsets = np.random.choice(len(artifacts) - max_duration, size=num_artifacts)
    artifact_offsets = artifact_onsets + artifact_durations

    # Create artifact mask
    artifact_idcs = np.array([False] * len(artifacts))
    for i in range(num_artifacts):
        artifact_idcs[artifact_onsets[i] : artifact_offsets[i]] = True

    artifacts[~artifact_idcs] = 0

    # Scale amplitude by the signal's standard deviation.
    if signal_sd is not None:
        amplitude *= signal_sd
    artifacts *= amplitude

    return artifacts


def create_powerline_noise(
    signal_len: int,
    signal_sd: float | None = None,
    sample_rate: float = 1000,
    frequency: float = 50,
    amplitude: float = 0.1,
):
    """Create powerline noise.

    Args:
        signal (npt.NDArray): Signal.
        sample_rate (float, optional): Sampling rate. Defaults to 1000.
        frequency (float, optional): Noise frequency. Defaults to 50.
        amplitude (float, optional): Noise amplitude. Defaults to 0.1.

    Returns:
        npt.NDArray: Powerline noise.
    """
    duration = signal_len / sample_rate
    noise = signal_simulate(
        duration=duration,
        sample_rate=sample_rate,
        frequency=frequency,
        amplitude=1,
    )
    if signal_sd is not None:
        amplitude *= signal_sd
    noise *= amplitude
    return noise


def create_noise_distortions(
    signal_len: int,
    signal_sd: float | None = None,
    sample_rate: float = 1000,
    frequencies: float | list[float] = 100,
    amplitudes: float | list[float] = 0.1,
    noise_shapes: str | list[str] = "laplace",
) -> npt.NDArray:
    """Create multiple noise distortions.

    Args:
        signal_len (int): Signal length.
        sample_rate (int, optional): Sampling rate. Defaults to 1000.
        frequencies (float|list[float], optional): Noise frequencies. Defaults to 100.
        amplitudes (float|list[float], optional): Noise amplitudes. Defaults to 0.1.
        noise_shapes (float|list[float], optional): Noise shapes. Defaults to "laplace".

    Returns:
        npt.NDArray: Noise distortions.
    """
    noise = np.zeros(signal_len)

    frequencies = frequencies if isinstance(frequencies, Iterable) else [frequencies]
    amplitudes = amplitudes if isinstance(amplitudes, Iterable) else [amplitudes]
    noise_shapes = [noise_shapes] if isinstance(noise_shapes, str) else noise_shapes

    for frequency, amplitude, noise_shape in zip(frequencies, amplitudes, noise_shapes):
        if signal_sd is not None:
            amplitude *= signal_sd
        noise += create_noise_distortion(
            signal_len=signal_len,
            sample_rate=sample_rate,
            frequency=frequency,
            amplitude=amplitude,
            noise_shape=noise_shape,
        )
    # END FOR
    return noise


def create_noise_distortion(
    signal_len: int,
    sample_rate: int = 1000,
    frequency: float = 100,
    amplitude: float = 0.1,
    noise_shape: str = "laplace",
) -> npt.NDArray:
    """Create noise distortion w/ given frequency and amplitude.

    Args:
        signal_len (int): Signal length.
        sample_rate (int, optional): Sampling rate. Defaults to 1000.
        frequency (float, optional): Noise frequency. Defaults to 100.
        amplitude (float, optional): Noise amplitude. Defaults to 0.1.
        noise_shape (str, optional): Noise shape. Defaults to "laplace".

    Returns:
        npt.NDArray: Noise distortion.
    """
    noise = np.zeros(signal_len)

    nyquist = sample_rate * 0.5
    duration = signal_len / sample_rate
    if frequency > nyquist or (1 / frequency) > duration:
        return noise

    noise_duration = int(duration * frequency)
    if noise_shape in ["normal", "gaussian"]:
        noise = np.random.normal(0, amplitude, noise_duration)
    elif noise_shape == "laplace":
        noise = np.random.laplace(0, amplitude, noise_duration)
    else:
        raise ValueError("'noise_shape' should be one of 'gaussian' or 'laplace'.")

    if len(noise) != signal_len:
        noise = scipy.ndimage.zoom(noise, signal_len / len(noise))
    return noise


def signal_simulate(
    duration: float = 10,
    sample_rate: float = 1000,
    frequency: float | list[float] = 1,
    amplitude: float | list[float] = 0.5,
    noise: float = 0,
) -> npt.NDArray:
    """Simulate a sinusoidal signal with noise.

    Args:
        duration (float, optional): Signal duration in seconds. Defaults to 10.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000.
        frequency (float, optional): Signal frequency in Hz. Defaults to 1.
        amplitude (float, optional): Signal amplitude. Defaults to 0.5.
        noise (float, optional): Noise amplitude. Defaults to 0.

    Returns:
        npt.NDArray: Simulated signal.
    """
    n_samples = int(np.rint(duration * sample_rate))
    period = 1 / sample_rate
    seconds = np.arange(n_samples) * period

    signal = np.zeros(seconds.size)
    frequencies = frequency if isinstance(frequency, Iterable) else [frequency]
    amplitudes = amplitude if isinstance(amplitude, Iterable) else [amplitude]

    for freq, amp in zip(frequencies, amplitudes):
        nyquist = sample_rate * 0.5
        if freq > nyquist or (1 / freq) > duration:
            continue
        signal += _signal_simulate_sinusoidal(x=seconds, frequency=freq, amplitude=amp)
        # Add random noise
        if noise > 0:
            signal += np.random.laplace(0, noise, len(signal))
    return signal


def _signal_simulate_sinusoidal(x: npt.NDArray, frequency: float = 100, amplitude: float = 0.5) -> npt.NDArray:
    """Simulate a sinusoidal signal.

    Args:
        x (npt.NDArray): Time vector.
        frequency (float, optional): Signal frequency in Hz. Defaults to 100.
        amplitude (float, optional): Signal amplitude. Defaults to 0.5.

    Returns:
        npt.NDArray: Simulated signal.
    """
    signal = amplitude * np.sin(2 * np.pi * frequency * x)
    return signal
