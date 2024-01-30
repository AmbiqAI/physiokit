import numpy as np
import numpy.typing as npt

from ..signal.distort import signal_simulate
from ..signal.smooth import signal_smooth_boxzen


def synthesize(
    duration: float = 10, sample_rate: float = 1000, respiratory_rate: int = 15, method: str = "breathmetrics"
) -> npt.NDArray:
    """Generate synthetic respiratory signal. Utilize pk.signal.noise methods to make more realistic.

    Args:
        duration (float, optional): Duration in seconds. Defaults to 10.
        sample_rate (float, optional): Sample rate in Hz. Defaults to 1000 Hz.
        respiratory_rate (int, optional): Respiratory rate in breaths per minute. Defaults to 15 bpm.

    Returns:
        npt.NDArray: Synthetic respiratory signal
    """

    length = duration * sample_rate

    if method.lower() == "sinusoidal":
        rsp = simulate_sinusoidal(duration=duration, sample_rate=sample_rate, respiratory_rate=respiratory_rate)
    else:
        rsp = simulate_breathmetrics(
            duration=duration,
            sample_rate=sample_rate,
            respiratory_rate=respiratory_rate,
        )
        rsp = rsp[0:length]

    return rsp


def simulate_sinusoidal(duration: float = 10, sample_rate: float = 1000, respiratory_rate: float = 15):
    """Simulate a basic sinusoidal respiratory signal.

    Args:
        duration (float, optional): Duration in seconds. Defaults to 10.
        sampling_rate (float, optional): Sample rate in Hz. Defaults to 1000 Hz.
        respiratory_rate (int, optional): Respiratory rate in breaths per minute. Defaults to 15 bpm.

    Returns:
        npt.NDArray: Sinusoidal respiratory signal
    """
    rsp = signal_simulate(
        duration=duration,
        sample_rate=sample_rate,
        frequency=respiratory_rate / 60,
        amplitude=0.5,
    )

    return rsp


def simulate_breathmetrics(
    duration: float = 10, sample_rate: float = 1000, respiratory_rate: float = 15
) -> npt.NDArray:
    """Simulate a respiratory signal using the breathmetrics method.

    Args:
        duration (float, optional): Duration in seconds. Defaults to 10.
        sample_rate (float, optional): Sample rate in Hz. Defaults to 1000 Hz.
        respiratory_rate (float, optional): Respiratory rate in breaths per minute. Defaults to 15 bpm.

    Returns:
        npt.NDArray: Simulated respiratory signal
    """
    n_cycles = int(respiratory_rate / 60 * duration)

    # Loop until it doesn't fail
    rsp = False
    while rsp is False:
        # Generate a longer than necessary signal so it won't be shorter
        rsp, _, __ = _simulate_breathmetrics_core(
            num_cycles=int(n_cycles * 1.5),
            sample_rate=sample_rate,
            breathing_rate=respiratory_rate / 60,
            signal_noise=0,
        )
    return rsp


def _simulate_breathmetrics_core(
    num_cycles: int = 100,
    sample_rate: float = 1000,
    breathing_rate: float = 0.25,
    average_amplitude: float = 0.5,
    amplitude_variance: float = 0.1,
    phase_variance: float = 0.1,
    inhale_pause_percent: float = 0.3,
    inhale_pause_avgLength: float = 0.2,
    inhale_pauseLength_variance: float = 0.5,
    exhale_pause_percent: float = 0.3,
    exhale_pause_avgLength: float = 0.2,
    exhale_pauseLength_variance: float = 0.5,
    pause_amplitude: float = 0.1,
    pause_amplitude_variance: float = 0.2,
    signal_noise: float = 0.1,
):
    """Simulates a recording of human airflow data by appending individually constructed sin waves and pauses in
    sequence. This is translated from the matlab code available `here.

    <https://github.com/zelanolab/breathmetrics/blob/master/simulateRespiratoryData.m>`_ by Noto, et al. (2018).

    Args:
        num_cycles (int, optional): Number of cycles to simulate. Defaults to 100.
        sampling_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        breathing_rate (float, optional): Breathing rate in Hz. Defaults to 0.25 Hz.
        average_amplitude (float, optional): Average amplitude of breaths. Defaults to 0.5.
        amplitude_variance (float, optional): Variance of amplitude. Defaults to 0.1.
        phase_variance (float, optional): Variance of phase. Defaults to 0.1.
        inhale_pause_percent (float, optional): Percent of breaths with inhale pause. Defaults to 0.3.
        inhale_pause_avgLength (float, optional): Average length of inhale pause. Defaults to 0.2.
        inhale_pauseLength_variance (float, optional): Variance of inhale pause length. Defaults to 0.5.
        exhale_pause_percent (float, optional): Percent of breaths with exhale pause. Defaults to 0.3.
        exhale_pause_avgLength (float, optional): Average length of exhale pause. Defaults to 0.2.
        exhale_pauseLength_variance (float, optional): Variance of exhale pause length. Defaults to 0.5.
        pause_amplitude (float, optional): Amplitude of pauses. Defaults to 0.1.
        pause_amplitude_variance (float, optional): Variance of pause amplitude. Defaults to 0.2.
        signal_noise (float, optional): Noise level. Defaults to 0.1.


    """
    # Define additional parameters
    sample_phase = sample_rate / breathing_rate
    inhale_pause_phase = np.round(inhale_pause_avgLength * sample_phase).astype(int)
    exhale_pause_phase = np.round(exhale_pause_avgLength * sample_phase).astype(int)

    # Normalize variance by average breath amplitude
    amplitude_variance_normed = average_amplitude * amplitude_variance
    amplitudes_with_noise = np.random.standard_normal(num_cycles) * amplitude_variance_normed + average_amplitude
    amplitudes_with_noise[amplitudes_with_noise < 0] = 0

    # Normalize phase by average breath length
    phase_variance_normed = phase_variance * sample_phase
    phases_with_noise = np.round(np.random.standard_normal(num_cycles) * phase_variance_normed + sample_phase).astype(
        int
    )
    phases_with_noise[phases_with_noise < 0] = 0

    # Normalize pause lengths by phase and variation
    inhale_pauseLength_variance_normed = inhale_pause_phase * inhale_pauseLength_variance
    inhale_pauseLengths_with_noise = np.round(
        np.random.standard_normal(num_cycles) * inhale_pauseLength_variance_normed + inhale_pause_phase
    ).astype(int)
    inhale_pauseLengths_with_noise[inhale_pauseLengths_with_noise < 0] = 0
    exhale_pauseLength_variance_normed = exhale_pause_phase * exhale_pauseLength_variance
    exhale_pauseLengths_with_noise = np.round(
        np.random.standard_normal(num_cycles) * exhale_pauseLength_variance_normed + inhale_pause_phase
    ).astype(int)

    # why inhale pause phase?
    exhale_pauseLengths_with_noise[exhale_pauseLengths_with_noise < 0] = 0

    # Normalize pause amplitudes
    pause_amplitude_variance_normed = pause_amplitude * pause_amplitude_variance

    # Initialize empty vector to fill with simulated data
    simulated_respiration = []

    # Initialize parameters to save
    inhale_onsets = np.zeros(num_cycles)
    exhale_onsets = np.zeros(num_cycles)

    inhale_pause_onsets = np.zeros(num_cycles)
    exhale_pause_onsets = np.zeros(num_cycles)

    inhale_lengths = np.zeros(num_cycles)
    inhale_pauseLengths = np.zeros(num_cycles)
    exhale_lengths = np.zeros(num_cycles)
    exhale_pauseLengths = np.zeros(num_cycles)

    inhale_peaks = np.zeros(num_cycles)
    exhale_troughs = np.zeros(num_cycles)

    i = 1
    for c in range(num_cycles):
        # Determine length of inhale pause for this cycle
        if np.random.uniform() < inhale_pause_percent:
            this_inhale_pauseLength = inhale_pauseLengths_with_noise[c]
            this_inhale_pause = np.random.standard_normal(this_inhale_pauseLength) * pause_amplitude_variance_normed
            this_inhale_pause[this_inhale_pause < 0] = 0
        else:
            this_inhale_pauseLength = 0
            this_inhale_pause = []

        # Determine length of exhale pause for this cycle
        if np.random.uniform() < exhale_pause_percent:
            this_exhale_pauseLength = exhale_pauseLengths_with_noise[c]
            this_exhale_pause = np.random.standard_normal(this_exhale_pauseLength) * pause_amplitude_variance_normed
            this_exhale_pause[this_exhale_pause < 0] = 0
        else:
            this_exhale_pauseLength = 0
            this_exhale_pause = []

        # Determine length of inhale and exhale for this cycle to main
        # breathing rate
        cycle_length = phases_with_noise[c] - (this_inhale_pauseLength + this_exhale_pauseLength)

        # If pauses are longer than the time alloted for this breath, set them
        # to 0 so a real breath can be simulated. This will deviate the
        # statistics from those initialized but is unavaoidable at the current
        # state
        if (cycle_length <= 0) or (cycle_length < min(phases_with_noise) / 4):
            this_inhale_pauseLength = 0
            this_inhale_pause = []
            this_exhale_pauseLength = 0
            this_exhale_pause = []
            cycle_length = phases_with_noise[c] - (this_inhale_pauseLength + this_exhale_pauseLength)

        # Compute inhale and exhale for this cycle
        this_cycle = np.sin(np.linspace(0, 2 * np.pi, cycle_length)) * amplitudes_with_noise[c]
        half_cycle = np.round(len(this_cycle) / 2).astype(int)
        this_inhale = this_cycle[0:half_cycle]
        this_inhale_length = len(this_inhale)
        this_exhale = this_cycle[half_cycle:]
        this_exhale_length = len(this_exhale)

        # Save parameters for checking
        inhale_lengths[c] = this_inhale_length
        inhale_pauseLengths[c] = this_inhale_pauseLength
        exhale_lengths[c] = this_exhale_length
        exhale_pauseLengths[c] = this_exhale_pauseLength
        inhale_onsets[c] = i
        exhale_onsets[c] = i + this_inhale_length + this_inhale_pauseLength

        if len(this_inhale_pause) > 0:
            inhale_pause_onsets[c] = i + this_inhale_length
        else:
            inhale_pause_onsets[c] = np.nan

        if len(this_exhale_pause) > 0:
            exhale_pause_onsets[c] = i + this_inhale_length + this_inhale_pauseLength + this_exhale_length
        else:
            exhale_pause_onsets[c] = np.nan

        # Compose breath from parameters
        this_breath = np.hstack([this_inhale, this_inhale_pause, this_exhale, this_exhale_pause])

        # Compute max flow for inhale and exhale for this breath
        max_ID = np.argmax(this_breath)
        min_ID = np.argmin(this_breath)
        inhale_peaks[c] = i + max_ID
        exhale_troughs[c] = i + min_ID

        # Append breath to simulated resperation vector
        simulated_respiration = np.hstack([simulated_respiration, this_breath])
        i = i + len(this_breath) - 1

    # Smooth signal
    simulated_respiration = signal_smooth_boxzen(simulated_respiration, size=sample_rate / 2)

    if signal_noise == 0:
        signal_noise = 0.0001

    noise_vector = np.random.uniform(size=simulated_respiration.shape) * average_amplitude
    simulated_respiration = simulated_respiration * (1 - signal_noise) + noise_vector * signal_noise
    raw_features = {
        "Inhale Onsets": inhale_onsets,
        "Exhale Onsets": exhale_onsets,
        "Inhale Pause Onsets": inhale_pause_onsets,
        "Exhale Pause Onsets": exhale_pause_onsets,
        "Inhale Lengths": inhale_lengths / sample_rate,
        "Inhale Pause Lengths": inhale_pauseLengths / sample_rate,
        "Exhale Lengths": exhale_lengths / sample_rate,
        "Exhale Pause Lengths": exhale_pauseLengths / sample_rate,
        "Inhale Peaks": inhale_peaks,
        "Exhale Troughs": exhale_troughs,
    }
    if len(inhale_pauseLengths[inhale_pauseLengths > 0]) > 0:
        avg_inhale_pauseLength = np.mean(inhale_pauseLengths[inhale_pauseLengths > 0])
    else:
        avg_inhale_pauseLength = 0

    if len(exhale_pauseLengths[exhale_pauseLengths > 0]) > 0:
        avg_exhale_pauseLength = np.mean(exhale_pauseLengths[exhale_pauseLengths > 0])
    else:
        avg_exhale_pauseLength = 0

    estimated_breathing_rate = (1 / np.mean(np.diff(inhale_onsets))) * sample_rate
    feature_stats = {
        "Breathing Rate": estimated_breathing_rate,
        "Average Inhale Length": np.mean(inhale_lengths / sample_rate),
        "Average Inhale Pause Length": avg_inhale_pauseLength / sample_rate,
        "Average Exhale Length": np.mean(exhale_lengths / sample_rate),
        "Average Exhale Pause Length": avg_exhale_pauseLength / sample_rate,
    }

    return simulated_respiration, raw_features, feature_stats
