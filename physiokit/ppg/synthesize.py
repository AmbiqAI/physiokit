import neurokit2 as nk
import numpy.typing as npt


def synthesize(
    duration: float = 10,
    sample_rate: int = 1000,
    heart_rate: int = 60,
    frequency_modulation: float = 0,
    ibi_randomness: float = 0,
) -> npt.NDArray:
    """Generate synthetic PPG signal. Utilize pk.signal.noise methods to make more realistic.

    Args:
        duration (float, optional): Duration in seconds. Defaults to 10.
        sample_rate (int, optional): Sample rate in Hz. Defaults to 1000 Hz.
        heart_rate (int, optional): Heart rate in BPM. Defaults to 60 BPM.
        frequency_modulation (float, optional): Frequency modulation in Hz. Defaults to 0 Hz.
        ibi_randomness (float, optional): IBI randomness in Hz. Defaults to 0 Hz.

    Returns:
        npt.NDArray: Synthetic PPG signal
    """

    return nk.ppg_simulate(
        duration=duration,
        sampling_rate=sample_rate,
        heart_rate=heart_rate,
        frequency_modulation=frequency_modulation,
        ibi_randomness=ibi_randomness,
        # Dont inject noise here
        drift=0,
        motion_amplitude=0,
        powerline_amplitude=0,
        burst_number=0,
        burst_amplitude=0,
        random_state=None,
        random_state_distort="spawn",
        show=False,
    )
