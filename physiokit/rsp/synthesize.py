import neurokit2 as nk
import numpy.typing as npt


def synthesize(duration: float = 10, sample_rate: int = 1000, respiratory_rate: int = 15) -> npt.NDArray:
    """Generate synthetic respiratory signal. Utilize pk.signal.noise methods to make more realistic.

    Args:
        duration (float, optional): Duration in seconds. Defaults to 10.
        sample_rate (int, optional): Sample rate in Hz. Defaults to 1000 Hz.
        respiratory_rate (int, optional): Respiratory rate in breaths per minute. Defaults to 15 bpm.

    Returns:
        npt.NDArray: Synthetic respiratory signal
    """

    return nk.rsp_simulate(
        duration=duration,
        sampling_rate=sample_rate,
        respiratory_rate=respiratory_rate,
        method="breathmetrics",
        # Dont inject noise here
        noise=0,
        random_state=None,
        random_state_distort="spawn",
    )
