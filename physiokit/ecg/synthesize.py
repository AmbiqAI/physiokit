import neurokit2 as nk


def synthesize(
    duration: float = 10,
    sample_rate: int = 1000,
    heart_rate: int = 60,
    heart_rate_std: int = 1,
    leads: int = 1,
):
    """Generate synthetic ECG signal. Utilize pk.signal.noise methods to make more realistic.

    Args:
        duration (float, optional): Duration in seconds. Defaults to 10 sec.
        sample_rate (int, optional): Sample rate in Hz. Defaults to 1000 Hz.
        heart_rate (int, optional): Heart rate in BPM. Defaults to 60 BPM.
        heart_rate_std (int, optional): Heart rate standard deviation in BPM. Defaults to 1 BPM.
        leads (int, optional): Number of leads. Defaults to 1.

    Returns:
        npt.NDArray: Synthetic ECG signal
    """
    # lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    ecg_data = nk.ecg_simulate(
        duration=duration,
        sampling_rate=sample_rate,
        heart_rate=heart_rate,
        heart_rate_std=heart_rate_std,
        method="ecgsyn" if leads == 1 else "multi",
        # Dont inject noise here
        noise=0,
    )
    # Return as numpy array
    if leads > 1:
        ecg_data = ecg_data.to_numpy().T

    return ecg_data
