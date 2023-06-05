import neurokit2 as nk

def synthesize(
        duration: float,
        sample_rate: int = 1000,
        heart_rate: int = 60,
        heart_rate_std: int = 1,
        leads: int = 1,
    ):
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
    # Single lead is returned as
    if leads > 1:
        ecg_data = ecg_data.to_numpy()

    return ecg_data
