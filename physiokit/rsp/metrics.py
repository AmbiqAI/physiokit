import numpy as np
import numpy.typing as npt

from ..signal import compute_fft
from .defines import RspDualMetrics
from .peaks import compute_rr_intervals, filter_rr_intervals, find_peaks


def compute_respiratory_rate(
    data: npt.NDArray, sample_rate: float = 1000, method: str = "fft", **kwargs: dict
) -> tuple[float, float]:
    """Compute respiratory rate in BPM from signal.

    Args:
        data (array): RSP signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        method (str, optional): Method to compute respiratory rate. Defaults to 'fft'.
        **kwargs (dict): Keyword arguments to pass to method.

    Returns:
        tuple[float, float]: Respiratory rate (BPM) and qos metric.
    """
    match method:
        case "fft":
            bpm, qos = compute_respiratory_rate_from_fft(data=data, sample_rate=sample_rate, **kwargs)
        case "peak":
            bpm, qos = compute_respiratory_rate_from_peaks(data=data, sample_rate=sample_rate, **kwargs)
        case _:
            raise NotImplementedError(f"Respiratory rate computation method {method} not implemented.")
    # END MATCH
    return bpm, qos


def compute_respiratory_rate_from_peaks(
    data: npt.NDArray, sample_rate: float = 1000, min_rr: float = 0.5, max_rr: float = 20, min_delta: float = 0.5
) -> tuple[float, float]:
    """Compute respiratory rate in BPM from peaks of PPG signal.

    Args:
        data (array): RSP signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        min_rr (float, optional): Minimum RR interval in seconds. Defaults to 0.5 s.
        max_rr (float, optional): Maximum RR interval in seconds. Defaults to 20 s.
        min_delta (float, optional): Minimum delta between RR intervals in seconds. Defaults to 0.5 s.

    Returns:
        float: Respiratory rate (BPM).
    """
    peaks = find_peaks(data=data, sample_rate=sample_rate)
    rri = compute_rr_intervals(peaks=peaks)
    rmask = filter_rr_intervals(rr_ints=rri, sample_rate=sample_rate, min_rr=min_rr, max_rr=max_rr, min_delta=min_delta)

    bpm = 60 / (np.nanmean(rri[rmask == 0]) / sample_rate)
    qos = rmask[rmask == 0].size / rmask.size
    return bpm, qos


def compute_respiratory_rate_from_fft(
    data: npt.NDArray,
    sample_rate: float = 1000,
    lowcut: float = 0.05,
    highcut: float = 3.0,
) -> tuple[float, float]:
    """Compute respiratory rate in BPM from FFT of respiratory signal.

    Args:
        data (array): RSP signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        lowcut (float, optional): Lowcut frequency in Hz. Defaults to 0.05 Hz.
        highcut (float, optional): Highcut frequency in Hz. Defaults to 3.0 Hz.

    Returns:
        float: Respiratory rate (BPM).
    """
    freqs, sp = compute_fft(data, sample_rate=sample_rate, window="blackman")
    l_idx = np.where(freqs >= lowcut)[0][0]
    r_idx = np.where(freqs >= highcut)[0][0]
    freqs = freqs[l_idx:r_idx]
    ps = 2 * np.abs(sp[l_idx:r_idx])
    fft_pk_idx = np.argmax(ps)
    bpm = freqs[fft_pk_idx] * 60
    qos = ps[fft_pk_idx] / np.sum(ps)
    return bpm, qos


def compute_dual_band_metrics(
    rc: npt.NDArray,
    ab: npt.NDArray,
    sample_rate: float = 1000,
    lowcut: float = 0.05,
    highcut: float = 3.0,
    fft_len: int | None = None,
    pwr_threshold: float = 0.80,
) -> RspDualMetrics:
    """Compute respiratory dual band metrics.

    Args:
        rc (array): Ribcage band.
        ab (array): Abdominal band.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        lowcut (float, optional): Lowcut frequency in Hz. Defaults to 0.05 Hz.
        highcut (float, optional): Highcut frequency in Hz. Defaults to 3.0 Hz.
        fft_len (int, optional): FFT length. Defaults to None.
        pwr_threshold (float, optional): Power threshold. Defaults to 0.80.

    Returns:
        RspDualMetrics: Respiratory dual band metrics.
    """

    # Remove DC
    rc = rc - rc.mean()
    ab = ab - ab.mean()

    # Compute Vt
    vt = rc + ab

    # Compute FFT
    freqs, rc_sp = compute_fft(rc, sample_rate=sample_rate, fft_len=fft_len, window="blackman")
    freqs, ab_sp = compute_fft(ab, sample_rate=sample_rate, fft_len=fft_len, window="blackman")
    freqs, vt_sp = compute_fft(vt, sample_rate=sample_rate, fft_len=fft_len, window="blackman")

    # Clip to frequency band of interest
    l_idx = np.where(freqs >= lowcut)[0][0]
    r_idx = np.where(freqs >= highcut)[0][0]
    freqs = freqs[l_idx:r_idx]
    rc_sp = rc_sp[l_idx:r_idx]
    ab_sp = ab_sp[l_idx:r_idx]
    vt_sp = vt_sp[l_idx:r_idx]

    # Compute power spectrum
    rc_ps = 2 * np.abs(rc_sp)
    ab_ps = 2 * np.abs(ab_sp)
    vt_ps = 2 * np.abs(vt_sp)

    # Compute Vtc (corrected Vt)
    vtc_ps = rc_ps + ab_ps

    # Find dominant frequency
    rc_pk_idx = np.argmax(rc_ps)
    ab_pk_idx = np.argmax(ab_ps)
    # vt_pk_idx = np.argmax(vt_ps)
    vtc_pk_idx = np.argmax(vtc_ps)

    # Find all peaks above threshold
    rc_pk_idxs = np.where(rc_ps >= pwr_threshold * rc_ps[rc_pk_idx])[0]
    rc_pk_vals = rc_ps[rc_pk_idxs]
    ab_pk_idxs = np.where(ab_ps >= pwr_threshold * ab_ps[ab_pk_idx])[0]
    ab_pk_vals = ab_ps[ab_pk_idxs]

    vtc_pk_idxs = np.where(vtc_ps >= pwr_threshold * vtc_ps[vtc_pk_idx])[0]
    vtc_pk_vals = vtc_ps[vtc_pk_idxs]

    # Compute respiratory rates
    rc_rr = 60 * np.sum(freqs[rc_pk_idxs] * rc_pk_vals) / np.sum(rc_pk_vals)
    ab_rr = 60 * np.sum(freqs[ab_pk_idxs] * ab_pk_vals) / np.sum(ab_pk_vals)
    vtc_rr = 60 * np.sum(freqs[vtc_pk_idxs] * vtc_pk_vals) / np.sum(vtc_pk_vals)

    # Compute phase angle
    vtc_dom_div = rc_sp[vtc_pk_idxs] / ab_sp[vtc_pk_idxs]
    vtc_dom_angle = np.arctan2(np.imag(vtc_dom_div), np.real(vtc_dom_div))
    angles = (180 / np.pi) * np.sum(vtc_dom_angle * vtc_pk_vals) / np.sum(vtc_pk_vals)
    phase = np.abs(angles)
    rc_lead = angles > 0

    # Compute LBI: 𝚫 Vtc / 𝚫 Vt
    lbi = np.clip(np.sum(vtc_ps[vtc_pk_idxs]) / np.sum(vt_ps[vtc_pk_idxs]), 1, 10)

    # Compute %RC
    rc_percent = 100 * np.mean(rc_ps[vtc_pk_idxs] / (rc_ps[vtc_pk_idxs] + ab_ps[vtc_pk_idxs]))

    # Compute QoS
    qos = np.sum(rc_ps * ab_ps) / (np.sum(rc_ps) + np.sum(ab_ps))

    return RspDualMetrics(
        rc_rr=rc_rr,
        ab_rr=ab_rr,
        vt_rr=vtc_rr,
        phase=phase,
        lbi=lbi,
        rc_lead=rc_lead,
        rc_percent=rc_percent,
        qos=qos,
        rc_pk_freq=freqs[rc_pk_idx],
        rc_pk_pwr=rc_ps[rc_pk_idx],
        ab_pk_freq=freqs[ab_pk_idx],
        ab_pk_pwr=ab_ps[ab_pk_idx],
        vt_pk_freq=freqs[vtc_pk_idx],
        vt_pk_pwr=vtc_ps[vtc_pk_idx],
    )
