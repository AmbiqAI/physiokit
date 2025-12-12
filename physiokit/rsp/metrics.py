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
        data (npt.NDArray): RSP signal (1-D).
        sample_rate (float): Sampling rate in Hz.
        method (str): Method to compute respiratory rate, ``"fft"`` or ``"peak"``.
        **kwargs: Keyword arguments to pass to the selected method.

    Returns:
        tuple[float, float]: (respiratory BPM, quality score).

    Example:
        >>> import numpy as np
        >>> rsp = np.sin(2 * np.pi * 0.25 * np.arange(0, 40, 0.1))
        >>> bpm, _ = compute_respiratory_rate(rsp, sample_rate=10)
        >>> int(round(bpm))
        15
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
        data (npt.NDArray): RSP signal (1-D).
        sample_rate (float): Sampling rate in Hz.
        min_rr (float): Minimum RR interval in seconds.
        max_rr (float): Maximum RR interval in seconds.
        min_delta (float): Minimum delta between RR intervals in seconds.

    Returns:
        tuple[float, float]: (respiratory BPM, quality score).

    Example:
        >>> import numpy as np
        >>> rsp = np.sin(2 * np.pi * 0.2 * np.arange(0, 50, 0.1))
        >>> bpm, qos = compute_respiratory_rate_from_peaks(rsp, sample_rate=10)
        >>> bpm > 10 and qos > 0
        True
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
        data (npt.NDArray): RSP signal (1-D).
        sample_rate (float): Sampling rate in Hz.
        lowcut (float): Lowcut frequency in Hz.
        highcut (float): Highcut frequency in Hz.

    Returns:
        tuple[float, float]: (respiratory BPM, quality score).

    Example:
        >>> import numpy as np
        >>> rsp = np.sin(2 * np.pi * 0.3 * np.arange(0, 30, 0.1))
        >>> int(round(compute_respiratory_rate_from_fft(rsp, sample_rate=10)[0]))
        18
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
        rc (npt.NDArray): Ribcage band (1-D).
        ab (npt.NDArray): Abdominal band (1-D).
        sample_rate (float): Sampling rate in Hz.
        lowcut (float): Lowcut frequency in Hz.
        highcut (float): Highcut frequency in Hz.
        fft_len (int | None): FFT length; defaults to signal length.
        pwr_threshold (float): Power threshold for peak inclusion.

    Returns:
        RspDualMetrics: Aggregated dual-band respiratory metrics.

    Example:
        >>> import numpy as np
        >>> t = np.arange(0, 20, 0.1)
        >>> rc = np.sin(2 * np.pi * 0.2 * t)
        >>> ab = 0.8 * np.sin(2 * np.pi * 0.2 * t + 0.1)
        >>> metrics = compute_dual_band_metrics(rc=rc, ab=ab, sample_rate=10)
        >>> round(metrics.vt_rr)
        12
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
