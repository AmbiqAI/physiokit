import numpy as np
import numpy.typing as npt
import scipy.stats

from .defines import HrvTimeMetrics


def compute_hrv_time(
    rr_intervals: npt.NDArray,
    sample_rate: float = 1000,
) -> HrvTimeMetrics:
    """Compute time domain HRV metrics.

    Args:
        rr_intervals (npt.NDArray): RR intervals.
        sample_rate (float, optional): Sampling rate. Defaults to 1000 Hz.

    Returns:
        HrvTimeMetrics: Time domain HRV metrics.
    """
    rri_ms = rr_intervals / sample_rate * 1000

    diff_rri_ms = np.diff(rri_ms)

    # Deviation based
    mean_nn = np.nanmean(rri_ms)
    sd_nn = np.nanstd(rri_ms, ddof=1)

    # Difference-based
    rms_sd = np.sqrt(np.nanmean(diff_rri_ms**2))
    sd_sd = np.nanstd(diff_rri_ms, ddof=1)

    # Normalized
    cv_nn = sd_nn / mean_nn
    cv_sd = rms_sd / mean_nn

    # Robust
    median_nn = np.nanmedian(rri_ms)
    mad_nn = 1.4826 * np.nanmedian(np.abs(rri_ms - median_nn))
    mcv_nn = mad_nn / median_nn
    iqr_nn = scipy.stats.iqr(rri_ms)
    prc20_nn = np.nanpercentile(rri_ms, q=20)
    prc80_nn = np.nanpercentile(rri_ms, q=80)

    # Extrema
    nn50 = np.sum(np.abs(diff_rri_ms) > 50)
    nn20 = np.sum(np.abs(diff_rri_ms) > 20)
    pnn50 = nn50 / (len(diff_rri_ms) + 1) * 100
    pnn20 = nn20 / (len(diff_rri_ms) + 1) * 100
    min_nn = np.nanmin(rri_ms)
    max_nn = np.nanmax(rri_ms)

    return HrvTimeMetrics(
        mean_nn=mean_nn,
        sd_nn=sd_nn,
        rms_sd=rms_sd,
        sd_sd=sd_sd,
        cv_nn=cv_nn,
        cv_sd=cv_sd,
        median_nn=median_nn,
        mad_nn=mad_nn,
        mcv_nn=mcv_nn,
        iqr_nn=iqr_nn,
        prc20_nn=prc20_nn,
        prc80_nn=prc80_nn,
        nn50=nn50,
        nn20=nn20,
        pnn50=pnn50,
        pnn20=pnn20,
        min_nn=min_nn,
        max_nn=max_nn,
    )
