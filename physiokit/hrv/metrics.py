import numpy.typing as npt
import numpy as np
import scipy.stats


def compute_hrv_time(
    rr_intervals: npt.NDArray,
    sample_rate: float = 1000,
    axis: int = -1
) -> float:
    rri_ms = rr_intervals / sample_rate * 1000

    diff_rri_ms = np.diff(rri_ms)

    # Deviation based
    mean_nn = np.nanmean(rri_ms)
    sdnn = np.nanstd(rri_ms, ddof=1)

    # Difference-based
    rmssd = np.sqrt(np.nanmean(diff_rri_ms**2))
    sdsd = np.nanstd(diff_rri_ms, ddof=1)

    # Normalized
    cvnn = sdnn / mean_nn
    cvsd = rmssd / mean_nn

    # Robust
    median_nn = np.nanmedian(rri_ms)

    madnn = 1.4826*np.nanmedian(np.abs(rri_ms - median_nn))


    mcvnn = madnn / median_nn
    iqrnn = scipy.stats.iqr(rri_ms)
    prc20nn = np.nanpercentile(rri_ms, q=20)
    prc80nn = np.nanpercentile(rri_ms, q=80)

    # Extreme-based
    nn50 = np.sum(np.abs(diff_rri_ms) > 50)
    nn20 = np.sum(np.abs(diff_rri_ms) > 20)
    pnn50 = nn50 / (len(diff_rri_ms) + 1) * 100
    pnn20 = nn20 / (len(diff_rri_ms) + 1) * 100
    min_nn = np.nanmin(rri_ms)
    max_nn = np.nanmax(rri_ms)

    # mean_nn, sdnn, rmssd, sdsd, cvnn, cvsd, madnn, mcvnn, iqrnn, prc20nn, prc80nn, nn50, nn20, pnn50, pnn20, min_nn, max_nn

    return sdnn

def compute_hrv_frequency(
    rr_intervals: npt.NDArray,
    sample_rate: float = 1000,
    axis: int = -1
):
    rri_ms = rr_intervals / sample_rate * 1000
