from functools import reduce

import numpy as np
import numpy.typing as npt
import scipy.interpolate
import scipy.signal

from .defines import HrvFrequencyBandMetrics, HrvFrequencyMetrics


def compute_hrv_frequency(
    peaks: npt.NDArray,
    rri: npt.NDArray,
    bands: list[tuple[float, float]],
    sample_rate: float = 1000,
) -> HrvFrequencyMetrics:
    """Compute the frequency domain HRV features."""

    # Interpolate to get evenly spaced samples
    ts = np.arange(peaks[0], peaks[-1], 1)
    rri_int = scipy.interpolate.interp1d(peaks, rri, kind="linear")(ts)

    fft_len = int(2 ** np.ceil(np.log2(ts.size)))
    fft_win = np.blackman(ts.size)
    amp_corr = 1.93

    freqs = np.fft.fftfreq(fft_len, 1 / sample_rate)
    rri_fft = np.fft.fft(fft_win * rri_int, fft_len) / ts.size
    rri_ps = 2 * amp_corr * np.abs(rri_fft)

    metrics = HrvFrequencyMetrics()
    for lowcut, highcut in bands:
        l_idx = np.where(freqs >= lowcut)[0][0]
        r_idx = np.where(freqs >= highcut)[0][0]
        f_idx = rri_ps[l_idx:r_idx].argmax() + l_idx
        metrics.bands.append(
            HrvFrequencyBandMetrics(
                peak_frequency=freqs[f_idx],
                peak_power=rri_ps[f_idx],
                total_power=rri_ps[l_idx:r_idx].sum(),
            )
        )
    # END FOR
    metrics.total_power = reduce(lambda x, y: x + y.total_power, metrics.bands, 0)
    return metrics
