from functools import reduce

import numpy as np
import numpy.typing as npt
import scipy.interpolate
import scipy.signal

from ..signal import compute_fft
from .defines import HrvFrequencyBandMetrics, HrvFrequencyMetrics


def compute_hrv_frequency(
    peaks: npt.NDArray,
    rri: npt.NDArray,
    bands: list[tuple[float, float]],
    sample_rate: float = 1000,
) -> HrvFrequencyMetrics:
    """Compute the frequency domain HRV features.

    Args:
        peaks (array): R peaks.
        rri (array): RR intervals.
        bands (list): List of frequency bands.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.

    Returns:
        HrvFrequencyMetrics: Frequency domain HRV features.
    """

    # Interpolate to get evenly spaced samples
    ts = np.arange(peaks[0], peaks[-1], 1)
    rri_int = scipy.interpolate.interp1d(peaks, rri, kind="linear")(ts)

    # NOTE: Use bands to determine amount of zero padding for freq bins
    fft_len = int(2 ** np.ceil(np.log2(max(ts.size, 32 * sample_rate))))

    freqs, rri_fft = compute_fft(rri_int, sample_rate=sample_rate, fft_len=fft_len, window="blackman", axis=-1)
    rri_ps = 2 * np.abs(rri_fft)

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
