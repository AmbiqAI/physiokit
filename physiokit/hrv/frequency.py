import numpy.typing as npt
import numpy as np
import scipy.stats

from .defines import HrvFrequencyMetrics


def compute_hrv_frequency(
    rr_intervals: npt.NDArray,
    sample_rate: float = 1000,
    axis: int = -1
):
    return HrvFrequencyMetrics()
