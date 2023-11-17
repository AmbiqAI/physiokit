from dataclasses import dataclass, field


# pylint: disable=too-many-instance-attributes
@dataclass
class HrvTimeMetrics:
    """Time domain HRV metric dataclass."""

    # Deviation-based
    mean_nn: float = 0  # Mean of normal intervals
    sd_nn: float = 0  # St. Dev of normal intervals

    # Difference-based
    rms_sd: float = 0  # RMS of successive differences between normal intervals
    sd_sd: float = 0  # St. Dev of successive differences between normal intervals

    # Normalized
    cv_nn: float = 0
    cv_sd: float = 0

    # Robust
    median_nn: float = 0  # Median of normal intervals
    mad_nn: float = 0  # Median absolute deviation of normal intervals
    mcv_nn: float = 0  # Median coefficient of variation of normal intervals
    iqr_nn: float = 0  # Interquartile range of normal intervals
    prc20_nn: float = 0  # 20th percentile of normal intervals
    prc80_nn: float = 0  # 80th percentile of normal intervals

    # Extrema
    nn50: int = 0  # Number of intervals > 50 ms
    nn20: int = 0  # Number of intervals > 20 ms
    pnn50: float = 0  # Percentage of intervals > 50 ms
    pnn20: float = 0  # Percentage of intervals > 20 ms
    min_nn: float = 0  # Minimum of normal intervals
    max_nn: float = 0  # Maximum of normal intervals


@dataclass
class HrvFrequencyBandMetrics:
    """HRV Frequency domain metrics dataclass."""

    peak_frequency: float = 0  # Peak of frequency band in Hz
    peak_power: float = 0  # Power of frequency band in ms^2
    total_power: float = 0  # Total power in ms^2


@dataclass
class HrvFrequencyMetrics:
    """Frequency domain HRV metric dataclass."""

    bands: list[HrvFrequencyBandMetrics] = field(default_factory=list)
    total_power: float = 0  # Total power in ms^2


@dataclass
class HrvNonlinearMetrics:
    """Non-linear HRV metric dataclass."""

    sd1: float = 0  # Short-term variability
    sd2: float = 0  # Long-term variability
    sd1_sd2_ratio: float = 0  # Ratio of short-term to long-term variability
