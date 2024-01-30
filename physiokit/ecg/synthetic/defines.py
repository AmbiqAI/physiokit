from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


class EcgPresets(StrEnum):
    """ECG synthetic presets"""

    SR = "SR"
    ant_STEMI = "ant_STEMI"
    LAHB = "LAHB"
    LPHB = "LPHB"
    high_take_off = "high_take_off"
    LBBB = "LBBB"
    random_morphology = "random_morphology"


class SyntheticSegments(IntEnum):
    """Synthetic Segment labels"""

    background = 0
    p_wave = 1
    qrs_complex = 2
    t_wave = 3
    pr_segment = 11  # End of P-wave to start of QRS
    st_segment = 12  # End of QRS to start of T-wave
    tp_segment = 13  # End of T-wave to start of P-wave
    tp_overlap = 14  # T-on-P

    # Below not currently used
    p_wave_biphasic = 16
    qrs_complex_wide_rsr = 17
    qrs_complex_wide = 18
    qrs_complex_inv = 19
    qrs_complex_wide_inv_rsr = 20
    qrs_complex_wide_inv = 21
    st_segment_upsloping = 22
    st_segment_downsloping = 23
    t_wave_inv = 24


class SyntheticFiducials(IntEnum):
    """Synthetic fiducials labels"""

    p_wave = 8
    q_wave = 9
    q_trough = 10
    r_peak = 11
    rpr_peak = 12
    s_trough = 13
    j_point = 14
    qt_segment = 15  # end
    t_wave = 16  # end


@dataclass
class SyntheticParameters:
    """Synthetic ECG parameters"""

    # pylint: disable=R0902
    p_length: int = 80
    pr_interval: int = 80  # This is really PR segment
    qrs_duration: int = 50
    noisiness: float = 0
    st_length: int = 20
    t_length: int = 0
    qt: int = 0
    qtc: float = 0
    flippers: list[int] = field(default_factory=list)
    p_voltages: list[float] = field(default_factory=list)
    p_biphasics: list[bool] = field(default_factory=list)
    p_leans: list[float] = field(default_factory=list)
    q_depths: list[float] = field(default_factory=list)
    r_heights: list[float] = field(default_factory=list)
    r_prime_presents: list[bool] = field(default_factory=list)
    r_prime_heights: list[float] = field(default_factory=list)
    r_to_r_prime_duration_ratio: list[float] = field(default_factory=list)
    s_presents: list[bool] = field(default_factory=list)
    s_depths: list[float] = field(default_factory=list)
    s_prime_heights: list[float] = field(default_factory=list)
    s_to_qrs_duration_ratio: list[int] = field(default_factory=list)
    st_deltas: list[float] = field(default_factory=list)
    j_points: list[float] = field(default_factory=list)
    t_heights: list[float] = field(default_factory=list)
    t_leans: list[float] = field(default_factory=list)
