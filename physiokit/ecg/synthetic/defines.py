from dataclasses import dataclass, field
from enum import Enum


class StrEnum(str, Enum):
    """Backport of StrEnum from Python 3.11 to 3.10"""

    def __str__(self: "StrEnum") -> str:
        return self.value


class EcgPreset(StrEnum):
    """ECG presets"""

    SR = "SR"
    AFIB = "AFIB"
    ant_STEMI = "ant_STEMI"
    LAHB = "LAHB"
    LPHB = "LPHB"
    high_take_off = "high_take_off"
    LBBB = "LBBB"
    random_morphology = "random_morphology"


@dataclass
class EcgPresetParameters:
    """ECG preset parameters"""

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
