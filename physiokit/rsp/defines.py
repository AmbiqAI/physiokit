from dataclasses import dataclass
from enum import IntEnum


class RspSegment(IntEnum):
    """RSP Segment labels"""

    background = 0
    inhale = 1
    exhale = 2


class RspFiducial(IntEnum):
    """RSP fiducials labels"""

    inhale_peak = 1
    exhale_trough = 2


# pylint: disable=too-many-instance-attributes
@dataclass
class RspDualMetrics:
    """Respiratory dual band metrics."""

    rc_rr: float = 0  # RC respiratory rate (BPM)
    ab_rr: float = 0  # AB respiratory rate (BPM)
    vt_rr: float = 0  # VT respiratory rate (BPM)
    phase: float = 0  # Phase angle (degrees)
    lbi: float = 0  # Labored breathing index
    rc_lead: bool = False  # RC leads AB
    rc_percent: float = 0  # Percent RC contribution
    qos: float = 0  # Quality of signal (0-1)

    rc_pk_freq: float = 0  # RC peak frequency (Hz)
    rc_pk_pwr: float = 0  # RC peak power
    ab_pk_freq: float = 0  # AB peak frequency (Hz)
    ab_pk_pwr: float = 0  # AB peak power
    vt_pk_freq: float = 0  # VT peak frequency (Hz)
    vt_pk_pwr: float = 0  # VT peak power
