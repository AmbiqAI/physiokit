from enum import IntEnum


class PpgSegment(IntEnum):
    """PPG Segment labels"""

    background = 0
    systolic = 1
    diastolic = 2


class PpgFiducial(IntEnum):
    """PPG fiducials labels"""

    systolic_peak = 1
    dicrotic_notch = 2
    diastolic_peak = 3
