from enum import IntEnum


class ECGSegment(IntEnum):
    """ECG Segment labels"""

    background = 0
    p_wave = 1
    qrs_complex = 2
    t_wave = 3
    u_wave = 4
    pr_segment = 5
    st_segment = 6


class EcgFiducials(IntEnum):
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
