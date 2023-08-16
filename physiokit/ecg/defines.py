from enum import IntEnum


class ECGSegments(IntEnum):
    """ECG Segment labels"""

    background = 0
    p_wave = 1
    q_wave = 2
    r_wave = 3
    s_wave = 4
    t_wave = 5
    u_wave = 6
    pr_interval = 7
    pr_segment = 8
    qrs_complex = 9
    st_segment = 10
    qt_segment = 11
