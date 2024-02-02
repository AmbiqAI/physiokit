from enum import IntEnum


class EcgSegment(IntEnum):
    """ECG Segment labels"""

    background = 0
    p_wave = 1
    qrs_complex = 2
    t_wave = 3
    u_wave = 4

    pr_segment = 11  # End of P-wave to start of QRS
    st_segment = 12  # End of QRS to start of T-wave
    tp_segment = 13  # End of T-wave to start of P-wave
    tp_overlap = 14  # T-on-P

    # Below not currently used
    # p_wave_biphasic = 16
    # qrs_complex_wide_rsr = 17
    # qrs_complex_wide = 18
    # qrs_complex_inv = 19
    # qrs_complex_wide_inv_rsr = 20
    # qrs_complex_wide_inv = 21
    # st_segment_upsloping = 22
    # st_segment_downsloping = 23
    # t_wave_inv = 24


class EcgFiducial(IntEnum):
    """ECG fiducial labels"""

    p_wave = 8
    q_wave = 9
    q_trough = 10
    r_peak = 11
    rpr_peak = 12
    s_trough = 13
    j_point = 14
    qt_segment = 15  # end
    t_wave = 16  # end
