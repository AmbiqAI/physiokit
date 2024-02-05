# pylint: skip-file
""" Generate synthetic ECG signals. Adapted from following:
    authors:
    - family-names: "Brisk"
      given-names: "Rob"
      orcid: "https://orcid.org/0000-0002-3865-0792"
    title: "WaSP-ECG"
    version: 1.0.0
    doi: 0.3389/fphys.2022.760000
    date-released: 2022-03-17
    url: "https://github.com/docbrisky/WaSP-ECG"
"""
import logging
import random

import numpy as np
import numpy.typing as npt
import scipy.signal

from ..defines import EcgFiducial, EcgSegment
from . import presets
from . import wave_generator as wg
from .defines import EcgPreset, EcgPresetParameters
from .helper_functions import evenly_spaced_y, smooth_and_noise

logger = logging.getLogger(__name__)


def simulate_brisk(
    signal_length: int = 10000,
    sample_rate: float = 1000,
    leads: int = 12,
    heart_rate: float = 60,
    preset: EcgPreset = EcgPreset.SR,
    impedance: float = 1.0,
    p_multiplier: float = 1.0,
    t_multiplier: float = 1.0,
    noise_multiplier: float = 1.0,
    voltage_factor: float = 300,
    parameters: EcgPresetParameters | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, EcgPresetParameters]:
    """Generate synthetic ECG signals via WaSP-ECG

    Args:
        signal_length (int, optional): Signal length in samples. Defaults to 10000.
        sample_rate (float, optional): Sampling frequency in Hz. Defaults to 1000.
        leads (int, optional): # ECG leads. Max is 12. Defaults to 12.
        heart_rate (int, optional): Heart rate (BPM). Defaults to 60.
        preset (str, optional): ECG Preset. Defaults to "SR".
        impedance (float, optional): Lead impedance to adjust y scale. Defaults to 1.0.
        p_multiplier (float, optional): P wave width multiplier. Defaults to 1.0.
        t_multiplier (float, optional): T wave width multiplier. Defaults to 1.0.
        noise_multiplier (float, optional): Noise multiplier. Defaults to 1.0.
        voltage_factor (float, optional): Voltage factor. Defaults to 300.
        parameters (EcgPresetParameters, optional): Preset parameters. Defaults to None.

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, SyntheticParameters]: x, y, segs, fids, params
    """
    duration = signal_length / sample_rate
    afib_var = random.uniform(0.05, 0.4)

    frequency = 1000
    base_gap = int((60 / heart_rate) * frequency)
    leads = min(leads, 12)

    if sample_rate > 1000:
        logger.warning("Clamping to maximum supported frequency of 1000 Hz")
        sample_rate = 1000

    sizer = int(duration * frequency) + frequency
    X = np.zeros((leads, sizer))
    Y = np.zeros((leads, sizer))
    Y_segs = np.zeros((leads, sizer))
    Y_fids = np.zeros((leads, sizer))

    if parameters is None:
        parameters = presets.generate_preset_parameters(preset, heart_rate)

    for h in range(leads):
        x = np.linspace(0, sizer, sizer)
        y = np.zeros(sizer)
        y_segs = np.zeros(sizer)
        y_fids = np.zeros(sizer)

        start = 0
        beat_counter = 0
        beat_start = 0
        beat_length = (
            parameters.p_length
            + parameters.pr_interval
            + parameters.qrs_duration
            + parameters.st_length
            + parameters.t_length
        )

        if beat_length > base_gap + parameters.qrs_duration + parameters.st_length + parameters.t_length:
            logger.exception("Error, heart rate too high for beat length")

        while beat_start < sizer:
            start = beat_start

            if preset == EcgPreset.AFIB:
                gap = base_gap + random.randint(int(-base_gap * afib_var), int(base_gap * afib_var))
            else:
                gap = base_gap

            ################################################
            ## P-Wave Segment
            ################################################
            if preset != EcgPreset.AFIB:
                x_p, y_p = wg.syn_p_wave(
                    p_length=parameters.p_length,
                    p_voltage=parameters.p_voltages[h] * 2,
                    p_biphasic=parameters.p_biphasics[h],
                    p_lean=parameters.p_leans[h],
                    flipper=parameters.flippers[h],
                )
                x_p = x_p * p_multiplier
                y[start : min(start + y_p.size, sizer)] += evenly_spaced_y(
                    x_p[: min(x_p.size, sizer - start)],
                    y_p[: min(x_p.size, sizer - start)],
                )
                # Mark P-wave
                y_segs[start : min(start + y_p.size, sizer)] = EcgSegment.p_wave
                # if parameters.p_biphasics[h]:
                #   y_segs[start:min(start + y_p.size,sizer)] = EcgSegment.p_wave_biphasic

                # T-on-P phenomenon
                overlap = beat_length - gap
                if overlap > 0 and beat_counter > 0:
                    y_segs[start : start + overlap] = EcgSegment.tp_overlap

                # Mark PR segment
                if start + y_p.size < sizer:
                    y_segs[start + y_p.size : start + parameters.pr_interval] = EcgSegment.pr_segment

                y_fids[start] = EcgFiducial.p_wave

                start = start + parameters.pr_interval
                if start >= sizer:
                    break
            # END IF

            ################################################
            ## QRS Complex
            ################################################

            x_qrs, y_qrs, wave_peak_list = wg.syn_qrs_complex(
                qrs_duration=parameters.qrs_duration,
                q_depth=parameters.q_depths[h],
                r_height=parameters.r_heights[h],
                r_prime_present=parameters.r_prime_presents[h],
                r_prime_height=parameters.r_prime_heights[h],
                r_to_r_prime_duration_ratio=parameters.r_to_r_prime_duration_ratio[h],
                s_prime_height=parameters.s_prime_heights[h],
                s_present=parameters.s_presents[h],
                s_depth=parameters.s_depths[h],
                s_to_qrs_duration_ratio=parameters.s_to_qrs_duration_ratio[h],
                flipper=parameters.flippers[h],
                j_point=parameters.j_points[h],
            )
            y[start : min(start + y_qrs.size, sizer)] = evenly_spaced_y(
                x_qrs[: min(x_qrs.size, sizer - start)],
                y_qrs[: min(x_qrs.size, sizer - start)],
            )
            y_segs[start : min(start + y_qrs.size, sizer)] = EcgSegment.qrs_complex

            # # check if QRS complex predominantly negative:
            # if (
            #     parameters.s_presents[h]
            #     and parameters.s_depths[h] > parameters.r_heights[h]
            #     and parameters.s_depths[h] > parameters.r_prime_heights[h]
            # ):
            #     # check if broad QRS
            #     if parameters.qrs_duration > 120:
            #         # check if RSR pattern:
            #         if (
            #             parameters.r_prime_presents[h]
            #             and parameters.r_prime_heights[h] > parameters.s_prime_heights[h]
            #             and parameters.r_heights[h] > parameters.s_prime_heights[h]
            #         ):
            #             y_segs[start : min(start + y_qrs.size, sizer)] = ECGSegment.qrs_complex_wide_inv_rsr
            #         # if no RSR pattern:
            #         else:
            #             y_segs[start : min(start + y_qrs.size, sizer)] = ECGSegment.qrs_complex_wide_inv
            #     # if QRS narrow:
            #     else:
            #         y_segs[start : min(start + y_qrs.size, sizer)] = ECGSegment.qrs_complex_inv
            # # if QRS predominantly positive:
            # else:
            #     # check if broad QRS
            #     if parameters.qrs_duration > 120:
            #         # check if RSR pattern:
            #         if wave_peak_list[2] > 0:
            #             y_segs[start : min(start + y_qrs.size, sizer)] = ECGSegment.qrs_complex_wide_rsr
            #         # if no RSR pattern:
            #         else:
            #             y_segs[start : min(start + y_qrs.size, sizer)] = ECGSegment.qrs_complex_wide

            for pk in range(len(wave_peak_list)):
                if wave_peak_list[pk] != -1 and start + wave_peak_list[pk] < sizer:
                    y_fids[start + wave_peak_list[pk]] = EcgFiducial(10 + pk)
            # END FOR

            # Mark onset of QRS complex
            y_fids[start] = EcgFiducial.q_wave

            start = start + x_qrs.size
            if start >= sizer:
                break

            ################################################
            ## ST Segment
            ################################################

            if parameters.st_length > 0:
                x_st, y_st = wg.syn_st_segment(
                    j_point=parameters.j_points[h],
                    st_delta=parameters.st_deltas[h],
                    st_length=parameters.st_length,
                    flipper=parameters.flippers[h],
                )
                y[start : min(start + x_st.size, sizer)] = evenly_spaced_y(
                    x_st[: min(x_st.size, sizer - start)],
                    y_st[: min(x_st.size, sizer - start)],
                )
                y_segs[start : start + x_st.size] = EcgSegment.st_segment

                # # check if upsloping ST segment:
                # if parameters.st_delta > 0:
                #     y_segs[start:min(start + x_st.size,sizer)] = ECGSegment.st_segment_upsloping
                # # if downsloping ST segment:
                # elif parameters.st_delta < 0:
                #     y_segs[start:min(start + x_st.size,sizer)] =  ECGSegment.st_segment_downsloping

                y_fids[start] = EcgFiducial.j_point

                start = start + x_st.size
                if start >= sizer:
                    break
            # END IF

            ################################################
            ## T Wave
            ################################################

            st_end = (
                parameters.j_points[h] + parameters.st_deltas[h]
                if (parameters.st_length) > 0
                else parameters.j_points[h]
            )
            x_t, y_t = wg.syn_t_wave(
                st_end=st_end,
                t_height=parameters.t_heights[h] * 0.1,
                t_length=parameters.t_length,
                flipper=parameters.flippers[h],
                t_lean=parameters.t_leans[h],
            )
            x_t = x_t * t_multiplier
            y[start : min(start + x_t.size, sizer)] = evenly_spaced_y(
                x_t[: min(x_t.size, sizer - start)], y_t[: min(x_t.size, sizer - start)]
            )

            # Correct between T wave and ST segment
            if parameters.st_length > 5 and min(start + x_t.size, sizer) - start > 5:
                t_grad = np.amax(np.abs(np.gradient(y[start : min(start + x_t.size, sizer)] * 1e5))) / 10

                st_grad = np.mean(np.abs(np.gradient(y[start - 5 : start - 1] * 1e5)))

                grad = np.abs(np.gradient(y[start : min(start + x_t.size, sizer)] * 1e5))

                for t_value in range(start, min(start + x_t.size, sizer), 1):
                    if abs(st_grad - grad[t_value - start]) < t_grad:
                        y_segs[t_value] = EcgSegment.st_segment
                    else:
                        y_segs[t_value : min(start + x_t.size, sizer)] = EcgSegment.t_wave
                        break
                    # END IF
                # END FOR
            else:
                y_segs[start : min(start + x_t.size, sizer)] = EcgSegment.t_wave
            # END IF

            # # if T wave inverted:
            # y_segs[start:min(start + x_t.size,sizer)] = ECGSegment.t_wave_inv

            # mark J point if no ST segment:
            if parameters.st_length == 0:
                y_fids[start] = EcgFiducial.j_point

            beat_start += gap
            beat_counter += 1
            y_segs[start + x_t.size : beat_start] = EcgSegment.tp_segment

            # calculate end of QT interval using max slope method
            if x_t.size > 1 and start + x_t.size < sizer:
                grad = np.gradient(y[max(start, x_t.size - 100) : max(start, x_t.size - 100) + x_t.size])
                if parameters.flippers[h] > 0:
                    max_slope_x_coordinate = np.argmin(grad)
                else:
                    max_slope_x_coordinate = np.argmax(grad)

                if y[start + max_slope_x_coordinate] != 0 and grad[max_slope_x_coordinate] != 0:
                    # x intercept of maximum slope = x value of maximum slope + (y value of maximum slope * -gradient)
                    qt_end = int(
                        max_slope_x_coordinate + (y[start + max_slope_x_coordinate] / -grad[max_slope_x_coordinate])
                    )
                    if start + qt_end < sizer and qt_end > 0:
                        y_fids[start + qt_end] = EcgFiducial.qt_segment
                y_fids[start + x_t.size] = EcgFiducial.t_wave
            # END IF
        # END WHILE

        y = smooth_and_noise(
            y,
            noise_multiplier=noise_multiplier,
            impedance=impedance,
        )
        X[h,] = x
        Y[h,] = y
        Y_segs[h,] = y_segs
        Y_fids[h,] = y_fids
    # END FOR

    delay_start = random.randint(0, frequency - 1)
    delay_end = -(frequency - delay_start)
    X = X[:, delay_start:delay_end]
    for lead in range(X.shape[0]):
        X[lead, :] = X[lead, :] - X[lead, 0]

    Y = voltage_factor * Y[:, delay_start:delay_end]
    Y_segs = Y_segs[:, delay_start:delay_end]
    Y_fids = Y_fids[:, delay_start:delay_end]

    if sample_rate < frequency:
        X, Y, Y_segs, Y_fids = _resample_syn_signals(X, Y, Y_segs, Y_fids, frequency, sample_rate)

    X = X[:, :signal_length]
    Y = Y[:, :signal_length]
    Y_segs = Y_segs[:, :signal_length]
    Y_fids = Y_fids[:, :signal_length]

    return X, Y, Y_segs.astype(int), Y_fids.astype(int), parameters


def _resample_syn_signals(
    x: npt.NDArray,
    y: npt.NDArray,
    y_segs: npt.NDArray,
    y_fids: npt.NDArray,
    sampling_frequency: int,
    target_frequency: int,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Resample synthetic ECG signal to target sampling rate

    Args:
        x (npt.NDArray): Time domain [leads x data]
        y (npt.NDArray): ECG data [leads x data]
        y_segs (npt.NDArray): Segmentations
        y_fids (npt.NDArray): Fiducials
        sampling_frequency (int): Original Fs
        target_frequency (int): Target Fs

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]: Resampled signals
    """
    numel = int((target_frequency / sampling_frequency) * x.shape[1])

    # Resample ECG using FFT method
    yr = scipy.signal.resample(y, numel, axis=1)
    xr = np.zeros_like(yr)
    yr_segs = np.zeros_like(yr)
    yr_fids = np.zeros_like(yr)
    for l in range(xr.shape[0]):
        # Time might not be evenly space so we interpolate it
        ts_fn = scipy.interpolate.interp1d(np.arange(x[l].size), x[l], fill_value="extrapolate")
        xr[l] = ts_fn(np.linspace(x[l][0], x[l][-1], numel))
        # Use nearest neighbor for segmentation
        segs_fn = scipy.interpolate.interp1d(x[l], y_segs[l], kind="nearest", fill_value="extrapolate")
        yr_segs[l] = segs_fn(xr[l])
        # Adjust fiducials by resampling ratio
        yr_fids_idxs = np.nonzero(y_fids[l])[0]
        yr_fids_adj_idxs = yr_fids_idxs * (target_frequency / sampling_frequency)
        yr_fids_adj_idxs = yr_fids_adj_idxs.astype(int)
        yr_fids[l, yr_fids_adj_idxs] = y_fids[l, yr_fids_idxs]
    return xr, yr, yr_segs, yr_fids
