import numpy as np
import numpy.typing as npt
import scipy.ndimage as spn

from ..signal import filter_signal, quotient_filter_mask

def compute_heart_rate(
    data: npt.NDArray,
    sample_rate: float = 1000,
    method: str = 'fft'
):
    """Compute heart rate from PPG signal.
    Args:
        data (array): PPG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        method (str, optional): Method to compute heart rate. Defaults to 'fft'.
    Returns:
        float: Heart rate in BPM.
    """

    if method == 'fft':
        return compute_heart_rate_from_fft(
            data=data,
            sample_rate=sample_rate
        )

    if method == 'peak':
        return compute_heartrate_from_peaks(
            data=data,
            sample_rate=sample_rate
        )

    raise NotImplementedError(f'Heart rate computation method {method} not implemented.')

def compute_heartrate_from_peaks(
        data: npt.NDArray,
        sample_rate: float = 1000,
    ) -> float:
    """Compute heart rate from peaks of PPG signal.
    Args:
        data (array): PPG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
    Returns:
        float: Heart rate in BPM.
    """
    peaks = find_peaks(
        data=data,
        sample_rate=sample_rate,
    )
    peaks = filter_peaks(
        peaks=peaks,
        sample_rate=sample_rate,
    )
    return 60*np.diff(peaks).mean()/sample_rate

def compute_heart_rate_from_fft(
        data: npt.NDArray,
        sample_rate: float = 1000,
    ) -> float:
    """Compute heart rate from FFT of PPG signal.
    Args:
        data (array): PPG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
    Returns:
        float: Heart rate in BPM.
    """
    lowcut = 0.7
    highcut = 4.0
    freqs, sp = compute_fft(data, sample_rate)
    l_idx = np.where(freqs >= lowcut)[0][0]
    r_idx = np.where(freqs >= highcut)[0][0]
    ps = 2*np.abs(sp)
    fft_pk_idx = np.argmax(ps[l_idx:r_idx]) + l_idx
    hr = freqs[fft_pk_idx]*60
    return hr

def compute_fft(
        data: npt.NDArray,
        sample_rate: float = 1000,
        axis: int = -1,
    ) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute FFT of PPG signal.
    Args:
        data (array): PPG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        axis (int, optional): Axis to compute FFT. Defaults to -1.
    Returns:
        tuple[array, array]: Frequencies and FFT of PPG signal.
    """
    data_len = data.shape[axis]
    fft_len = int(2**np.ceil(np.log2(data_len)))
    fft_win = np.blackman(data_len)
    amp_corr = 1.93

    freqs = np.fft.fftfreq(fft_len, 1 / sample_rate)
    sp = amp_corr*np.fft.fft(fft_win*data, fft_len, axis=axis)/data_len
    return freqs, sp


def compute_spo2_from_perfusion(
        dc1: float,
        ac1: float,
        dc2: float,
        ac2: float,
        coefs: tuple[float, float, float] = (1, 0, 0)
    ) -> float:
    """Compute SpO2 from ratio of perfusion indexes (AC/DC).
        MAX30101: [1.5958422, -34.6596622, 112.6898759]
        MAX8614X: [-16.666666, 8.333333, 100]
    Args:
        dc1 (float): DC component of 1st PPG signal (e.g RED).
        ac1 (float): AC component of 1st PPG signal (e.g RED).
        dc2 (float): DC component of 2nd PPG signal (e.g. IR).
        ac2 (float): AC component of 2nd PPG signal (e.g. IR).
        coefs (tuple[float, float, float], optional): Calibration coefficients. Defaults to (1, 0, 0).
    Returns:
        float: SpO2 value clipped to [50, 100].
    """
    r = (ac1/dc1)/(ac2/dc2)
    spo2 = coefs[0]*r**2 + coefs[1]*r + coefs[2]
    return max(min(spo2, 100), 50)

def compute_spo2_in_time(
        ppg1: npt.NDArray,
        ppg2: npt.NDArray,
        coefs: tuple[float, float, float] = (1, 0, 0),
        sample_rate: float = 1000,
        lowcut: float = 0.7,
        highcut: float = 4,
        axis: int = -1
    ) -> float:
    """Compute SpO2 from PPG signals in time domain.

    Args:
        ppg1 (array): 1st PPG signal (e.g RED).
        ppg2 (array): 2nd PPG signal (e.g. IR).
        sampling_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        coefs (tuple[float, float, float], optional): Calibration coefficients. Defaults to (1, 0, 0).
    Returns:
        float: SpO2 value
    """

    # Compute DC
    ppg1_dc = np.mean(ppg1)
    ppg2_dc = np.mean(ppg2)

    # Bandpass filter
    ppg1_clean = filter_signal(
        data=ppg1,
        lowcut=lowcut,
        highcut=highcut,
        sample_rate=sample_rate,
        order=3,
        forward_backward=True
    )

    ppg2_clean = filter_signal(
        data=ppg2,
        lowcut=lowcut,
        highcut=highcut,
        sample_rate=sample_rate,
        order=3,
        forward_backward=True
    )

    # Compute AC via RMS
    ppg1_ac = np.sqrt(np.mean(ppg1_clean**2))
    ppg2_ac = np.sqrt(np.mean(ppg2_clean**2))

    spo2 = compute_spo2_from_perfusion(
        dc1=ppg1_dc,
        ac1=ppg1_ac,
        dc2=ppg2_dc,
        ac2=ppg2_ac,
        coefs=coefs
    )
    return spo2


def compute_spo2_in_frequency(
        ppg1: npt.NDArray,
        ppg2: npt.NDArray,
        coefs: tuple[float, float, float] = (1, 0, 0),
        sample_rate: float = 1000,
        lowcut: float = 0.7,
        highcut: float = 4.0,
        axis: int = -1
    ) -> float:
    """Compute SpO2 from PPG signals in frequency domain.
    Args:
        ppg1 (array): 1st PPG signal (e.g RED).
        ppg2 (array): 2nd PPG signal (e.g. IR).
        coefs (tuple[float, float, float], optional): Calibration coefficients. Defaults to (1, 0, 0).
        sampling_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        lowcut (float, optional): Lowcut frequency in Hz. Defaults to 0.7 Hz.
        highcut (float, optional): Highcut frequency in Hz. Defaults to 4.0 Hz.
        axis (int, optional): Axis along which to compute the FFT. Defaults to -1.
    Returns:
        float: SpO2 value
    """

    data_len = ppg1.shape[axis]
    fft_len = int(2**np.ceil(np.log2(data_len)))
    fft_win = np.blackman(data_len)
    amp_corr = 1.93

    freqs = np.fft.fftfreq(fft_len, 1 / sample_rate)
    l_idx = np.where(freqs >= lowcut)[0][0]
    r_idx = np.where(freqs >= highcut)[0][0]

    # Compute DC
    ppg1_dc = np.mean(ppg1)
    ppg2_dc = np.mean(ppg2)

    # Bandpass filter
    ppg1_clean = filter_signal(
        data=ppg1,
        lowcut=lowcut,
        highcut=highcut,
        sample_rate=sample_rate,
        order=3,
        forward_backward=True
    )

    ppg2_clean = filter_signal(
        data=ppg2,
        lowcut=lowcut,
        highcut=highcut,
        sample_rate=sample_rate,
        order=3,
        forward_backward=True
    )

    ppg1_fft = np.fft.fft(fft_win*ppg1_clean, fft_len, axis=axis)/data_len
    ppg2_fft = np.fft.fft(fft_win*ppg2_clean, fft_len, axis=axis)/data_len

    ppg1_ps = 2*amp_corr*np.abs(ppg1_fft)
    ppg2_ps = 2*amp_corr*np.abs(ppg2_fft)

    fft_pk_idx = np.argmax(ppg1_ps[l_idx:r_idx] + ppg2_ps[l_idx:r_idx]) + l_idx

    # Compute AC
    ppg1_ac = ppg1_ps[fft_pk_idx]
    ppg2_ac = ppg2_ps[fft_pk_idx]

    spo2 = compute_spo2_from_perfusion(
        dc1=ppg1_dc,
        ac1=ppg1_ac,
        dc2=ppg2_dc,
        ac2=ppg2_ac,
        coefs=coefs
    )

    return spo2


def find_peaks(
    data: npt.NDArray,
    sample_rate: float = 1000,
    peak_window: float = 0.111,
    beat_window: float = 0.667,
    beat_offset: float = 0.02,
    peak_delay: float = 0.3,
):
    """Find systolic peaks in PPG signal.
    Implementation based on Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection in
    Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions. PLoS ONE 8(10): e76585.
    doi:10.1371/journal.pone.0076585.
    Assumes input data is bandpass filtered with a lowcut of .5 Hz and a highcut of 8 Hz.
    Args:
        data (array): PPG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        peak_window (float, optional): Peak window in seconds. Defaults to 0.111 s.
        beat_window (float, optional): Beat window in seconds. Defaults to 0.667 s.
        beat_offset (float, optional): Beat offset in seconds. Defaults to 0.02 s.
        peak_delay (float, optional): Peak delay in seconds. Defaults to 0.3 s.
    """

    # Clip negative values and square the signal
    sqrd = np.where(data > 0, data**2, 0)

    # Apply 1st moving average filter
    ma_peak_kernel = int(np.rint(peak_window * sample_rate))
    ma_peak = spn.uniform_filter1d(sqrd, ma_peak_kernel, mode="nearest")

    # Apply 2nd moving average filter
    ma_beat_kernel = int(np.rint(beat_window * sample_rate))
    ma_beat = spn.uniform_filter1d(sqrd, ma_beat_kernel, mode="nearest")

    # Thresholds
    min_height = ma_beat + beat_offset * np.mean(sqrd)
    min_width = int(np.rint(peak_window * sample_rate))
    min_delay = int(np.rint(peak_delay * sample_rate))

    # Identify wave boundaries
    waves = ma_peak > min_height
    beg_waves = np.where(np.logical_and(np.logical_not(waves[0:-1]), waves[1:]))[0]
    end_waves = np.where(np.logical_and(waves[0:-1], np.logical_not(waves[1:])))[0]
    end_waves = end_waves[end_waves > beg_waves[0]]

    # Identify systolic peaks
    peaks = []
    for i in range(min(beg_waves.size, end_waves.size)):
        beg, end = beg_waves[i], end_waves[i]
        peak = beg + np.argmax(data[beg:end])
        peak_width = end - beg
        peak_delay = peak - peaks[-1] if len(peaks) else min_delay

        # Enforce minimum length and delay between peaks
        if (peak_width < min_width) or (peak_delay < min_delay):
            continue
        peaks.append(peak)
    # END FOR

    return np.array(peaks, dtype=int)


def filter_peaks(
        peaks: npt.NDArray,
        sample_rate: float = 1000,
    ) -> npt.NDArray:
    """Filter out peaks with RR intervals outside of normal range.
    Args:
        peaks (array): Systolic peaks.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
    Returns:
        npt.NDArray: Filtered peaks.
    """
    lowcut = 0.3*sample_rate
    highcut = 2*sample_rate

    # Capture RR intervals
    rr_ints = np.diff(peaks)
    rr_ints = np.hstack((rr_ints[0], rr_ints))

    # Filter out peaks with RR intervals outside of normal range
    rr_mask = np.where((rr_ints < lowcut) | (rr_ints > highcut), 1, 0)

    # Filter out peaks that deviate more than 30%
    rr_mask = quotient_filter_mask(rr_ints, mask=rr_mask, lowcut=0.7, highcut=1.3)
    filt_peaks = peaks[np.where(rr_mask == 0)[0]]
    return filt_peaks
