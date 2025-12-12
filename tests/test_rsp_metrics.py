import numpy as np
import pytest

from physiokit.rsp.metrics import compute_dual_band_metrics, compute_respiratory_rate
from physiokit.rsp.peaks import filter_peaks, find_peaks


def test_rsp_find_peaks_and_filter():
    sample_rate = 20
    freq = 0.2
    t = np.arange(0, 40, 1 / sample_rate)
    data = 0.5 + np.sin(2 * np.pi * freq * t)

    peaks = find_peaks(data, sample_rate=sample_rate)
    filt = filter_peaks(peaks, sample_rate=sample_rate)

    assert peaks.size > 5
    assert filt.size <= peaks.size


def test_compute_respiratory_rate_fft_and_peak():
    sample_rate = 20
    freq = 0.25
    t = np.arange(0, 20, 1 / sample_rate)
    data = np.sin(2 * np.pi * freq * t)

    fft_bpm, fft_qos = compute_respiratory_rate(data=data, sample_rate=sample_rate, method="fft", lowcut=0.1, highcut=1.0)
    peak_bpm, peak_qos = compute_respiratory_rate(data=data + 0.5, sample_rate=sample_rate, method="peak")

    target_bpm = freq * 60
    assert abs(fft_bpm - target_bpm) < 5
    assert abs(peak_bpm - target_bpm) < 5
    assert fft_qos > 0
    assert peak_qos > 0


def test_compute_respiratory_rate_invalid_method():
    with pytest.raises(NotImplementedError):
        compute_respiratory_rate(data=np.zeros(10), sample_rate=20, method="bad")


def test_rsp_find_peaks_handles_short_signal():
    data = np.array([0.1, 0.2, 0.1])
    with pytest.raises(IndexError):
        find_peaks(data, sample_rate=10)


def test_compute_dual_band_metrics():
    sample_rate = 20
    freq = 0.3
    t = np.arange(0, 20, 1 / sample_rate)
    rc = np.sin(2 * np.pi * freq * t)
    ab = 0.8 * np.sin(2 * np.pi * freq * t + 0.5)

    metrics = compute_dual_band_metrics(rc=rc, ab=ab, sample_rate=sample_rate)

    target_bpm = freq * 60
    assert abs(metrics.rc_rr - target_bpm) < 5
    assert abs(metrics.ab_rr - target_bpm) < 5
    assert abs(metrics.vt_rr - target_bpm) < 5
    assert 0 <= metrics.rc_percent <= 100
    assert 1 <= metrics.lbi <= 10
    assert metrics.qos > 0
