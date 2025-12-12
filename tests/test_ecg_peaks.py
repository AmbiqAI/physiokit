import random

import numpy as np
import pytest

from physiokit.ecg.metrics import compute_heart_rate
from physiokit.ecg.metrics import compute_pulse_arrival_time, derive_respiratory_rate
from physiokit.ecg.peaks import compute_rr_intervals, find_peaks, filter_rr_intervals
from physiokit.ecg.synthesize import synthesize


def test_find_peaks_handles_unclosed_qrs(monkeypatch):
    """Ensure peak finder works when QRS mask never closes."""
    data = np.array([0.0, 1.0, 3.0, 2.0, 1.0])
    qrs_mask = np.ones_like(data, dtype=float)

    # Force a QRS mask that starts at index 0 and never drops low.
    monkeypatch.setattr("physiokit.ecg.peaks.moving_gradient_filter", lambda *args, **kwargs: qrs_mask)

    peaks = find_peaks(data=data, sample_rate=1000)

    assert peaks.tolist() == [2]


def test_compute_heart_rate_on_synthesized_signal():
    random.seed(0)
    np.random.seed(0)

    sample_rate = 250
    target_bpm = 72
    duration_seconds = 12
    signal_length = sample_rate * duration_seconds

    ecg, _, _ = synthesize(
        signal_length=signal_length, sample_rate=sample_rate, leads=1, heart_rate=target_bpm, noise_multiplier=0.0
    )
    data = ecg[0]

    peaks = find_peaks(data=data, sample_rate=sample_rate)
    assert peaks.size > 5

    bpm, qos = compute_heart_rate(data=data, sample_rate=sample_rate)

    assert abs(bpm - target_bpm) < 10
    assert qos > 0.4


def test_filter_rr_intervals_marks_outliers():
    peaks = np.array([0, 10, 40])
    rr_ints = compute_rr_intervals(peaks)

    mask = filter_rr_intervals(rr_ints, sample_rate=10, min_rr=1.0, max_rr=4.0, min_delta=0.2)

    assert mask.tolist() == [0, 0, 1]


def test_derive_respiratory_rate_from_rri():
    sample_rate = 50
    peaks = np.arange(0, 400, 50)
    rri = 50 + 10 * np.sin(2 * np.pi * 0.2 * np.arange(peaks.size))

    rsp_bpm, qos = derive_respiratory_rate(peaks=peaks, rri=rri, sample_rate=sample_rate, method="rifv")

    assert 9 <= rsp_bpm <= 15
    assert qos > 0


def test_compute_pulse_arrival_time(monkeypatch):
    sample_rate = 100
    ecg = np.zeros(250)
    ppg = np.zeros_like(ecg)
    ecg_peaks = np.array([50, 150])
    ppg_peaks = ecg_peaks + 7
    ppg[ppg_peaks] = 5

    monkeypatch.setattr("physiokit.ecg.metrics.find_peaks", lambda ecg, sample_rate=sample_rate: ecg_peaks)

    lag = compute_pulse_arrival_time(ecg, ppg, sample_rate=sample_rate, min_delay=0.02)

    assert lag == 7


def test_compute_heart_rate_invalid_method():
    with pytest.raises(NotImplementedError):
        compute_heart_rate(data=np.zeros(10), sample_rate=100, method="unknown")


def test_derive_respiratory_rate_requires_min_peaks():
    peaks = np.array([10, 20, 30])
    with pytest.raises(ValueError):
        derive_respiratory_rate(peaks=peaks, rri=np.ones_like(peaks), sample_rate=100)
