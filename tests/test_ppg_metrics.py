import numpy as np
import pytest

from physiokit.ppg.metrics import (
    compute_heart_rate,
    compute_spo2_from_perfusion,
    compute_spo2_in_frequency,
    compute_spo2_in_time,
    derive_respiratory_rate,
)
from physiokit.ppg.peaks import filter_peaks, find_peaks
from physiokit.ppg.synthesize import synthesize


def test_ppg_find_peaks_and_filter():
    np.random.seed(0)
    sample_rate = 50
    ppg, _, _ = synthesize(signal_length=1000, sample_rate=sample_rate, heart_rate=70)

    peaks = find_peaks(ppg, sample_rate=sample_rate)
    filt = filter_peaks(peaks, sample_rate=sample_rate)

    assert peaks.size > 5
    assert filt.size <= peaks.size


def test_compute_heart_rate_fft_matches_frequency():
    sample_rate = 50
    freq = 1.2
    t = np.arange(0, 10, 1 / sample_rate)
    data = np.sin(2 * np.pi * freq * t)

    bpm, qos = compute_heart_rate(data=data, sample_rate=sample_rate, method="fft", lowcut=0.5, highcut=3.0)

    assert abs(bpm - freq * 60) < 5
    assert qos > 0.2


def test_compute_heart_rate_invalid_method():
    with pytest.raises(NotImplementedError):
        compute_heart_rate(data=np.zeros(10), sample_rate=50, method="bad")


def test_compute_heart_rate_from_peaks_on_synth():
    np.random.seed(1)
    target_bpm = 75
    sample_rate = 50
    ppg, _, _ = synthesize(signal_length=1500, sample_rate=sample_rate, heart_rate=target_bpm)

    bpm, qos = compute_heart_rate(data=ppg, sample_rate=sample_rate, method="peak")

    assert abs(bpm - target_bpm) < 15
    assert qos > 0.2


def test_derive_respiratory_rate_riiv():
    sample_rate = 50
    resp_freq = 0.2
    heart_freq = 1.0
    t = np.arange(0, 30, 1 / sample_rate)
    ppg = (1 + 0.5 * np.sin(2 * np.pi * resp_freq * t)) * np.sin(2 * np.pi * heart_freq * t)

    period = int(sample_rate / heart_freq)
    peaks = np.arange(period // 4, t.size, period)
    troughs = peaks + period // 2
    troughs = troughs[troughs < t.size]

    rsp_bpm, qos = derive_respiratory_rate(
        ppg=ppg, peaks=peaks[: troughs.size], troughs=troughs, sample_rate=sample_rate, method="riiv"
    )

    assert abs(rsp_bpm - resp_freq * 60) < 3
    assert qos > 0


def test_derive_respiratory_rate_requires_min_peaks():
    peaks = np.array([1, 2, 3])
    troughs = np.array([1, 2, 3])
    ppg = np.zeros(5)
    with pytest.raises(ValueError):
        derive_respiratory_rate(ppg=ppg, peaks=peaks, troughs=troughs, sample_rate=10, method="riiv")


def test_spo2_estimations_are_consistent():
    sample_rate = 50
    t = np.arange(0, 10, 1 / sample_rate)
    ppg1 = 1.0 + 0.2 * np.sin(2 * np.pi * 1.0 * t)
    ppg2 = 1.0 + 0.2 * np.sin(2 * np.pi * 1.0 * t + 0.3)
    coefs = (1.0, 0.0, 90.0)

    perf_spo2 = compute_spo2_from_perfusion(dc1=1.0, ac1=0.2, dc2=1.0, ac2=0.2, coefs=coefs)
    time_spo2 = compute_spo2_in_time(ppg1=ppg1, ppg2=ppg2, coefs=coefs, sample_rate=sample_rate)
    freq_spo2 = compute_spo2_in_frequency(ppg1=ppg1, ppg2=ppg2, coefs=coefs, sample_rate=sample_rate)

    assert 85 <= perf_spo2 <= 95
    assert abs(time_spo2 - perf_spo2) < 5
    assert abs(freq_spo2 - perf_spo2) < 5
