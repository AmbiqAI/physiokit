import numpy as np

from physiokit.signal import moving_gradient_filter, quotient_filter_mask, resample_signal
from physiokit.signal.filter import resample_categorical


def test_resample_signal_and_categorical_shapes():
    data = np.arange(100).reshape(10, 10)
    resampled = resample_signal(data, sample_rate=10, target_rate=20, axis=1)
    assert resampled.shape[1] == 20

    labels = np.array([0, 1, 1, 2, 2, 3])
    labels_resampled = resample_categorical(labels, sample_rate=6, target_rate=3, axis=0)
    assert labels_resampled.shape[0] == 3
    # Nearest neighbor resampling preserves categorical values
    assert set(labels_resampled.tolist()).issubset(set(labels.tolist()))


def test_moving_gradient_filter_and_quotient_filter_mask():
    data = np.array([0, 1, 3, 1, 0], dtype=float)
    filt = moving_gradient_filter(data, sample_rate=10, sig_window=0.2, avg_window=0.5, sig_prom_weight=1.0)
    assert filt.shape == data.shape
    assert filt.max() >= 0

    rr_ints = np.array([10, 10, 30, 10], dtype=float)
    mask = quotient_filter_mask(rr_ints, iterations=2, lowcut=0.5, highcut=1.5)
    assert mask.tolist() == [0, 0, 1, 1]
