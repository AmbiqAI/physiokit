import numpy as np

from physiokit.imu.metrics import compute_counts, compute_enmo, compute_tilt_angles


def test_compute_enmo_and_tilt_angles():
    x = np.array([0, 1, 2], dtype=float)
    y = np.array([0, 1, 2], dtype=float)
    z = np.array([1, 1, 2], dtype=float)

    enmo = compute_enmo(x, y, z)
    angle_x, angle_y, angle_z = compute_tilt_angles(x, y, z, in_radians=False)

    assert enmo[0] == 0
    assert np.all(enmo >= 0)
    assert 30 < angle_x[1] < 50
    assert 30 < angle_y[1] < 50
    assert 80 < angle_z[0] <= 90


def test_compute_counts(monkeypatch):
    # Use identity filter to isolate count aggregation logic.
    monkeypatch.setattr("physiokit.imu.metrics._count_bpf_filter", lambda data: data)

    data = np.ones((300, 3)) * 10
    counts = compute_counts(data, sample_rate=30, epoch_len=1, min_thresh=4, max_thresh=128)

    assert counts.shape == (10, 3)
    assert np.all(counts == 100)
