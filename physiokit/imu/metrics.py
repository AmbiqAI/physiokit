import numpy as np
import numpy.typing as npt
import scipy.signal

from ..signal import resample_signal


def compute_enmo(x: npt.NDArray, y: npt.NDArray, z: npt.NDArray) -> npt.NDArray:
    """Compute ENMO from x, y, and z accelerometer data.

    Reference: https://doi.org/10.1371/journal.pone.0142533

    Args:
        x (npt.NDArray): x-axis accelerometer data
        y (npt.NDArray): y-axis accelerometer data
        z (npt.NDArray): z-axis accelerometer data

    Returns:
        npt.NDArray: ENMO data
    """
    enmo = np.maximum(np.sqrt(x**2 + y**2 + z**2) - 1, 0)
    return enmo


def compute_tilt_angles(
    x: npt.NDArray, y: npt.NDArray, z: npt.NDArray, in_radians: bool = True
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compute tilt angles from x, y, and z accelerometer data.

    Reference: https://doi.org/10.1371/journal.pone.0142533

    Args:
        x (npt.NDArray): x-axis accelerometer data
        y (npt.NDArray): y-axis accelerometer data
        z (npt.NDArray): z-axis accelerometer data
        in_radians (bool, optional): If True, return angles in radians. Defaults to True.

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]: Tilt angles in radians or degrees

    """
    x2 = x**2
    y2 = y**2
    z2 = z**2
    factor = 1 if in_radians else 180.0 / np.pi
    angle_x = np.arctan2(x, np.sqrt(y2 + z2)) * factor
    angle_y = np.arctan2(y, np.sqrt(x2 + z2)) * factor
    angle_z = np.arctan2(z, np.sqrt(x2 + y2)) * factor
    return angle_x, angle_y, angle_z


def _count_bpf_filter(data: npt.NDArray) -> npt.NDArray:
    """Bandpass filter for computing counts. Specific for Actigraph GT3X+.

    Args:
        data (npt.NDArray): 2-D raw accelerometer data [ts x axis] in G.

    Returns:
        npt.NDArray: 2-D filtered accelerometer data [ts x axis] in G.
    """
    b = np.array(
        [
            -0.009341062898525,
            -0.025470289659360,
            -0.004235264826105,
            0.044152415456420,
            0.036493718347760,
            -0.011893961934740,
            -0.022917390623150,
            -0.006788163862310,
            0.000000000000000,
        ]
    )
    a = np.array(
        [
            1.00000000000000000000,
            -3.63367395910957000000,
            5.03689812757486000000,
            -3.09612247819666000000,
            0.50620507633883000000,
            0.32421701566682000000,
            -0.15685485875559000000,
            0.01949130205890000000,
            0.00000000000000000000,
        ]
    )

    zi = scipy.signal.lfilter_zi(b, a).reshape((1, -1))
    zi = zi.repeat(data.shape[0], axis=0) * data[:, 0].reshape((-1, 1))

    data, _ = scipy.signal.lfilter(b, a, data, zi=zi)

    data = ((3.0 / 4096.0) / (2.6 / 256.0) * 237.5) * data
    # 17.127404 is used in ActiLife and 17.128125 is used in firmware.
    return data


def compute_counts(
    data: npt.NDArray, sample_rate: float = 1000, epoch_len: int = 10, min_thresh: int = 4, max_thresh: int = 128
) -> npt.NDArray:
    """Compute counts from raw accelerometer data.

    Reference: https://doi.org/10.1038/s41598-022-16003-x

    Args:
        data (npt.NDArray): 2-D raw accelerometer data [ts x axis] in G.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        epoch_len (int, optional): Epoch length in seconds. Defaults to 10.
        min_thresh (int, optional): Minimum threshold. Defaults to 4.
        max_thresh (int, optional): Maximum threshold. Defaults to 128.

    Returns:
        npt.NDArray: 2-D counts data [ts x axis] in counts.
    """

    # 1. Resample to 30 Hz
    data = resample_signal(data, sample_rate=sample_rate, target_rate=30, axis=0)

    # 2. Bandpass filter
    data = _count_bpf_filter(data)

    # 3. Rectify, & threshold
    data = np.abs(data)
    data[data < min_thresh] = 0
    data[data > max_thresh] = max_thresh
    data = np.floor(data)

    # 4. Downsample to 10 Hz by taking moving average (n=3)
    data = np.nanmean(data.reshape((-1, 3, data.shape[1])), axis=1)

    # 5. Find counts by summing over epoch length
    counts = np.zeros((data.shape[0] // (10 * epoch_len),) + data.shape[1:], dtype=int)
    for i in range(0, counts.shape[0]):
        counts[i] = np.sum(data[i * 10 * epoch_len : (i + 1) * 10 * epoch_len], axis=0)
    # END FOR

    return counts
