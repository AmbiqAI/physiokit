import numpy as np


def rescale_signal(x, old_min: float, old_max: float, new_min: float, new_max: float, clip: bool = True):
    """Rescale signal to new range.
    Args:
        x (npt.NDArray): Signal
        old_min (float): Old minimum
        old_max (float): Old maximum
        new_min (float): New minimum
        new_max (float): New maximum
        clip (bool, optional): Clip values to range. Defaults to True.
    Returns:
        npt.NDArray: Rescaled signal
    """
    if clip:
        x = np.clip(x, old_min, old_max)
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
