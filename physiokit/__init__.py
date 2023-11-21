from importlib.metadata import version

__version__ = version(__name__)

from . import ecg, hrv, imu, ppg, rsp, signal
