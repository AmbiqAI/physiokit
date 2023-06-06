try:
    from importlib.metadata import version
    __version__ = version(__name__)
except ImportError:
    __version__ = "0.0.0"

from . import ecg
from . import ppg
from . import rsp
from . import signal
