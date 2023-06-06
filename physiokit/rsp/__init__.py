"""Respiratory signal module for PhysioKit."""
from .synthesize import synthesize
from .clean import clean
from .metrics import (
    compute_respiratory_rate,
    compute_respiratory_rate_from_fft,
    compute_respiratory_rate_from_peaks
)
from .peaks import (
    find_peaks,
    filter_peaks
)
