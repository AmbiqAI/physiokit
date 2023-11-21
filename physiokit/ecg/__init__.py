"""ECG module for PhysioKit"""
from .clean import clean, square_filter_mask
from .metrics import (
    compute_heart_rate,
    compute_heart_rate_from_peaks,
    derive_respiratory_rate,
)
from .peaks import compute_rr_intervals, filter_peaks, filter_rr_intervals, find_peaks
from .synthesize import synthesize
from .synthetic import generate_afib, generate_nsr, SyntheticFiducials, SyntheticSegments, SyntheticParameters, EcgPresets
