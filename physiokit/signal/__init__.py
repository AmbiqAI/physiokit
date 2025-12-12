"""

# :material-chart-bell-curve: Signal API

Shared signal-processing primitives for filtering, noise injection, smoothing, and transforms.

## Available Tools

- **[filter](./filter)**: Butterworth filtering, resampling, normalization, gradient/quotient filters (`filter_signal`, `get_butter_sos`, `generate_arm_biquad_sos`, `resample_signal`, `normalize_signal`, `moving_gradient_filter`, `quotient_filter_mask`, `remove_baseline_wander`).
- **[noise](./noise)**: Additive noise sources and perturbations (`add_noise_sources`, `add_baseline_wander`, `add_burst_noise`, `add_motion_noise`, `add_emg_noise`, `add_powerline_noise`, `add_lead_noise`).
- **[distort](./distort)**: Distortion generators and drifts (`add_distortions`, `create_linear_drift`, `create_noise_artifacts`, `create_noise_distortions`, `create_powerline_noise`).
- **[smooth](./smooth)**: Smoothing utilities (`signal_smooth_boxcar`, `signal_smooth_boxzen`, `signal_smooth_conv`, `signal_smooth_median`, `signal_smooth_savgol`).
- **[transform](./transform)**: FFT and rescaling helpers (`compute_fft`, `rescale_signal`).

Copyright 2025 Ambiq. All Rights Reserved.

"""
from .distort import (
    add_distortions,
    create_linear_drift,
    create_noise_artifacts,
    create_noise_distortions,
    create_powerline_noise,
)
from .filter import (
    filter_signal,
    generate_arm_biquad_sos,
    get_butter_sos,
    moving_gradient_filter,
    normalize_signal,
    quotient_filter_mask,
    remove_baseline_wander,
    resample_signal,
)
from .noise import (
    add_baseline_wander,
    add_burst_noise,
    add_emg_noise,
    add_lead_noise,
    add_motion_noise,
    add_noise_sources,
    add_powerline_noise,
)
from .smooth import (
    signal_smooth_boxcar,
    signal_smooth_boxzen,
    signal_smooth_conv,
    signal_smooth_median,
    signal_smooth_savgol,
)
from .transform import compute_fft, rescale_signal
