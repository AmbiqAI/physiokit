"""

# :material-compass: IMU API

Inertial metrics for accelerometer-derived activity summaries.

## Available Tools

- **[metrics](./metrics)**: ENMO, tilt angles, and activity counts (`compute_enmo`, `compute_tilt_angles`, `compute_counts`).

Copyright 2025 Ambiq. All Rights Reserved.

"""

from .metrics import compute_counts, compute_enmo, compute_tilt_angles
