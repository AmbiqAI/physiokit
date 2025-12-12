"""

# :material-waveform: ECG Synthetic API

Utilities to generate synthetic ECG beats and parameter presets.

## Available Tools

- **[rhythm_generator](./rhythm_generator)**: WaSP-based ECG synthesis (`simulate_brisk`).
- **[presets](./presets)**: Parameter generation (`generate_preset_parameters`).
- **[defines](./defines)**: Preset enums and parameter dataclass (`EcgPreset`, `EcgPresetParameters`).

Copyright 2025 Ambiq. All Rights Reserved.

"""
from .defines import EcgPreset, EcgPresetParameters
from .presets import generate_preset_parameters
from .rhythm_generator import simulate_brisk
