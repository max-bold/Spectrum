"""Reusable audio signal generation and analysis primitives."""

from .audioio import AudioDevice, list_devices, play_and_record
from .generators import (
    PinkNoiseThread,
    log_chirp,
    pink_noise,
    pink_noise_zi,
    white_noise,
)
from .levels import as_channels, normalize_peak, peak_levels
from .phase import break_phase_wraps, phase_derivative, wrap_phase
from .smoothing import SmoothingWindow, grid_smooth, log_smooth, log_window
from .spectrum import (
    AnalysisMethod,
    ReferenceMode,
    SpectrumConfig,
    SpectrumResult,
    analyze_spectrum,
    magnitude_db,
    phase_degrees,
    power_db,
)
from .types import ASignal, FrequencyBand

__all__ = [
    "AnalysisMethod",
    "AudioDevice",
    "ASignal",
    "FrequencyBand",
    "PinkNoiseThread",
    "ReferenceMode",
    "SmoothingWindow",
    "SpectrumConfig",
    "SpectrumResult",
    "analyze_spectrum",
    "as_channels",
    "break_phase_wraps",
    "grid_smooth",
    "list_devices",
    "log_chirp",
    "log_smooth",
    "log_window",
    "magnitude_db",
    "normalize_peak",
    "peak_levels",
    "phase_derivative",
    "wrap_phase",
    "phase_degrees",
    "pink_noise",
    "pink_noise_zi",
    "play_and_record",
    "power_db",
    "white_noise",
]
