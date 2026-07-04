from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .types import FrequencyBand


@dataclass(frozen=True)
class ImpedanceConfig:
    sample_rate: int = 48_000
    duration: float = 5.0
    reference_resistor: float = 3.25
    calibration_resistor: float = 10.4
    band: FrequencyBand = FrequencyBand()
    points: int = 1024


@dataclass(frozen=True)
class ImpedanceResult:
    frequency: NDArray[np.float64]
    impedance: NDArray[np.complex128]


def calculate_impedance_from_transfer(
    transfer: NDArray[np.number],
    reference_resistor: float,
) -> NDArray[np.complex128]:
    """Calculate load impedance from divider transfer function."""
    if reference_resistor <= 0:
        raise ValueError("Reference resistor must be positive")
    transfer = np.asarray(transfer, dtype=np.complex128)
    impedance = np.full(transfer.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    valid = np.abs(1.0 - transfer) >= 1e-12
    impedance[valid] = reference_resistor * transfer[valid] / (1.0 - transfer[valid])
    return impedance
