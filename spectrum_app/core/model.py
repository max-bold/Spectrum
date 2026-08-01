from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np


AxisScale = Literal["linear", "log"]


def _new_id() -> str:
    return str(uuid4())


@dataclass(frozen=True)
class AxisSpec:
    """Describes the physical quantity and its plot representation."""

    quantity: str
    unit: str
    scale: AxisScale = "linear"


@dataclass
class GraphData:
    """One calculated series that can be displayed by the plot workspace."""

    name: str
    x: np.ndarray
    y: np.ndarray
    x_axis: AxisSpec
    y_axis: AxisSpec
    id: str = field(default_factory=_new_id)


@dataclass
class Measurement:
    """A stored measurement produced by one application module."""

    module_id: str
    name: str
    id: str = field(default_factory=_new_id)
    settings: dict[str, Any] = field(default_factory=dict)
    module_state: dict[str, Any] = field(default_factory=dict)
    graphs: list[GraphData] = field(default_factory=list)


@dataclass
class AppState:
    """Persistent project data and application UI state."""

    project_path: Path | None = None
    measurements: list[Measurement] = field(default_factory=list)
    active_measurement_id: str | None = None
    visible_graph_ids: list[str] = field(default_factory=list)
    interface_state: dict[str, Any] = field(default_factory=dict)
    measuring: bool = False
