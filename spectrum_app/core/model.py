from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


def _new_id() -> str:
    return str(uuid4())


class AxisSpec(str, Enum):
    FREQ = "frequency"
    LEVEL = "level"
    IMPEDANCE = "impedance"
    PHASE = "phase"
    THD = "thd"


class PlotType(str, Enum):
    LINE = "line"
    BARS = "bars"


@dataclass
class GraphData:
    """One calculated series that can be displayed by the plot workspace."""

    name: str
    x: np.ndarray
    y: np.ndarray
    x_axis: AxisSpec
    y_axis: AxisSpec
    id: str = field(default_factory=_new_id)
    plot_type: PlotType = PlotType.LINE


@dataclass
class Measurement:
    """A stored measurement produced by one application module."""

    module_id: str
    name: str
    id: str = field(default_factory=_new_id)
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
    graph_data_changed: bool = True
