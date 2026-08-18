from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np


def _new_id() -> str:
    return str(uuid4())


GraphColor = tuple[int, int, int, int]

GRAPH_COLORS: tuple[GraphColor, ...] = (
    (255, 255, 255, 255),  # white
    (230, 70, 70, 255),  # red
    (70, 130, 240, 255),  # blue
    (70, 200, 100, 255),  # green
    (240, 215, 70, 255),  # yellow
    (230, 120, 40, 255),  # orange
    (70, 210, 210, 255),  # cyan
    (215, 80, 210, 255),  # magenta
    (150, 100, 240, 255),  # violet
    (130, 210, 60, 255),  # lime
    (250, 150, 190, 255),  # pink
    (70, 170, 230, 255),  # sky blue
    (240, 170, 50, 255),  # amber
    (100, 220, 170, 255),  # mint
    (210, 110, 80, 255),  # coral
    (130, 170, 255, 255),  # light blue
    (190, 220, 90, 255),  # chartreuse
    (220, 130, 240, 255),  # lavender
    (90, 200, 200, 255),  # teal
    (245, 180, 130, 255),  # peach
)

_graph_color_counter = count()


def next_graph_color() -> GraphColor:
    """Return the next standard graph color, cycling through the palette."""
    return GRAPH_COLORS[next(_graph_color_counter) % len(GRAPH_COLORS)]


def normalize_graph_color(value: object) -> GraphColor:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Graph color must contain four RGBA components")
    color = tuple(int(component) for component in value)
    if any(component < 0 or component > 255 for component in color):
        raise ValueError("Graph color components must be between 0 and 255")
    return cast(GraphColor, color)


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
    color: GraphColor = field(default_factory=next_graph_color)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "color" not in state:
            self.color = next_graph_color()


@dataclass
class Measurement:
    """A stored measurement produced by one application module."""

    module_id: str
    name: str
    id: str = field(default_factory=_new_id)
    module_state: dict[str, Any] = field(default_factory=dict)
    graphs: list[GraphData] = field(default_factory=list)
    graph_colors: dict[str, GraphColor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.remember_graph_colors()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "graph_colors" not in state:
            self.graph_colors = {}
        self.remember_graph_colors()

    def color_for_graph(self, name: str) -> GraphColor:
        """Return the persistent color assigned to a logical graph name."""
        color = self.graph_colors.get(name)
        if color is None:
            color = next_graph_color()
            self.graph_colors[name] = color
        return color

    def remember_graph_colors(self) -> None:
        """Preserve styles before graph objects are removed or replaced."""
        for graph in self.graphs:
            if not hasattr(graph, "color"):
                graph.color = next_graph_color()
            self.graph_colors.setdefault(graph.name, graph.color)


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
