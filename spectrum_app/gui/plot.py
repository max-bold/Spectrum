from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dearpygui.dearpygui as dpg
import numpy as np

from audioanalysis import break_phase_wraps, phase_derivative, wrap_phase
from spectrum_app.core.model import AxisSpec, GraphData, Measurement, PlotType
from spectrum_app.core.plot_export import PlotExportError, PlotExporter

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class Plot:
    EXPORT_WAIT_FRAMES = 2
    WATERMARK_TEXT = "BM Spectrum"
    WATERMARK_RIGHT_MARGIN = 36
    WATERMARK_TOP_MARGIN = 20
    WATERMARK_COLOR = (180, 180, 180, 110)
    WATERMARK_SIZE = 15
    BAR_COLOR = (70, 145, 215, 210)
    BAR_FILL = (70, 145, 215, 140)
    WATERMARK_BLOCKING_ITEM_TYPES = {
        "mvAppItemType::mvWindowAppItem",
        "mvAppItemType::mvFileDialog",
    }
    AXIS_ORDER = (
        AxisSpec.LEVEL,
        AxisSpec.IMPEDANCE,
        AxisSpec.PHASE,
        AxisSpec.THD,
    )

    def __init__(self, app: "SpectrumApplication") -> None:
        self.app = app
        self.tag = "app::plot"
        self.x_axis = "app::plot::x_axis"
        self.watermark_layer = "app::plot::watermark_layer"
        self.watermark = "app::plot::watermark"
        self.y_axes = [
            "app::plot::y_axis_1",
            "app::plot::y_axis_2",
            "app::plot::y_axis_3",
        ]
        self.series_tags: list[str] = []
        self._series_layout: dict[str, tuple[int | str, PlotType]] = {}
        self._bar_data: dict[
            str,
            tuple[np.ndarray, np.ndarray, int | str, float],
        ] = {}
        self._topology: tuple[tuple[str, AxisSpec, PlotType], ...] = ()
        self._pending_axis_fits: set[AxisSpec] = set()
        self.axis_warning = "app::plot::axis_warning"
        self.axis_warning_text = "app::plot::axis_warning_text"
        self.export_dialog = "app::plot::export_dialog"
        self.exporter = PlotExporter()
        self._pending_export_path: Path | None = None
        self._export_wait_frames = 0
        self._built = False

    def build(self, width: int, height: int) -> None:
        with dpg.plot(  # pyright: ignore[reportGeneralTypeIssues]
            width=width,
            height=height,
            tag=self.tag,
        ):
            dpg.add_plot_legend()
            dpg.add_plot_axis(
                dpg.mvXAxis,
                label="Frequency [Hz]",
                tag=self.x_axis,
                scale=dpg.mvPlotScale_Log10,
            )
            for axis_type, axis_tag in zip(
                (dpg.mvYAxis, dpg.mvYAxis2, dpg.mvYAxis3),
                self.y_axes,
            ):
                dpg.add_plot_axis(
                    axis_type,
                    tag=axis_tag,
                    show=False,
                )
        dpg.add_viewport_drawlist(tag=self.watermark_layer, front=True)
        dpg.draw_text(
            (0, 0),
            self.WATERMARK_TEXT,
            tag=self.watermark,
            parent=self.watermark_layer,
            color=self.WATERMARK_COLOR,
            size=self.WATERMARK_SIZE,
            show=False,
        )
        self._built = True

    def build_export(self, export_menu: int | str) -> None:
        dpg.add_menu_item(
            label="Plot",
            parent=export_menu,
            callback=self.show_export_dialog,
        )
        dpg.add_file_dialog(
            tag=self.export_dialog,
            show=False,
            modal=True,
            width=700,
            height=400,
            default_filename="plot.png",
            callback=self.export_png,
        )
        dpg.add_file_extension(".png", parent=self.export_dialog)

    def show_export_dialog(self, sender=None, app_data=None, user_data=None) -> None:
        dpg.show_item(self.export_dialog)

    def export_png(
        self,
        sender: int | str,
        app_data: dict[str, Any],
        user_data=None,
    ) -> None:
        value = app_data.get("file_path_name")
        if not value:
            return
        if dpg.does_item_exist(sender):
            dpg.hide_item(sender)
        self.app.main_window.set_status_text(f"Exporting plot: {value}")
        self._pending_export_path = Path(value)
        self._export_wait_frames = self.EXPORT_WAIT_FRAMES

    def _process_pending_export(self) -> None:
        path = self._pending_export_path
        if path is None:
            return
        if self._export_wait_frames > 0:
            self._export_wait_frames -= 1
            return
        self._pending_export_path = None
        try:
            self.exporter.export(
                path,
                self.tag,
                on_complete=self._export_completed,
            )
        except PlotExportError as error:
            self.app.main_window.set_status_text(f"Plot export error: {error}")
            return

    def _export_completed(self, path: Path | None, error: str | None) -> None:
        if error is not None or path is None:
            self.app.main_window.set_status_text(
                f"Plot export error: {error or 'Unknown error'}"
            )
            return
        self.app.main_window.set_status_text(f"Plot exported: {path}")

    def update(self) -> None:
        self._process_pending_export()
        if self.app.app_state.graph_data_changed:
            self.app.app_state.graph_data_changed = False
            try:
                self._redraw()
            except Exception:
                self.app.app_state.graph_data_changed = True
                raise
        self._refresh_bar_baselines()
        self._update_watermark()

    def request_axis_autoscale(self, axis_spec: AxisSpec) -> None:
        """Fit an axis after its displayed data is updated on the next redraw."""
        if any(spec == axis_spec for _, spec, _ in self._topology):
            self._pending_axis_fits.add(axis_spec)

    def _update_watermark(self) -> None:
        if not self._built:
            return
        if self._watermark_is_obscured():
            dpg.configure_item(self.watermark, show=False)
            return
        rect_min = dpg.get_item_rect_min(self.tag)
        rect_max = dpg.get_item_rect_max(self.tag)
        if len(rect_min) < 2 or len(rect_max) < 2:
            return
        if rect_max[0] <= rect_min[0] or rect_max[1] <= rect_min[1]:
            dpg.configure_item(self.watermark, show=False)
            return

        text_width, _ = dpg.get_text_size(self.WATERMARK_TEXT)
        dpg.configure_item(
            self.watermark,
            pos=(
                float(rect_max[0] - text_width - self.WATERMARK_RIGHT_MARGIN),
                float(rect_min[1] + self.WATERMARK_TOP_MARGIN),
            ),
            show=True,
        )

    def _watermark_is_obscured(self) -> bool:
        main_window = self.app.main_window.tag
        for item in dpg.get_windows():
            if dpg.get_item_alias(item) == main_window:
                continue
            if dpg.get_item_type(item) not in self.WATERMARK_BLOCKING_ITEM_TYPES:
                continue
            if dpg.is_item_shown(item):
                return True
        return False

    def _redraw(self) -> None:
        dpg.set_axis_limits(self.x_axis, *self.app.settings.frequency_range)
        visible_graphs = list(self._visible_graphs())
        grouped_graphs = {
            axis_spec: [item for item in visible_graphs if item[1].y_axis == axis_spec]
            for axis_spec in self.AXIS_ORDER
        }
        visible_axis_specs = [
            axis_spec for axis_spec in self.AXIS_ORDER if grouped_graphs[axis_spec]
        ]

        unsupported_specs = {
            graph.y_axis
            for _, graph in visible_graphs
            if graph.y_axis not in self.AXIS_ORDER
        }
        if unsupported_specs:
            names = ", ".join(spec.value for spec in unsupported_specs)
            raise ValueError(f"Unsupported Y axis: {names}")

        if len(visible_axis_specs) > len(self.y_axes):
            self._show_axis_warning(visible_axis_specs)
            return

        self._hide_axis_warning()

        desired = [
            (measurement, graph, visible_axis_specs.index(graph.y_axis))
            for measurement, graph in visible_graphs
        ]
        topology = tuple(
            (
                graph.id,
                graph.y_axis,
                getattr(graph, "plot_type", PlotType.LINE),
            )
            for _, graph, _ in desired
        )
        topology_changed = topology != self._topology
        desired_tags = {self._series_tag(graph.id) for _, graph, _ in desired}
        for series_tag in list(self.series_tags):
            if series_tag not in desired_tags:
                dpg.delete_item(series_tag)
                self.series_tags.remove(series_tag)
                self._series_layout.pop(series_tag, None)
                self._bar_data.pop(series_tag, None)

        for index, axis_tag in enumerate(self.y_axes):
            if index >= len(visible_axis_specs):
                dpg.configure_item(axis_tag, show=False)
                continue

            axis_spec = visible_axis_specs[index]
            dpg.configure_item(
                axis_tag,
                show=True,
                label=self._axis_label(axis_spec),
                scale=self._axis_scale(axis_spec),
                opposite=index > 0,
                no_side_switch=True,
            )

        for measurement, graph, axis_index in desired:
            axis_tag = self.y_axes[axis_index]
            series_tag = self._series_tag(graph.id)
            x, y = self._display_data(graph)
            label = f"{measurement.name}: {graph.name}"
            plot_type = getattr(graph, "plot_type", PlotType.LINE)
            layout = axis_tag, plot_type
            if self._series_layout.get(series_tag) != layout:
                if series_tag in self.series_tags:
                    dpg.delete_item(series_tag)
                    self.series_tags.remove(series_tag)
                self._bar_data.pop(series_tag, None)
                if plot_type == PlotType.BARS:
                    self._add_bar_series(series_tag, axis_tag, label, x, y)
                else:
                    dpg.add_line_series(
                        x.tolist(),
                        y.tolist(),
                        label=label,
                        tag=series_tag,
                        parent=axis_tag,
                    )
                self.series_tags.append(series_tag)
                self._series_layout[series_tag] = layout
            else:
                dpg.set_item_label(series_tag, label)
                if plot_type == PlotType.BARS:
                    self._set_bar_data(series_tag, axis_tag, x, y)
                else:
                    dpg.set_value(series_tag, [x.tolist(), y.tolist()])

        pending_axis_fits = self._pending_axis_fits
        self._pending_axis_fits = set()
        for index, axis_spec in enumerate(visible_axis_specs):
            if topology_changed or axis_spec in pending_axis_fits:
                dpg.fit_axis_data(self.y_axes[index])
        self._topology = topology

    def _add_bar_series(
        self,
        tag: str,
        parent: int | str,
        label: str,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Draw variable-width bars that remain even on a logarithmic X axis."""
        top_x, top_y, baseline = self._bar_series_data(parent, x, y)
        self._bar_data[tag] = (
            top_x,
            top_y,
            parent,
            float(baseline[0]) if len(baseline) else 0.0,
        )
        dpg.add_custom_series(
            top_x.tolist(),
            top_y.tolist(),
            3,
            y1=baseline.tolist(),
            label=label,
            tag=tag,
            parent=parent,
            callback=self._draw_bar_series,
            tooltip=False,
        )

    def _set_bar_data(
        self,
        tag: str,
        parent: int | str,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        top_x, top_y, baseline = self._bar_series_data(parent, x, y)
        self._bar_data[tag] = (
            top_x,
            top_y,
            parent,
            float(baseline[0]) if len(baseline) else 0.0,
        )
        dpg.set_value(
            tag,
            [top_x.tolist(), top_y.tolist(), baseline.tolist()],
        )

    def _bar_series_data(
        self,
        parent: int | str,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left, right = self._log_bar_edges(x)
        top_x = np.column_stack((left, right)).reshape(-1)
        top_y = np.repeat(np.asarray(y, dtype=np.float64), 2)
        lower = self._axis_lower_limit(parent, top_y)
        baseline = np.full_like(top_y, lower)
        return top_x, top_y, baseline

    def _draw_bar_series(self, sender, app_data, user_data=None) -> None:
        if len(app_data) < 4:
            return
        transformed_x = app_data[1]
        transformed_y = app_data[2]
        transformed_baseline = app_data[3]
        dpg.delete_item(sender, children_only=True, slot=2)
        dpg.push_container_stack(sender)
        try:
            for index in range(0, len(transformed_x) - 1, 2):
                left = float(transformed_x[index]) + 1.5
                right = float(transformed_x[index + 1]) - 1.5
                if right < left:
                    left = right = (left + right) / 2.0
                dpg.draw_rectangle(
                    (left, transformed_y[index]),
                    (right, transformed_baseline[index + 1]),
                    color=self.BAR_COLOR,
                    fill=self.BAR_FILL,
                )
        finally:
            dpg.pop_container_stack()

    def _refresh_bar_baselines(self) -> None:
        for tag, (top_x, top_y, axis_tag, previous) in list(self._bar_data.items()):
            if tag not in self.series_tags:
                self._bar_data.pop(tag, None)
                continue
            lower = self._axis_lower_limit(axis_tag, top_y)
            if np.isclose(lower, previous, rtol=1e-9, atol=1e-12):
                continue
            self._bar_data[tag] = top_x, top_y, axis_tag, lower
            dpg.set_value(
                tag,
                [
                    top_x.tolist(),
                    top_y.tolist(),
                    np.full_like(top_y, lower).tolist(),
                ],
            )

    @staticmethod
    def _axis_lower_limit(axis_tag: int | str, y: np.ndarray) -> float:
        try:
            lower, _ = dpg.get_axis_limits(axis_tag)
            if np.isfinite(lower):
                return float(lower)
        except (AttributeError, SystemError, TypeError, ValueError):
            pass
        finite = np.asarray(y, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        return float(np.min(finite)) if finite.size else 0.0

    @staticmethod
    def _log_bar_edges(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centers = np.asarray(x, dtype=np.float64)
        if centers.ndim != 1 or not len(centers):
            return np.empty(0), np.empty(0)
        if np.any(centers <= 0.0) or np.any(np.diff(centers) <= 0.0):
            raise ValueError("Bar frequencies must be positive and increasing")
        if len(centers) == 1:
            return centers / np.sqrt(2.0), centers * np.sqrt(2.0)
        boundaries = np.sqrt(centers[:-1] * centers[1:])
        left = np.concatenate(([centers[0] ** 2 / boundaries[0]], boundaries))
        right = np.concatenate((boundaries, [centers[-1] ** 2 / boundaries[-1]]))
        return left, right

    def _display_data(self, graph: GraphData) -> tuple[np.ndarray, np.ndarray]:
        if graph.y_axis != AxisSpec.PHASE:
            return graph.x, graph.y
        if self.app.settings.phase_unit == "deg/dec":
            return graph.x, phase_derivative(graph.x, graph.y)
        return break_phase_wraps(graph.x, wrap_phase(graph.y))

    def _visible_graphs(self) -> Iterable[tuple[Measurement, GraphData]]:
        visible_ids = set(self.app.app_state.visible_graph_ids)
        for measurement in self.app.app_state.measurements:
            for graph in measurement.graphs:
                if graph.id in visible_ids:
                    yield measurement, graph

    def _axis_label(self, axis_spec: AxisSpec) -> str:
        if axis_spec == AxisSpec.LEVEL:
            return "Spectrum [dB]"
        if axis_spec == AxisSpec.IMPEDANCE:
            return "Impedance [Ohm]"
        if axis_spec == AxisSpec.PHASE:
            return f"Phase [{self.app.settings.phase_unit}]"
        return "THD [%]"

    def _axis_scale(self, axis_spec: AxisSpec) -> int:
        if (
            axis_spec == AxisSpec.IMPEDANCE
            and self.app.settings.impedance_scale == "log"
        ):
            return dpg.mvPlotScale_Log10
        if axis_spec == AxisSpec.THD and self.app.settings.thd_scale == "log":
            return dpg.mvPlotScale_Log10
        return dpg.mvPlotScale_Linear

    def _show_axis_warning(self, axis_specs: list[AxisSpec]) -> None:
        axis_names = ", ".join(self._axis_label(spec) for spec in axis_specs)
        message = (
            "Dear PyGui supports no more than three Y axes.\n"
            f"Visible axes: {axis_names}.\n"
            "Hide at least one measurement to update the plot."
        )

        if dpg.does_item_exist(self.axis_warning):
            dpg.set_value(self.axis_warning_text, message)
            dpg.configure_item(self.axis_warning, show=True)
            return

        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            label="Too many plot axes",
            tag=self.axis_warning,
            modal=True,
            no_resize=True,
            no_collapse=True,
            width=420,
            height=130,
        ):
            dpg.add_text(message, tag=self.axis_warning_text, wrap=390)
            dpg.add_button(
                label="OK",
                width=-1,
                callback=self._close_axis_warning,
            )

    def _hide_axis_warning(self) -> None:
        if dpg.does_item_exist(self.axis_warning):
            dpg.configure_item(self.axis_warning, show=False)

    def _close_axis_warning(self, sender=None, app_data=None, user_data=None) -> None:
        dpg.configure_item(self.axis_warning, show=False)

    @staticmethod
    def _series_tag(graph_id: str) -> str:
        return f"app::plot::series::{graph_id}"
