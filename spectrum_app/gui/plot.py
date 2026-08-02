from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dearpygui.dearpygui as dpg
import numpy as np

from audioanalysis import break_phase_wraps, phase_derivative, wrap_phase
from spectrum_app.core.model import AxisSpec, GraphData, Measurement
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
        self._update_watermark()

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
            axis_spec: [
                item for item in visible_graphs if item[1].y_axis == axis_spec
            ]
            for axis_spec in self.AXIS_ORDER
        }
        visible_axis_specs = [
            axis_spec
            for axis_spec in self.AXIS_ORDER
            if grouped_graphs[axis_spec]
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

        for series_tag in self.series_tags:
            dpg.delete_item(series_tag)
        self.series_tags.clear()

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
            )
            for measurement, graph in grouped_graphs[axis_spec]:
                series_tag = self._series_tag(graph.id)
                x, y = self._display_data(graph)
                dpg.add_line_series(
                    x.tolist(),
                    y.tolist(),
                    label=f"{measurement.name}: {graph.name}",
                    tag=series_tag,
                    parent=axis_tag,
                )
                self.series_tags.append(series_tag)

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
