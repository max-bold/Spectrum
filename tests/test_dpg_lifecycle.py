from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from spectrum_app import SpectrumApplication
from spectrum_app.core.audio import AudioDevice
from spectrum_app.core.dpg import DearPyGuiRuntime
from spectrum_app.core.model import AxisSpec, GraphData, PlotType
from spectrum_app.gui.error import ErrorDialog
from spectrum_app.gui.main_window import MainWindow
from spectrum_app.modules.spectrum import SpectrumModule
from spectrum_app.modules.phase.settings import PhaseSettingsWindow
from spectrum_app.modules.spectrum.settings import SpectrumSettingsWindow
from spectrum_app.modules.thd.settings import THDSettingsWindow


class FakeContainer:
    def __init__(self, backend: "FakeDpgBackend", kind: str, config: dict) -> None:
        self.backend = backend
        self.kind = kind
        self.config = config

    def __enter__(self) -> "FakeContainer":
        self.backend.calls.append((f"enter_{self.kind}", self.config))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.backend.calls.append((f"exit_{self.kind}", self.config))


class FakeDpgBackend:
    mvXAxis = 0
    mvYAxis = 3
    mvYAxis2 = 4
    mvYAxis3 = 5
    mvPlotScale_Linear = 0
    mvPlotScale_Log10 = 2
    mvButton = 1
    mvThemeCol_Button = 2
    mvThemeCol_ButtonHovered = 3
    mvThemeCol_ButtonActive = 4
    mvThemeCat_Core = 5

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.running_values = iter((True, False))
        self.shown_windows: list[str] = []

    def create_context(self) -> None:
        self.calls.append(("create_context",))

    def configure_app(self, **kwargs) -> None:
        self.calls.append(("configure_app", kwargs))

    def _container(self, kind: str, **kwargs) -> FakeContainer:
        self.calls.append((kind, kwargs))
        return FakeContainer(self, kind, kwargs)

    def add_window(self, **kwargs) -> str:
        self.calls.append(("add_window", kwargs))
        return kwargs["tag"]

    def add_menu_bar(self, **kwargs) -> str:
        self.calls.append(("add_menu_bar", kwargs))
        return "menu_bar"

    def add_group(self, **kwargs) -> str:
        self.calls.append(("add_group", kwargs))
        return "group"

    def window(self, **kwargs) -> FakeContainer:
        return self._container("window", **kwargs)

    def menu_bar(self) -> FakeContainer:
        return self._container("menu_bar")

    def child_window(self, **kwargs) -> FakeContainer:
        return self._container("child_window", **kwargs)

    def plot(self, **kwargs) -> FakeContainer:
        return self._container("plot", **kwargs)

    def group(self, **kwargs) -> FakeContainer:
        return self._container("group", **kwargs)

    def drawlist(self, **kwargs) -> FakeContainer:
        return self._container("drawlist", **kwargs)

    def collapsing_header(self, **kwargs) -> FakeContainer:
        return self._container("collapsing_header", **kwargs)

    def table(self, **kwargs) -> FakeContainer:
        return self._container("table", **kwargs)

    def table_row(self, **kwargs) -> FakeContainer:
        return self._container("table_row", **kwargs)

    def texture_registry(self, **kwargs) -> FakeContainer:
        return self._container("texture_registry", **kwargs)

    def theme(self, **kwargs) -> FakeContainer:
        return self._container("theme", **kwargs)

    def theme_component(self, component: int) -> FakeContainer:
        return self._container("theme_component", component=component)

    def tab_bar(self, **kwargs) -> FakeContainer:
        return self._container("tab_bar", **kwargs)

    def tab(self, **kwargs) -> FakeContainer:
        return self._container("tab", **kwargs)

    def item_handler_registry(self, **kwargs) -> FakeContainer:
        return self._container("item_handler_registry", **kwargs)

    def add_menu(self, **kwargs) -> str:
        self.calls.append(("add_menu", kwargs))
        return kwargs["tag"]

    def add_menu_item(self, **kwargs) -> str:
        self.calls.append(("add_menu_item", kwargs))
        return kwargs.get("tag", "menu_item")

    def add_file_dialog(self, **kwargs) -> str:
        self.calls.append(("add_file_dialog", kwargs))
        return kwargs.get("tag", "file_dialog")

    def add_file_extension(self, extension: str, **kwargs) -> str:
        self.calls.append(("add_file_extension", extension, kwargs))
        return "file_extension"

    def add_plot_legend(self, **kwargs) -> str:
        self.calls.append(("add_plot_legend", kwargs))
        return "plot_legend"

    def add_plot_axis(self, axis: int, **kwargs) -> str:
        self.calls.append(("add_plot_axis", axis, kwargs))
        return kwargs.get("tag", "plot_axis")

    def add_viewport_drawlist(self, **kwargs) -> str:
        self.calls.append(("add_viewport_drawlist", kwargs))
        return kwargs.get("tag", "viewport_drawlist")

    def add_line_series(self, x: list[float], y: list[float], **kwargs) -> str:
        self.calls.append(("add_line_series", x, y, kwargs))
        return kwargs.get("tag", "line_series")

    def add_custom_series(
        self,
        x: list[float],
        y: list[float],
        channel_count: int,
        **kwargs,
    ) -> str:
        self.calls.append(("add_custom_series", x, y, channel_count, kwargs))
        return kwargs.get("tag", "custom_series")

    def add_table_column(self, **kwargs) -> str:
        self.calls.append(("add_table_column", kwargs))
        return "table_column"

    def add_checkbox(self, **kwargs) -> str:
        self.calls.append(("add_checkbox", kwargs))
        return kwargs.get("tag", "checkbox")

    def add_input_text(self, **kwargs) -> str:
        self.calls.append(("add_input_text", kwargs))
        return kwargs.get("tag", "input_text")

    def add_combo(self, items: list[str], **kwargs) -> str:
        self.calls.append(("add_combo", items, kwargs))
        return kwargs.get("tag", "combo")

    def add_input_float(self, **kwargs) -> str:
        self.calls.append(("add_input_float", kwargs))
        return kwargs.get("tag", "input_float")

    def add_slider_float(self, **kwargs) -> str:
        self.calls.append(("add_slider_float", kwargs))
        return kwargs.get("tag", "slider_float")

    def add_input_intx(self, **kwargs) -> str:
        self.calls.append(("add_input_intx", kwargs))
        return kwargs.get("tag", "input_intx")

    def add_input_int(self, **kwargs) -> str:
        self.calls.append(("add_input_int", kwargs))
        return kwargs.get("tag", "input_int")

    def add_item_deactivated_after_edit_handler(self, **kwargs) -> str:
        self.calls.append(("add_item_deactivated_after_edit_handler", kwargs))
        return kwargs.get("tag", "item_deactivated_after_edit_handler")

    def add_radio_button(self, items: tuple[str, ...], **kwargs) -> str:
        self.calls.append(("add_radio_button", items, kwargs))
        return kwargs.get("tag", "radio_button")

    def add_separator(self, **kwargs) -> str:
        self.calls.append(("add_separator", kwargs))
        return "separator"

    def add_button(self, **kwargs) -> str:
        self.calls.append(("add_button", kwargs))
        return kwargs.get("tag", "button")

    def add_image_button(self, texture_tag: str, **kwargs) -> str:
        self.calls.append(("add_image_button", texture_tag, kwargs))
        return kwargs.get("tag", "image_button")

    def does_item_exist(self, item: str) -> bool:
        self.calls.append(("does_item_exist", item))
        return False

    def load_image(self, path: str) -> tuple[int, int, int, list[float]]:
        self.calls.append(("load_image", path))
        return 16, 16, 4, [0.0] * (16 * 16 * 4)

    def add_static_texture(
        self, width: int, height: int, data: list[float], **kwargs
    ) -> str:
        self.calls.append(("add_static_texture", width, height, kwargs))
        return kwargs.get("tag", "texture")

    def add_theme_color(self, target: int, color: tuple, **kwargs) -> str:
        self.calls.append(("add_theme_color", target, color, kwargs))
        return "theme_color"

    def delete_item(self, item: str, **kwargs) -> None:
        if kwargs:
            self.calls.append(("delete_item", item, kwargs))
        else:
            self.calls.append(("delete_item", item))

    def add_child_window(self, **kwargs) -> str:
        self.calls.append(("add_child_window", kwargs))
        return kwargs.get("tag", "child_window")

    def add_text(self, value: str, **kwargs) -> str:
        self.calls.append(("add_text", value, kwargs))
        return kwargs.get("tag", "text")

    def set_value(self, tag: str, value: str) -> None:
        self.calls.append(("set_value", tag, value))

    def get_value(self, tag: str):
        self.calls.append(("get_value", tag))
        return None

    def get_item_pos(self, tag: str) -> list[int]:
        self.calls.append(("get_item_pos", tag))
        return [10, 20]

    def get_item_rect_size(self, tag: str) -> list[int]:
        self.calls.append(("get_item_rect_size", tag))
        return [1200, 800]

    def get_item_rect_min(self, tag: str) -> list[int]:
        self.calls.append(("get_item_rect_min", tag))
        return [20, 40]

    def get_item_rect_max(self, tag: str) -> list[int]:
        self.calls.append(("get_item_rect_max", tag))
        return [820, 640]

    def get_text_size(self, text: str) -> tuple[int, int]:
        self.calls.append(("get_text_size", text))
        return (88, 15)

    def get_windows(self) -> list[str]:
        self.calls.append(("get_windows",))
        return [MainWindow.TAG, *self.shown_windows]

    def get_item_alias(self, item: str) -> str:
        self.calls.append(("get_item_alias", item))
        return item

    def get_item_type(self, item: str) -> str:
        self.calls.append(("get_item_type", item))
        return "mvAppItemType::mvWindowAppItem"

    def is_item_shown(self, item: str) -> bool:
        self.calls.append(("is_item_shown", item))
        return item == MainWindow.TAG or item in self.shown_windows

    def get_item_state(self, tag: str) -> dict[str, list[int]]:
        self.calls.append(("get_item_state", tag))
        return {"rect_size": [1200, 200]}

    def draw_rectangle(self, point1, point2, **kwargs) -> str:
        self.calls.append(("draw_rectangle", point1, point2, kwargs))
        return f"rectangle::{len(self.calls)}"

    def push_container_stack(self, item: str) -> None:
        self.calls.append(("push_container_stack", item))

    def pop_container_stack(self) -> None:
        self.calls.append(("pop_container_stack",))

    def draw_text(self, position, text: str, **kwargs) -> str:
        self.calls.append(("draw_text", position, text, kwargs))
        return f"draw_text::{len(self.calls)}"

    def draw_line(self, point1, point2, **kwargs) -> str:
        self.calls.append(("draw_line", point1, point2, kwargs))
        return f"line::{len(self.calls)}"

    def set_item_pos(self, tag: str, position: list[float]) -> None:
        self.calls.append(("set_item_pos", tag, position))

    def set_item_label(self, tag: str, label: str) -> None:
        self.calls.append(("set_item_label", tag, label))

    def bind_item_theme(self, tag: str, theme: str) -> None:
        self.calls.append(("bind_item_theme", tag, theme))

    def bind_item_handler_registry(self, tag: str, registry: str) -> None:
        self.calls.append(("bind_item_handler_registry", tag, registry))

    def configure_item(self, tag: str, **kwargs) -> None:
        self.calls.append(("configure_item", tag, kwargs))

    def show_item(self, tag: str) -> None:
        self.calls.append(("show_item", tag))

    def hide_item(self, tag: str) -> None:
        self.calls.append(("hide_item", tag))

    def set_axis_limits(self, tag: str, low: float, high: float) -> None:
        self.calls.append(("set_axis_limits", tag, low, high))

    def get_axis_limits(self, tag: str) -> tuple[float, float]:
        self.calls.append(("get_axis_limits", tag))
        return -48.0, 0.0

    def fit_axis_data(self, tag: str) -> None:
        self.calls.append(("fit_axis_data", tag))

    def create_viewport(self, **kwargs) -> None:
        self.calls.append(("create_viewport", kwargs))

    def setup_dearpygui(self) -> None:
        self.calls.append(("setup_dearpygui",))

    def show_viewport(self) -> None:
        self.calls.append(("show_viewport",))

    def set_primary_window(self, tag: str, value: bool) -> None:
        self.calls.append(("set_primary_window", tag, value))

    def is_dearpygui_running(self) -> bool:
        self.calls.append(("is_dearpygui_running",))
        return next(self.running_values)

    def get_callback_queue(self) -> list:
        self.calls.append(("get_callback_queue",))
        return []

    def run_callbacks(self, jobs: list) -> None:
        self.calls.append(("run_callbacks", jobs))

    def render_dearpygui_frame(self) -> None:
        self.calls.append(("render_dearpygui_frame",))

    def destroy_context(self) -> None:
        self.calls.append(("destroy_context",))


class DearPyGuiLifecycleTests(unittest.TestCase):
    def test_error_dialog_is_modal_centered_and_reusable(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        dialog = ErrorDialog(app)

        with (
            patch("spectrum_app.gui.error.dpg", backend),
            patch.object(backend, "does_item_exist", return_value=True),
        ):
            dialog.build()
            dialog.show("Measurement failed", "Detailed diagnostic")

        error_window = next(
            call
            for call in backend.calls
            if call[0] == "window" and call[1].get("tag") == ErrorDialog.TAG
        )
        self.assertTrue(error_window[1]["modal"])
        self.assertIn(
            ("set_value", ErrorDialog.MESSAGE, "Detailed diagnostic"),
            backend.calls,
        )
        self.assertIn(
            (
                "configure_item",
                ErrorDialog.TAG,
                {"label": "Measurement failed"},
            ),
            backend.calls,
        )
        self.assertIn(
            ("configure_item", ErrorDialog.TAG, {"show": True}),
            backend.calls,
        )

    def test_application_runs_dpg_lifecycle(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        app.dpg = DearPyGuiRuntime(backend)
        app.main_window = MainWindow(app)
        app.frame_callbacks.append(lambda: backend.calls.append(("frame_callback",)))

        with (
            patch.object(app._audio_service, "start"),
            patch.object(app._audio_service, "shutdown"),
            patch("spectrum_app.gui.main_window.dpg", backend),
            patch("spectrum_app.gui.error.dpg", backend),
            patch("spectrum_app.gui.app_state.dpg", backend),
            patch("spectrum_app.gui.measurement.dpg", backend),
            patch("spectrum_app.gui.plot.dpg", backend),
            patch("spectrum_app.gui.settings.dpg", backend),
            patch("spectrum_app.gui.project.dpg", backend),
            patch("spectrum_app.gui.measurement_io.dpg", backend),
            patch("spectrum_app.modules.spectrum.view.dpg", backend),
            patch("spectrum_app.modules.spectrum.settings.dpg", backend),
            patch("spectrum_app.modules.phase.view.dpg", backend),
            patch("spectrum_app.modules.phase.settings.dpg", backend),
            patch("spectrum_app.modules.rta.view.dpg", backend),
            patch("spectrum_app.modules.rta.settings.dpg", backend),
            patch("spectrum_app.modules.thd.view.dpg", backend),
            patch("spectrum_app.gui.controls.level_meter.dpg", backend),
            patch("spectrum_app.modules.thd.settings.dpg", backend),
        ):
            app.run()

        call_names = [call[0] for call in backend.calls]
        self.assertEqual(call_names[:2], ["create_context", "configure_app"])
        self.assertLess(call_names.index("window"), call_names.index("create_viewport"))
        self.assertLess(
            call_names.index("run_callbacks"), call_names.index("frame_callback")
        )
        self.assertLess(
            call_names.index("frame_callback"),
            call_names.index("render_dearpygui_frame"),
        )
        self.assertEqual(call_names[-1], "destroy_context")
        self.assertFalse(app.running)
        spectrum_settings_item = next(
            call
            for call in backend.calls
            if call[0] == "add_menu_item"
            and call[1].get("tag") == SpectrumSettingsWindow.MENU_ITEM
        )
        self.assertEqual(spectrum_settings_item[1]["label"], "Spectrum")
        self.assertEqual(
            spectrum_settings_item[1]["parent"],
            app.main_window.settings_menu,
        )
        thd_settings_item = next(
            call
            for call in backend.calls
            if call[0] == "add_menu_item"
            and call[1].get("tag") == THDSettingsWindow.MENU_ITEM
        )
        self.assertEqual(thd_settings_item[1]["label"], "THD")
        phase_settings_item = next(
            call
            for call in backend.calls
            if call[0] == "add_menu_item"
            and call[1].get("tag") == PhaseSettingsWindow.MENU_ITEM
        )
        self.assertEqual(phase_settings_item[1]["label"], "Phase")
        export_menu = next(
            call
            for call in backend.calls
            if call[0] == "add_menu"
            and call[1].get("tag") == app.main_window.export_menu
        )
        self.assertEqual(export_menu[1]["label"], "Export")
        import_menu = next(
            call
            for call in backend.calls
            if call[0] == "add_menu"
            and call[1].get("tag") == app.main_window.import_menu
        )
        self.assertEqual(import_menu[1]["label"], "Import")
        plot_export = next(
            call
            for call in backend.calls
            if call[0] == "add_menu_item"
            and call[1].get("parent") == app.main_window.export_menu
        )
        self.assertEqual(plot_export[1]["label"], "Plot")
        measurement_import = next(
            call
            for call in backend.calls
            if call[0] == "add_menu_item"
            and call[1].get("parent") == app.main_window.import_menu
        )
        self.assertEqual(measurement_import[1]["label"], "Measurement")
        measurement_export = next(
            call
            for call in backend.calls
            if call[0] == "add_menu_item"
            and call[1].get("parent") == app.main_window.export_menu
            and call is not plot_export
        )
        self.assertEqual(measurement_export[1]["label"], "Measurement")

    def test_main_window_exposes_module_hosts_and_status(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        app.dpg = DearPyGuiRuntime(backend)
        app.main_window = MainWindow(app)

        with (
            patch("spectrum_app.gui.main_window.dpg", backend),
            patch("spectrum_app.gui.error.dpg", backend),
            patch("spectrum_app.gui.app_state.dpg", backend),
            patch("spectrum_app.gui.measurement.dpg", backend),
            patch("spectrum_app.gui.plot.dpg", backend),
            patch("spectrum_app.gui.settings.dpg", backend),
            patch("spectrum_app.gui.project.dpg", backend),
            patch("spectrum_app.gui.measurement_io.dpg", backend),
        ):
            app.dpg.create_context()
            app.main_window.build()
            app.main_window.plot.update()
            app.main_window.set_status_text("Ready")
            settings_item = next(
                call
                for call in backend.calls
                if call[0] == "add_menu_item"
                and call[1].get("parent") == app.main_window.settings_menu
            )
            settings_item[1]["callback"]()

        self.assertEqual(settings_item[1]["label"], "Application")

        menu_calls = [call[1] for call in backend.calls if call[0] == "add_menu"]
        self.assertEqual(
            [(call["label"], call["tag"]) for call in menu_calls],
            [
                ("File", app.main_window.file_menu),
                ("Tools", app.main_window.tools_menu),
                ("Settings", app.main_window.settings_menu),
                ("Import", app.main_window.import_menu),
                ("Export", app.main_window.export_menu),
            ],
        )

        host_tags = {
            call[1]["tag"]
            for call in backend.calls
            if call[0] == "child_window" and "tag" in call[1]
        }
        plot_tags = {
            call[1]["tag"]
            for call in backend.calls
            if call[0] == "plot" and "tag" in call[1]
        }
        self.assertTrue(
            {
                app.main_window.bottom_host,
                app.main_window.control_panel_host,
                app.main_window.module_gui_host,
                app.main_window.appstate_host,
            }.issubset(host_tags)
        )
        self.assertIn(app.main_window.plot_host, plot_tags)
        watermark_layer = next(
            call
            for call in backend.calls
            if call[0] == "add_viewport_drawlist"
            and call[1].get("tag") == app.main_window.plot.watermark_layer
        )
        self.assertTrue(watermark_layer[1]["front"])
        watermark = next(
            call
            for call in backend.calls
            if call[0] == "draw_text"
            and call[3].get("tag") == app.main_window.plot.watermark
        )
        self.assertEqual(watermark[2], "BM Spectrum")
        self.assertEqual(
            watermark[3]["parent"],
            app.main_window.plot.watermark_layer,
        )
        self.assertLess(watermark[3]["color"][3], 255)
        self.assertIn(
            (
                "configure_item",
                app.main_window.plot.watermark,
                {"pos": (696.0, 60.0), "show": True},
            ),
            backend.calls,
        )

        backend.shown_windows.append(app.main_window.settings_window.tag)
        with patch("spectrum_app.gui.plot.dpg", backend):
            app.main_window.plot.update()
        self.assertIn(
            (
                "configure_item",
                app.main_window.plot.watermark,
                {"show": False},
            ),
            backend.calls,
        )
        self.assertIn(
            ("set_value", app.main_window.status, "Ready"),
            backend.calls,
        )
        self.assertIn(
            (
                "set_item_pos",
                app.main_window.settings_window.tag,
                [310.0, 195.0],
            ),
            backend.calls,
        )

    def test_measurement_selector_adds_default_measurements(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        first = app.create_measurement()
        window = MainWindow(app)

        with (
            patch("spectrum_app.gui.main_window.dpg", backend),
            patch("spectrum_app.gui.error.dpg", backend),
            patch("spectrum_app.gui.app_state.dpg", backend),
            patch("spectrum_app.gui.measurement.dpg", backend),
            patch("spectrum_app.gui.plot.dpg", backend),
            patch("spectrum_app.gui.settings.dpg", backend),
            patch("spectrum_app.gui.project.dpg", backend),
            patch("spectrum_app.gui.measurement_io.dpg", backend),
        ):
            window.build()
            add_button_call = next(
                call
                for call in backend.calls
                if call[0] == "add_button"
                and call[1].get("tag") == window.app_state_panel.add_measurement_button
            )
            add_button_call[1]["callback"]()

            self.assertEqual(len(app.app_state.measurements), 2)
            self.assertEqual(app.app_state.measurements[0], first)
            self.assertTrue(
                all(
                    measurement.module_id == app.DEFAULT_MODULE_ID
                    for measurement in app.app_state.measurements
                )
            )
            self.assertEqual(
                app.app_state.active_measurement_id,
                app.app_state.measurements[-1].id,
            )

            delete_button_call = next(
                call
                for call in backend.calls
                if call[0] == "add_image_button"
                and call[2].get("tag") == window.app_state_panel._delete_tag(first.id)
            )
            self.assertEqual(
                delete_button_call[1],
                window.app_state_panel.delete_icon,
            )
            delete_button_call[2]["callback"](None, None, first.id)

        self.assertNotIn(first, app.app_state.measurements)
        self.assertIn(
            ("delete_item", window.app_state_panel._row_tag(first.id)),
            backend.calls,
        )

    def test_measurement_visibility_checkbox_tracks_new_module_graph(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement()
        window = MainWindow(app)

        with (
            patch("spectrum_app.gui.main_window.dpg", backend),
            patch("spectrum_app.gui.error.dpg", backend),
            patch("spectrum_app.gui.app_state.dpg", backend),
            patch("spectrum_app.gui.measurement.dpg", backend),
            patch("spectrum_app.gui.plot.dpg", backend),
            patch("spectrum_app.gui.settings.dpg", backend),
            patch("spectrum_app.gui.project.dpg", backend),
            patch("spectrum_app.gui.measurement_io.dpg", backend),
        ):
            window.build()
            graph = GraphData(
                "Spectrum",
                np.array([20.0]),
                np.array([-20.0]),
                AxisSpec.FREQ,
                AxisSpec.LEVEL,
            )
            measurement.graphs.append(graph)
            app.app_state.visible_graph_ids.append(graph.id)
            app.app_state.graph_data_changed = True
            backend.calls.clear()
            window.update()

        self.assertIn(
            (
                "set_value",
                window.app_state_panel._visible_tag(measurement.id),
                True,
            ),
            backend.calls,
        )

    def test_run_button_updates_measuring_state_and_theme(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        app.create_measurement()
        window = MainWindow(app)

        with (
            patch("spectrum_app.gui.main_window.dpg", backend),
            patch("spectrum_app.gui.error.dpg", backend),
            patch("spectrum_app.gui.app_state.dpg", backend),
            patch("spectrum_app.gui.measurement.dpg", backend),
            patch("spectrum_app.gui.plot.dpg", backend),
            patch("spectrum_app.gui.settings.dpg", backend),
            patch("spectrum_app.gui.project.dpg", backend),
            patch("spectrum_app.gui.measurement_io.dpg", backend),
            patch("spectrum_app.modules.spectrum.view.dpg", backend),
            patch("spectrum_app.modules.spectrum.settings.dpg", backend),
        ):
            window.build()
            spectrum_module = app.module_manager.module("spectrum")
            spectrum_module.initialize(app)
            window.measurement_panel.modules_initialized()
            run_button_call = next(
                call
                for call in backend.calls
                if call[0] == "add_button"
                and call[1].get("tag") == window.measurement_panel.run_button
            )
            with patch.object(
                SpectrumModule,
                "start_measurement",
                autospec=True,
                side_effect=lambda module: setattr(
                    module.app.app_state,
                    "measuring",
                    True,
                ),
            ) as start_measurement:
                run_button_call[1]["callback"]()
                start_measurement.assert_called_once_with(spectrum_module)

            self.assertTrue(app.app_state.measuring)
            self.assertIn(
                ("set_item_label", window.measurement_panel.run_button, "STOP"),
                backend.calls,
            )
            self.assertIn(
                (
                    "bind_item_theme",
                    window.measurement_panel.run_button,
                    window.measurement_panel.red_button_theme,
                ),
                backend.calls,
            )
            window.measurement_panel.shutdown()
            spectrum_module.shutdown()

    def test_plot_redraws_three_supported_graph_types(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement()
        frequency = AxisSpec.FREQ
        frequencies = np.array([20.0, 200.0, 2000.0])
        measurement.graphs = [
            GraphData(
                "Spectrum",
                frequencies,
                np.array([-40.0, -20.0, -10.0]),
                frequency,
                AxisSpec.LEVEL,
            ),
            GraphData(
                "Impedance",
                frequencies,
                np.array([8.0, 10.0, 12.0]),
                frequency,
                AxisSpec.IMPEDANCE,
            ),
            GraphData(
                "Phase",
                frequencies,
                np.array([-10.0, 5.0, 20.0]),
                frequency,
                AxisSpec.PHASE,
            ),
        ]
        app.app_state.visible_graph_ids = [graph.id for graph in measurement.graphs]
        app.settings.impedance_scale = "log"
        app.settings.phase_unit = "deg/dec"
        app.settings.frequency_range = (30.0, 18_000.0)
        window = MainWindow(app)

        with (
            patch("spectrum_app.gui.main_window.dpg", backend),
            patch("spectrum_app.gui.error.dpg", backend),
            patch("spectrum_app.gui.app_state.dpg", backend),
            patch("spectrum_app.gui.measurement.dpg", backend),
            patch("spectrum_app.gui.plot.dpg", backend),
            patch("spectrum_app.gui.settings.dpg", backend),
            patch("spectrum_app.gui.project.dpg", backend),
            patch("spectrum_app.gui.measurement_io.dpg", backend),
        ):
            window.build()
            window.plot.update()

        series_calls = [call for call in backend.calls if call[0] == "add_line_series"]
        self.assertEqual(len(series_calls), 3)
        self.assertEqual(
            [call[3]["parent"] for call in series_calls],
            window.plot.y_axes,
        )
        self.assertTrue(np.all(np.isfinite(series_calls[2][2])))
        self.assertNotEqual(series_calls[2][2], measurement.graphs[2].y.tolist())
        axis_configurations = {
            call[1]: call[2]
            for call in backend.calls
            if call[0] == "configure_item" and call[1] in window.plot.y_axes
        }
        self.assertEqual(
            axis_configurations[window.plot.y_axes[0]]["label"],
            "Spectrum [dB]",
        )
        self.assertEqual(
            axis_configurations[window.plot.y_axes[1]]["scale"],
            backend.mvPlotScale_Log10,
        )
        self.assertEqual(
            axis_configurations[window.plot.y_axes[2]]["label"],
            "Phase [deg/dec]",
        )
        self.assertFalse(axis_configurations[window.plot.y_axes[0]]["opposite"])
        self.assertTrue(axis_configurations[window.plot.y_axes[1]]["opposite"])
        self.assertTrue(axis_configurations[window.plot.y_axes[2]]["opposite"])
        self.assertIn(
            ("set_axis_limits", window.plot.x_axis, 30.0, 18_000.0),
            backend.calls,
        )
        self.assertFalse(app.app_state.graph_data_changed)

    def test_plot_updates_existing_series_without_refit(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement()
        graph = GraphData(
            "Spectrum",
            np.array([20.0, 200.0]),
            np.array([-30.0, -20.0]),
            AxisSpec.FREQ,
            AxisSpec.LEVEL,
        )
        measurement.graphs = [graph]
        app.app_state.visible_graph_ids = [graph.id]
        window = MainWindow(app)

        with patch("spectrum_app.gui.plot.dpg", backend):
            window.plot.build(width=800, height=600)
            window.plot.update()
            backend.calls.clear()
            graph.y = np.array([-25.0, -15.0])
            app.app_state.graph_data_changed = True
            window.plot.update()

        series_tag = window.plot._series_tag(graph.id)
        self.assertIn(
            ("set_value", series_tag, [[20.0, 200.0], [-25.0, -15.0]]),
            backend.calls,
        )
        self.assertFalse(any(call[0] == "add_line_series" for call in backend.calls))
        self.assertFalse(any(call[0] == "delete_item" for call in backend.calls))
        self.assertFalse(any(call[0] == "fit_axis_data" for call in backend.calls))

    def test_phase_unit_change_refits_visible_phase_axis(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement()
        graph = GraphData(
            "Phase",
            np.array([20.0, 200.0, 2_000.0]),
            np.array([-20.0, 15.0, 65.0]),
            AxisSpec.FREQ,
            AxisSpec.PHASE,
        )
        measurement.graphs = [graph]
        app.app_state.visible_graph_ids = [graph.id]
        window = app.main_window

        with patch("spectrum_app.gui.plot.dpg", backend):
            window.plot.build(width=800, height=600)
            window.plot.update()
            backend.calls.clear()

            window.settings_window._set_phase_unit(  # pyright: ignore[reportPrivateUsage]
                window.settings_window.phase_unit,
                "deg/dec",
            )
            window.plot.update()

        self.assertIn(("fit_axis_data", window.plot.y_axes[0]), backend.calls)

    def test_plot_accepts_logarithmic_bar_graphs(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement()
        graph = GraphData(
            "RTA",
            np.array([20.0, 200.0, 2_000.0]),
            np.array([-30.0, -20.0, -10.0]),
            AxisSpec.FREQ,
            AxisSpec.LEVEL,
            plot_type=PlotType.BARS,
        )
        measurement.graphs = [graph]
        app.app_state.visible_graph_ids = [graph.id]
        window = MainWindow(app)

        with patch("spectrum_app.gui.plot.dpg", backend):
            window.plot.build(width=800, height=600)
            window.plot.update()

        call = next(item for item in backend.calls if item[0] == "add_custom_series")
        self.assertEqual(call[3], 3)
        edges = np.asarray(call[1]).reshape(-1, 2)
        np.testing.assert_allclose(edges[:-1, 1], edges[1:, 0], rtol=1e-12)
        self.assertTrue(np.all(edges > 0.0))
        np.testing.assert_array_equal(call[4]["y1"], -48.0)

        backend.calls.clear()
        with patch("spectrum_app.gui.plot.dpg", backend):
            call[4]["callback"](
                call[4]["tag"],
                [{}, [10.0, 20.0], [30.0, 30.0], [100.0, 100.0]],
            )
        rectangle = next(item for item in backend.calls if item[0] == "draw_rectangle")
        self.assertEqual(rectangle[1], (11.5, 30.0))
        self.assertEqual(rectangle[2], (18.5, 100.0))

    def test_settings_window_updates_application_settings(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        window = MainWindow(app)

        with patch("spectrum_app.gui.settings.dpg", backend):
            window.settings_window.build()
            phase_control = next(
                call
                for call in backend.calls
                if call[0] == "add_radio_button"
                and call[2].get("tag") == window.settings_window.phase_unit
            )
            frequency_range = next(
                call
                for call in backend.calls
                if call[0] == "add_input_intx"
                and call[1].get("tag") == window.settings_window.frequency_range
            )
            app.app_state.graph_data_changed = False
            phase_control[2]["callback"](
                window.settings_window.phase_unit,
                "deg/dec",
                None,
            )
            frequency_range[1]["callback"](
                window.settings_window.frequency_range,
                [10, 20_000, 0, 0],
                None,
            )

        self.assertEqual(app.settings.phase_unit, "deg/dec")
        self.assertEqual(app.settings.frequency_range, (10.0, 20_000.0))
        self.assertTrue(app.app_state.graph_data_changed)

    def test_audio_settings_build_and_update_routing_matrices(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        input_device = AudioDevice(
            id="input",
            index=0,
            name="Input",
            host_api="Test",
            sample_rate=48_000,
            input_channels=4,
            output_channels=0,
        )
        output_device = AudioDevice(
            id="output",
            index=1,
            name="Output",
            host_api="Test",
            sample_rate=48_000,
            input_channels=0,
            output_channels=4,
        )
        app._audio_service._input_devices = (  # pyright: ignore[reportPrivateUsage]
            input_device,
        )
        app._audio_service._output_devices = (  # pyright: ignore[reportPrivateUsage]
            output_device,
        )
        app._audio_service._default_input_id = (  # pyright: ignore[reportPrivateUsage]
            input_device.id
        )
        app._audio_service._default_output_id = (  # pyright: ignore[reportPrivateUsage]
            output_device.id
        )
        window = MainWindow(app)

        with patch("spectrum_app.gui.settings.dpg", backend):
            window.settings_window.build()

            input_combo = next(
                call
                for call in backend.calls
                if call[0] == "add_combo"
                and call[2].get("tag") == window.settings_window.input_device
            )
            output_combo = next(
                call
                for call in backend.calls
                if call[0] == "add_combo"
                and call[2].get("tag") == window.settings_window.output_device
            )
            input_combo[2]["callback"](
                input_combo[2]["tag"],
                input_device.label,
                None,
            )
            output_combo[2]["callback"](
                output_combo[2]["tag"],
                output_device.label,
                None,
            )

            input_route = next(
                call
                for call in backend.calls
                if call[0] == "add_checkbox"
                and call[1].get("tag") == window.settings_window._input_route_tag(0, 2)
            )
            input_route[1]["callback"](
                input_route[1]["tag"],
                True,
                input_route[1]["user_data"],
            )

            output_route = next(
                call
                for call in backend.calls
                if call[0] == "add_checkbox"
                and call[1].get("tag") == window.settings_window._output_route_tag(1)
            )
            output_route[1]["callback"](
                output_route[1]["tag"],
                False,
                output_route[1]["user_data"],
            )

        self.assertEqual(app.settings.input_routing, (2, 1))
        self.assertEqual(app.settings.input_block_size, 4_800)
        self.assertEqual(app.settings.output_block_size, 4_800)
        self.assertEqual(
            app.settings.output_routing,
            (True, False, True, True),
        )
        self.assertIn(
            (
                "set_value",
                window.settings_window._input_route_tag(0, 0),
                False,
            ),
            backend.calls,
        )

    def test_module_change_requires_confirmation_when_measurement_has_data(
        self,
    ) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement("spectrum")
        measurement.graphs = [
            GraphData(
                "Spectrum",
                np.array([20.0]),
                np.array([-20.0]),
                AxisSpec.FREQ,
                AxisSpec.LEVEL,
            )
        ]
        window = MainWindow(app)

        with (
            patch("spectrum_app.gui.main_window.dpg", backend),
            patch("spectrum_app.gui.error.dpg", backend),
            patch("spectrum_app.gui.app_state.dpg", backend),
            patch("spectrum_app.gui.measurement.dpg", backend),
            patch("spectrum_app.gui.plot.dpg", backend),
            patch("spectrum_app.gui.settings.dpg", backend),
            patch("spectrum_app.gui.project.dpg", backend),
            patch("spectrum_app.gui.measurement_io.dpg", backend),
        ):
            window.build()
            panel = window.measurement_panel
            panel._set_module(panel.module_combo, "Phase")
            self.assertEqual(measurement.module_id, "spectrum")
            self.assertTrue(measurement.graphs)
            panel._confirm_module_change()

        self.assertEqual(measurement.module_id, "phase")
        self.assertEqual(measurement.graphs, [])
        self.assertTrue(
            any(
                call[0] == "configure_item"
                and call[1] == window.measurement_panel.module_change_dialog
                and call[2].get("show") is True
                for call in backend.calls
            )
        )

    def test_plot_keeps_previous_render_and_warns_about_fourth_axis(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement()
        x = np.array([20.0, 1000.0])
        frequency = AxisSpec.FREQ
        measurement.graphs = [
            GraphData(
                "Spectrum", x, np.array([-40.0, -10.0]), frequency, AxisSpec.LEVEL
            ),
            GraphData(
                "Impedance",
                x,
                np.array([8.0, 12.0]),
                frequency,
                AxisSpec.IMPEDANCE,
            ),
            GraphData("Phase", x, np.array([-10.0, 20.0]), frequency, AxisSpec.PHASE),
        ]
        app.app_state.visible_graph_ids = [graph.id for graph in measurement.graphs]
        window = MainWindow(app)

        with (
            patch("spectrum_app.gui.main_window.dpg", backend),
            patch("spectrum_app.gui.error.dpg", backend),
            patch("spectrum_app.gui.app_state.dpg", backend),
            patch("spectrum_app.gui.measurement.dpg", backend),
            patch("spectrum_app.gui.plot.dpg", backend),
            patch("spectrum_app.gui.settings.dpg", backend),
            patch("spectrum_app.gui.project.dpg", backend),
            patch("spectrum_app.gui.measurement_io.dpg", backend),
        ):
            window.build()
            window.plot.update()
            previous_series = window.plot.series_tags.copy()
            thd = GraphData(
                "THD",
                x,
                np.array([0.1, 1.0]),
                frequency,
                AxisSpec.THD,
            )
            measurement.graphs.append(thd)
            app.app_state.visible_graph_ids.append(thd.id)
            app.app_state.graph_data_changed = True
            backend.calls.clear()

            window.plot.update()

        self.assertEqual(window.plot.series_tags, previous_series)
        self.assertFalse(any(call[0] == "delete_item" for call in backend.calls))
        self.assertTrue(
            any(
                call[0] == "window"
                and call[1].get("tag") == window.plot.axis_warning
                and call[1].get("modal") is True
                for call in backend.calls
            )
        )
        self.assertFalse(app.app_state.graph_data_changed)

    def test_plot_wraps_phase_and_inserts_line_breaks(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement()
        phase = GraphData(
            "Phase",
            np.array([10.0, 100.0, 1000.0]),
            np.array([170.0, 190.0, 200.0]),
            AxisSpec.FREQ,
            AxisSpec.PHASE,
        )
        measurement.graphs = [phase]
        app.app_state.visible_graph_ids = [phase.id]
        window = MainWindow(app)

        with patch("spectrum_app.gui.plot.dpg", backend):
            window.plot.build(width=800, height=600)
            window.plot.update()

        series_call = next(
            call for call in backend.calls if call[0] == "add_line_series"
        )
        np.testing.assert_allclose(
            series_call[1],
            [10.0, np.nan, 100.0, 1000.0],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            series_call[2],
            [170.0, np.nan, -170.0, -160.0],
            equal_nan=True,
        )
        np.testing.assert_array_equal(phase.y, [170.0, 190.0, 200.0])

    def test_plot_export_waits_for_dialog_to_disappear(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        app.app_state.graph_data_changed = False
        window = MainWindow(app)
        app.main_window.set_status_text = lambda text: None
        exporter = MagicMock()
        window.plot.exporter = exporter

        with patch("spectrum_app.gui.plot.dpg", backend):
            window.plot.export_png(
                window.plot.export_dialog,
                {"file_path_name": "D:/plot.png"},
            )
            window.plot.update()
            window.plot.update()
            exporter.export.assert_not_called()

            window.plot.update()

        exporter.export.assert_called_once()
        self.assertEqual(
            exporter.export.call_args.args[:2], (Path("D:/plot.png"), window.plot.tag)
        )

    def test_destroy_context_is_safe_before_initialization(self) -> None:
        backend = FakeDpgBackend()
        runtime = DearPyGuiRuntime(backend)

        runtime.destroy_context()

        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
