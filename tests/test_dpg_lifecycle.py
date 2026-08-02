import unittest
from unittest.mock import patch

import numpy as np

from spectrum_app import SpectrumApplication
from spectrum_app.core.dpg import DearPyGuiRuntime
from spectrum_app.core.model import AxisSpec, GraphData
from spectrum_app.gui.main_window import MainWindow


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

    def add_menu(self, **kwargs) -> str:
        self.calls.append(("add_menu", kwargs))
        return kwargs["tag"]

    def add_menu_item(self, **kwargs) -> str:
        self.calls.append(("add_menu_item", kwargs))
        return kwargs.get("tag", "menu_item")

    def add_plot_legend(self, **kwargs) -> str:
        self.calls.append(("add_plot_legend", kwargs))
        return "plot_legend"

    def add_plot_axis(self, axis: int, **kwargs) -> str:
        self.calls.append(("add_plot_axis", axis, kwargs))
        return kwargs.get("tag", "plot_axis")

    def add_line_series(
        self, x: list[float], y: list[float], **kwargs
    ) -> str:
        self.calls.append(("add_line_series", x, y, kwargs))
        return kwargs.get("tag", "line_series")

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

    def add_input_intx(self, **kwargs) -> str:
        self.calls.append(("add_input_intx", kwargs))
        return kwargs.get("tag", "input_intx")

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

    def delete_item(self, item: str) -> None:
        self.calls.append(("delete_item", item))

    def add_child_window(self, **kwargs) -> str:
        self.calls.append(("add_child_window", kwargs))
        return kwargs.get("tag", "child_window")

    def add_text(self, value: str, **kwargs) -> str:
        self.calls.append(("add_text", value, kwargs))
        return kwargs.get("tag", "text")

    def set_value(self, tag: str, value: str) -> None:
        self.calls.append(("set_value", tag, value))

    def get_item_pos(self, tag: str) -> list[int]:
        self.calls.append(("get_item_pos", tag))
        return [10, 20]

    def get_item_rect_size(self, tag: str) -> list[int]:
        self.calls.append(("get_item_rect_size", tag))
        return [1200, 800]

    def set_item_pos(self, tag: str, position: list[float]) -> None:
        self.calls.append(("set_item_pos", tag, position))

    def set_item_label(self, tag: str, label: str) -> None:
        self.calls.append(("set_item_label", tag, label))

    def bind_item_theme(self, tag: str, theme: str) -> None:
        self.calls.append(("bind_item_theme", tag, theme))

    def configure_item(self, tag: str, **kwargs) -> None:
        self.calls.append(("configure_item", tag, kwargs))

    def set_axis_limits(self, tag: str, low: float, high: float) -> None:
        self.calls.append(("set_axis_limits", tag, low, high))

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
    def test_application_runs_dpg_lifecycle(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        app.dpg = DearPyGuiRuntime(backend)
        app.main_window = MainWindow(app)
        app.frame_callbacks.append(
            lambda: backend.calls.append(("frame_callback",))
        )

        with (
            patch("spectrum_app.gui.main_window.dpg", backend),
            patch("spectrum_app.gui.app_state.dpg", backend),
            patch("spectrum_app.gui.measurement.dpg", backend),
            patch("spectrum_app.gui.plot.dpg", backend),
            patch("spectrum_app.gui.settings.dpg", backend),
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

    def test_main_window_exposes_module_hosts_and_status(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        app.dpg = DearPyGuiRuntime(backend)
        app.main_window = MainWindow(app)

        with (
            patch("spectrum_app.gui.main_window.dpg", backend),
            patch("spectrum_app.gui.app_state.dpg", backend),
            patch("spectrum_app.gui.measurement.dpg", backend),
            patch("spectrum_app.gui.plot.dpg", backend),
            patch("spectrum_app.gui.settings.dpg", backend),
        ):
            app.dpg.create_context()
            app.main_window.build()
            app.main_window.set_status_text("Ready")
            settings_item = next(
                call
                for call in backend.calls
                if call[0] == "add_menu_item"
                and call[1].get("parent") == app.main_window.settings_menu
            )
            settings_item[1]["callback"]()

        menu_calls = [call[1] for call in backend.calls if call[0] == "add_menu"]
        self.assertEqual(
            [(call["label"], call["tag"]) for call in menu_calls],
            [
                ("File", app.main_window.file_menu),
                ("Tools", app.main_window.tools_menu),
                ("Settings", app.main_window.settings_menu),
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
        self.assertIn(
            ("set_value", app.main_window.status, "Ready"),
            backend.calls,
        )
        self.assertIn(
            (
                "set_item_pos",
                app.main_window.settings_window.tag,
                [276.5, 220.0],
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
            patch("spectrum_app.gui.app_state.dpg", backend),
            patch("spectrum_app.gui.measurement.dpg", backend),
            patch("spectrum_app.gui.plot.dpg", backend),
            patch("spectrum_app.gui.settings.dpg", backend),
        ):
            window.build()
            add_button_call = next(
                call
                for call in backend.calls
                if call[0] == "add_button"
                and call[1].get("tag")
                == window.app_state_panel.add_measurement_button
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
                and call[2].get("tag")
                == window.app_state_panel._delete_tag(first.id)
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

    def test_run_button_updates_measuring_state_and_theme(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        app.create_measurement()
        window = MainWindow(app)

        with (
            patch("spectrum_app.gui.main_window.dpg", backend),
            patch("spectrum_app.gui.app_state.dpg", backend),
            patch("spectrum_app.gui.measurement.dpg", backend),
            patch("spectrum_app.gui.plot.dpg", backend),
            patch("spectrum_app.gui.settings.dpg", backend),
        ):
            window.build()
            run_button_call = next(
                call
                for call in backend.calls
                if call[0] == "add_button"
                and call[1].get("tag") == window.measurement_panel.run_button
            )
            run_button_call[1]["callback"]()

        self.assertTrue(app.app_state.measuring)
        self.assertIn(
            ("set_item_label", window.measurement_panel.run_button, "ON"),
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
        app.app_state.visible_graph_ids = [
            graph.id for graph in measurement.graphs
        ]
        app.settings.impedance_scale = "log"
        app.settings.phase_unit = "deg/dec"
        app.settings.frequency_range = (30.0, 18_000.0)
        window = MainWindow(app)

        with (
            patch("spectrum_app.gui.main_window.dpg", backend),
            patch("spectrum_app.gui.app_state.dpg", backend),
            patch("spectrum_app.gui.measurement.dpg", backend),
            patch("spectrum_app.gui.plot.dpg", backend),
            patch("spectrum_app.gui.settings.dpg", backend),
        ):
            window.build()
            window.plot.update()

        series_calls = [
            call for call in backend.calls if call[0] == "add_line_series"
        ]
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
        self.assertIn(
            ("set_axis_limits", window.plot.x_axis, 30.0, 18_000.0),
            backend.calls,
        )
        self.assertFalse(app.app_state.graph_data_changed)

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
            GraphData(
                "Phase", x, np.array([-10.0, 20.0]), frequency, AxisSpec.PHASE
            ),
        ]
        app.app_state.visible_graph_ids = [
            graph.id for graph in measurement.graphs
        ]
        window = MainWindow(app)

        with (
            patch("spectrum_app.gui.main_window.dpg", backend),
            patch("spectrum_app.gui.app_state.dpg", backend),
            patch("spectrum_app.gui.measurement.dpg", backend),
            patch("spectrum_app.gui.plot.dpg", backend),
            patch("spectrum_app.gui.settings.dpg", backend),
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

    def test_destroy_context_is_safe_before_initialization(self) -> None:
        backend = FakeDpgBackend()
        runtime = DearPyGuiRuntime(backend)

        runtime.destroy_context()

        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
