import unittest

from spectrum_app import SpectrumApplication
from spectrum_app.core.model import AppState
from spectrum_app.core.settings import AppSettings


class RecordingApplication(SpectrumApplication):
    def __init__(self, fail_in_main_loop: bool = False) -> None:
        super().__init__()
        self.events: list[str] = []
        self.fail_in_main_loop = fail_in_main_loop

    def _initialize(self) -> None:
        self.events.append("initialize")

    def _run_main_loop(self) -> None:
        self.events.append("main_loop")
        if self.fail_in_main_loop:
            raise RuntimeError("main loop failed")

    def _shutdown(self) -> None:
        self.events.append("shutdown")


class SpectrumApplicationTests(unittest.TestCase):
    def test_application_owns_app_state(self) -> None:
        app = SpectrumApplication()

        self.assertIsInstance(app.app_state, AppState)
        self.assertIsInstance(app.settings, AppSettings)

    def test_settings_changes_request_plot_redraw(self) -> None:
        app = SpectrumApplication()
        app.app_state.graph_data_changed = False

        app.settings.phase_unit = "deg/dec"

        self.assertTrue(app.app_state.graph_data_changed)

    def test_create_measurement_uses_default_module_and_makes_it_active(self) -> None:
        app = SpectrumApplication()

        measurement = app.create_measurement()

        self.assertEqual(measurement.module_id, app.DEFAULT_MODULE_ID)
        self.assertEqual(measurement.name, "Measurement 1")
        self.assertEqual(app.app_state.measurements, [measurement])
        self.assertEqual(app.app_state.active_measurement_id, measurement.id)

    def test_delete_active_measurement_selects_its_neighbour(self) -> None:
        app = SpectrumApplication()
        first = app.create_measurement()
        second = app.create_measurement()

        app.delete_measurement(second.id)

        self.assertEqual(app.app_state.measurements, [first])
        self.assertEqual(app.app_state.active_measurement_id, first.id)

        app.delete_measurement(first.id)

        self.assertEqual(app.app_state.measurements, [])
        self.assertIsNone(app.app_state.active_measurement_id)

    def test_run_executes_complete_lifecycle(self) -> None:
        app = RecordingApplication()

        app.run()

        self.assertEqual(app.events, ["initialize", "main_loop", "shutdown"])
        self.assertFalse(app.running)

    def test_run_shuts_down_when_main_loop_fails(self) -> None:
        app = RecordingApplication(fail_in_main_loop=True)

        with self.assertRaisesRegex(RuntimeError, "main loop failed"):
            app.run()

        self.assertEqual(app.events, ["initialize", "main_loop", "shutdown"])
        self.assertFalse(app.running)

    def test_run_rejects_nested_execution(self) -> None:
        app = SpectrumApplication()
        app._running = True

        with self.assertRaisesRegex(RuntimeError, "already running"):
            app.run()

    def test_frame_callbacks_can_modify_the_callback_list(self) -> None:
        app = SpectrumApplication()
        calls: list[str] = []

        def first_callback() -> None:
            calls.append("first")
            app.frame_callbacks.pop()

        app.frame_callbacks.append(first_callback)
        app.frame_callbacks.append(lambda: calls.append("second"))

        app._process_frame_callbacks()

        self.assertEqual(calls, ["first", "second"])
        self.assertEqual(app.frame_callbacks, [first_callback])


if __name__ == "__main__":
    unittest.main()
