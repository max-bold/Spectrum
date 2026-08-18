from threading import current_thread
import time
from typing import cast
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.typing import NDArray

from audioanalysis import ASignal
from spectrum_app import SpectrumApplication
from spectrum_app.core.audio import AudioInput, AudioOutput
from spectrum_app.core.model import AxisSpec, PlotType
from spectrum_app.modules.rta import RTAModule
from spectrum_app.modules.rta.settings import RTASettings, RTASettingsWindow
from spectrum_app.modules.rta.view import RTAView
from tests.test_dpg_lifecycle import FakeDpgBackend


class ContinuousRTAInput:
    sample_rate = 8_000
    blocksize = 64

    def __init__(self) -> None:
        self.position = 0
        self.read_threads: set[str] = set()

    def open(self) -> bool:
        return True

    def read(self, samples: int) -> NDArray[np.float32]:
        self.read_threads.add(current_thread().name)
        indexes = np.arange(self.position, self.position + samples)
        self.position += samples
        block = np.column_stack(
            (
                0.1 * np.sin(2.0 * np.pi * 500.0 * indexes / self.sample_rate),
                0.05 * np.sin(2.0 * np.pi * 1_000.0 * indexes / self.sample_rate),
            )
        )
        return np.asarray(block, dtype=np.float32)

    def close(self) -> bool:
        return True


class RecordingRTAOutput:
    sample_rate = 8_000
    blocksize = 64

    def __init__(self) -> None:
        self.write_threads: set[str] = set()
        self.written: list[np.ndarray] = []

    def open(self) -> bool:
        return True

    def write(self, data: NDArray) -> None:
        self.write_threads.add(current_thread().name)
        self.written.append(np.asarray(data, dtype=np.float32))

    def close(self) -> bool:
        return True


class RTAModuleTests(unittest.TestCase):
    def test_intermediate_band_input_is_normalized_without_gui_exception(self) -> None:
        self.assertEqual(RTAModule._normalize_setting("band", (0, 0)), (1, 2))
        self.assertEqual(
            RTAModule._normalize_setting("band", (1_000, 100)),
            (1_000, 1_001),
        )

    def test_view_builds_rta_controls_and_compact_level_meter(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement("rta")
        module = cast(RTAModule, app.module_manager.module("rta"))

        with (
            patch("spectrum_app.modules.rta.view.dpg", backend),
            patch("spectrum_app.modules.rta.settings.dpg", backend),
            patch("spectrum_app.gui.controls.level_meter.dpg", backend),
        ):
            module.initialize(app)
            module.activate(measurement)
            try:
                self.assertTrue(measurement.module_state["noise"])
                self.assertEqual(measurement.module_state["points"], 31)
                self.assertEqual(measurement.module_state["smoothing_octaves"], 0.1)
                self.assertTrue(
                    any(
                        call[0] == "drawlist" and call[1].get("tag") == RTAView.METER
                        for call in backend.calls
                    )
                )
                meter_size = next(
                    call
                    for call in backend.calls
                    if call[0] == "configure_item" and call[1] == RTAView.METER
                )
                self.assertEqual(meter_size[2]["height"], 180)
                slider = next(
                    call
                    for call in backend.calls
                    if call[0] == "add_slider_float"
                    and call[1].get("tag") == RTAView.LEVEL
                )
                self.assertEqual(slider[1]["min_value"], -10.0)
                self.assertEqual(slider[1]["max_value"], 10.0)
                self.assertTrue(slider[1]["clamped"])
                self.assertTrue(
                    any(
                        call[0] == "group"
                        and call[1].get("tag") == RTAView.SMOOTHING_GROUP
                        for call in backend.calls
                    )
                )
                assert module._view is not None
                backend.calls.clear()
                with patch.object(backend, "does_item_exist", return_value=True):
                    module._view.set_enabled(False)
                self.assertIn(
                    (
                        "configure_item",
                        RTAView.GENERATOR_GROUP,
                        {"enabled": False},
                    ),
                    backend.calls,
                )
                self.assertIn(
                    ("configure_item", RTAView.FFT_GROUP, {"enabled": False}),
                    backend.calls,
                )
            finally:
                module.deactivate()
                module.shutdown()

    def test_continuous_measurement_uses_three_workers_and_publishes_bars(self) -> None:
        app = SpectrumApplication()
        audio_input = ContinuousRTAInput()
        audio_output = RecordingRTAOutput()
        app.audio_input = cast(AudioInput, audio_input)
        app.audio_output = cast(AudioOutput, audio_output)
        app.main_window.set_status_text = lambda text: None
        measurement = app.create_measurement("rta")
        measurement.module_state.update(
            {
                "band": (50, 3_000),
                "window_width": 0.05,
                "window_hop": 0.01,
                "points": 31,
            }
        )
        module = cast(RTAModule, app.module_manager.module("rta"))
        analysis_threads: set[str] = set()

        from spectrum_app.modules.rta import jobs

        original_analyze = jobs.analyze_rta

        def observe_analysis(recording, config):
            analysis_threads.add(current_thread().name)
            return original_analyze(recording, config)

        with (
            patch.object(RTAView, "build"),
            patch.object(RTAView, "destroy"),
            patch.object(RTAView, "update"),
            patch.object(RTAView, "set_enabled"),
            patch.object(RTAView, "update_levels"),
            patch.object(RTASettingsWindow, "build"),
            patch.object(RTASettingsWindow, "destroy"),
            patch(
                "spectrum_app.modules.rta.jobs.analyze_rta",
                side_effect=observe_analysis,
            ),
        ):
            module.initialize(app)
            module.settings.pre_silence = 0.01
            module.settings.fade_in = 0.02
            module.settings.fade_out = 0.02
            module.activate(measurement)
            try:
                module.start_measurement()
                deadline = time.monotonic() + 5.0
                while not measurement.graphs and time.monotonic() < deadline:
                    module.update()
                    time.sleep(0.002)
                module.stop_measurement()
                while app.app_state.measuring and time.monotonic() < deadline:
                    module.update()
                    time.sleep(0.002)
                module.update()

                self.assertFalse(app.app_state.measuring)
                self.assertEqual(audio_input.read_threads, {"rta-io"})
                self.assertEqual(audio_output.write_threads, {"rta-io"})
                self.assertEqual(analysis_threads, {"rta-analyzer"})
                self.assertTrue(audio_output.written)
                self.assertTrue(all(block.ndim == 1 for block in audio_output.written))
                self.assertEqual(len(measurement.graphs), 1)
                self.assertEqual(measurement.graphs[0].y_axis, AxisSpec.LEVEL)
                self.assertEqual(measurement.graphs[0].plot_type, PlotType.BARS)
                self.assertEqual(measurement.graphs[0].y.shape, (31,))
                self.assertIsInstance(measurement.module_state["recording"], ASignal)
            finally:
                module.deactivate()
                module.shutdown()

    def test_settings_defaults_and_persistence(self) -> None:
        changes: list[str] = []
        app = SpectrumApplication()
        settings = RTASettings(app.settings, changes.append)

        self.assertEqual(settings.mode, "mono")
        self.assertEqual(settings.plot_type, "auto")
        self.assertEqual(settings.window_function, "hann")
        self.assertEqual(settings.pre_silence, 0.1)
        self.assertEqual(settings.fade_in, 0.5)
        self.assertEqual(settings.fade_out, 0.5)

        settings.mode = "stereo"
        settings.plot_type = "line"
        settings.window_function = "blackman"
        settings.pre_silence = 0.2
        settings.fade_in = 0.3
        settings.fade_out = 0.4

        self.assertEqual(app.settings.module_setting("rta", "mode"), "stereo")
        self.assertEqual(
            changes,
            [
                "mode",
                "plot_type",
                "window_function",
                "pre_silence",
                "fade_in",
                "fade_out",
            ],
        )

    def test_smoothing_settings_update_running_analyzer_config(self) -> None:
        app = SpectrumApplication()
        app.audio_input = cast(AudioInput, ContinuousRTAInput())
        app.audio_output = cast(AudioOutput, RecordingRTAOutput())
        app.main_window.set_status_text = lambda text: None
        measurement = app.create_measurement("rta")
        measurement.module_state["band"] = (50, 3_000)
        module = cast(RTAModule, app.module_manager.module("rta"))

        with (
            patch.object(RTAView, "build"),
            patch.object(RTAView, "destroy"),
            patch.object(RTAView, "update"),
            patch.object(RTAView, "set_enabled"),
            patch.object(RTAView, "update_levels"),
            patch.object(RTASettingsWindow, "build"),
            patch.object(RTASettingsWindow, "destroy"),
        ):
            module.initialize(app)
            module.activate(measurement)
            worker = MagicMock()
            worker.is_alive.return_value = True
            module._io_worker = worker
            module._revision = 7
            try:
                module.set_setting("points", 128)
                module.set_setting("smoothing_octaves", 0.5)

                self.assertEqual(module._revision, 7)
                self.assertIsNotNone(module._runtime_analysis_config)
                assert module._runtime_analysis_config is not None
                self.assertEqual(module._runtime_analysis_config.points, 128)
                self.assertEqual(
                    module._runtime_analysis_config.smoothing_width,
                    0.5,
                )
            finally:
                module._io_worker = None
                module.deactivate()
                module.shutdown()

    def test_bar_width_matches_grid_step_and_line_defaults_to_point_one(self) -> None:
        app = SpectrumApplication()
        measurement = app.create_measurement("rta")
        module = cast(RTAModule, app.module_manager.module("rta"))

        with (
            patch.object(RTASettingsWindow, "build"),
            patch.object(RTASettingsWindow, "destroy"),
        ):
            module.initialize(app)
            try:
                module._ensure_state(measurement.module_state)
                bar_config = module._build_analysis_config(
                    measurement.module_state
                )
                expected = np.log2(20_000.0 / 20.0) / (31 - 1)
                self.assertAlmostEqual(bar_config.smoothing_width, expected)

                app.settings.set_module_setting("rta", "plot_type", "line")
                line_config = module._build_analysis_config(
                    measurement.module_state
                )
                self.assertEqual(line_config.smoothing_width, 0.1)
            finally:
                module.shutdown()


if __name__ == "__main__":
    unittest.main()
