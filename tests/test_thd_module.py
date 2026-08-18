from threading import current_thread
import time
from typing import cast
import unittest
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

from audioanalysis import (
    ASignal,
    FrequencyBand,
    SemiAnalogTHDConfig,
    generate_semi_analog_thd_sweep,
)
from spectrum_app import SpectrumApplication
from spectrum_app.core.audio import AudioInput, AudioOutput
from spectrum_app.core.model import AxisSpec
from spectrum_app.modules.thd import THDModule
from spectrum_app.modules.thd.settings import THDSettings, THDSettingsWindow
from spectrum_app.modules.thd.view import THDView
from tests.test_dpg_lifecycle import FakeDpgBackend


class PreparedTHDInput:
    sample_rate = 8_000
    blocksize = 512

    def __init__(self, data: NDArray[np.float32]) -> None:
        self.data = data.reshape(-1, 1)
        self.position = 0
        self.read_threads: list[str] = []
        self.opened = False

    def open(self) -> bool:
        self.opened = True
        return True

    def read(self, samples: int) -> NDArray[np.float32]:
        self.read_threads.append(current_thread().name)
        end = self.position + samples
        block = self.data[self.position:end]
        self.position = end
        if len(block) < samples:
            block = np.vstack(
                (
                    block,
                    np.zeros((samples - len(block), 1), dtype=np.float32),
                )
            )
        return block

    def close(self) -> bool:
        self.opened = False
        return True


class RecordingTHDOutput:
    sample_rate = 8_000
    blocksize = 512

    def __init__(self) -> None:
        self.write_threads: list[str] = []
        self.written: list[NDArray[np.float32]] = []
        self.opened = False

    def open(self) -> bool:
        self.opened = True
        return True

    def write(self, data: NDArray) -> None:
        self.write_threads.append(current_thread().name)
        self.written.append(np.asarray(data, dtype=np.float32))

    def close(self) -> bool:
        self.opened = False
        return True


class THDModuleTests(unittest.TestCase):
    def test_module_builds_discussed_controls_and_settings_window(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement("thd")
        module = cast(THDModule, app.module_manager.module("thd"))
        self.assertEqual(module.name, "THD+N")

        with (
            patch("spectrum_app.modules.thd.view.dpg", backend),
            patch("spectrum_app.gui.controls.level_meter.dpg", backend),
            patch("spectrum_app.modules.thd.settings.dpg", backend),
        ):
            module.initialize(app)
            module.activate(measurement)
            try:
                settings_item = next(
                    call
                    for call in backend.calls
                    if call[0] == "add_menu_item"
                    and call[1].get("tag") == THDSettingsWindow.MENU_ITEM
                )
                self.assertEqual(settings_item[1]["label"], "THD")
                self.assertEqual(
                    settings_item[1]["parent"],
                    app.main_window.settings_menu,
                )
                self.assertTrue(
                    any(
                        call[0] == "drawlist"
                        and call[1].get("tag") == THDView.METER
                        for call in backend.calls
                    )
                )
                self.assertTrue(
                    any(
                        call[0] == "add_line_series"
                        and call[3].get("tag") == THDView.LEVEL_SERIES
                        for call in backend.calls
                    )
                )
                level_plot = next(
                    call
                    for call in backend.calls
                    if call[0] == "plot"
                    and call[1].get("tag") == THDView.LEVEL_PLOT
                )
                self.assertNotIn("label", level_plot[1])
                level_axes = [
                    call
                    for call in backend.calls
                    if call[0] == "add_plot_axis"
                    and call[2].get("tag")
                    in (THDView.LEVEL_X_AXIS, THDView.LEVEL_Y_AXIS)
                ]
                self.assertTrue(
                    all("label" not in call[2] for call in level_axes)
                )
                self.assertAlmostEqual(
                    measurement.module_state["smoothing_octaves"],
                    0.1,
                )
            finally:
                module.deactivate()
                module.shutdown()

    def test_complete_measurement_uses_separate_io_and_analysis_threads(self) -> None:
        config = SemiAnalogTHDConfig(
            sample_rate=8_000,
            duration=4.0,
            band=FrequencyBand(50.0, 2_500.0),
            smoothing_octaves=1.0 / 3.0,
            segment_seconds=0.25,
            overlap=0.75,
            fade_in_seconds=0.1,
            fade_out_seconds=0.1,
            notch_ratio=1.5,
            points=128,
        )
        generator = generate_semi_analog_thd_sweep(config).as_array(np.float64)[:, 0]
        response = generator + 0.01 * np.square(generator)
        recording = np.concatenate(
            (
                np.zeros(round(0.2 * config.sample_rate)),
                response,
                np.zeros(round(0.5 * config.sample_rate)),
            )
        ).astype(np.float32)
        audio_input = PreparedTHDInput(recording)
        audio_output = RecordingTHDOutput()
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        app.audio_input = cast(AudioInput, audio_input)
        app.audio_output = cast(AudioOutput, audio_output)
        app.main_window.set_status_text = lambda text: None
        for key, value in (
            ("segment_seconds", 0.25),
            ("overlap_percent", 75.0),
            ("fade_in_seconds", 0.1),
            ("fade_out_seconds", 0.1),
            ("notch_ratio", 1.5),
            ("points", 128),
        ):
            app.settings.set_module_setting("thd", key, value)
        measurement = app.create_measurement("thd")
        measurement.module_state.update(
            {
                "band": (50, 2_500),
                "duration": 4.0,
                "smoothing_octaves": 1.0 / 3.0,
            }
        )
        module = cast(THDModule, app.module_manager.module("thd"))

        with (
            patch("spectrum_app.modules.thd.view.dpg", backend),
            patch("spectrum_app.gui.controls.level_meter.dpg", backend),
            patch("spectrum_app.modules.thd.settings.dpg", backend),
        ):
            module.initialize(app)
            module.activate(measurement)
            try:
                module.start_measurement()
                self._wait_for_measurement(module, app)

                self.assertFalse(app.app_state.measuring)
                self.assertEqual(len(measurement.graphs), 1)
                self.assertEqual(measurement.graphs[0].name, "THD+N")
                self.assertEqual(measurement.graphs[0].y_axis, AxisSpec.THD)
                self.assertEqual(measurement.graphs[0].y.shape, (128,))
                self.assertGreater(
                    float(measurement.module_state["integrated_ratio"]),
                    0.0,
                )
                self.assertIsInstance(measurement.module_state["recording"], ASignal)
                self.assertIsInstance(measurement.module_state["generator"], ASignal)
                self.assertGreater(len(measurement.module_state["level_time"]), 0)
                self.assertEqual(set(audio_input.read_threads), {"thd-acquisition"})
                self.assertEqual(set(audio_output.write_threads), {"thd-output"})
                self.assertTrue(audio_output.written)
                self.assertTrue(
                    all(block.ndim == 1 for block in audio_output.written)
                )
            finally:
                module.deactivate()
                module.shutdown()

    def test_advanced_settings_are_saved_in_app_settings(self) -> None:
        changes: list[str] = []
        app = SpectrumApplication()
        settings = THDSettings(app.settings, changes.append)

        settings.segment_seconds = 0.5
        settings.overlap_percent = 80.0
        settings.fade_in_seconds = 0.25
        settings.fade_out_seconds = 0.75
        settings.notch_ratio = 1.75
        settings.points = 640

        self.assertEqual(settings.segment_seconds, 0.5)
        self.assertEqual(settings.overlap_percent, 80.0)
        self.assertEqual(settings.fade_in_seconds, 0.25)
        self.assertEqual(settings.fade_out_seconds, 0.75)
        self.assertEqual(settings.notch_ratio, 1.75)
        self.assertEqual(settings.points, 640)
        self.assertEqual(
            changes,
            [
                "segment_seconds",
                "overlap_percent",
                "fade_in_seconds",
                "fade_out_seconds",
                "notch_ratio",
                "points",
            ],
        )

    @staticmethod
    def _wait_for_measurement(
        module: THDModule,
        app: SpectrumApplication,
    ) -> None:
        deadline = time.monotonic() + 8.0
        while app.app_state.measuring and time.monotonic() < deadline:
            module.update()
            time.sleep(0.002)
        module.update()
        if app.app_state.measuring:
            raise AssertionError("THD measurement did not complete")


if __name__ == "__main__":
    unittest.main()
