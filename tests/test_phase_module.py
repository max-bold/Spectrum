from threading import current_thread
import time
from typing import cast
import unittest
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

from audioanalysis import ASignal
from spectrum_app import SpectrumApplication
from spectrum_app.core.audio import AudioInput, AudioOutput
from spectrum_app.core.model import AxisSpec
from spectrum_app.modules.phase import PhaseModule
from spectrum_app.modules.phase.settings import PhaseSettings, PhaseSettingsWindow
from spectrum_app.modules.phase.view import PhaseView
from tests.test_dpg_lifecycle import FakeDpgBackend


class PreparedPhaseInput:
    sample_rate = 8_000
    block_size = 256

    def __init__(self) -> None:
        self.position = 0
        self.read_threads: set[str] = set()
        self.data = np.zeros((800, 2), dtype=np.float32)
        self.data[108, 0] = 0.5
        self.data[100, 1] = 0.5

    def open(self) -> bool:
        return True

    def read(self, samples: int) -> NDArray[np.float32]:
        self.read_threads.add(current_thread().name)
        end = self.position + samples
        block = self.data[self.position : end]
        self.position = end
        if len(block) < samples:
            block = np.vstack(
                (block, np.zeros((samples - len(block), 2), dtype=np.float32))
            )
        return block

    def close(self) -> bool:
        return True


class RecordingPhaseOutput:
    sample_rate = 8_000
    block_size = 256

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


class PhaseModuleTests(unittest.TestCase):
    def test_view_builds_phase_controls_meter_and_compact_response(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement("phase")
        module = cast(PhaseModule, app.module_manager.module("phase"))

        with (
            patch("spectrum_app.modules.phase.view.dpg", backend),
            patch("spectrum_app.modules.phase.settings.dpg", backend),
            patch("spectrum_app.gui.controls.level_meter.dpg", backend),
        ):
            module.initialize(app)
            module.activate(measurement)
            try:
                self.assertAlmostEqual(
                    measurement.module_state["smoothing_octaves"],
                    1.0 / 3.0,
                )
                self.assertEqual(measurement.module_state["points"], 1024)
                self.assertEqual(
                    measurement.module_state["delay_correction_meters"],
                    0.0,
                )
                self.assertTrue(
                    any(
                        call[0] == "drawlist" and call[1].get("tag") == PhaseView.METER
                        for call in backend.calls
                    )
                )
                response_plot = next(
                    call
                    for call in backend.calls
                    if call[0] == "plot"
                    and call[1].get("tag") == PhaseView.RESPONSE_PLOT
                )
                self.assertNotIn("label", response_plot[1])
                self.assertEqual(response_plot[1]["width"], -1)
                control_tags = {
                    call[1].get("tag")
                    for call in backend.calls
                    if call[0] in ("add_input_intx", "add_input_float", "add_input_int")
                }
                self.assertTrue(
                    {
                        PhaseView.BAND,
                        PhaseView.DURATION,
                        PhaseView.SMOOTHING,
                        PhaseView.POINTS,
                        PhaseView.DELAY_FIT,
                        PhaseView.DELAY_CORRECTION,
                    }.issubset(control_tags)
                )
                labels = [call[1] for call in backend.calls if call[0] == "add_text"]
                self.assertNotIn("Output level", labels)
            finally:
                module.deactivate()
                module.shutdown()

    def test_measurement_uses_workers_and_publishes_only_unwrapped_phase(self) -> None:
        app = SpectrumApplication()
        audio_input = PreparedPhaseInput()
        audio_output = RecordingPhaseOutput()
        app.audio_input = cast(AudioInput, audio_input)
        app.audio_output = cast(AudioOutput, audio_output)
        app.main_window.set_status_text = lambda text: None
        measurement = app.create_measurement("phase")
        measurement.module_state.update(
            {
                "band": (50, 3_000),
                "duration": 0.1,
                "delay_fit_band": (100, 2_500),
                "points": 128,
            }
        )
        module = cast(PhaseModule, app.module_manager.module("phase"))
        analysis_threads: set[str] = set()

        from spectrum_app.modules.phase import jobs

        original_analyze = jobs.analyze_phase

        def observe_analysis(recording, config):
            analysis_threads.add(current_thread().name)
            return original_analyze(recording, config)

        with (
            patch.object(PhaseView, "build"),
            patch.object(PhaseView, "destroy"),
            patch.object(PhaseView, "update"),
            patch.object(PhaseView, "set_enabled"),
            patch.object(PhaseView, "update_levels"),
            patch.object(PhaseView, "update_result"),
            patch.object(PhaseSettingsWindow, "build"),
            patch.object(PhaseSettingsWindow, "destroy"),
            patch(
                "spectrum_app.modules.phase.jobs.analyze_phase",
                side_effect=observe_analysis,
            ),
        ):
            module.initialize(app)
            module.settings.pre_silence = 0.0
            module.settings.post_silence = 0.0
            module.settings.fade = 0.0
            module.activate(measurement)
            try:
                module.start_measurement()
                deadline = time.monotonic() + 5.0
                while app.app_state.measuring and time.monotonic() < deadline:
                    module.update()
                    time.sleep(0.002)
                module.update()

                self.assertFalse(app.app_state.measuring)
                self.assertEqual(audio_input.read_threads, {"phase-acquisition"})
                self.assertEqual(audio_output.write_threads, {"phase-output"})
                self.assertEqual(analysis_threads, {"phase-analyzer"})
                self.assertTrue(audio_output.written)
                self.assertTrue(all(block.ndim == 1 for block in audio_output.written))
                self.assertLessEqual(
                    max(float(np.max(np.abs(block))) for block in audio_output.written),
                    0.900001,
                )
                self.assertIsInstance(measurement.module_state["recording"], ASignal)
                self.assertIsInstance(measurement.module_state["generator"], ASignal)
                self.assertEqual(len(measurement.graphs), 1)
                self.assertEqual(measurement.graphs[0].name, "Phase")
                self.assertEqual(measurement.graphs[0].y_axis, AxisSpec.PHASE)
                self.assertEqual(measurement.graphs[0].y.shape, (128,))
            finally:
                module.deactivate()
                module.shutdown()

    def test_settings_defaults_and_persistence(self) -> None:
        changes: list[str] = []
        app = SpectrumApplication()
        settings = PhaseSettings(app.settings, changes.append)

        self.assertEqual(settings.pre_silence, 0.5)
        self.assertEqual(settings.post_silence, 0.5)
        self.assertEqual(settings.fade, 0.5)
        settings.pre_silence = 0.25
        settings.post_silence = 0.75
        settings.fade = 0.1

        self.assertEqual(
            app.settings.module_setting("phase", "pre_silence"),
            0.25,
        )
        self.assertEqual(changes, ["pre_silence", "post_silence", "fade"])


if __name__ == "__main__":
    unittest.main()
