from threading import current_thread
import time
from typing import cast
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.typing import NDArray

from audioanalysis import ASignal, AnalysisMethod, FrequencyBand
from spectrum_app import SpectrumApplication
from spectrum_app.core.audio import AudioInput, AudioOutput
from spectrum_app.core.model import AxisSpec
from spectrum_app.modules.spectrum import SpectrumModule
from spectrum_app.modules.spectrum.jobs import SpectrumAcquisition
from spectrum_app.modules.spectrum.settings import SpectrumSettingsWindow
from spectrum_app.modules.spectrum.view import SpectrumView
from tests.test_dpg_lifecycle import FakeDpgBackend


class FakeAudioInput:
    sample_rate = 8_000
    blocksize = 256

    def __init__(self) -> None:
        self.position = 0
        self.thread_names: set[str] = set()

    def open(self) -> bool:
        return True

    def read(self, samples: int) -> NDArray[np.float32]:
        self.thread_names.add(current_thread().name)
        time.sleep(0.002)
        indexes = np.arange(self.position, self.position + samples)
        self.position += samples
        signal = 0.5 * np.sin(2 * np.pi * 1_000 * indexes / self.sample_rate)
        return np.column_stack((signal, signal)).astype(np.float32)

    def close(self) -> bool:
        return True


class FakeAudioOutput:
    sample_rate = 8_000
    blocksize = 256

    def __init__(self) -> None:
        self.thread_names: set[str] = set()
        self.written_samples = 0

    def open(self) -> bool:
        return True

    def write(self, data: NDArray) -> None:
        self.thread_names.add(current_thread().name)
        self.written_samples += len(data)

    def close(self) -> bool:
        return True


class SpectrumModuleTests(unittest.TestCase):
    def test_generators_include_silence_and_fades(self) -> None:
        audio_output = FakeAudioOutput()
        audio_output.sample_rate = 1_000
        worker = SpectrumAcquisition(
            cast(AudioInput, FakeAudioInput()),
            cast(AudioOutput, audio_output),
            generator_mode="pink noise",
            band=FrequencyBand(10.0, 100.0),
            duration=1.0,
            pre_silence=0.2,
            post_silence=0.3,
            fade_in=0.1,
            fade_out=0.1,
            online_interval=None,
            on_level=lambda *args: None,
            on_snapshot=lambda *args: None,
            on_complete=lambda *args: None,
        )

        flat = ASignal(np.ones(1_200, dtype=np.float32), 1_000)
        with patch(
            "spectrum_app.modules.spectrum.jobs.pink_noise",
            return_value=(flat, None),
        ):
            signal = worker._generate_signal()

        data = signal.as_array()[:, 0]
        self.assertEqual(signal.sample_count, 1_700)
        self.assertTrue(np.all(data[:200] == 0.0))
        self.assertEqual(data[200], 0.0)
        self.assertEqual(data[299], 1.0)
        self.assertEqual(data[1_399], 0.0)
        self.assertTrue(np.all(data[1_400:] == 0.0))

    def test_log_chirp_expands_band_outside_fades(self) -> None:
        audio_output = FakeAudioOutput()
        audio_output.sample_rate = 1_000
        worker = SpectrumAcquisition(
            cast(AudioInput, FakeAudioInput()),
            cast(AudioOutput, audio_output),
            generator_mode="log chirp",
            band=FrequencyBand(10.0, 100.0),
            duration=1.0,
            pre_silence=0.0,
            post_silence=0.0,
            fade_in=0.1,
            fade_out=0.2,
            online_interval=None,
            on_level=lambda *args: None,
            on_snapshot=lambda *args: None,
            on_complete=lambda *args: None,
        )

        def fake_chirp(samples, sample_rate, band):
            self.assertAlmostEqual(band.low, 10.0 / 10.0**0.1)
            self.assertAlmostEqual(band.high, 100.0 * 10.0**0.2)
            return ASignal(np.ones(samples, dtype=np.float32), sample_rate)

        with patch(
            "spectrum_app.modules.spectrum.jobs.log_chirp",
            side_effect=fake_chirp,
        ):
            signal = worker._generate_signal()

        self.assertEqual(signal.sample_count, 1_300)
        self.assertEqual(signal.as_array()[0, 0], 0.0)
        self.assertEqual(signal.as_array()[-1, 0], 0.0)

    def test_measurement_uses_separate_io_and_analysis_threads(self) -> None:
        app = SpectrumApplication()
        audio_input = FakeAudioInput()
        audio_output = FakeAudioOutput()
        app.audio_input = cast(AudioInput, audio_input)
        app.audio_output = cast(AudioOutput, audio_output)
        app.main_window.set_status_text = lambda text: None
        measurement = app.create_measurement()
        measurement.module_state.update(
            {
                "duration": 0.02,
                "band": (20, 3_000),
                "reference": "none",
                "points": 100,
            }
        )
        module = cast(SpectrumModule, app.module_manager.module("spectrum"))
        module.ONLINE_INTERVAL = 0.02
        analysis_threads: set[str] = set()
        analysis_methods: list[AnalysisMethod] = []

        from spectrum_app.modules.spectrum import jobs

        original_analyze = jobs.analyze_spectrum

        def observe_analysis(signal: ASignal, config):
            analysis_threads.add(current_thread().name)
            analysis_methods.append(config.method)
            return original_analyze(signal, config)

        with (
            patch.object(SpectrumView, "build"),
            patch.object(SpectrumView, "destroy"),
            patch.object(SpectrumView, "set_enabled"),
            patch.object(SpectrumView, "update_levels"),
            patch.object(SpectrumSettingsWindow, "build"),
            patch.object(SpectrumSettingsWindow, "destroy"),
            patch(
                "spectrum_app.modules.spectrum.jobs.analyze_spectrum",
                side_effect=observe_analysis,
            ),
        ):
            module.initialize(app)
            module.settings.welch_samples = 256
            module.settings.pre_silence = 0.2
            module.settings.post_silence = 0.2
            module.settings.fade_in = 0.0
            module.settings.fade_out = 0.0
            module.activate(measurement)
            app.app_state.graph_data_changed = False
            try:
                module.start_measurement()
                deadline = time.monotonic() + 5.0
                while app.app_state.measuring and time.monotonic() < deadline:
                    module.update()
                    time.sleep(0.005)
                module.update()

                self.assertFalse(app.app_state.measuring)
                self.assertEqual(audio_input.thread_names, {"spectrum-acquisition"})
                self.assertEqual(audio_output.thread_names, {"spectrum-output"})
                self.assertEqual(analysis_threads, {"spectrum-analyzer"})
                self.assertIn(AnalysisMethod.WELCH, analysis_methods)
                self.assertIn(AnalysisMethod.PERIODOGRAM, analysis_methods)
                self.assertGreater(audio_output.written_samples, 0)
                self.assertIsInstance(measurement.module_state["recording"], ASignal)
                self.assertIsInstance(measurement.module_state["generator"], ASignal)
                self.assertGreater(len(measurement.module_state["level_time"]), 0)
                self.assertEqual(measurement.module_state["level_values"].shape[1], 2)
                self.assertEqual(len(measurement.graphs), 1)
                self.assertEqual(measurement.graphs[0].y_axis, AxisSpec.LEVEL)
                self.assertTrue(app.app_state.graph_data_changed)
            finally:
                module.deactivate()
                module.shutdown()

    def test_online_welch_can_be_disabled_in_application_settings(self) -> None:
        app = SpectrumApplication()
        audio_input = FakeAudioInput()
        audio_output = FakeAudioOutput()
        app.audio_input = cast(AudioInput, audio_input)
        app.audio_output = cast(AudioOutput, audio_output)
        app.main_window.set_status_text = lambda text: None
        measurement = app.create_measurement()
        measurement.module_state.update(
            {"band": (20, 3_000), "reference": "none"}
        )
        module = cast(SpectrumModule, app.module_manager.module("spectrum"))
        acquisition = MagicMock()
        acquisition.is_alive.return_value = False

        with (
            patch.object(SpectrumView, "build"),
            patch.object(SpectrumView, "destroy"),
            patch.object(SpectrumView, "set_enabled"),
            patch.object(SpectrumView, "update_levels"),
            patch.object(SpectrumSettingsWindow, "build"),
            patch.object(SpectrumSettingsWindow, "destroy"),
            patch(
                "spectrum_app.modules.spectrum.module.SpectrumAcquisition",
                return_value=acquisition,
            ) as acquisition_class,
        ):
            module.initialize(app)
            module.settings.online_welch = False
            module.settings.fade_in = 0.0
            module.settings.fade_out = 0.0
            module.activate(measurement)
            try:
                module.start_measurement()

                self.assertIsNone(
                    acquisition_class.call_args.kwargs["online_interval"]
                )
                self.assertEqual(
                    acquisition_class.call_args.kwargs["pre_silence"],
                    module.settings.pre_silence,
                )
                self.assertEqual(
                    acquisition_class.call_args.kwargs["post_silence"],
                    module.settings.post_silence,
                )
                self.assertEqual(
                    acquisition_class.call_args.kwargs["fade_in"],
                    module.settings.fade_in,
                )
                self.assertEqual(
                    acquisition_class.call_args.kwargs["fade_out"],
                    module.settings.fade_out,
                )
                self.assertIn("on_level", acquisition_class.call_args.kwargs)
                acquisition.start.assert_called_once_with()
            finally:
                module.deactivate()
                module.shutdown()

    def test_view_builds_two_channel_compact_level_plot(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement("spectrum")
        module = cast(SpectrumModule, app.module_manager.module("spectrum"))

        with (
            patch("spectrum_app.modules.spectrum.view.dpg", backend),
            patch("spectrum_app.modules.spectrum.settings.dpg", backend),
        ):
            module.initialize(app)
            module.activate(measurement)
            try:
                level_plot = next(
                    call
                    for call in backend.calls
                    if call[0] == "plot"
                    and call[1].get("tag") == SpectrumView.LEVEL_PLOT
                )
                self.assertNotIn("label", level_plot[1])
                level_axes = [
                    call
                    for call in backend.calls
                    if call[0] == "add_plot_axis"
                    and call[2].get("tag")
                    in (SpectrumView.LEVEL_X_AXIS, SpectrumView.LEVEL_Y_AXIS)
                ]
                self.assertTrue(
                    all("label" not in call[2] for call in level_axes)
                )
                series = [
                    call
                    for call in backend.calls
                    if call[0] == "add_line_series"
                    and call[3].get("tag")
                    in (SpectrumView.LEVEL_SERIES_1, SpectrumView.LEVEL_SERIES_2)
                ]
                self.assertEqual(len(series), 2)
            finally:
                module.deactivate()
                module.shutdown()


if __name__ == "__main__":
    unittest.main()
