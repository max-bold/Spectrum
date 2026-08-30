from threading import current_thread
import time
from typing import cast
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import hann

from audioanalysis import (
    ASignal,
    AnalysisMethod,
    FrequencyBand,
    ReferenceMode,
    SpectrumConfig,
    SpectrumResult,
)
from spectrum_app import SpectrumApplication
from spectrum_app.core.audio import AudioInput, AudioOutput
from spectrum_app.core.model import AxisSpec
from spectrum_app.modules.spectrum import SpectrumModule
from spectrum_app.modules.spectrum.jobs import (
    HANN_SQUARED_OVERLAP_GAIN,
    ONLINE_OVERLAP_DIVISOR,
    OnlineAnalysisRequest,
    SpectrumAcquisition,
    SpectrumAnalyzer,
)
from spectrum_app.modules.spectrum.settings import SpectrumSettings, SpectrumSettingsWindow
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
    def test_online_hann_squared_overlap_is_constant(self) -> None:
        samples = 256
        hop = samples // ONLINE_OVERLAP_DIVISOR
        window = hann(samples, sym=False)
        overlap = sum(np.roll(window**2, shift) for shift in range(0, samples, hop))

        np.testing.assert_allclose(overlap, HANN_SQUARED_OVERLAP_GAIN)

    def test_online_welch_sums_chirp_and_averages_pink_noise(self) -> None:
        frequency = np.linspace(0.0, 4_000.0, 5)
        bucket = np.ones((5, 2), dtype=np.float64)
        signal = ASignal(np.ones((64, 2), dtype=np.float32), 8_000)
        config = SpectrumConfig(band=FrequencyBand(100.0, 3_000.0))

        def preserve_linear_spectrum(frequency, spectrum, config):
            return SpectrumResult(frequency, spectrum[:, 0])

        with (
            patch(
                "spectrum_app.modules.spectrum.jobs._hann_periodogram",
                return_value=(frequency, bucket),
            ),
            patch(
                "spectrum_app.modules.spectrum.jobs.smooth_power_spectrum",
                side_effect=preserve_linear_spectrum,
            ),
            patch(
                "spectrum_app.modules.spectrum.jobs.ONLINE_PUBLISH_SECONDS",
                0.0,
            ),
        ):
            chirp_analyzer = SpectrumAnalyzer()
            chirp_request = OnlineAnalysisRequest(
                1,
                signal,
                None,
                config,
                False,
            )
            chirp_analyzer._analyze_online(chirp_request)
            chirp_result = chirp_analyzer._analyze_online(chirp_request)

            pink_analyzer = SpectrumAnalyzer()
            pink_request = OnlineAnalysisRequest(
                1,
                signal,
                None,
                config,
                True,
            )
            pink_analyzer._analyze_online(pink_request)
            pink_result = pink_analyzer._analyze_online(pink_request)

        self.assertIsNotNone(chirp_result)
        self.assertIsNotNone(pink_result)
        assert chirp_result is not None
        assert pink_result is not None
        np.testing.assert_allclose(
            chirp_result.values,
            10.0 * np.log10(2.0 / HANN_SQUARED_OVERLAP_GAIN),
        )
        np.testing.assert_allclose(pink_result.values, 0.0)

    def test_online_generator_reference_uses_the_same_bucket_accumulation(self) -> None:
        samples = 256
        hop = samples // ONLINE_OVERLAP_DIVISOR
        generator = ASignal(
            np.random.default_rng(42).normal(size=1024).astype(np.float32),
            8_000,
        )
        analyzer = SpectrumAnalyzer()
        config = SpectrumConfig(
            reference=ReferenceMode.CHANNEL_B,
            band=FrequencyBand(100.0, 3_000.0),
            points=64,
        )
        result = None

        with patch(
            "spectrum_app.modules.spectrum.jobs.ONLINE_PUBLISH_SECONDS",
            0.0,
        ):
            for start in range(0, generator.sample_count - samples + 1, hop):
                generator_bucket = generator.trim(samples, start=start).as_array()
                recording = ASignal(
                    np.column_stack((generator_bucket[:, 0], np.zeros(samples))),
                    generator.sample_rate,
                )
                result = analyzer._analyze_online(
                    OnlineAnalysisRequest(
                        1,
                        recording,
                        generator,
                        config,
                        False,
                    )
                )

        self.assertIsNotNone(result)
        assert result is not None
        finite = np.isfinite(result.values)
        self.assertTrue(np.any(finite))
        np.testing.assert_allclose(result.values[finite], 0.0, atol=1e-10)

    def test_acquisition_feeds_each_new_audio_block_to_online_analysis(self) -> None:
        blocks: list[np.ndarray] = []
        worker = SpectrumAcquisition(
            cast(AudioInput, FakeAudioInput()),
            cast(AudioOutput, FakeAudioOutput()),
            generator_mode="log chirp",
            band=FrequencyBand(20.0, 3_000.0),
            duration=0.08,
            pre_silence=0.0,
            post_silence=0.0,
            fade_in=0.0,
            fade_out=0.0,
            online_samples=64,
            on_level=lambda *args: None,
            on_snapshot=lambda signal, generator: blocks.append(
                signal.as_array().copy()
            ),
            on_complete=lambda *args: None,
        )

        worker.run()

        self.assertEqual([len(block) for block in blocks], [256, 256, 128])
        self.assertEqual(sum(len(block) for block in blocks), 640)

    def test_online_analyzer_drains_all_windows_without_gui_polling(self) -> None:
        samples = 64
        hop = samples // ONLINE_OVERLAP_DIVISOR
        data = np.arange(256, dtype=np.float32)
        signal = ASignal(np.column_stack((data, data)), 8_000)
        starts: list[float] = []
        analyzer = SpectrumAnalyzer()
        analyzer.start()

        def observe_window(window: ASignal):
            starts.append(float(window.as_array()[0, 0]))
            return np.linspace(0.0, 4_000.0, 5), np.ones((5, 2))

        try:
            with (
                patch(
                    "spectrum_app.modules.spectrum.jobs._hann_periodogram",
                    side_effect=observe_window,
                ),
                patch(
                    "spectrum_app.modules.spectrum.jobs.smooth_power_spectrum",
                    side_effect=lambda frequency, spectrum, config: SpectrumResult(
                        frequency,
                        spectrum[:, 0],
                    ),
                ),
            ):
                analyzer.feed_online(
                    1,
                    signal,
                    samples,
                    None,
                    SpectrumConfig(
                        band=FrequencyBand(100.0, 3_000.0),
                        points=4,
                    ),
                    running_mean=False,
                )
                expected = 1 + (signal.sample_count - samples) // hop
                deadline = time.monotonic() + 2.0
                while len(starts) < expected and time.monotonic() < deadline:
                    time.sleep(0.001)

            self.assertEqual(len(starts), expected)
            np.testing.assert_allclose(starts, np.arange(expected) * hop)
        finally:
            analyzer.shutdown()

    def test_weighting_is_only_applied_without_a_reference(self) -> None:
        app = SpectrumApplication()
        measurement = app.create_measurement("spectrum")
        module = cast(SpectrumModule, app.module_manager.module("spectrum"))
        recording = ASignal(np.ones((256, 2), dtype=np.float32), 8_000)
        generator = ASignal(np.ones(256, dtype=np.float32), 8_000)

        with (
            patch.object(SpectrumView, "build"),
            patch.object(SpectrumView, "destroy"),
            patch.object(SpectrumView, "set_enabled"),
            patch.object(SpectrumView, "hide_repeat_dialog"),
            patch.object(SpectrumSettingsWindow, "build"),
            patch.object(SpectrumSettingsWindow, "destroy"),
        ):
            module.initialize(app)
            module.activate(measurement)
            try:
                self.assertEqual(measurement.module_state["reference"], "generator")
                measurement.module_state["weighting"] = "pink"

                measurement.module_state["reference"] = "none"
                _, config = module._analysis_input(
                    recording,
                    generator,
                    AnalysisMethod.PERIODOGRAM,
                )
                self.assertTrue(config.pink_weighting)

                for reference in ("channel b", "generator"):
                    measurement.module_state["reference"] = reference
                    _, config = module._analysis_input(
                        recording,
                        generator,
                        AnalysisMethod.PERIODOGRAM,
                    )
                    self.assertFalse(config.pink_weighting)
            finally:
                module.deactivate()
                module.shutdown()

    def test_analyzer_averages_linear_power_before_converting_to_db(self) -> None:
        analyzer = SpectrumAnalyzer()
        analyzer.start()
        signal_a = ASignal(np.ones(64, dtype=np.float32), 8_000)
        signal_b = ASignal(np.ones(64, dtype=np.float32), 8_000)
        frequency = np.asarray([100.0, 1_000.0])
        results = (
            SpectrumResult(frequency, np.asarray([1.0, 4.0])),
            SpectrumResult(frequency, np.asarray([9.0, 16.0])),
        )
        try:
            with patch(
                "spectrum_app.modules.spectrum.jobs.analyze_spectrum",
                side_effect=results,
            ):
                analyzer.submit(
                    1,
                    True,
                    (signal_a, signal_b),
                    SpectrumConfig(),
                )
                response = None
                deadline = time.monotonic() + 2.0
                while response is None and time.monotonic() < deadline:
                    response = analyzer.poll()
                    time.sleep(0.001)

            self.assertIsNotNone(response)
            assert response is not None
            result = response[2]
            self.assertIsNotNone(result)
            assert result is not None
            np.testing.assert_allclose(
                result.values,
                10.0 * np.log10(np.asarray([5.0, 10.0])),
            )
        finally:
            analyzer.shutdown()

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
            online_samples=None,
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
            online_samples=None,
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
        analysis_threads: set[str] = set()
        online_analysis_threads: set[str] = set()
        analysis_methods: list[AnalysisMethod] = []

        from spectrum_app.modules.spectrum import jobs

        original_analyze = jobs.analyze_spectrum
        original_periodogram = jobs.periodogram

        def observe_analysis(signal: ASignal, config):
            analysis_threads.add(current_thread().name)
            analysis_methods.append(config.method)
            return original_analyze(signal, config)

        def observe_online_periodogram(*args, **kwargs):
            online_analysis_threads.add(current_thread().name)
            return original_periodogram(*args, **kwargs)

        with (
            patch.object(SpectrumView, "build"),
            patch.object(SpectrumView, "destroy"),
            patch.object(SpectrumView, "set_enabled"),
            patch.object(SpectrumView, "update_levels"),
            patch.object(SpectrumView, "show_repeat_dialog"),
            patch.object(SpectrumView, "hide_repeat_dialog"),
            patch.object(SpectrumSettingsWindow, "build"),
            patch.object(SpectrumSettingsWindow, "destroy"),
            patch(
                "spectrum_app.modules.spectrum.jobs.analyze_spectrum",
                side_effect=observe_analysis,
            ),
            patch(
                "spectrum_app.modules.spectrum.jobs.periodogram",
                side_effect=observe_online_periodogram,
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
                self.assertEqual(online_analysis_threads, {"spectrum-analyzer"})
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
            patch.object(SpectrumView, "show_repeat_dialog"),
            patch.object(SpectrumView, "hide_repeat_dialog"),
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
                    acquisition_class.call_args.kwargs["online_samples"]
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

    def test_manual_multiple_measurement_continues_and_breaks_with_average(self) -> None:
        app = SpectrumApplication()
        app.audio_input = cast(AudioInput, FakeAudioInput())
        app.audio_output = cast(AudioOutput, FakeAudioOutput())
        app.main_window.set_status_text = lambda text: None
        measurement = app.create_measurement("spectrum")
        measurement.module_state.update(
            {
                "band": (20, 1_000),
                "duration": 1.0,
                "reference": "none",
                "multiple": True,
                "count": 3,
                "auto": False,
                "points": 100,
            }
        )
        module = cast(SpectrumModule, app.module_manager.module("spectrum"))
        acquisitions = [MagicMock(), MagicMock()]
        for acquisition in acquisitions:
            acquisition.is_alive.return_value = False
        recording_a = ASignal(
            np.column_stack((np.ones(256), np.ones(256))),
            8_000,
        )
        recording_b = ASignal(
            np.column_stack((np.full(256, 2.0), np.ones(256))),
            8_000,
        )
        generator = ASignal(np.ones(256), 8_000)

        with (
            patch.object(SpectrumView, "build"),
            patch.object(SpectrumView, "destroy"),
            patch.object(SpectrumView, "set_enabled"),
            patch.object(SpectrumView, "update_levels"),
            patch.object(SpectrumView, "show_repeat_dialog") as show_repeat,
            patch.object(SpectrumView, "hide_repeat_dialog"),
            patch.object(SpectrumSettingsWindow, "build"),
            patch.object(SpectrumSettingsWindow, "destroy"),
            patch(
                "spectrum_app.modules.spectrum.module.SpectrumAcquisition",
                side_effect=acquisitions,
            ),
        ):
            module.initialize(app)
            module.settings.online_welch = False
            module.settings.fade_in = 0.0
            module.settings.fade_out = 0.0
            module.activate(measurement)
            try:
                module.start_measurement()
                self.assertEqual(acquisitions[0].start.call_count, 1)

                module._receive_completion(
                    measurement,
                    recording_a,
                    generator,
                    np.empty(0),
                    np.empty((0, 0)),
                    None,
                )
                module.update()
                show_repeat.assert_called_once_with(1, 3)
                self.assertTrue(app.app_state.measuring)

                module.continue_multiple_measurement()
                self.assertEqual(acquisitions[1].start.call_count, 1)
                module._receive_completion(
                    measurement,
                    recording_b,
                    generator,
                    np.empty(0),
                    np.empty((0, 0)),
                    None,
                )
                module.update()
                show_repeat.assert_called_with(2, 3)

                module.break_multiple_measurement()
                deadline = time.monotonic() + 2.0
                while app.app_state.measuring and time.monotonic() < deadline:
                    module.update()
                    time.sleep(0.001)

                self.assertFalse(app.app_state.measuring)
                self.assertIsNone(measurement.module_state["recording"])
                self.assertEqual(len(measurement.module_state["recordings"]), 2)
                self.assertEqual(len(measurement.module_state["generators"]), 2)
                self.assertEqual(len(measurement.graphs), 1)
            finally:
                module.deactivate()
                module.shutdown()

    def test_measurement_pause_is_saved_in_application_settings(self) -> None:
        app = SpectrumApplication()
        settings = SpectrumSettings(app.settings)

        self.assertEqual(settings.measurement_pause, 5.0)
        settings.measurement_pause = 2.5
        self.assertEqual(settings.measurement_pause, 2.5)
        self.assertEqual(
            app.settings.module_setting("spectrum", "measurement_pause"),
            2.5,
        )

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
                multiple = next(
                    call
                    for call in backend.calls
                    if call[0] == "add_checkbox"
                    and call[1].get("tag") == SpectrumView.MULTIPLE
                )
                self.assertFalse(multiple[1]["default_value"])
                count = next(
                    call
                    for call in backend.calls
                    if call[0] == "add_input_int"
                    and call[1].get("tag") == SpectrumView.COUNT
                )
                self.assertEqual(count[1]["min_value"], 2)
                self.assertEqual(count[1]["max_value"], 100)
                repeat_dialog = next(
                    call
                    for call in backend.calls
                    if call[0] == "window"
                    and call[1].get("tag") == SpectrumView.REPEAT_DIALOG
                )
                self.assertTrue(repeat_dialog[1]["modal"])
                weighting_group = next(
                    call
                    for call in backend.calls
                    if call[0] == "group"
                    and call[1].get("tag") == SpectrumView.WEIGHTING_GROUP
                )
                self.assertFalse(weighting_group[1]["show"])
            finally:
                module.deactivate()
                module.shutdown()


if __name__ == "__main__":
    unittest.main()
