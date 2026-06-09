import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from threading import Event, Lock
from unittest.mock import patch

import numpy as np
import sounddevice as sd
from utils.windows import log_filter2

from spectrum_app.impedance.imp_measure import (
    CalibrationStage,
    ImpedanceAppState,
    MeasurementConfig,
    MeasurementState,
    calculate_channel_correction,
    calculate_impedance,
    estimate_reference_resistor,
    export_impedance_plot,
    fit_impedance,
    generate_measurement_signal,
    play_and_record,
    require_valid_reference_calibration,
    resolve_sample_rate,
    speaker_impedance,
)


class ImpedanceMathTests(unittest.TestCase):
    def test_log_filter_preserves_complex_components(self) -> None:
        frequency = np.linspace(0.0, 24000.0, 513)
        values = (
            np.linspace(1.0, 2.0, 513)
            + 1j * np.linspace(-0.5, 0.75, 513)
        )
        output_frequency, filtered = log_filter2(
            frequency,
            values,
            band=(20.0, 20000.0),
            n_output=64,
        )
        _, filtered_real = log_filter2(
            frequency,
            values.real,
            band=(20.0, 20000.0),
            n_output=64,
        )
        _, filtered_imag = log_filter2(
            frequency,
            values.imag,
            band=(20.0, 20000.0),
            n_output=64,
        )
        self.assertEqual(output_frequency.size, 64)
        self.assertTrue(np.iscomplexobj(filtered))
        np.testing.assert_allclose(
            filtered,
            filtered_real + 1j * filtered_imag,
            equal_nan=True,
        )

    def test_calibration_recovers_known_resistance(self) -> None:
        sample_rate = 48000
        reference_resistor = 10.0
        calibration_resistor = 20.0
        load_resistance = 8.0
        config = MeasurementConfig(
            sample_rate=sample_rate,
            duration=1.0,
            reference_resistor=reference_resistor,
            calibration_resistor=calibration_resistor,
            f_min=20.0,
            f_max=20000.0,
            points=128,
        )
        generator = np.random.default_rng(1)
        channel_1 = generator.normal(0.0, 0.2, sample_rate)
        channel_gain = 0.8
        _, channel_correction = calculate_channel_correction(
            channel_1,
            channel_1 * channel_gain,
            config,
        )
        calibration_ratio = calibration_resistor / (
            reference_resistor + calibration_resistor
        )
        _, _, estimated_reference, diagnostics = estimate_reference_resistor(
            channel_1,
            channel_1 * calibration_ratio * channel_gain,
            config,
            channel_correction,
        )
        require_valid_reference_calibration(diagnostics)
        self.assertAlmostEqual(estimated_reference, reference_resistor, places=5)
        load_ratio = load_resistance / (
            reference_resistor + load_resistance
        )
        _, impedance = calculate_impedance(
            channel_1,
            channel_1 * load_ratio * channel_gain,
            config,
            channel_correction,
            estimated_reference,
        )
        self.assertAlmostEqual(
            float(np.median(np.abs(impedance))),
            load_resistance,
            places=5,
        )

    def test_same_signal_cannot_pass_resistor_calibration(self) -> None:
        config = MeasurementConfig(
            sample_rate=48000,
            duration=1.0,
            reference_resistor=10.0,
            calibration_resistor=20.0,
            f_min=20.0,
            f_max=20000.0,
            points=128,
        )
        signal = np.random.default_rng(2).normal(0.0, 0.2, 48000)
        _, correction = calculate_channel_correction(
            signal,
            signal,
            config,
        )
        with self.assertWarns(RuntimeWarning):
            _, _, _, diagnostics = estimate_reference_resistor(
                signal,
                signal,
                config,
                correction,
            )
        with self.assertRaisesRegex(ValueError, "resistor network is invalid"):
            require_valid_reference_calibration(diagnostics)

    def test_fit_series_rl_model(self) -> None:
        frequency = np.geomspace(20.0, 20000.0, 128)
        expected = np.array([6.8, 0.0004])
        measured = np.abs(speaker_impedance(frequency, expected, 0))
        result = fit_impedance(
            frequency,
            measured,
            sections=0,
            max_evaluations=300,
        )
        np.testing.assert_allclose(
            result.physical_params,
            expected,
            rtol=1e-3,
        )

    def test_export_impedance_plot(self) -> None:
        frequency = np.geomspace(20.0, 20000.0, 64)
        impedance = 8.0 + 1j * 2.0 * np.pi * frequency * 0.0003
        with TemporaryDirectory() as directory:
            path = export_impedance_plot(
                Path(directory) / "impedance",
                frequency,
                impedance,
            )
            self.assertEqual(path.suffix, ".png")
            self.assertGreater(path.stat().st_size, 10000)

    def test_streaming_recording_reports_block_levels(self) -> None:
        signal = np.linspace(-0.5, 0.5, 10, dtype=np.float32)
        config = MeasurementConfig(
            sample_rate=48000,
            duration=len(signal) / 48000,
            f_min=20.0,
            f_max=20000.0,
            block_size=4,
            recording_tail=0.0,
        )
        published_levels: list[tuple[float, float]] = []

        streams = {}

        class FakeInputStream:
            def __init__(
                self,
                *,
                blocksize,
                callback,
                finished_callback,
                **kwargs,
            ) -> None:
                self.blocksize = blocksize
                self.callback = callback
                self.finished_callback = finished_callback
                streams["input"] = self

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback) -> None:
                return None

        class FakeOutputStream:
            def __init__(
                self,
                *,
                blocksize,
                callback,
                finished_callback,
                **kwargs,
            ) -> None:
                self.blocksize = blocksize
                self.callback = callback
                self.finished_callback = finished_callback

            def __enter__(self):
                input_stream = streams["input"]
                block_number = 1
                while True:
                    left = np.full(
                        self.blocksize,
                        block_number / 10,
                        dtype=np.float32,
                    )
                    right = np.full(
                        self.blocksize,
                        block_number / 20,
                        dtype=np.float32,
                    )
                    indata = np.column_stack((left, right))
                    outdata = np.empty_like(indata)
                    try:
                        input_stream.callback(
                            indata,
                            input_stream.blocksize,
                            None,
                            None,
                        )
                    except sd.CallbackStop:
                        input_stream.finished_callback()
                    try:
                        self.callback(
                            outdata,
                            self.blocksize,
                            None,
                            None,
                        )
                    except sd.CallbackStop:
                        self.finished_callback()
                        break
                    block_number += 1
                return self

            def __exit__(self, exc_type, exc_value, traceback) -> None:
                return None

        with (
            patch(
                "spectrum_app.impedance.imp_measure.sd.InputStream",
                FakeInputStream,
            ),
            patch(
                "spectrum_app.impedance.imp_measure.sd.OutputStream",
                FakeOutputStream,
            ),
            patch(
                "spectrum_app.impedance.imp_measure.sd.check_input_settings",
            ),
            patch(
                "spectrum_app.impedance.imp_measure.sd.check_output_settings",
            ),
        ):
            recording = play_and_record(
                signal,
                config,
                published_levels.append,
            )

        self.assertEqual(recording.shape, (10, 2))
        self.assertEqual(len(published_levels), 3)
        np.testing.assert_allclose(
            published_levels,
            ((0.1, 0.05), (0.2, 0.1), (0.3, 0.15)),
            rtol=1e-6,
        )
        np.testing.assert_allclose(recording[-2:, 0], 0.3)

    def test_generator_does_not_depend_on_filter_settings(self) -> None:
        first = MeasurementConfig(
            sample_rate=48000,
            duration=0.1,
            f_min=100.0,
            f_max=10000.0,
            window_width=0.1,
            points=64,
        )
        second = MeasurementConfig(
            **{
                **first.__dict__,
                "window_width": 1.0,
                "points": 256,
            }
        )
        np.testing.assert_array_equal(
            generate_measurement_signal(first),
            generate_measurement_signal(second),
        )

    @patch("spectrum_app.impedance.imp_measure.sd.query_devices")
    def test_sample_rate_comes_from_audio_devices(
        self,
        query_devices,
    ) -> None:
        query_devices.side_effect = (
            {"default_samplerate": 48000.0},
            {"default_samplerate": 48000.0},
        )
        self.assertEqual(resolve_sample_rate(), 48000)

    @patch("spectrum_app.impedance.imp_measure.sd.query_devices")
    def test_different_default_sample_rates_are_rejected(
        self,
        query_devices,
    ) -> None:
        query_devices.side_effect = (
            {"default_samplerate": 48000.0},
            {"default_samplerate": 44100.0},
        )
        with self.assertRaisesRegex(ValueError, "default sample rates differ"):
            resolve_sample_rate()


class ImpedanceStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls = 0
        self.reference_resistor = 10.0
        self.calibration_resistor = 20.0
        self.load_resistance = 8.0

        def recorder(
            signal: np.ndarray,
            config: MeasurementConfig,
            level_callback,
        ) -> np.ndarray:
            if self.calls == 0:
                ratio = 1.0
            elif self.calls == 1:
                ratio = self.calibration_resistor / (
                    self.reference_resistor + self.calibration_resistor
                )
            else:
                ratio = self.load_resistance / (
                    self.reference_resistor + self.load_resistance
                )
            self.calls += 1
            recording = np.column_stack(
                (signal, signal * ratio)
            ).astype(np.float32)
            for start in range(0, len(recording), config.block_size):
                block = recording[start : start + config.block_size]
                level_callback(
                    (
                        float(np.max(np.abs(block[:, 0]))),
                        float(np.max(np.abs(block[:, 1]))),
                    )
                )
            return recording

        self.state = ImpedanceAppState(recorder=recorder)
        self.config = MeasurementConfig(
            sample_rate=48000,
            duration=0.1,
            reference_resistor=self.reference_resistor,
            calibration_resistor=self.calibration_resistor,
            f_min=100.0,
            f_max=10000.0,
            points=64,
            recording_tail=0.0,
            spice_min_sections=0,
            spice_max_sections=0,
            spice_max_evaluations=200,
        )

    def complete_calibration(self) -> None:
        self.assertTrue(self.state.start_calibration(self.config))
        self.state.wait(10)
        snapshot = self.state.snapshot()
        self.assertEqual(snapshot.state, MeasurementState.CALIBRATING)
        self.assertEqual(
            snapshot.calibration_stage,
            CalibrationStage.WAITING_REFERENCE,
        )
        self.assertTrue(self.state.continue_calibration())
        self.state.wait(10)
        self.assertEqual(
            self.state.snapshot().state,
            MeasurementState.CALIBRATED,
        )

    def test_state_workflow(self) -> None:
        self.assertEqual(
            self.state.snapshot().state,
            MeasurementState.UNCALIBRATED,
        )
        self.assertFalse(self.state.start_measurement(self.config))
        self.complete_calibration()
        self.assertTrue(self.state.start_measurement(self.config))
        self.assertEqual(
            self.state.snapshot().state,
            MeasurementState.MEASURING,
        )
        self.state.wait(10)
        snapshot = self.state.snapshot()
        self.assertEqual(
            snapshot.state,
            MeasurementState.MEASURING_COMPLETED,
        )
        self.assertAlmostEqual(
            float(np.median(np.abs(snapshot.impedance))),
            self.load_resistance,
            places=5,
        )
        self.assertIsNotNone(snapshot.spice_values)
        self.assertEqual(snapshot.levels, (0.0, 0.0))

    def test_settings_change_requires_recalibration(self) -> None:
        self.complete_calibration()
        changed = MeasurementConfig(
            **{
                **self.config.__dict__,
                "duration": self.config.duration * 2,
            }
        )
        with self.assertRaisesRegex(ValueError, "recalibrate"):
            self.state.start_measurement(changed)

    def test_acoustic_loopback_does_not_complete_calibration(self) -> None:
        def recorder(signal, config, level_callback):
            recording = np.column_stack((signal, signal)).astype(np.float32)
            level_callback((0.5, 0.5))
            return recording

        state = ImpedanceAppState(recorder=recorder)
        self.assertTrue(state.start_calibration(self.config))
        state.wait(10)
        self.assertEqual(
            state.snapshot().calibration_stage,
            CalibrationStage.WAITING_REFERENCE,
        )
        self.assertTrue(state.continue_calibration())
        with self.assertWarns(RuntimeWarning):
            state.wait(10)
        snapshot = state.snapshot()
        self.assertEqual(snapshot.state, MeasurementState.UNCALIBRATED)
        self.assertIn("resistor network is invalid", snapshot.error)

    def test_invalidating_calibration_clears_measurement(self) -> None:
        self.complete_calibration()
        self.state.start_measurement(self.config)
        self.state.wait(10)

        self.assertTrue(self.state.invalidate_calibration("IO changed"))
        snapshot = self.state.snapshot()

        self.assertEqual(snapshot.state, MeasurementState.UNCALIBRATED)
        self.assertEqual(snapshot.status, "IO changed")
        self.assertIsNone(snapshot.frequency)
        self.assertIsNone(snapshot.impedance)
        self.assertIsNone(snapshot.spice_values)

    def test_filter_settings_reprocess_raw_recordings(self) -> None:
        self.complete_calibration()
        self.state.start_measurement(self.config)
        self.state.wait(10)
        self.assertEqual(self.calls, 3)

        changed = MeasurementConfig(
            **{
                **self.config.__dict__,
                "points": self.config.points * 2,
                "window_width": 0.2,
            }
        )
        self.assertTrue(self.state.request_reprocess(changed))
        self.assertTrue(self.state.snapshot().processing)
        self.state.wait(10)
        snapshot = self.state.snapshot()

        self.assertEqual(
            snapshot.state,
            MeasurementState.MEASURING_COMPLETED,
        )
        self.assertFalse(snapshot.processing)
        self.assertEqual(len(snapshot.frequency), changed.points)
        self.assertEqual(self.calls, 3)
        self.assertAlmostEqual(
            float(np.median(np.abs(snapshot.impedance))),
            self.load_resistance,
            places=5,
        )
        self.assertFalse(self.state.request_reprocess(changed))

    def test_reprocess_uses_latest_filter_settings(self) -> None:
        self.complete_calibration()
        self.state.start_measurement(self.config)
        self.state.wait(10)

        first_started = Event()
        release_first = Event()
        calls_lock = Lock()
        processed_points: list[int] = []
        original = self.state._process_recordings

        def delayed_process(
            channel_calibration,
            calibration,
            measurement,
            config,
        ):
            with calls_lock:
                processed_points.append(config.points)
                call_number = len(processed_points)
            if call_number == 1:
                first_started.set()
                release_first.wait(5)
            return original(
                channel_calibration,
                calibration,
                measurement,
                config,
            )

        first = MeasurementConfig(
            **{**self.config.__dict__, "points": 96}
        )
        latest = MeasurementConfig(
            **{**self.config.__dict__, "points": 128}
        )
        with patch.object(
            self.state,
            "_process_recordings",
            side_effect=delayed_process,
        ):
            self.assertTrue(self.state.request_reprocess(first))
            self.assertTrue(first_started.wait(5))
            self.assertTrue(self.state.request_reprocess(latest))
            release_first.set()
            self.state.wait(10)

        snapshot = self.state.snapshot()
        self.assertFalse(snapshot.processing)
        self.assertEqual(processed_points, [96, 128])
        self.assertEqual(len(snapshot.frequency), latest.points)
        self.assertEqual(self.calls, 3)

    def test_levels_are_published_while_calibrating(self) -> None:
        level_published = Event()
        release_recorder = Event()
        calls = 0

        def recorder(signal, config, level_callback):
            nonlocal calls
            level_callback((0.4, 0.2))
            level_published.set()
            release_recorder.wait(5)
            if calls == 0:
                ratio = 1.0
            else:
                ratio = self.calibration_resistor / (
                    self.reference_resistor + self.calibration_resistor
                )
            calls += 1
            return np.column_stack((signal, signal * ratio)).astype(np.float32)

        state = ImpedanceAppState(recorder=recorder)
        self.assertTrue(state.start_calibration(self.config))
        self.assertTrue(level_published.wait(5))
        snapshot = state.snapshot()
        self.assertEqual(snapshot.state, MeasurementState.CALIBRATING)
        self.assertEqual(snapshot.levels, (0.4, 0.2))
        release_recorder.set()
        state.wait(10)
        self.assertEqual(
            state.snapshot().calibration_stage,
            CalibrationStage.WAITING_REFERENCE,
        )
        self.assertTrue(state.continue_calibration())
        state.wait(10)
        self.assertEqual(
            state.snapshot().state,
            MeasurementState.CALIBRATED,
        )
        self.assertEqual(state.snapshot().levels, (0.0, 0.0))

    def test_level_updates_are_smoothed_and_rate_limited(self) -> None:
        with patch(
            "spectrum_app.impedance.imp_measure.monotonic",
            side_effect=(0.0, 0.02, 0.09),
        ):
            self.state._update_levels((0.4, 0.2))
            first = self.state.snapshot()
            self.state._update_levels((0.8, 0.6))
            throttled = self.state.snapshot()
            self.state._update_levels((0.6, 0.4))
            smoothed = self.state.snapshot()

        self.assertEqual(first.levels, (0.4, 0.2))
        self.assertEqual(throttled.revision, first.revision)
        self.assertEqual(throttled.levels, first.levels)
        self.assertEqual(smoothed.revision, first.revision + 1)
        np.testing.assert_allclose(smoothed.levels, (0.54, 0.34))


if __name__ == "__main__":
    unittest.main()
