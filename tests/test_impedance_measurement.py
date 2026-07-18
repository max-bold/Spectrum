import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from threading import Event, Lock
from unittest.mock import patch

import numpy as np
import sounddevice as sd
from utils.windows import log_filter2

from spectrum_app.impedance.imp_measure import (
    CHANNEL_CALIBRATION_DURATION,
    CalibrationStage,
    ImpedanceAppState,
    MeasurementConfig,
    MeasurementState,
    PhaseDisplayMode,
    analyze_recording_levels,
    calculate_channel_correction,
    calculate_calibration_impedance,
    calculate_impedance,
    channel_calibration_frequencies,
    current_phase_angle,
    current_phase_derivative,
    estimate_reference_resistor,
    export_impedance_plot,
    fit_impedance,
    fit_impedance_auto,
    generate_channel_calibration_signal,
    generate_level_test_signal,
    generate_measurement_signal,
    impedance_axis_limits,
    phase_axis_limits,
    play_and_record,
    phase_plot_data,
    require_valid_reference_calibration,
    resolve_sample_rate,
    speaker_impedance,
    validate_channel_similarity,
)


class ImpedanceMathTests(unittest.TestCase):
    def test_default_measurement_duration_is_twenty_seconds(self) -> None:
        self.assertEqual(MeasurementConfig().duration, 20.0)

    def test_input_level_error_identifies_channel_without_signal(self) -> None:
        recording = np.column_stack((np.zeros(100), np.full(100, 0.1)))

        with self.assertRaisesRegex(
            ValueError,
            r"Channel 1 \(L\): no signal",
        ):
            analyze_recording_levels(recording)

    def test_input_level_error_distinguishes_quiet_signal(self) -> None:
        recording = np.column_stack(
            (np.full(100, 0.1), np.full(100, 5e-5))
        )

        with self.assertRaisesRegex(
            ValueError,
            r"Channel 2 \(R\): signal is too quiet",
        ):
            analyze_recording_levels(recording)

    def test_input_level_error_reports_clipping_per_channel(self) -> None:
        recording = np.column_stack((np.full(100, 0.1), np.ones(100)))

        with self.assertRaisesRegex(
            ValueError,
            r"Channel 2 \(R\): clipping detected",
        ):
            analyze_recording_levels(recording, raise_on_clipping=True)

    def test_input_level_error_reports_different_channel_failures(self) -> None:
        recording = np.column_stack((np.zeros(100), np.ones(100)))

        with self.assertRaises(ValueError) as caught:
            analyze_recording_levels(recording, raise_on_clipping=True)

        message = str(caught.exception)
        self.assertIn("Channel 1 (L): no signal", message)
        self.assertIn("Channel 2 (R): clipping detected", message)

    def test_calibration_impedance_preserves_complex_response(self) -> None:
        reference_by_frequency = np.array(
            [3.0 + 0.0j, 3.0 + 0.3j, np.nan + 1j * np.nan]
        )

        impedance = calculate_calibration_impedance(
            reference_by_frequency,
            reference_resistor=3.0,
            calibration_resistor=10.0,
        )

        np.testing.assert_allclose(
            impedance[:2],
            30.0 / reference_by_frequency[:2],
        )
        self.assertTrue(np.isnan(impedance[2]))

    def test_impedance_axis_always_includes_zero(self) -> None:
        lower, upper = impedance_axis_limits(
            np.array([6.0, 8.0, 20.0, np.nan])
        )
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(upper, 21.0)

    def test_phase_axis_limits_include_data_with_padding(self) -> None:
        lower, upper = phase_axis_limits(
            np.array([-100.0, 20.0, 200.0, np.nan])
        )

        self.assertAlmostEqual(lower, -115.0)
        self.assertAlmostEqual(upper, 215.0)

    def test_current_phase_is_unwrapped(self) -> None:
        expected = np.linspace(-270.0, 270.0, 128)
        impedance = np.exp(-1j * np.deg2rad(expected))
        phase = current_phase_angle(impedance)
        np.testing.assert_allclose(
            np.diff(phase),
            np.diff(expected),
            atol=1e-10,
        )

    def test_phase_derivative_uses_log_frequency(self) -> None:
        frequency = np.geomspace(20.0, 20000.0, 256)
        expected_derivative = 45.0
        current_phase = 10.0 + expected_derivative * np.log10(frequency)
        impedance = np.exp(-1j * np.deg2rad(current_phase))
        derivative = current_phase_derivative(frequency, impedance)
        np.testing.assert_allclose(
            derivative[16:-16],
            expected_derivative,
            atol=1e-6,
        )
        displayed, label, series = phase_plot_data(
            frequency,
            impedance,
            PhaseDisplayMode.DERIVATIVE,
        )
        np.testing.assert_allclose(displayed, derivative)
        self.assertIn("deg/decade", label)
        self.assertEqual(series, "Phase derivative")

    def test_multitone_correction_recovers_gain_and_delay(self) -> None:
        config = MeasurementConfig(
            sample_rate=48000,
            duration=1.0,
            f_min=20.0,
            f_max=20000.0,
            points=128,
        )
        channel_1 = generate_channel_calibration_signal(config).astype(float)
        delay_samples = 37
        gain = 0.35
        channel_2 = (
            np.pad(channel_1[:-delay_samples], (delay_samples, 0)) * gain
        )
        frequency, correction = calculate_channel_correction(
            channel_1,
            channel_2,
            config,
        )
        expected = gain * np.exp(
            -2j
            * np.pi
            * frequency
            * delay_samples
            / config.sample_rate
        )
        np.testing.assert_allclose(correction, expected, rtol=1e-4, atol=1e-5)

    def test_channel_calibration_signal_uses_fixed_duration(self) -> None:
        config = MeasurementConfig(
            sample_rate=48000,
            duration=0.1,
            f_min=100.0,
            f_max=10000.0,
            points=128,
        )
        signal = generate_channel_calibration_signal(config)
        self.assertEqual(
            len(signal),
            int(config.sample_rate * CHANNEL_CALIBRATION_DURATION),
        )

    def test_level_test_signal_is_loopable_multitone(self) -> None:
        config = MeasurementConfig(
            sample_rate=48000,
            duration=0.1,
            f_min=20.0,
            f_max=20000.0,
        )
        signal = generate_level_test_signal(config)
        self.assertEqual(len(signal), config.sample_rate)
        self.assertLessEqual(float(np.max(np.abs(signal))), 0.9)
        self.assertGreater(float(np.max(np.abs(signal))), 0.1)

    def test_multitone_rejects_different_rms_profiles(self) -> None:
        config = MeasurementConfig(
            sample_rate=48000,
            duration=1.0,
            f_min=20.0,
            f_max=20000.0,
            points=128,
        )
        frequencies = channel_calibration_frequencies(config)
        time = np.arange(config.sample_rate, dtype=float) / config.sample_rate
        phases = (
            np.pi
            * np.arange(frequencies.size)
            * (np.arange(frequencies.size) - 1)
            / frequencies.size
        )
        channel_1 = np.sum(
            np.cos(
                2 * np.pi * frequencies[:, None] * time + phases[:, None]
            ),
            axis=0,
        )
        levels = np.geomspace(0.1, 2.0, frequencies.size)
        channel_2 = np.sum(
            levels[:, None]
            * np.cos(
                2 * np.pi * frequencies[:, None] * time + phases[:, None]
            ),
            axis=0,
        )
        channel_1 *= 0.5 / np.max(np.abs(channel_1))
        channel_2 *= 0.5 / np.max(np.abs(channel_2))
        with self.assertRaisesRegex(ValueError, "level profiles"):
            calculate_channel_correction(channel_1, channel_2, config)

    def test_channel_similarity_allows_gain_polarity_and_delay(self) -> None:
        sample_rate = 48000
        signal = np.random.default_rng(3).normal(0.0, 0.2, sample_rate)
        delayed = np.pad(signal[:-37], (37, 0)) * -0.35
        similarity = validate_channel_similarity(
            signal,
            delayed,
            sample_rate,
        )
        self.assertGreater(similarity, 0.99)

    def test_channel_similarity_rejects_different_signals(self) -> None:
        sample_rate = 48000
        generator = np.random.default_rng(4)
        channel_1 = generator.normal(0.0, 0.2, sample_rate)
        channel_2 = generator.normal(0.0, 0.2, sample_rate)
        with self.assertRaisesRegex(ValueError, "contain different signals"):
            validate_channel_similarity(
                channel_1,
                channel_2,
                sample_rate,
            )

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
        channel_calibration = generate_channel_calibration_signal(config)
        _, channel_correction = calculate_channel_correction(
            channel_calibration,
            channel_calibration * channel_gain,
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

    def test_impedance_recovers_frequency_dependent_load_from_sweep(self) -> None:
        config = MeasurementConfig(
            sample_rate=48000,
            duration=1.0,
            reference_resistor=3.0,
            calibration_resistor=9.8,
            f_min=50.0,
            f_max=15000.0,
            window_width=0.05,
            points=128,
        )
        channel_1 = generate_measurement_signal(config).astype(float)
        fft_frequency = np.fft.rfftfreq(
            channel_1.size,
            1.0 / config.sample_rate,
        )
        source = np.fft.rfft(channel_1 - np.mean(channel_1))
        expected_fft_impedance = (
            8.0 + 1j * 2.0 * np.pi * fft_frequency * 0.0004
        )
        transfer = expected_fft_impedance / (
            config.reference_resistor + expected_fft_impedance
        )
        channel_2 = np.fft.irfft(source * transfer, n=channel_1.size)
        frequency, impedance = calculate_impedance(
            channel_1,
            channel_2,
            config,
            np.ones(config.points, dtype=np.complex128),
            config.reference_resistor,
        )
        expected = 8.0 + 1j * 2.0 * np.pi * frequency * 0.0004
        relative_error = np.abs((impedance - expected) / expected)
        self.assertLess(float(np.nanmedian(relative_error)), 1e-3)

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
        channel_calibration = generate_channel_calibration_signal(config)
        _, correction = calculate_channel_correction(
            channel_calibration,
            channel_calibration,
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

    def test_wrong_entered_rc_rr_ratio_fails_with_measured_ratio(self) -> None:
        config = MeasurementConfig(
            sample_rate=48000,
            duration=1.0,
            reference_resistor=10.0,
            calibration_resistor=20.0,
            f_min=20.0,
            f_max=20000.0,
            points=128,
        )
        actual_reference_resistor = 15.0
        signal = np.random.default_rng(3).normal(0.0, 0.2, 48000)
        transfer = config.calibration_resistor / (
            actual_reference_resistor + config.calibration_resistor
        )

        with self.assertWarns(RuntimeWarning):
            _, _, _, diagnostics = estimate_reference_resistor(
                signal,
                signal * transfer,
                config,
                np.ones(config.points, dtype=np.complex128),
            )

        with self.assertRaises(ValueError) as caught:
            require_valid_reference_calibration(diagnostics)

        message = str(caught.exception)
        self.assertIn("Entered Rc/Rr: 2", message)
        self.assertIn("measured Rc/Rr: 1.333", message)
        self.assertIn("difference: 33.3%", message)

    def test_five_percent_rc_rr_ratio_error_is_allowed(self) -> None:
        config = MeasurementConfig(
            sample_rate=48000,
            duration=1.0,
            reference_resistor=10.0,
            calibration_resistor=20.0,
            f_min=20.0,
            f_max=20000.0,
            points=128,
        )
        entered_ratio = config.calibration_resistor / config.reference_resistor
        measured_ratio = entered_ratio * 1.05
        actual_reference = config.calibration_resistor / measured_ratio
        signal = np.random.default_rng(4).normal(0.0, 0.2, 48000)
        transfer = config.calibration_resistor / (
            actual_reference + config.calibration_resistor
        )

        _, _, _, diagnostics = estimate_reference_resistor(
            signal,
            signal * transfer,
            config,
            np.ones(config.points, dtype=np.complex128),
        )

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

    def test_calibration_publishes_calibration_resistor_impedance(self) -> None:
        self.complete_calibration()

        snapshot = self.state.snapshot()

        self.assertIsNotNone(snapshot.frequency)
        self.assertIsNotNone(snapshot.impedance)
        self.assertAlmostEqual(
            float(np.median(np.abs(snapshot.impedance))),
            self.calibration_resistor,
            places=5,
        )
        self.assertIn("showing measured Rcal", snapshot.status)
        self.assertFalse(snapshot.modeling)
        self.assertIsNone(snapshot.spice_values)

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
        self.assertIsNone(snapshot.spice_values)
        self.assertTrue(self.state.request_spice_model())
        self.state.wait(10)
        self.assertIsNotNone(self.state.snapshot().spice_values)
        self.assertEqual(snapshot.levels, (0.0, 0.0))

    def test_measurement_can_be_stopped(self) -> None:
        self.complete_calibration()
        recording_started = Event()
        release_recording = Event()
        original_recorder = self.state._recorder

        def blocking_recorder(signal, config, level_callback):
            recording_started.set()
            release_recording.wait(5)
            return original_recorder(signal, config, level_callback)

        self.state._recorder = blocking_recorder
        self.assertTrue(self.state.start_measurement(self.config))
        self.assertTrue(recording_started.wait(5))

        self.assertTrue(self.state.stop_measurement())
        self.assertIn("Stopping", self.state.snapshot().status)
        release_recording.set()
        self.state.wait(10)

        snapshot = self.state.snapshot()
        self.assertEqual(snapshot.state, MeasurementState.CALIBRATED)
        self.assertEqual(snapshot.status, "Measurement stopped")
        self.assertIsNotNone(snapshot.frequency)
        self.assertIsNotNone(snapshot.impedance)
        self.assertFalse(self.state.stop_measurement())

    def test_spice_model_is_calculated_only_when_requested(self) -> None:
        self.complete_calibration()
        fit_started = Event()
        release_fit = Event()

        def delayed_fit(*args, **kwargs):
            fit_started.set()
            release_fit.wait(5)
            return fit_impedance_auto(*args, **kwargs)

        with patch(
            "spectrum_app.impedance.imp_measure.fit_impedance_auto",
            side_effect=delayed_fit,
        ):
            self.assertTrue(self.state.start_measurement(self.config))
            self.state.wait(10)
            snapshot = self.state.snapshot()
            self.assertIsNotNone(snapshot.frequency)
            self.assertFalse(snapshot.modeling)
            self.assertIsNone(snapshot.spice_values)

            self.assertTrue(self.state.request_spice_model())
            self.assertTrue(fit_started.wait(5))

            snapshot = self.state.snapshot()
            self.assertEqual(
                snapshot.state,
                MeasurementState.MEASURING_COMPLETED,
            )
            self.assertIsNotNone(snapshot.frequency)
            self.assertIsNotNone(snapshot.impedance)
            self.assertTrue(snapshot.modeling)
            self.assertIsNone(snapshot.spice_values)

            release_fit.set()
            self.state.wait(10)

        snapshot = self.state.snapshot()
        self.assertFalse(snapshot.modeling)
        self.assertIsNotNone(snapshot.spice_values)

    def test_new_measurement_discards_previous_spice_model(self) -> None:
        self.complete_calibration()
        self.assertTrue(self.state.start_measurement(self.config))
        self.state.wait(10)
        self.assertTrue(self.state.request_spice_model())
        self.state.wait(10)
        self.assertIsNotNone(self.state.snapshot().spice_values)

        self.assertTrue(self.state.start_measurement(self.config))
        snapshot = self.state.snapshot()

        self.assertEqual(snapshot.state, MeasurementState.MEASURING)
        self.assertIsNone(snapshot.spice_values)
        self.state.wait(10)

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

    def test_different_first_stage_signals_fail_calibration(self) -> None:
        generator = np.random.default_rng(5)

        def recorder(signal, config, level_callback):
            recording = np.column_stack(
                (
                    generator.normal(0.0, 0.2, len(signal)),
                    generator.normal(0.0, 0.2, len(signal)),
                )
            ).astype(np.float32)
            level_callback((0.5, 0.5))
            return recording

        state = ImpedanceAppState(recorder=recorder)
        self.assertTrue(state.start_calibration(self.config))
        state.wait(10)
        snapshot = state.snapshot()
        self.assertEqual(snapshot.state, MeasurementState.UNCALIBRATED)
        self.assertEqual(snapshot.calibration_stage, CalibrationStage.IDLE)
        self.assertIn("not the generated multitone signal", snapshot.error)

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
