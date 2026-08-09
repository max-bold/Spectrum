import unittest

import numpy as np

from audioanalysis import (
    ASignal,
    ChannelCalibration,
    FrequencyBand,
    ImpedanceConfig,
    analyze_recording_levels,
    calculate_channel_correction,
    calculate_impedance,
    channel_calibration_config,
    fit_impedance,
    generate_channel_calibration_signal,
    generate_level_test_signal,
    generate_measurement_signal,
    interpolate_channel_calibration,
    require_valid_reference_calibration,
    speaker_impedance,
)
from audioanalysis.impedance import estimate_reference_resistor


class ImpedanceMathTests(unittest.TestCase):
    def test_channel_calibration_can_be_interpolated_without_raw_audio(self) -> None:
        source_frequency = np.geomspace(20.0, 20_000.0, 32)
        source = np.power(10.0, 0.5 / 20.0) * np.exp(
            -1j * 2.0 * np.pi * source_frequency * 20e-6
        )
        target_frequency = np.geomspace(20.0, 20_000.0, 128)

        result = interpolate_channel_calibration(
            ChannelCalibration(source_frequency, source),
            target_frequency,
        )

        self.assertEqual(result.correction.shape, (128,))
        np.testing.assert_allclose(np.abs(result.correction), np.abs(source[0]))
        np.testing.assert_allclose(
            np.unwrap(np.angle(result.correction)),
            -2.0 * np.pi * target_frequency * 20e-6,
            atol=1e-12,
        )

    def test_invalid_reference_calibration_has_readable_multiline_details(self) -> None:
        diagnostics = {
            "fatal_warnings": [
                "estimated Rref is not positive",
                "not enough resistive frequency points",
            ],
            "rr_estimated": -0.0002724,
            "rr_nominal": 3.25,
            "rr_real_cv": 1.102,
            "rr_imag_to_real_ratio": 0.865,
            "rc_rr_entered": 3.2,
            "rc_rr_measured": float("inf"),
            "rc_rr_error_rel": float("inf"),
            "valid_points_count": 110,
            "resistive_points_count": 0,
        }

        with self.assertRaises(ValueError) as raised:
            require_valid_reference_calibration(diagnostics)

        message = str(raised.exception)
        self.assertIn("Problems:\n- Estimated Rref is not positive", message)
        self.assertIn("\n\nMeasured values:\n", message)
        self.assertIn("Frequency points: 110 valid, 0 resistive", message)

    def test_generators_return_asignal(self) -> None:
        config = ImpedanceConfig(
            sample_rate=8_000,
            duration=0.1,
            band=FrequencyBand(20, 3_000),
        )

        self.assertIsInstance(generate_measurement_signal(config), ASignal)
        self.assertIsInstance(generate_channel_calibration_signal(config), ASignal)
        self.assertIsInstance(generate_level_test_signal(config), ASignal)

    def test_input_level_error_identifies_channel(self) -> None:
        recording = ASignal(
            np.column_stack((np.zeros(100), np.full(100, 0.1))),
            48_000,
        )

        with self.assertRaisesRegex(ValueError, r"Channel 1 \(L\): no signal"):
            analyze_recording_levels(recording)

    def test_multitone_correction_recovers_gain_and_delay(self) -> None:
        config = ImpedanceConfig(
            sample_rate=8_000,
            duration=0.1,
            band=FrequencyBand(20, 3_000),
            points=128,
        )
        signal = generate_channel_calibration_signal(config).as_array()[:, 0]
        delay_samples = 17
        gain = 0.35
        delayed = np.pad(signal[:-delay_samples], (delay_samples, 0)) * gain
        calibration = calculate_channel_correction(
            ASignal(np.column_stack((signal, delayed)), config.sample_rate),
            channel_calibration_config(config),
        )
        expected = gain * np.exp(
            -2j
            * np.pi
            * calibration.frequency
            * delay_samples
            / config.sample_rate
        )

        np.testing.assert_allclose(
            calibration.correction,
            expected,
            rtol=1e-3,
            atol=1e-4,
        )

    def test_calibration_and_measurement_recover_known_resistance(self) -> None:
        config = ImpedanceConfig(
            sample_rate=8_000,
            duration=1.0,
            reference_resistor=10.0,
            calibration_resistor=20.0,
            band=FrequencyBand(20, 3_000),
            points=128,
        )
        source = np.random.default_rng(1).normal(0.0, 0.2, config.sample_rate)
        channel_signal = generate_channel_calibration_signal(config).as_array()[:, 0]
        channel_gain = 0.8
        channel_calibration = calculate_channel_correction(
            ASignal(
                np.column_stack((channel_signal, channel_signal * channel_gain)),
                config.sample_rate,
            ),
            channel_calibration_config(config),
        )
        calibration_ratio = config.calibration_resistor / (
            config.reference_resistor + config.calibration_resistor
        )
        reference = estimate_reference_resistor(
            ASignal(
                np.column_stack(
                    (source, source * calibration_ratio * channel_gain)
                ),
                config.sample_rate,
            ),
            config,
            channel_calibration.correction,
        )
        require_valid_reference_calibration(reference.diagnostics)

        load_resistance = 8.0
        load_ratio = load_resistance / (
            config.reference_resistor + load_resistance
        )
        result = calculate_impedance(
            ASignal(
                np.column_stack((source, source * load_ratio * channel_gain)),
                config.sample_rate,
            ),
            config,
            channel_calibration.correction,
            reference.reference_resistor,
        )

        self.assertAlmostEqual(reference.reference_resistor, 10.0, places=4)
        self.assertAlmostEqual(
            float(np.nanmedian(result.magnitude)),
            load_resistance,
            places=4,
        )

    def test_impedance_recovers_frequency_dependent_load(self) -> None:
        config = ImpedanceConfig(
            sample_rate=8_000,
            duration=1.0,
            reference_resistor=3.0,
            calibration_resistor=9.8,
            band=FrequencyBand(50, 3_000),
            window_width=0.05,
            points=128,
        )
        source = generate_measurement_signal(config).as_array()[:, 0].astype(float)
        fft_frequency = np.fft.rfftfreq(source.size, 1.0 / config.sample_rate)
        expected_fft = 8.0 + 1j * 2.0 * np.pi * fft_frequency * 0.0004
        transfer = expected_fft / (config.reference_resistor + expected_fft)
        channel_2 = np.fft.irfft(
            np.fft.rfft(source - np.mean(source)) * transfer,
            n=source.size,
        )
        result = calculate_impedance(
            ASignal(np.column_stack((source, channel_2)), config.sample_rate),
            config,
            np.ones(config.points, dtype=np.complex128),
            config.reference_resistor,
        )
        expected = 8.0 + 1j * 2.0 * np.pi * result.frequency * 0.0004
        relative_error = np.abs((result.impedance - expected) / expected)

        self.assertLess(float(np.nanmedian(relative_error)), 2e-3)

    def test_spice_fit_series_rl_model(self) -> None:
        frequency = np.geomspace(20.0, 20_000.0, 128)
        expected = np.asarray([6.8, 0.0004])
        measured = np.abs(speaker_impedance(frequency, expected, 0))

        result = fit_impedance(
            frequency,
            measured,
            sections=0,
            max_evaluations=300,
        )

        np.testing.assert_allclose(result.physical_params, expected, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
