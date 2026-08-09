import unittest

import numpy as np

from audioanalysis import (
    ASignal,
    FrequencyBand,
    PhaseConfig,
    analyze_phase,
    break_phase_wraps,
    estimate_phase_delay,
    phase_derivative,
    wrap_phase,
)


class PhaseDerivativeTests(unittest.TestCase):
    def test_phase_is_wrapped_to_signed_degrees(self) -> None:
        phase = np.array([-181.0, -180.0, 0.0, 179.0, 180.0, 181.0, np.nan])

        result = wrap_phase(phase)

        np.testing.assert_allclose(
            result,
            np.array([179.0, -180.0, 0.0, 179.0, -180.0, -179.0, np.nan]),
            equal_nan=True,
        )

    def test_phase_wraps_are_broken_with_nan_points(self) -> None:
        x = np.array([10.0, 100.0, 1000.0])
        phase = wrap_phase(np.array([170.0, 190.0, 200.0]))

        result_x, result_phase = break_phase_wraps(x, phase)

        np.testing.assert_allclose(
            result_x,
            np.array([10.0, np.nan, 100.0, 1000.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            result_phase,
            np.array([170.0, np.nan, -170.0, -160.0]),
            equal_nan=True,
        )

    def test_derivative_is_calculated_per_frequency_decade(self) -> None:
        frequency = np.geomspace(10.0, 10_000.0, 7)
        phase = 30.0 * np.log10(frequency) + 5.0

        result = phase_derivative(frequency, phase, smoothing_sigma=0)

        np.testing.assert_allclose(result, 30.0)

    def test_invalid_points_remain_nan(self) -> None:
        frequency = np.array([10.0, 0.0, 100.0, 1000.0])
        phase = np.array([0.0, 5.0, 10.0, 20.0])

        result = phase_derivative(frequency, phase, smoothing_sigma=0)

        self.assertTrue(np.isnan(result[1]))
        np.testing.assert_allclose(result[[0, 2, 3]], 10.0)

    def test_frequency_must_be_strictly_increasing(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            phase_derivative(
                np.array([10.0, 100.0, 50.0]),
                np.array([0.0, 10.0, 20.0]),
                smoothing_sigma=0,
            )


class PhaseAnalysisTests(unittest.TestCase):
    def test_delay_fit_uses_fourth_power_error(self) -> None:
        frequency = np.linspace(1.0, 100.0, 101)
        phase = np.zeros_like(frequency)
        phase[-1] = 1.0
        transfer = np.exp(1j * phase)
        magnitude = np.ones_like(frequency)

        delay = estimate_phase_delay(
            frequency,
            transfer,
            magnitude,
            magnitude,
            FrequencyBand(1.0, 100.0),
        )
        l2_slope, _ = np.polyfit(frequency, phase, 1)
        l2_delay = -float(l2_slope) / (2.0 * np.pi)

        self.assertGreater(abs(delay), abs(l2_delay) * 4.0)

    def test_impulse_delay_is_estimated_and_removed_from_unwrapped_phase(self) -> None:
        sample_rate = 48_000
        samples = 4096
        delay_samples = 48
        reference = np.zeros(samples, dtype=np.float32)
        measured = np.zeros(samples, dtype=np.float32)
        reference[200] = 0.5
        measured[200 + delay_samples] = 0.5
        recording = ASignal(
            np.column_stack((measured, reference)),
            sample_rate,
        )

        result = analyze_phase(
            recording,
            PhaseConfig(
                band=FrequencyBand(100.0, 15_000.0),
                delay_fit_band=FrequencyBand(200.0, 10_000.0),
                points=128,
                smoothing_octaves=1.0 / 3.0,
            ),
        )

        self.assertAlmostEqual(
            result.estimated_delay_seconds,
            delay_samples / sample_rate,
            places=5,
        )
        self.assertEqual(result.frequency.shape, (128,))
        self.assertLess(float(np.nanmax(np.abs(result.phase_degrees))), 3.0)

    def test_phase_analysis_requires_logical_a_and_b(self) -> None:
        recording = ASignal(np.ones(1024), 48_000)

        with self.assertRaisesRegex(ValueError, "channels A and B"):
            analyze_phase(recording, PhaseConfig())


if __name__ == "__main__":
    unittest.main()
