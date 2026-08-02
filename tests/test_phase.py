import unittest

import numpy as np

from audioanalysis.phase import break_phase_wraps, phase_derivative, wrap_phase


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


if __name__ == "__main__":
    unittest.main()
