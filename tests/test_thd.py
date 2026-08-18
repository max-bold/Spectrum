import unittest
from typing import Any, cast

import numpy as np

from audioanalysis import (
    ASignal,
    FrequencyBand,
    SemiAnalogTHDConfig,
    analyze_semi_analog_thd,
    fundamental_rejection_response,
    generate_semi_analog_thd_sweep,
)


class SemiAnalogTHDTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SemiAnalogTHDConfig(
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
        self.sweep = generate_semi_analog_thd_sweep(self.config)

    def test_sweep_has_fixed_level_and_asignal_metadata(self) -> None:
        self.assertIsInstance(self.sweep, ASignal)
        self.assertEqual(self.sweep.sample_rate, self.config.sample_rate)
        self.assertEqual(self.sweep.channel_count, 1)
        self.assertEqual(self.sweep.sample_count, self.config.sample_count)
        self.assertAlmostEqual(self.config.total_duration, 4.2)
        self.assertAlmostEqual(float(self.sweep.max()[0]), 0.9, places=5)

        start, stop = self.config.sweep_band
        sweep_rate = np.log(stop / start) / self.config.total_duration
        self.assertAlmostEqual(
            start * np.exp(sweep_rate * self.config.fade_in_seconds),
            self.config.band.low,
        )
        self.assertAlmostEqual(
            start
            * np.exp(
                sweep_rate
                * (self.config.fade_in_seconds + self.config.duration)
            ),
            self.config.band.high,
        )

    def test_clean_sweep_has_low_leakage_floor(self) -> None:
        result = analyze_semi_analog_thd(
            self.sweep,
            self.config,
        )

        self.assertLess(result.integrated_percent, 0.2)
        self.assertEqual(result.frequency.shape, (self.config.points,))
        self.assertEqual(result.percent.shape, (self.config.points,))
        self.assertAlmostEqual(result.frequency[0], self.config.band.low)
        self.assertAlmostEqual(result.frequency[-1], self.config.band.high)

    def test_nonlinearity_is_measured_above_clean_leakage(self) -> None:
        clean = self.sweep.as_array(np.float64)[:, 0]
        distorted = ASignal(clean + 0.1 * np.square(clean), self.config.sample_rate)

        result = analyze_semi_analog_thd(
            distorted,
            self.config,
        )

        self.assertGreater(result.integrated_percent, 2.0)
        self.assertAlmostEqual(
            result.integrated_ratio,
            np.sqrt(
                np.sum(result.residual_energy) / np.sum(result.total_energy)
            ),
        )
        self.assertTrue(np.all(result.residual_energy <= result.total_energy))

    def test_tracking_tolerates_recording_delay(self) -> None:
        clean = self.sweep.as_array(np.float64)[:, 0]
        delayed = ASignal(np.pad(clean, (347, 0)), self.config.sample_rate)

        result = analyze_semi_analog_thd(
            delayed,
            self.config,
        )

        self.assertLess(result.integrated_percent, 0.2)

    def test_only_channel_one_is_analyzed(self) -> None:
        clean = self.sweep.as_array(np.float64)[:, 0]
        rng = np.random.default_rng(1234)
        stereo = ASignal(
            np.column_stack((clean, rng.normal(size=len(clean)))),
            self.config.sample_rate,
        )

        result = analyze_semi_analog_thd(
            stereo,
            self.config,
        )

        self.assertLess(result.integrated_percent, 0.2)

    def test_analysis_rejects_bare_array(self) -> None:
        with self.assertRaisesRegex(TypeError, "ASignal"):
            analyze_semi_analog_thd(
                cast(Any, np.zeros(self.config.sample_count)),
                self.config,
            )

    def test_rejection_window_has_fixed_frequency_ratio(self) -> None:
        frequency = np.array([50.0, 100.0 / 1.5, 100.0, 150.0, 151.0])

        response = fundamental_rejection_response(frequency, 100.0, 1.5)

        np.testing.assert_array_equal(response, [1.0, 0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
