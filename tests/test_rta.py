import unittest

import numpy as np

from audioanalysis import (
    ASignal,
    FrequencyBand,
    RTAConfig,
    analyze_rta,
    compensate_log_band_density,
)
from spectrum_app.modules.rta.jobs import RTANoiseGenerator


class RTAMathTests(unittest.TestCase):
    def test_log_band_compensation_flattens_inverse_frequency_density(self) -> None:
        frequency = np.array([100.0, 1_000.0, 10_000.0])
        density = 1.0 / frequency

        compensated = compensate_log_band_density(frequency, density)

        np.testing.assert_allclose(compensated, compensated[0])

    def test_analyze_rta_returns_one_smoothed_series_per_channel(self) -> None:
        sample_rate = 8_000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        signal = ASignal(
            np.column_stack(
                (
                    0.5 * np.sin(2.0 * np.pi * 500.0 * time),
                    0.25 * np.sin(2.0 * np.pi * 1_000.0 * time),
                )
            ),
            sample_rate,
        )

        result = analyze_rta(
            signal,
            RTAConfig(
                band=FrequencyBand(50, 3_000),
                points=31,
                fft_window="hann",
            ),
        )

        self.assertEqual(result.frequency.shape, (31,))
        self.assertEqual(result.level_db.shape, (31, 2))
        self.assertTrue(np.all(np.diff(result.frequency) > 0.0))
        self.assertTrue(np.all(np.isfinite(result.level_db)))

    def test_noise_envelope_spans_arbitrary_block_boundaries(self) -> None:
        sample_rate = 100
        clipped: list[bool] = []
        generator = RTANoiseGenerator(
            sample_rate=sample_rate,
            block_size=7,
            band=FrequencyBand(5, 40),
            level_db=0.0,
            pre_silence=0.1,
            fade_in=0.2,
            fade_out=0.15,
            on_clipping=lambda message: clipped.append(True),
        )
        generator.start()
        blocks: list[np.ndarray] = []
        samples = 0
        while samples < 40:
            block = generator.take()
            self.assertIsNotNone(block)
            assert block is not None
            blocks.append(block)
            samples += len(block)
        generator.request_stop()
        while True:
            block = generator.take()
            if block is None:
                break
            blocks.append(block)
        generator.join(timeout=1.0)
        output = np.concatenate(blocks)

        np.testing.assert_array_equal(output[:10], 0.0)
        self.assertEqual(float(output[10]), 0.0)
        self.assertAlmostEqual(float(output[-1]), 0.0, places=7)
        self.assertFalse(generator.is_alive())
        self.assertLessEqual(float(np.max(np.abs(output))), 1.0)


if __name__ == "__main__":
    unittest.main()
