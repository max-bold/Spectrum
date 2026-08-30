import unittest

import numpy as np

from audioanalysis import (
    ASignal,
    FrequencyBand,
    RTAConfig,
    analyze_rta,
    compensate_log_band_density,
    periodic_pink_noise,
)
from spectrum_app.modules.rta.jobs import RTANoiseGenerator
from spectrum_app.modules.rta.types import PERIODIC_IFFT_GENERATOR


class RTAMathTests(unittest.TestCase):
    def test_periodic_pink_noise_is_flat_after_pink_compensation(self) -> None:
        sample_rate = 48_000
        samples = sample_rate
        band = FrequencyBand(20.0, 20_000.0)

        signal = periodic_pink_noise(
            samples,
            sample_rate,
            band,
            rng=np.random.default_rng(0),
        )

        data = signal.as_array(np.float64)[:, 0]
        frequency = np.fft.rfftfreq(samples, 1.0 / sample_rate)
        magnitude = np.abs(np.fft.rfft(data))
        inside = (frequency >= band.low) & (frequency <= band.high)
        compensated = magnitude[inside] * np.sqrt(frequency[inside])
        relative_db = 20.0 * np.log10(compensated / np.max(compensated))

        self.assertEqual(signal.sample_count, samples)
        self.assertEqual(signal.channel_count, 1)
        self.assertAlmostEqual(float(np.max(np.abs(data))), 0.9, places=6)
        self.assertGreaterEqual(float(np.min(relative_db)), -1.0)

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
            period_samples=sample_rate,
            band=FrequencyBand(5, 40),
            generator=PERIODIC_IFFT_GENERATOR,
            level_db=0.0,
            pre_silence=0.1,
            fade_in=0.2,
            fade_out=0.15,
            on_clipping=lambda message: clipped.append(True),
            rng=np.random.default_rng(0),
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
        self.assertFalse(clipped)
        self.assertLessEqual(float(np.max(np.abs(output))), 1.0)

    def test_periodic_generator_repeats_across_arbitrary_block_boundaries(
        self,
    ) -> None:
        sample_rate = 8_000
        period_samples = 23
        band = FrequencyBand(100, 3_000)
        expected = periodic_pink_noise(
            period_samples,
            sample_rate,
            band,
            rng=np.random.default_rng(5),
        ).as_array(np.float64)[:, 0]
        generator = RTANoiseGenerator(
            sample_rate=sample_rate,
            block_size=7,
            period_samples=period_samples,
            band=band,
            generator=PERIODIC_IFFT_GENERATOR,
            level_db=0.0,
            pre_silence=0.0,
            fade_in=0.0,
            fade_out=0.0,
            on_clipping=lambda message: None,
            rng=np.random.default_rng(5),
        )

        generator.start()
        blocks = []
        for _ in range(8):
            block = generator.take()
            self.assertIsNotNone(block)
            assert block is not None
            blocks.append(block)
        generator.request_stop()
        while generator.take() is not None:
            pass
        generator.join(timeout=1.0)

        output = np.concatenate(blocks).astype(np.float64)
        np.testing.assert_allclose(output[:period_samples], expected, atol=1e-7)
        np.testing.assert_allclose(
            output[period_samples : 2 * period_samples],
            expected,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
