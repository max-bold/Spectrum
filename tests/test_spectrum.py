import unittest
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from scipy.signal import periodogram, welch

from audioanalysis import (
    ASignal,
    AnalysisMethod,
    FrequencyBand,
    ReferenceMode,
    SpectrumConfig,
    analyze_spectrum,
    calculate_power_spectrum,
    extend_log_sweep_band,
    log_chirp,
    pink_noise,
    pinking_sos,
    white_noise,
)


class SpectrumAnalysisTests(unittest.TestCase):
    def test_analyzer_excludes_out_of_band_fft_bins_before_smoothing(self) -> None:
        frequency = np.arange(0.0, 24_001.0)
        spectrum = np.ones((len(frequency), 2), dtype=np.float64)
        spectrum[frequency > 20_000.0, 0] = 1e12
        signal = ASignal(np.ones((128, 2), dtype=np.float32), 48_000)

        with patch(
            "audioanalysis.spectrum.calculate_power_spectrum",
            return_value=(frequency, spectrum),
        ):
            result = analyze_spectrum(
                signal,
                SpectrumConfig(
                    reference=ReferenceMode.CHANNEL_B,
                    band=FrequencyBand(20.0, 20_000.0),
                    window_width=0.3,
                    points=1024,
                ),
            )

        np.testing.assert_allclose(result.values, 1.0)

    def test_periodogram_matches_scipy_for_every_channel(self) -> None:
        rng = np.random.default_rng(1234)
        signal = ASignal(rng.normal(size=(2048, 2)), 48_000)

        frequency, values = calculate_power_spectrum(signal)
        expected_frequency, expected_values = periodogram(
            signal.as_array(np.float64),
            signal.sample_rate,
            axis=0,
        )

        np.testing.assert_allclose(frequency, expected_frequency)
        np.testing.assert_allclose(values, expected_values)

    def test_welch_matches_scipy_and_limits_window_to_recording(self) -> None:
        rng = np.random.default_rng(5678)
        signal = ASignal(rng.normal(size=(1024, 2)), 44_100)

        frequency, values = calculate_power_spectrum(
            signal,
            method=AnalysisMethod.WELCH,
            welch_samples=8192,
        )
        expected_frequency, expected_values = welch(
            signal.as_array(np.float64),
            signal.sample_rate,
            window="hann",
            nperseg=signal.sample_count,
            axis=0,
        )

        np.testing.assert_allclose(frequency, expected_frequency)
        np.testing.assert_allclose(values, expected_values)

    def test_analyzer_uses_sample_rate_from_asignal(self) -> None:
        sample_rate = 48_000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        signal = ASignal(np.sin(2 * np.pi * 1000.0 * time), sample_rate)

        result = analyze_spectrum(
            signal,
            SpectrumConfig(
                band=FrequencyBand(100.0, 10_000.0),
                points=512,
            ),
        )

        self.assertEqual(result.frequency.shape, (512,))
        self.assertEqual(result.values.shape, (512,))
        peak_frequency = result.frequency[np.nanargmax(result.values)]
        self.assertAlmostEqual(peak_frequency, 1000.0, delta=100.0)

    def test_analyzer_rejects_bare_ndarray(self) -> None:
        with self.assertRaisesRegex(TypeError, "ASignal"):
            analyze_spectrum(
                cast(Any, np.zeros((1024, 1))),
                SpectrumConfig(),
            )

    def test_generators_return_asignal(self) -> None:
        band = FrequencyBand((20.0, 20_000.0))
        chirp = log_chirp(2048, 48_000, band)
        noise, zi = pink_noise(
            2048,
            48_000,
            band,
            rng=np.random.default_rng(42),
        )

        self.assertIsInstance(chirp, ASignal)
        self.assertIsInstance(noise, ASignal)
        self.assertEqual(chirp.sample_rate, 48_000)
        self.assertEqual(noise.sample_rate, 48_000)
        self.assertEqual(zi.ndim, 2)

    def test_frequency_band_accepts_two_numbers_or_tuple(self) -> None:
        self.assertEqual(
            FrequencyBand(30.0, 18_000.0),
            FrequencyBand((30.0, 18_000.0)),
        )

    def test_log_sweep_fades_are_extended_outside_the_working_band(self) -> None:
        band = FrequencyBand(20.0, 20_000.0)
        duration = 20.0
        fade = 0.5

        extended = extend_log_sweep_band(band, duration, fade, fade)
        sweep_rate = np.log(extended.high / extended.low) / (
            duration + 2.0 * fade
        )

        self.assertAlmostEqual(
            extended.low * np.exp(sweep_rate * fade),
            band.low,
        )
        self.assertAlmostEqual(
            extended.low * np.exp(sweep_rate * (fade + duration)),
            band.high,
        )

    def test_generators_are_mono_and_pink_noise_always_returns_state(self) -> None:
        chirp = log_chirp(128)
        white = white_noise(128, rng=np.random.default_rng(1))
        pink, zi = pink_noise(128, rng=np.random.default_rng(2))

        self.assertEqual(chirp.channel_count, 1)
        self.assertEqual(white.channel_count, 1)
        self.assertEqual(pink.channel_count, 1)
        self.assertEqual(zi.shape, (pinking_sos(44_100).shape[0], 2))

    def test_pink_noise_state_makes_consecutive_blocks_continuous(self) -> None:
        first_rng = np.random.default_rng(123)
        first, zi = pink_noise(256, rng=first_rng)
        second, _ = pink_noise(256, rng=first_rng, zi=zi)

        complete, _ = pink_noise(512, rng=np.random.default_rng(123))

        np.testing.assert_allclose(
            np.vstack((first.as_array(), second.as_array())),
            complete.as_array(),
        )


if __name__ == "__main__":
    unittest.main()
