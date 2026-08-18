import unittest

import numpy as np

from audioanalysis import SmoothingWindow, grid_smooth, log_smooth, log_window
from audioanalysis.smoothing import WINDOW_EDGE_WEIGHT, log_window_old


class SpectrumSmoothingTests(unittest.TestCase):
    def test_log_window_old_retains_original_gaussian_formula(self) -> None:
        center = 1.0
        frequency_step = 1e-5
        width = 0.3
        half_width = width / 2.0
        low = center / (2.0**half_width)
        high = center * (2.0**half_width)
        frequencies = np.arange(low, high, frequency_step)
        offsets = np.log2(frequencies / center)
        expected = np.exp(-((offsets / half_width * 4.0) ** 2) / 2.0)

        weights, start, end = log_window_old(
            SmoothingWindow.GAUSSIAN,
            center,
            frequency_step,
            width,
        )

        np.testing.assert_array_equal(weights, expected)
        self.assertEqual(start, int(np.rint(low / frequency_step)))
        self.assertEqual(end, start + len(expected))

    def test_tapered_windows_use_width_as_fwhm(self) -> None:
        center = 1.0
        frequency_step = 1e-5
        width = 0.3

        for window in (
            SmoothingWindow.COSINE,
            SmoothingWindow.GAUSSIAN,
            SmoothingWindow.TRIANGULAR,
        ):
            with self.subTest(window=window):
                weights, start, end = log_window(
                    window,
                    center,
                    frequency_step,
                    width,
                )
                frequencies = np.arange(start, end) * frequency_step
                offsets = np.log2(frequencies / center)
                left = np.argmin(np.abs(offsets + width / 2.0))
                right = np.argmin(np.abs(offsets - width / 2.0))

                self.assertAlmostEqual(float(weights[left]), 0.5, delta=1e-4)
                self.assertAlmostEqual(float(weights[right]), 0.5, delta=1e-4)

    def test_tapered_windows_end_near_minus_30_db(self) -> None:
        center = 1.0
        frequency_step = 1e-5
        width = 0.3

        for window in (
            SmoothingWindow.COSINE,
            SmoothingWindow.GAUSSIAN,
            SmoothingWindow.TRIANGULAR,
        ):
            with self.subTest(window=window):
                weights, _, _ = log_window(
                    window,
                    center,
                    frequency_step,
                    width,
                )
                edge_weights = weights[[0, -1]]

                self.assertTrue(np.all(edge_weights >= WINDOW_EDGE_WEIGHT - 1e-12))
                self.assertTrue(np.all(edge_weights < WINDOW_EDGE_WEIGHT * 1.1))

    def test_flat_window_support_is_one_width(self) -> None:
        center = 1.0
        frequency_step = 1e-5
        width = 0.3

        weights, start, end = log_window(
            SmoothingWindow.FLAT,
            center,
            frequency_step,
            width,
        )
        frequencies = np.arange(start, end) * frequency_step
        support_width = np.log2(frequencies[-1] / frequencies[0])

        np.testing.assert_array_equal(weights, np.ones_like(weights))
        self.assertAlmostEqual(float(support_width), width, delta=5e-5)

    def test_log_smoothing_preserves_constant_channel_values(self) -> None:
        frequency = np.linspace(0.0, 24_000.0, 24_001)
        values = np.column_stack(
            (np.ones_like(frequency), np.full_like(frequency, 2.0))
        )

        grid, smoothed = log_smooth(
            frequency,
            values,
            band=(20.0, 20_000.0),
            window=SmoothingWindow.GAUSSIAN,
            width=0.1,
            points=128,
        )

        self.assertEqual(grid.shape, (128,))
        self.assertEqual(smoothed.shape, (128, 2))
        np.testing.assert_allclose(smoothed[:, 0], 1.0)
        np.testing.assert_allclose(smoothed[:, 1], 2.0)

    def test_grid_smooth_supports_every_new_window_shape(self) -> None:
        frequency = np.linspace(0.0, 40_000.0, 100_000)
        values = np.ones_like(frequency)
        grid = np.geomspace(20.0, 20_000.0, 30)
        width = np.log2(grid[-1] / grid[0]) / (len(grid) - 1)

        for window in SmoothingWindow:
            with self.subTest(window=window):
                smoothed = grid_smooth(
                    frequency,
                    values,
                    grid,
                    window=window,
                    width=float(width),
                )

                np.testing.assert_allclose(smoothed, 1.0)

    def test_grid_smooth_preserves_complex_values_with_offset_grid(self) -> None:
        frequency = np.linspace(10.0, 1010.0, 1001)
        values = np.ones_like(frequency) + 2j * np.ones_like(frequency)
        grid = np.geomspace(20.0, 900.0, 32)

        smoothed = grid_smooth(
            frequency,
            values,
            grid,
            window=SmoothingWindow.GAUSSIAN,
            width=0.2,
        )

        np.testing.assert_allclose(smoothed, 1.0 + 2.0j)

    def test_grid_smooth_uses_a_uniform_log_frequency_measure(self) -> None:
        frequency = np.linspace(0.0, 4_000.0, 400_001)
        values = np.zeros_like(frequency)
        values[1:] = np.log2(frequency[1:])
        grid = np.geomspace(50.0, 2_000.0, 20)

        for window in SmoothingWindow:
            with self.subTest(window=window):
                smoothed = grid_smooth(
                    frequency,
                    values,
                    grid,
                    window=window,
                    width=0.2,
                )

                np.testing.assert_allclose(
                    smoothed,
                    np.log2(grid),
                    atol=3e-5,
                    rtol=0.0,
                )

    def test_neighboring_windows_cover_the_log_grid(self) -> None:
        frequency = np.linspace(0.0, 40_000.0, 100_000)
        grid = np.geomspace(20.0, 20_000.0, 30)
        width = float(np.log2(grid[-1] / grid[0]) / (len(grid) - 1))
        frequency_step = float(frequency[1] - frequency[0])
        interior = (frequency >= grid[1]) & (frequency <= grid[-2])
        tolerances = {
            SmoothingWindow.FLAT: 0.0,
            SmoothingWindow.COSINE: 1.1e-3,
            SmoothingWindow.GAUSSIAN: 0.126,
            SmoothingWindow.TRIANGULAR: 1.1e-3,
        }

        for window, tolerance in tolerances.items():
            with self.subTest(window=window):
                total = np.zeros_like(frequency)
                for center in grid:
                    weights, start, end = log_window(
                        window,
                        float(center),
                        frequency_step,
                        width,
                    )
                    data_start = max(0, start)
                    data_end = min(len(frequency), end)
                    weight_start = data_start - start
                    weight_end = weight_start + data_end - data_start
                    total[data_start:data_end] += weights[weight_start:weight_end]

                self.assertLessEqual(
                    float(np.max(np.abs(total[interior] - 1.0))),
                    tolerance,
                )


if __name__ == "__main__":
    unittest.main()
