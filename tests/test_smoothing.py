import unittest

import numpy as np

from audioanalysis import SmoothingWindow, log_smooth


class SpectrumSmoothingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
