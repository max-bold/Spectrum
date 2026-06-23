import unittest

import numpy as np

from spectrum_app.analysis import record_levels


class RecordLevelTests(unittest.TestCase):
    def test_record_levels_returns_chunk_peaks_for_stereo_record(self) -> None:
        record = np.array(
            [
                [0.1, -0.2],
                [-0.4, 0.3],
                [0.2, -0.8],
                [-0.7, 0.1],
            ],
            dtype=np.float32,
        )

        timestamps, levels = record_levels(record, sample_rate=4, time_step=0.5)

        np.testing.assert_allclose(timestamps, [0.0, 0.5])
        np.testing.assert_allclose(levels, [[0.4, 0.7], [0.3, 0.8]])

    def test_record_levels_duplicates_mono_records(self) -> None:
        record = np.array([[0.1], [-0.5], [0.2]], dtype=np.float32)

        timestamps, levels = record_levels(record, sample_rate=3, time_step=1.0)

        np.testing.assert_allclose(timestamps, [0.0])
        np.testing.assert_allclose(levels, [[0.5], [0.5]])


if __name__ == "__main__":
    unittest.main()
