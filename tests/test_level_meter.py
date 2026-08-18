import unittest
from unittest.mock import patch

from spectrum_app.gui.controls.level_meter import (
    METER_GREEN,
    METER_RED,
    METER_YELLOW,
    add_level_meter,
)
from tests.test_dpg_lifecycle import FakeDpgBackend


class LevelMeterTests(unittest.TestCase):
    def test_single_channel_meter_is_compact_and_uses_all_color_zones(self) -> None:
        backend = FakeDpgBackend()

        with patch("spectrum_app.gui.controls.level_meter.dpg", backend):
            meter = add_level_meter(
                "parent",
                "meter",
                "height_source",
                labels=("1",),
            )
            meter.set_levels(0.9)

        drawlist = next(call for call in backend.calls if call[0] == "drawlist")
        self.assertEqual(drawlist[1]["width"], 42)
        fills = {
            call[3]["fill"]
            for call in backend.calls
            if call[0] == "draw_rectangle" and "fill" in call[3]
        }
        self.assertTrue({METER_GREEN, METER_YELLOW, METER_RED}.issubset(fills))


if __name__ == "__main__":
    unittest.main()
