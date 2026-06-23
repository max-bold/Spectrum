import unittest

import dearpygui.dearpygui as dpg

from spectrum_app.fonts import bind_app_font, find_app_font


class FontTests(unittest.TestCase):
    def test_app_uses_default_font(self) -> None:
        dpg.create_context()
        try:
            self.assertIsNone(find_app_font())
            self.assertIsNone(bind_app_font())
        finally:
            dpg.destroy_context()


if __name__ == "__main__":
    unittest.main()
