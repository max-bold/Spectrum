import unittest

import dearpygui.dearpygui as dpg

from spectrum_app.fonts import bind_app_font, find_app_font


class FontTests(unittest.TestCase):
    def test_app_font_contains_cyrillic(self) -> None:
        font_path = find_app_font()
        self.assertIsNotNone(font_path)

        dpg.create_context()
        try:
            self.assertEqual(bind_app_font(), font_path)
        finally:
            dpg.destroy_context()


if __name__ == "__main__":
    unittest.main()
