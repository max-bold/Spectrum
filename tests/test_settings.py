import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from spectrum_app.core.settings import AppSettings


class AppSettingsTests(unittest.TestCase):
    def test_changes_are_saved_and_loaded_automatically(self) -> None:
        changes: list[str] = []
        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings = AppSettings(on_change=lambda: changes.append("changed"))

            self.assertFalse(settings.load(path))
            settings.frequency_range = (10.0, 30_000.0)
            settings.impedance_scale = "log"
            settings.thd_scale = "log"
            settings.phase_unit = "deg/dec"
            settings.input_device = "WASAPI\x1fInput"
            settings.output_device = "WASAPI\x1fOutput"
            settings.input_block_size = 2048
            settings.output_block_size = 4096
            settings.set_module_setting("spectrum", "generator_mode", "pink noise")
            settings.set_module_setting("spectrum", "welch_samples", 4096)
            settings.set_module_setting("spectrum", "online_welch", False)

            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data, settings.to_dict())
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            self.assertEqual(len(changes), 11)

            loaded = AppSettings()
            self.assertTrue(loaded.load(path))
            self.assertEqual(loaded.frequency_range, (10.0, 30_000.0))
            self.assertEqual(loaded.impedance_scale, "log")
            self.assertEqual(loaded.thd_scale, "log")
            self.assertEqual(loaded.phase_unit, "deg/dec")
            self.assertEqual(loaded.input_device, "WASAPI\x1fInput")
            self.assertEqual(loaded.output_device, "WASAPI\x1fOutput")
            self.assertEqual(loaded.input_block_size, 2048)
            self.assertEqual(loaded.output_block_size, 4096)
            self.assertEqual(
                loaded.module_setting("spectrum", "generator_mode"),
                "pink noise",
            )
            self.assertEqual(
                loaded.module_setting("spectrum", "welch_samples"),
                4096,
            )
            self.assertFalse(loaded.module_setting("spectrum", "online_welch"))

    def test_invalid_file_falls_back_to_defaults(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text("not json", encoding="utf-8")
            settings = AppSettings()

            self.assertFalse(settings.load(path))

            self.assertEqual(
                settings.frequency_range,
                AppSettings.DEFAULT_FREQUENCY_RANGE,
            )
            self.assertEqual(settings.impedance_scale, "linear")
            self.assertEqual(settings.thd_scale, "linear")
            self.assertEqual(settings.phase_unit, "deg")

    def test_frequency_range_is_validated(self) -> None:
        settings = AppSettings()

        with self.assertRaisesRegex(ValueError, "positive and increasing"):
            settings.frequency_range = (1000.0, 20.0)

    def test_block_sizes_are_validated(self) -> None:
        settings = AppSettings()

        with self.assertRaisesRegex(ValueError, "positive"):
            settings.input_block_size = 0


if __name__ == "__main__":
    unittest.main()
