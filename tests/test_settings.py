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

            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data, settings.to_dict())
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            self.assertEqual(len(changes), 4)

            loaded = AppSettings()
            self.assertTrue(loaded.load(path))
            self.assertEqual(loaded.frequency_range, (10.0, 30_000.0))
            self.assertEqual(loaded.impedance_scale, "log")
            self.assertEqual(loaded.thd_scale, "log")
            self.assertEqual(loaded.phase_unit, "deg/dec")

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


if __name__ == "__main__":
    unittest.main()
