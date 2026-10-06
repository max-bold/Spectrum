from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from spectrum_app.core.settings import AppSettings
from spectrum_app.modules.impedance.settings import ImpedanceSettings


class ImpedanceSettingsTests(unittest.TestCase):
    def test_accuracy_defaults_clamps_and_survives_reload(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            app_settings = AppSettings()
            app_settings.load(path)
            settings = ImpedanceSettings(app_settings)
            self.assertEqual(settings.spice_accuracy_percent, 2.0)
            settings.spice_accuracy_percent = 0
            self.assertEqual(settings.spice_accuracy_percent, 0.1)
            settings.spice_accuracy_percent = 100
            self.assertEqual(settings.spice_accuracy_percent, 20.0)
            settings.spice_accuracy_percent = 1.5
            app_settings.save()
            loaded = AppSettings()
            self.assertTrue(loaded.load(path))
            self.assertEqual(ImpedanceSettings(loaded).spice_accuracy_percent, 1.5)
            for value in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    settings.spice_accuracy_percent = value
            app_settings.set_module_setting("impedance", "spice_accuracy_percent", "bad")
            self.assertEqual(settings.spice_accuracy_percent, 2.0)


if __name__ == "__main__":
    unittest.main()
