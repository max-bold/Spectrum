from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from spectrum_app import SpectrumApplication
from spectrum_app.modules.manager import ModuleManager
from spectrum_app.modules.impedance import ImpedanceModule
from spectrum_app.modules.spectrum import SpectrumModule


class ModuleManagerTests(unittest.TestCase):
    def test_application_discovers_explicit_module_export(self) -> None:
        app = SpectrumApplication()

        self.assertEqual(app.module_manager.module_ids, ("impedance", "spectrum"))
        self.assertIsInstance(
            app.module_manager.module("impedance"),
            ImpedanceModule,
        )
        self.assertIsInstance(
            app.module_manager.module("spectrum"),
            SpectrumModule,
        )

    def test_manager_is_only_a_discovery_registry(self) -> None:
        with TemporaryDirectory() as directory:
            manager = ModuleManager(path=Path(directory))
            manager.discover()

        self.assertEqual(manager.modules, ())
        with self.assertRaisesRegex(ValueError, "Unknown module"):
            manager.module("missing")


if __name__ == "__main__":
    unittest.main()
