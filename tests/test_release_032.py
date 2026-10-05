from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
import json
from unittest.mock import MagicMock, patch

import numpy as np

from audioanalysis import ASignal
from spectrum_app import SpectrumApplication
from spectrum_app.core.dpg import DearPyGuiRuntime
from spectrum_app.core.model import Measurement
from spectrum_app.gui.measurement import MeasurementPanel
from spectrum_app.gui.native_files import choose_file, NativeFileDialogError
from spectrum_app.modules.spectrum.module import SpectrumModule
from spectrum_app.modules.spectrum.settings import SpectrumSettings
from spectrum_app.startup import StartupSplash
from tests.test_dpg_lifecycle import FakeDpgBackend


class Release032Tests(unittest.TestCase):
    def test_empty_spectrum_switches_to_rta_without_confirmation(self) -> None:
        app = SpectrumApplication()
        measurement = app.create_measurement()
        measurement.module_state = deepcopy(SpectrumModule.DEFAULT_STATE)
        panel = app.main_window.measurement_panel
        with patch("spectrum_app.gui.measurement.dpg", FakeDpgBackend()), patch.object(panel, "update"):
            panel._set_module(panel.module_combo, "RTA")
        self.assertEqual(measurement.module_id, "rta")
        self.assertIsNone(panel._pending_module_change)

    def test_recordings_and_results_still_need_confirmation(self) -> None:
        measurement = Measurement("spectrum", "Test")
        measurement.module_state = deepcopy(SpectrumModule.DEFAULT_STATE)
        self.assertFalse(MeasurementPanel._has_measurement_data(measurement))
        measurement.module_state["recordings"] = [ASignal(np.ones(32), 8000)]
        self.assertTrue(MeasurementPanel._has_measurement_data(measurement))
        measurement.module_state = {"calibration_resistor": 10.4, "generator_mode": "log chirp"}
        self.assertFalse(MeasurementPanel._has_measurement_data(measurement))
        measurement.module_state["channel_calibration_recording"] = ASignal(np.ones((32, 2)), 8000)
        self.assertTrue(MeasurementPanel._has_measurement_data(measurement))

    def test_sweep_error_explains_fades_and_nyquist(self) -> None:
        app = SpectrumApplication()
        app.audio_input = SimpleNamespace(sample_rate=48000)
        app.audio_output = SimpleNamespace(sample_rate=48000)
        module = SpectrumModule()
        module._app = app
        module._settings = SpectrumSettings(app.settings)
        module.settings.generator_mode = "log chirp"
        module.settings.fade_in = 0.5
        module.settings.fade_out = 0.5
        state = dict(SpectrumModule.DEFAULT_STATE, duration=1.0)
        with self.assertRaises(ValueError) as error:
            module._validate_audio_settings(state)
        message = str(error.exception)
        self.assertIn("Requested range: 20.0 - 20000.0 Hz\n", message)
        self.assertIn("Range with fade-in/out:", message)
        self.assertIn("24000.0 Hz", message)
        self.assertTrue(message.isascii())
        import re
        values = re.findall(r"\d+\.\d+", message)
        self.assertTrue(values)
        self.assertTrue(all(len(value.split(".")[1]) == 1 for value in values))
        self.assertIn("Reduce the upper frequency", message)

    def test_native_file_save_and_cancel_release_owner(self) -> None:
        root = MagicMock()
        with patch("tkinter.Tk", return_value=root), patch("tkinter.filedialog.asksaveasfilename", return_value="C:/demo.bms") as dialog:
            path = choose_file(title="Save", extension=".bms", description="Project", save=True,
                               initial=Path("C:/project.bms"))
        self.assertEqual(path, Path("C:/demo.bms"))
        self.assertEqual(dialog.call_args.kwargs["defaultextension"], ".bms")
        self.assertEqual(dialog.call_args.kwargs["initialfile"], "project.bms")
        root.destroy.assert_called_once()
        with patch("tkinter.Tk", return_value=MagicMock()), patch("tkinter.filedialog.askopenfilename", return_value=""):
            self.assertIsNone(choose_file(title="Open", extension=".bms", description="Project"))

    def test_native_cancel_does_not_open_project(self) -> None:
        app = SpectrumApplication()
        with patch("spectrum_app.gui.project.choose_file", return_value=None), patch.object(app, "load_project") as load:
            app.main_window.project_dialogs.show_open_dialog()
        load.assert_not_called()

    def test_native_dialog_error_releases_owner(self) -> None:
        import tkinter as tk
        root = MagicMock()
        with patch("tkinter.Tk", return_value=root), patch("tkinter.filedialog.askopenfilename", side_effect=tk.TclError("failed")):
            with self.assertRaisesRegex(NativeFileDialogError, "system file dialog"):
                choose_file(title="Open", extension=".bms", description="Project")
        root.destroy.assert_called_once()

    def test_bundled_splash_update_and_idempotent_close(self) -> None:
        bundled = MagicMock()
        bundled.is_alive.return_value = True
        splash = StartupSplash()
        with patch("spectrum_app.startup.sys.frozen", True, create=True), patch.dict("sys.modules", {"pyi_splash": bundled}), patch("spectrum_app.startup.subprocess.Popen") as process:
            splash.start()
            splash.update("Drawing main window", 0.95)
        process.assert_not_called()
        bundled.update_text.assert_called_with("Drawing main window  (95%)")
        splash.close()
        splash.close()
        bundled.close.assert_called_once()

    def test_splash_closes_after_first_render(self) -> None:
        backend = FakeDpgBackend()
        def ready():
            backend.calls.append(("startup_ready",))
        app = SpectrumApplication(startup_ready=ready)
        app.dpg = DearPyGuiRuntime(backend)
        app.dpg.create_context()
        app.main_window.update = MagicMock()
        app._run_main_loop()
        events = [call[0] for call in backend.calls]
        self.assertLess(events.index("render_dearpygui_frame"), events.index("startup_ready"))
        self.assertEqual(events.count("startup_ready"), 1)
        app.dpg.destroy_context()

    def test_splash_helper_status_and_cleanup(self) -> None:
        process = MagicMock()
        splash = StartupSplash()
        with patch("spectrum_app.startup.subprocess.Popen", return_value=process):
            splash.start()
        splash.update("Building interface", 0.5)
        status = splash._status
        self.assertEqual(json.loads(status.read_text())["progress"], 0.5)
        splash.close()
        splash.close()
        self.assertFalse(status.exists())
        process.wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
