from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from spectrum_app import SpectrumApplication
from spectrum_app.core.model import AppState, AxisSpec, GraphData, Measurement
from spectrum_app.core.project import ProjectError, load_project, save_project


class ProjectStorageTests(unittest.TestCase):
    def test_app_state_is_saved_and_loaded_as_binary_project(self) -> None:
        graph = GraphData(
            name="Spectrum",
            x=np.array([20.0, 1000.0]),
            y=np.array([-30.0, -10.0]),
            x_axis=AxisSpec.FREQ,
            y_axis=AxisSpec.LEVEL,
        )
        measurement = Measurement(
            module_id="spectrum",
            name="Stored measurement",
            module_state={"calibration": 1.25, "window": "hann"},
            graphs=[graph],
        )
        state = AppState(
            measurements=[measurement],
            active_measurement_id=measurement.id,
            visible_graph_ids=[graph.id],
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "session"
            saved_path = save_project(state, path)
            loaded = load_project(saved_path)

            self.assertEqual(saved_path.suffix, ".bms")
            self.assertFalse(saved_path.with_suffix(".bms.tmp").exists())
            self.assertEqual(loaded.project_path, saved_path)
            self.assertEqual(loaded.measurements[0].name, measurement.name)
            self.assertEqual(
                loaded.measurements[0].module_state,
                measurement.module_state,
            )
            np.testing.assert_array_equal(loaded.measurements[0].graphs[0].x, graph.x)
            np.testing.assert_array_equal(loaded.measurements[0].graphs[0].y, graph.y)

    def test_invalid_project_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.bms"
            path.write_bytes(b"not a pickle")

            with self.assertRaisesRegex(ProjectError, "Cannot load"):
                load_project(path)

    def test_application_load_replaces_state(self) -> None:
        app = SpectrumApplication()
        measurement = app.create_measurement()
        measurement.name = "Saved"

        with TemporaryDirectory() as directory:
            path = Path(directory) / "application.bms"
            app.save_project(path)
            app.app_state.measurements.clear()
            app.app_state.active_measurement_id = None

            app.load_project(path)

            self.assertEqual(app.app_state.measurements[0].name, "Saved")
            self.assertEqual(app.app_state.project_path, path)
            self.assertTrue(app.app_state.graph_data_changed)
            self.assertFalse(app.app_state.measuring)
            self.assertEqual(app.app_state.active_measurement_id, measurement.id)

    def test_unavailable_module_does_not_replace_current_state(self) -> None:
        app = SpectrumApplication()
        original_state = app.app_state
        unknown_state = AppState(
            measurements=[Measurement(module_id="missing", name="Unknown")]
        )

        with TemporaryDirectory() as directory:
            path = save_project(unknown_state, Path(directory) / "unknown.bms")

            with self.assertRaisesRegex(ProjectError, "unavailable modules"):
                app.load_project(path)

        self.assertIs(app.app_state, original_state)


if __name__ == "__main__":
    unittest.main()
