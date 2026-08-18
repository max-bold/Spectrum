from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from audioanalysis import ASignal, FrequencyBand
from spectrum_app import SpectrumApplication
from spectrum_app.core.measurement_io import (
    AUDIO_ARRAY_ENCODING,
    MEASUREMENT_FORMAT,
    MeasurementIOError,
    load_measurement,
    save_measurement,
)
from spectrum_app.core.model import AxisSpec, GraphData, Measurement, PlotType


class MeasurementIOTests(unittest.TestCase):
    def test_json_round_trip_preserves_audio_arrays_and_module_types(self) -> None:
        samples = np.sin(np.linspace(0.0, 20.0, 48_000, dtype=np.float32))
        recording = ASignal(np.column_stack((samples, samples * 0.5)), 48_000)
        graph = GraphData(
            "THD+N",
            np.array([20.0, 1000.0, 20_000.0]),
            np.array([0.1, 0.2, 0.3]),
            AxisSpec.FREQ,
            AxisSpec.THD,
            plot_type=PlotType.BARS,
        )
        measurement = Measurement(
            module_id="thd",
            name="Portable measurement",
            module_state={
                "recording": recording,
                "complex": np.array([1.0 + 2.0j, 3.0 - 4.0j]),
                "signature": (48_000, FrequencyBand(20.0, 20_000.0)),
                "special": float("inf"),
                "$bmm": "ordinary module value",
            },
            graphs=[graph],
            graph_colors={
                "THD+N": graph.color,
                "Phase": (10, 20, 30, 255),
            },
        )

        with TemporaryDirectory() as directory:
            saved_path = save_measurement(
                measurement,
                Path(directory) / "measurement",
            )
            loaded = load_measurement(saved_path)
            text = saved_path.read_text(encoding="utf-8")

        self.assertEqual(saved_path.suffix, ".bmm")
        self.assertIn(f'"format": "{MEASUREMENT_FORMAT}"', text)
        self.assertIn(AUDIO_ARRAY_ENCODING, text)
        self.assertNotEqual(loaded.id, measurement.id)
        self.assertNotEqual(loaded.graphs[0].id, graph.id)
        self.assertEqual(loaded.module_id, measurement.module_id)
        self.assertEqual(loaded.name, measurement.name)
        loaded_recording = loaded.module_state["recording"]
        self.assertIsInstance(loaded_recording, ASignal)
        self.assertEqual(loaded_recording.sample_rate, 48_000)
        np.testing.assert_array_equal(
            loaded_recording.as_array(),
            recording.as_array(),
        )
        np.testing.assert_array_equal(
            loaded.module_state["complex"],
            measurement.module_state["complex"],
        )
        self.assertEqual(loaded.module_state["signature"][1], FrequencyBand())
        self.assertTrue(np.isinf(loaded.module_state["special"]))
        self.assertEqual(loaded.module_state["$bmm"], "ordinary module value")
        np.testing.assert_array_equal(loaded.graphs[0].x, graph.x)
        np.testing.assert_array_equal(loaded.graphs[0].y, graph.y)
        self.assertEqual(loaded.graphs[0].plot_type, PlotType.BARS)
        self.assertEqual(loaded.graphs[0].color, graph.color)
        self.assertEqual(loaded.graph_colors, measurement.graph_colors)

    def test_invalid_measurement_document_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.bmm"
            path.write_text(
                '{"format": "bm-spectrum-measurement", "version": 99}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                MeasurementIOError,
                "unsupported measurement format version",
            ):
                load_measurement(path)

    def test_application_adds_import_as_active_and_visible(self) -> None:
        app = SpectrumApplication()
        existing = app.create_measurement()
        graph = GraphData(
            "Spectrum",
            np.array([20.0, 1000.0]),
            np.array([-30.0, -10.0]),
            AxisSpec.FREQ,
            AxisSpec.LEVEL,
        )
        imported = Measurement(
            module_id="spectrum",
            name="Imported",
            graphs=[graph],
        )

        app.add_imported_measurement(imported)

        self.assertEqual(app.app_state.measurements, [existing, imported])
        self.assertEqual(app.app_state.active_measurement_id, imported.id)
        self.assertIn(graph.id, app.app_state.visible_graph_ids)
        self.assertTrue(app.app_state.graph_data_changed)

    def test_application_rejects_measurement_for_missing_module(self) -> None:
        app = SpectrumApplication()
        imported = Measurement(module_id="missing", name="Unknown")

        with self.assertRaisesRegex(MeasurementIOError, "unavailable module"):
            app.add_imported_measurement(imported)

        self.assertNotIn(imported, app.app_state.measurements)


if __name__ == "__main__":
    unittest.main()
