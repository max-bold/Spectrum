import unittest

import numpy as np

from spectrum_app.core.model import (
    GRAPH_COLORS,
    AppState,
    AxisSpec,
    GraphData,
    Measurement,
)


class CoreModelTests(unittest.TestCase):
    def test_standard_graph_palette_starts_with_white(self) -> None:
        self.assertEqual(GRAPH_COLORS[0], (255, 255, 255, 255))
        self.assertEqual(len(GRAPH_COLORS), 20)
        self.assertEqual(len(set(GRAPH_COLORS)), 20)

    def test_measurement_remembers_color_when_graph_is_recreated(self) -> None:
        color = (12, 34, 56, 255)
        measurement = Measurement(module_id="spectrum", name="Measurement 1")
        measurement.graphs.append(
            GraphData(
                "Spectrum",
                np.array([20.0]),
                np.array([-10.0]),
                AxisSpec.FREQ,
                AxisSpec.LEVEL,
                color=color,
            )
        )

        measurement.remember_graph_colors()
        measurement.graphs.clear()
        replacement = GraphData(
            "Spectrum",
            np.array([20.0]),
            np.array([-20.0]),
            AxisSpec.FREQ,
            AxisSpec.LEVEL,
            color=measurement.color_for_graph("Spectrum"),
        )

        self.assertEqual(replacement.color, color)

    def test_measurement_can_store_multiple_graphs(self) -> None:
        frequency = AxisSpec.FREQ
        level = AxisSpec.LEVEL
        phase = AxisSpec.PHASE
        measurement = Measurement(module_id="spectrum", name="Measurement 1")

        measurement.graphs.append(
            GraphData(
                name="Spectrum",
                x=np.array([20.0, 1000.0]),
                y=np.array([-40.0, -10.0]),
                x_axis=frequency,
                y_axis=level,
            )
        )
        measurement.graphs.append(
            GraphData(
                name="Phase",
                x=np.array([20.0, 1000.0]),
                y=np.array([0.0, 45.0]),
                x_axis=frequency,
                y_axis=phase,
            )
        )

        self.assertEqual(len(measurement.graphs), 2)
        self.assertNotEqual(measurement.graphs[0].id, measurement.graphs[1].id)

    def test_mutable_defaults_are_not_shared(self) -> None:
        first = AppState()
        second = AppState()

        first.measurements.append(Measurement("spectrum", "Measurement 1"))
        first.interface_state["panel"] = "bottom"

        self.assertEqual(second.measurements, [])
        self.assertEqual(second.interface_state, {})


if __name__ == "__main__":
    unittest.main()
