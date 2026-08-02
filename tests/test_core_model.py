import unittest

import numpy as np

from spectrum_app.core.model import AppState, AxisSpec, GraphData, Measurement


class CoreModelTests(unittest.TestCase):
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
