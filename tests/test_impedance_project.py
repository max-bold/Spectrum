import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from spectrum_app.impedance.imp_measure import (
    CalibrationStage,
    ImpedanceAppState,
    MeasurementConfig,
    MeasurementState,
    PhaseDisplayMode,
)
from spectrum_app.impedance.project import (
    load_impedance_project,
    save_impedance_project,
)


class ImpedanceProjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls = 0
        self.reference_resistor = 10.0
        self.calibration_resistor = 20.0
        self.load_resistance = 8.0

        def recorder(signal, config, level_callback):
            if self.calls == 0:
                ratio = 1.0
            elif self.calls == 1:
                ratio = self.calibration_resistor / (
                    self.reference_resistor + self.calibration_resistor
                )
            else:
                ratio = self.load_resistance / (
                    self.reference_resistor + self.load_resistance
                )
            self.calls += 1
            recording = np.column_stack((signal, signal * ratio)).astype(
                np.float32
            )
            level_callback(
                (
                    float(np.max(np.abs(recording[:, 0]))),
                    float(np.max(np.abs(recording[:, 1]))),
                )
            )
            return recording

        self.state = ImpedanceAppState(recorder=recorder)
        self.config = MeasurementConfig(
            sample_rate=48000,
            duration=0.1,
            reference_resistor=self.reference_resistor,
            calibration_resistor=self.calibration_resistor,
            f_min=100.0,
            f_max=10000.0,
            points=64,
            recording_tail=0.0,
            spice_min_sections=0,
            spice_max_sections=0,
            spice_max_evaluations=200,
        )

    def complete_measurement(self) -> None:
        self.assertTrue(self.state.start_calibration(self.config))
        self.state.wait(10)
        self.assertTrue(self.state.continue_calibration())
        self.state.wait(10)
        self.assertTrue(self.state.start_measurement(self.config))
        self.state.wait(10)

    def test_bmi_round_trip_restores_complete_measurement(self) -> None:
        self.complete_measurement()
        self.assertTrue(self.state.request_spice_model())
        self.state.wait(10)
        original = self.state.export_project(PhaseDisplayMode.DERIVATIVE)

        with TemporaryDirectory() as directory:
            saved_path = save_impedance_project(
                Path(directory) / "speaker",
                original,
            )
            self.assertEqual(saved_path.suffix, ".bmi")
            self.assertEqual(saved_path.read_bytes()[:2], b"PK")
            loaded = load_impedance_project(saved_path)

        self.assertEqual(loaded.state, MeasurementState.MEASURING_COMPLETED)
        self.assertEqual(loaded.phase_mode, PhaseDisplayMode.DERIVATIVE)
        self.assertEqual(loaded.result_config, self.config)
        self.assertEqual(
            loaded.channel_calibration_recording_sample_rate,
            self.config.sample_rate,
        )
        self.assertEqual(
            loaded.calibration_signal_sample_rate,
            self.config.sample_rate,
        )
        self.assertEqual(
            loaded.measurement_recording_sample_rate,
            self.config.sample_rate,
        )
        for name in (
            "channel_calibration_recording",
            "channel_calibration_signal",
            "calibration_recording",
            "calibration_signal",
            "measurement_recording",
            "measurement_signal",
            "calibration_frequency",
            "calibration_impedance",
            "calibration_phase",
            "calibration_phase_derivative",
            "channel_correction",
            "frequency",
            "impedance",
            "phase",
            "phase_derivative",
        ):
            np.testing.assert_array_equal(
                getattr(loaded, name),
                getattr(original, name),
            )
        self.assertAlmostEqual(
            loaded.reference_resistor_estimated,
            original.reference_resistor_estimated,
        )
        self.assertEqual(
            loaded.reference_diagnostics,
            original.reference_diagnostics,
        )
        np.testing.assert_array_equal(
            loaded.fit_result.physical_params,
            original.fit_result.physical_params,
        )
        self.assertEqual(loaded.spice_values, original.spice_values)

        restored_state = ImpedanceAppState()
        restored_state.restore_project(loaded)
        snapshot = restored_state.snapshot()
        self.assertEqual(snapshot.state, MeasurementState.MEASURING_COMPLETED)
        np.testing.assert_array_equal(snapshot.frequency, original.frequency)
        np.testing.assert_array_equal(snapshot.impedance, original.impedance)
        self.assertEqual(snapshot.spice_values, original.spice_values)

    def test_stage_one_project_can_be_restored(self) -> None:
        self.assertTrue(self.state.start_calibration(self.config))
        self.state.wait(10)
        project = self.state.export_project(PhaseDisplayMode.ANGLE)

        self.assertEqual(project.state, MeasurementState.CALIBRATING)
        self.assertEqual(
            project.calibration_stage,
            CalibrationStage.WAITING_REFERENCE,
        )

        restored_state = ImpedanceAppState()
        restored_state.restore_project(project)
        snapshot = restored_state.snapshot()
        self.assertEqual(snapshot.state, MeasurementState.CALIBRATING)
        self.assertEqual(
            snapshot.calibration_stage,
            CalibrationStage.WAITING_REFERENCE,
        )

    def test_invalid_bmi_file_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.bmi"
            path.write_bytes(b"not a BMI project")
            with self.assertRaisesRegex(ValueError, "Cannot read BMI project"):
                load_impedance_project(path)


if __name__ == "__main__":
    unittest.main()
