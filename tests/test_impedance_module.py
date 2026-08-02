import unittest
from unittest.mock import patch
from typing import cast
import time

import numpy as np
from numpy.typing import NDArray

from audioanalysis import (
    ASignal,
    FrequencyBand,
    ImpedanceConfig,
    ImpedanceResult,
    channel_calibration_config,
    generate_channel_calibration_signal,
    generate_measurement_signal,
)
from spectrum_app import SpectrumApplication
from spectrum_app.core.audio import AudioInput, AudioOutput
from spectrum_app.core.model import AxisSpec
from spectrum_app.modules.impedance import ImpedanceModule
from spectrum_app.modules.impedance.view import ImpedanceView
from tests.test_dpg_lifecycle import FakeDpgBackend


class PreparedAudioInput:
    sample_rate = 8_000
    channels = 2
    block_size = 512

    def __init__(self) -> None:
        self.data = np.zeros((1, 2), dtype=np.float32)
        self.position = 0

    def prepare(self, channel_1: NDArray, channel_2: NDArray) -> None:
        self.data = np.column_stack((channel_1, channel_2)).astype(np.float32)
        self.position = 0

    def open(self) -> bool:
        return True

    def read(self, samples: int) -> NDArray[np.float32]:
        end = self.position + samples
        block = self.data[self.position:end]
        self.position = end
        if len(block) < samples:
            block = np.vstack(
                (
                    block,
                    np.zeros((samples - len(block), 2), dtype=np.float32),
                )
            )
        return block

    def close(self) -> bool:
        return True


class DiscardingAudioOutput:
    sample_rate = 8_000
    channels = 2
    block_size = 512

    def open(self) -> bool:
        return True

    def write(self, data: NDArray) -> None:
        pass

    def close(self) -> bool:
        return True


class ImpedanceModuleTests(unittest.TestCase):
    def test_main_action_requests_calibration_until_module_is_ready(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement("impedance")
        module = cast(ImpedanceModule, app.module_manager.module("impedance"))

        with patch("spectrum_app.modules.impedance.view.dpg", backend):
            module.initialize(app)
            module.activate(measurement)
            try:
                self.assertEqual(module.measurement_button_label, "Calibrate")
                module.start_measurement()
                self.assertIn(
                    (
                        "configure_item",
                        ImpedanceView.CALIBRATION_DIALOG,
                        {"show": True},
                    ),
                    backend.calls,
                )

                measurement.module_state["workflow"] = "calibrated"
                self.assertEqual(module.measurement_button_label, "MEASURE")
            finally:
                module.deactivate()
                module.shutdown()

    def test_complete_calibration_and_measurement_workflow(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        audio_input = PreparedAudioInput()
        app.audio_input = cast(AudioInput, audio_input)
        app.audio_output = cast(AudioOutput, DiscardingAudioOutput())
        app.main_window.set_status_text = lambda text: None
        measurement = app.create_measurement("impedance")
        measurement.module_state.update(
            {
                "band": (20, 3_000),
                "duration": 0.1,
                "reference_resistor": 10.0,
                "calibration_resistor": 20.0,
                "points": 128,
            }
        )
        module = cast(ImpedanceModule, app.module_manager.module("impedance"))
        config = ImpedanceConfig(
            sample_rate=8_000,
            duration=0.1,
            reference_resistor=10.0,
            calibration_resistor=20.0,
            band=FrequencyBand(20, 3_000),
            points=128,
        )

        with patch("spectrum_app.modules.impedance.view.dpg", backend):
            module.initialize(app)
            module.activate(measurement)
            try:
                channel_signal = generate_channel_calibration_signal(
                    config
                ).as_array()[:, 0]
                audio_input.prepare(channel_signal, channel_signal * 0.8)
                module.continue_calibration()
                self._wait_for_workflow(module, measurement, "waiting_reference")

                measurement_signal = generate_measurement_signal(config).as_array()[:, 0]
                calibration_ratio = 20.0 / 30.0
                audio_input.prepare(
                    measurement_signal,
                    measurement_signal * calibration_ratio * 0.8,
                )
                module.continue_calibration()
                self._wait_for_workflow(module, measurement, "calibrated")

                load_resistance = 8.0
                load_ratio = load_resistance / (10.0 + load_resistance)
                audio_input.prepare(
                    measurement_signal,
                    measurement_signal * load_ratio * 0.8,
                )
                module.start_measurement()
                self._wait_for_workflow(module, measurement, "completed")

                self.assertFalse(app.app_state.measuring)
                self.assertIsInstance(
                    measurement.module_state["measurement_recording"],
                    ASignal,
                )
                self.assertAlmostEqual(
                    float(np.nanmedian(measurement.graphs[0].y)),
                    load_resistance,
                    places=3,
                )
                self.assertEqual(
                    [graph.y_axis for graph in measurement.graphs],
                    [AxisSpec.IMPEDANCE, AxisSpec.PHASE],
                )
            finally:
                module.deactivate()
                module.shutdown()

    def test_activate_adds_tools_and_restores_graphs(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement("impedance")
        frequency = np.geomspace(20.0, 20_000.0, 32)
        impedance = 8.0 + 1j * 2.0 * np.pi * frequency * 0.0004
        measurement.module_state.update(
            {
                "workflow": "completed",
                "frequency": frequency,
                "impedance": impedance,
            }
        )
        module = cast(ImpedanceModule, app.module_manager.module("impedance"))
        self.assertIsInstance(module, ImpedanceModule)

        with patch("spectrum_app.modules.impedance.view.dpg", backend):
            module.initialize(app)
            module.activate(measurement)

            spice_item = next(
                call
                for call in backend.calls
                if call[0] == "add_menu_item"
                and call[1].get("tag") == ImpedanceView.TOOLS_ITEM
            )
            self.assertEqual(spice_item[1]["label"], "SPICE Fit")
            self.assertEqual(spice_item[1]["parent"], app.main_window.tools_menu)
            test_tone_item = next(
                call
                for call in backend.calls
                if call[0] == "add_menu_item"
                and call[1].get("tag") == ImpedanceView.TEST_TONE_ITEM
            )
            self.assertEqual(test_tone_item[1]["label"], "Test tone")
            self.assertEqual(
                test_tone_item[1]["parent"], app.main_window.tools_menu
            )
            self.assertEqual(
                [graph.y_axis for graph in measurement.graphs],
                [AxisSpec.IMPEDANCE, AxisSpec.PHASE],
            )
            self.assertEqual(len(app.app_state.visible_graph_ids), 2)

            with patch.object(backend, "does_item_exist", return_value=True):
                module.deactivate()
            module.shutdown()

        self.assertIn(("delete_item", ImpedanceView.TOOLS_ITEM), backend.calls)
        self.assertIn(("delete_item", ImpedanceView.TEST_TONE_ITEM), backend.calls)

    def test_graphs_use_magnitude_and_unwrapped_phase(self) -> None:
        app = SpectrumApplication()
        measurement = app.create_measurement("impedance")
        module = cast(ImpedanceModule, app.module_manager.module("impedance"))
        module.initialize(app)
        module._measurement = measurement  # pyright: ignore[reportPrivateUsage]
        frequency = np.geomspace(20.0, 20_000.0, 8)
        expected_phase = np.linspace(-220.0, 220.0, 8)
        impedance = np.asarray(
            10.0 * np.exp(-1j * np.deg2rad(expected_phase)),
            dtype=np.complex128,
        )

        module._update_graphs(  # pyright: ignore[reportPrivateUsage]
            measurement,
            ImpedanceResult(frequency, impedance),
        )

        np.testing.assert_allclose(measurement.graphs[0].y, 10.0)
        np.testing.assert_allclose(
            np.diff(measurement.graphs[1].y),
            np.diff(expected_phase),
        )
        module.shutdown()

    @staticmethod
    def _wait_for_workflow(
        module: ImpedanceModule,
        measurement,
        workflow: str,
    ) -> None:
        deadline = time.monotonic() + 5.0
        while (
            measurement.module_state.get("workflow") != workflow
            and time.monotonic() < deadline
        ):
            module.update()
            time.sleep(0.002)
        module.update()
        if measurement.module_state.get("workflow") != workflow:
            raise AssertionError(
                f"Expected {workflow}, got {measurement.module_state.get('workflow')}: "
                f"{measurement.module_state.get('status')}"
            )


if __name__ == "__main__":
    unittest.main()
