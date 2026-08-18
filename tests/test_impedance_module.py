import unittest
from unittest.mock import MagicMock, patch
from typing import cast
import time

import numpy as np
from numpy.typing import NDArray

from audioanalysis import (
    ASignal,
    FrequencyBand,
    ImpedanceConfig,
    ImpedanceResult,
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
    blocksize = 512

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
    blocksize = 512

    def open(self) -> bool:
        return True

    def write(self, data: NDArray) -> None:
        pass

    def close(self) -> bool:
        return True


class ImpedanceModuleTests(unittest.TestCase):
    def test_calibration_error_uses_popup_and_short_status(self) -> None:
        app = SpectrumApplication()
        app.main_window.set_status_text = MagicMock()
        app.main_window.show_error = MagicMock()
        measurement = app.create_measurement("impedance")
        module = cast(ImpedanceModule, app.module_manager.module("impedance"))

        with (
            patch.object(ImpedanceView, "build"),
            patch.object(ImpedanceView, "destroy"),
            patch.object(ImpedanceView, "update"),
            patch.object(ImpedanceView, "update_status"),
            patch.object(ImpedanceView, "set_enabled"),
            patch.object(ImpedanceView, "hide_calibration"),
            patch.object(ImpedanceView, "show_calibration_stage") as show_stage,
        ):
            module.initialize(app)
            module.activate(measurement)
            try:
                measurement.module_state["workflow"] = "waiting_reference"
                measurement.module_state["channel_correction"] = np.ones(
                    8,
                    dtype=np.complex128,
                )
                module._fail(
                    "Detailed calibration diagnostics",
                    fallback="waiting_reference",
                    calibration=True,
                )

                self.assertEqual(
                    measurement.module_state["status"],
                    "Calibration failed",
                )
                self.assertEqual(
                    measurement.module_state["workflow"],
                    "uncalibrated",
                )
                self.assertIsNone(measurement.module_state["channel_correction"])
                app.main_window.show_error.assert_called_once_with(
                    "Impedance calibration failed",
                    "Detailed calibration diagnostics",
                )
                module.show_calibration()
                show_stage.assert_called_once_with(1)
            finally:
                module.deactivate()
                module.shutdown()

    def test_main_action_requests_calibration_until_module_is_ready(self) -> None:
        backend = FakeDpgBackend()
        app = SpectrumApplication()
        measurement = app.create_measurement("impedance")
        module = cast(ImpedanceModule, app.module_manager.module("impedance"))

        with (
            patch("spectrum_app.modules.impedance.view.dpg", backend),
            patch("spectrum_app.gui.controls.level_meter.dpg", backend),
        ):
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
                self.assertIn(
                    (
                        "set_item_pos",
                        ImpedanceView.CALIBRATION_DIALOG,
                        [330.0, 250.0],
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

        with (
            patch("spectrum_app.modules.impedance.view.dpg", backend),
            patch("spectrum_app.gui.controls.level_meter.dpg", backend),
        ):
            module.initialize(app)
            module.activate(measurement)
            try:
                channel_signal = generate_channel_calibration_signal(
                    config
                ).as_array()[:, 0]
                audio_input.prepare(channel_signal, channel_signal * 0.8)
                module.continue_calibration()
                self._wait_for_workflow(module, measurement, "waiting_reference")

                channel_recording = measurement.module_state[
                    "channel_calibration_recording"
                ]
                channel_correction = measurement.module_state["channel_correction"]
                module.set_setting("window_width", 0.15)
                self.assertFalse(app.app_state.measuring)
                self.assertEqual(
                    measurement.module_state["workflow"],
                    "waiting_reference",
                )
                self.assertEqual(
                    measurement.module_state["status"],
                    "Smoothing will apply to calibration Stage 2",
                )
                self.assertIs(
                    measurement.module_state["channel_calibration_recording"],
                    channel_recording,
                )
                self.assertIs(
                    measurement.module_state["channel_correction"],
                    channel_correction,
                )

                measurement_signal = generate_measurement_signal(config).as_array()[:, 0]
                calibration_ratio = 20.0 / 30.0
                audio_input.prepare(
                    measurement_signal,
                    measurement_signal * calibration_ratio * 0.8,
                )
                module.continue_calibration()
                self._wait_for_workflow(module, measurement, "calibrated")

                calibration_recordings = (
                    measurement.module_state["channel_calibration_recording"],
                    measurement.module_state["reference_calibration_recording"],
                )
                calibration_graph_ids = [graph.id for graph in measurement.graphs]
                with patch(
                    "spectrum_app.modules.impedance.module.calculate_channel_correction",
                    side_effect=AssertionError("Stage 1 must not be reanalyzed"),
                ):
                    module.set_setting("points", 96)
                    self._wait_for_idle(module, app)

                self.assertEqual(measurement.module_state["workflow"], "calibrated")
                self.assertEqual(
                    measurement.module_state["status"],
                    "Calibration reprocessed",
                )
                self.assertEqual(len(measurement.graphs[0].x), 96)
                self.assertEqual(
                    [graph.id for graph in measurement.graphs],
                    calibration_graph_ids,
                )
                self.assertIs(
                    measurement.module_state["channel_calibration_recording"],
                    calibration_recordings[0],
                )
                self.assertIs(
                    measurement.module_state["reference_calibration_recording"],
                    calibration_recordings[1],
                )

                load_resistance = 8.0
                load_ratio = load_resistance / (10.0 + load_resistance)
                audio_input.prepare(
                    measurement_signal,
                    measurement_signal * load_ratio * 0.8,
                )
                module.start_measurement()
                self._wait_for_workflow(module, measurement, "completed")

                measurement_recording = measurement.module_state[
                    "measurement_recording"
                ]
                measurement_graph_ids = [graph.id for graph in measurement.graphs]
                with patch(
                    "spectrum_app.modules.impedance.module.calculate_channel_correction",
                    side_effect=AssertionError("Stage 1 must not be reanalyzed"),
                ):
                    module.set_setting("window_width", 0.2)
                    self._wait_for_idle(module, app)

                self.assertFalse(app.app_state.measuring)
                self.assertEqual(
                    measurement.module_state["status"],
                    "Measurement reprocessed",
                )
                self.assertIs(
                    measurement.module_state["measurement_recording"],
                    measurement_recording,
                )
                self.assertEqual(
                    [graph.id for graph in measurement.graphs],
                    measurement_graph_ids,
                )
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

        with (
            patch("spectrum_app.modules.impedance.view.dpg", backend),
            patch("spectrum_app.gui.controls.level_meter.dpg", backend),
        ):
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
            calibrate_item = next(
                call
                for call in backend.calls
                if call[0] == "add_menu_item"
                and call[1].get("tag") == ImpedanceView.CALIBRATE_ITEM
            )
            self.assertEqual(calibrate_item[1]["label"], "Calibrate")
            meter = next(
                call
                for call in backend.calls
                if call[0] == "drawlist"
                and call[1].get("tag") == ImpedanceView.LEVEL_METER
            )
            self.assertEqual(meter[1]["width"], 55)

            smoothing_inputs = {
                call[1]["tag"]: call[1]
                for call in backend.calls
                if call[0] in ("add_input_float", "add_input_int")
                and call[1].get("tag")
                in (ImpedanceView.WINDOW_WIDTH, ImpedanceView.POINTS)
            }
            self.assertTrue(
                smoothing_inputs[ImpedanceView.WINDOW_WIDTH]["on_enter"]
            )
            self.assertTrue(smoothing_inputs[ImpedanceView.POINTS]["on_enter"])
            self.assertIn(
                (
                    "bind_item_handler_registry",
                    ImpedanceView.WINDOW_WIDTH,
                    ImpedanceView.WINDOW_WIDTH_HANDLERS,
                ),
                backend.calls,
            )
            self.assertIn(
                (
                    "bind_item_handler_registry",
                    ImpedanceView.POINTS,
                    ImpedanceView.POINTS_HANDLERS,
                ),
                backend.calls,
            )

            measurement.module_state["workflow"] = "completed"
            calibrate_item[1]["callback"]()
            self.assertIn(
                (
                    "configure_item",
                    ImpedanceView.CALIBRATION_RESISTORS,
                    {"show": True},
                ),
                backend.calls,
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
        self.assertIn(("delete_item", ImpedanceView.CALIBRATE_ITEM), backend.calls)

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

    @staticmethod
    def _wait_for_idle(module: ImpedanceModule, app: SpectrumApplication) -> None:
        deadline = time.monotonic() + 5.0
        while app.app_state.measuring and time.monotonic() < deadline:
            module.update()
            time.sleep(0.002)
        module.update()
        if app.app_state.measuring:
            raise AssertionError("Impedance reprocessing did not finish")


if __name__ == "__main__":
    unittest.main()
