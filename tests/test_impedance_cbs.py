import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from spectrum_app.impedance import cbs
from spectrum_app.impedance.imp_measure import (
    MeasurementConfig,
    MeasurementState,
    PhaseDisplayMode,
    WindowFunction,
)


class PhaseAxisTests(unittest.TestCase):
    def test_angle_uses_fixed_phase_limits(self) -> None:
        with patch.object(cbs, "dpg") as dpg:
            ui = unittest.mock.Mock(
                phase_axis="phase-axis",
                unlock_phase_axis_frames=1,
            )
            cbs.configure_phase_axis(
                ui,
                PhaseDisplayMode.ANGLE,
                [-20.0, 30.0],
            )

        dpg.set_axis_limits.assert_called_once_with(
            "phase-axis",
            -180.0,
            180.0,
        )
        dpg.fit_axis_data.assert_not_called()
        self.assertEqual(ui.unlock_phase_axis_frames, 0)

    def test_derivative_sets_limits_and_schedules_unlock(self) -> None:
        with patch.object(cbs, "dpg") as dpg:
            ui = unittest.mock.Mock(
                phase_axis="phase-axis",
                unlock_phase_axis_frames=0,
            )
            cbs.configure_phase_axis(
                ui,
                PhaseDisplayMode.DERIVATIVE,
                [-100.0, 200.0],
            )

        dpg.set_axis_limits.assert_called_once_with(
            "phase-axis",
            -115.0,
            215.0,
        )
        dpg.set_axis_limits_auto.assert_not_called()
        dpg.fit_axis_data.assert_not_called()
        self.assertEqual(ui.unlock_phase_axis_frames, 2)

    def test_pending_axes_follow_main_app_frame_sequence(self) -> None:
        ui = unittest.mock.Mock(
            impedance_axis="z-axis",
            phase_axis="phase-axis",
            unlock_impedance_axis_frames=2,
            unlock_phase_axis_frames=2,
        )
        with patch.object(cbs, "dpg") as dpg:
            cbs.update_pending_axis_limits(ui)
            dpg.set_axis_limits_auto.assert_not_called()
            dpg.fit_axis_data.assert_not_called()

            dpg.reset_mock()
            cbs.update_pending_axis_limits(ui)

        self.assertEqual(
            dpg.set_axis_limits_auto.call_args_list,
            [
                unittest.mock.call("z-axis"),
                unittest.mock.call("phase-axis"),
            ],
        )
        dpg.fit_axis_data.assert_not_called()
        self.assertEqual(ui.unlock_impedance_axis_frames, 0)
        self.assertEqual(ui.unlock_phase_axis_frames, 0)


class MeasurementUiTests(unittest.TestCase):
    def test_measure_button_stops_active_measurement(self) -> None:
        state = unittest.mock.Mock()
        state.snapshot.return_value = SimpleNamespace(
            state=MeasurementState.MEASURING,
        )
        ui = SimpleNamespace(state=state)

        with patch.object(cbs, "pause_io_updater") as pause_io:
            cbs.start_measurement(None, None, ui)

        state.stop_measurement.assert_called_once_with()
        state.start_measurement.assert_not_called()
        pause_io.assert_not_called()


class ProjectUiTests(unittest.TestCase):
    def test_save_always_opens_path_dialog_for_loaded_project(self) -> None:
        ui = SimpleNamespace(
            project_path=Path("C:/measurements/opened.bmi"),
            save_project_dialog="save-dialog",
        )

        with (
            patch.object(cbs, "dpg") as dpg,
            patch.object(cbs, "start_project_save") as start_save,
        ):
            cbs.save_project_menu(None, None, ui)

        dpg.show_item.assert_called_once_with("save-dialog")
        start_save.assert_not_called()

    def test_project_dialog_path_gets_bmi_extension(self) -> None:
        path = cbs.project_path_from_dialog(
            {
                "current_path": "C:/measurements",
                "file_name": "woofer",
            },
            ensure_extension=True,
        )

        self.assertEqual(path, Path("C:/measurements/woofer.bmi"))

    def test_project_controls_are_restored(self) -> None:
        config = MeasurementConfig(
            duration=12.0,
            reference_resistor=3.0,
            calibration_resistor=9.8,
            f_min=30.0,
            f_max=18000.0,
            window_width=0.2,
            points=512,
            window_function=WindowFunction.COSINE,
            block_size=2048,
        )
        settings = SimpleNamespace(
            audio=SimpleNamespace(block_size=1024),
            save=unittest.mock.Mock(),
        )
        ui = SimpleNamespace(
            settings=settings,
            block_size_input="block-size",
            phase_angle_menu_item="angle-menu",
            phase_derivative_menu_item="dir-menu",
            phase_mode=PhaseDisplayMode.ANGLE,
        )
        project = SimpleNamespace(
            result_config=config,
            calibration_config=config,
            phase_mode=PhaseDisplayMode.DERIVATIVE,
        )

        with patch.object(cbs, "dpg") as dpg:
            cbs.apply_project_controls(ui, project)

        self.assertEqual(settings.audio.block_size, 2048)
        settings.save.assert_called_once_with()
        self.assertEqual(ui.phase_mode, PhaseDisplayMode.DERIVATIVE)
        self.assertIn(
            unittest.mock.call("impedance_duration_input", 12.0),
            dpg.set_value.call_args_list,
        )
        self.assertIn(
            unittest.mock.call("impedance_freq_length_input", 512),
            dpg.set_value.call_args_list,
        )
        self.assertIn(
            unittest.mock.call("impedance_window_func_input", "cosine"),
            dpg.set_value.call_args_list,
        )

if __name__ == "__main__":
    unittest.main()
