import unittest
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import dearpygui.dearpygui as dpg
import numpy as np

from spectrum_app.files import (
    LegacyProjectError,
    PROJECT_PROGRESS_DIALOG,
    PROJECT_PROGRESS_BAR,
    PROJECT_PROGRESS_TEXT,
    PROJECT_WARNING_DIALOG,
    PROJECT_WARNING_TEXT,
    apply_project_view,
    complete_project_open_if_analysis_finished,
    collect_project_records,
    collect_project_view,
    collect_project_audio,
    encode_project_array,
    load_prepared_project_records,
    project_audio_from_record,
    project_path_from_dialog,
    process_project_results,
    validate_project_view,
    read_project_file,
    save_project_as,
    start_next_project_reanalysis,
    unlock_fft_yaxis_if_needed,
    update_project_progress,
    validate_project,
)
from spectrum_app.state import AppState
from spectrum_app.ui.sync import flush_project_warning, sync_record_selection
from spectrum_app.callbacks import record_used_click


def drain_project_results(state: AppState) -> None:
    for _ in range(100):
        process_project_results(state)
        if state.project_results.empty() and not state.project_progress_active:
            return
        time.sleep(0.01)
    process_project_results(state)


class ProjectFileTests(unittest.TestCase):
    def project_settings(self) -> dict:
        return {
            "analyzer_mode": "periodogram",
            "ref": "channel b",
            "weighting": "none",
            "welch_n": 8192,
            "band": [20.0, 20000.0],
            "window_width": 0.1,
            "freq_length": 1024,
            "window_func": "gaussian",
        }

    def test_project_audio_round_trip_preserves_source_record(self) -> None:
        state = AppState()
        state.records = [
            np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32),
        ]
        state.generator_signals = [
            np.array([0.5, -0.6], dtype=np.float32),
        ]
        state.record_sample_rates = [48000]

        audio = collect_project_audio(state, 0)
        self.assertIsNotNone(audio)
        self.assertIsInstance(audio["record"], dict)
        self.assertIsInstance(audio["record"]["data"], str)
        record, sample_rate, generator_signal = project_audio_from_record(
            {"audio": audio}
        )

        self.assertEqual(sample_rate, 48000)
        np.testing.assert_allclose(record, state.records[0])
        np.testing.assert_allclose(generator_signal, state.generator_signals[0])

    def test_validate_project_reports_legacy_project_version(self) -> None:
        with self.assertRaisesRegex(LegacyProjectError, "supported up to BM Spectrum 0.2.2"):
            validate_project(
                {
                    "format": "bm-spectrum-project",
                    "version": 1,
                    "records": [],
                }
            )

    def test_validate_project_accepts_current_project_audio(self) -> None:
        records = validate_project(
            {
                "format": "bm-spectrum-project",
                "version": 4,
                "records": [
                    {
                        "name": "record",
                        "visible": True,
                        "used": True,
                        "settings": self.project_settings(),
                        "audio": {
                            "sample_rate": 48000,
                            "record": encode_project_array(
                                np.array([[0.1, -0.2]], dtype=np.float32)
                            ),
                            "generator_signal": encode_project_array(
                                np.array([0.3], dtype=np.float32)
                            ),
                        },
                    }
                ],
            }
        )

        self.assertEqual(len(records), 1)

    def test_collect_project_records_stores_settings_not_fft_samples(self) -> None:
        dpg.create_context()
        try:
            state = AppState()
            state.records = [np.empty((0, 2), np.float32)]
            state.generator_signals = [np.empty(0, np.float32)]
            state.record_sample_rates = [0]
            state.record_settings = [self.project_settings()]
            with dpg.window():
                with dpg.plot():
                    dpg.add_plot_axis(dpg.mvXAxis)
                    with dpg.plot_axis(dpg.mvYAxis):
                        state.lines.append(dpg.add_line_series([1.0], [2.0]))
                state.lines_table_rows.append(
                    [
                        dpg.add_checkbox(default_value=True),
                        dpg.add_checkbox(default_value=True),
                        dpg.add_input_text(default_value="record 1"),
                    ]
                )

            record = collect_project_records(state)[0]
        finally:
            dpg.destroy_context()

        self.assertIn("settings", record)
        self.assertNotIn("x", record)
        self.assertNotIn("y", record)

    def test_collect_project_view_stores_fft_y_axis_limits(self) -> None:
        state = AppState()
        state.fft_yaxis = 123

        with patch("spectrum_app.files.dpg.get_axis_limits", return_value=[-60.0, 6.0]):
            view = collect_project_view(state)

        self.assertEqual(view, {"fft_yaxis": [-60.0, 6.0]})

    def test_validate_project_view_ignores_invalid_limits(self) -> None:
        self.assertEqual(validate_project_view({"fft_yaxis": [0.0, 0.0]}), {})
        self.assertEqual(validate_project_view({"fft_yaxis": [-30.0, 3.0]}), {"fft_yaxis": [-30.0, 3.0]})

    def test_save_project_writes_fft_y_axis_limits(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "project.bms"
            dpg.create_context()
            try:
                state = AppState()
                with dpg.window():
                    with dpg.plot():
                        dpg.add_plot_axis(dpg.mvXAxis)
                        with dpg.plot_axis(dpg.mvYAxis) as yaxis:
                            state.fft_yaxis = yaxis
                            state.lines.append(dpg.add_line_series([], []))
                    state.lines_table_rows.append(
                        [
                            dpg.add_checkbox(default_value=True),
                            dpg.add_checkbox(default_value=False),
                            dpg.add_input_text(default_value="record 1"),
                        ]
                    )
                with patch(
                    "spectrum_app.files.dpg.get_axis_limits",
                    return_value=[-55.0, 5.0],
                ):
                    save_project_as(
                        None,
                        {"current_path": directory, "file_name": "project.bms"},
                        state,
                    )
                drain_project_results(state)
            finally:
                dpg.destroy_context()

            project = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(project["view"]["fft_yaxis"], [-55.0, 5.0])

    def test_apply_project_view_restores_fft_y_axis_limits(self) -> None:
        dpg.create_context()
        try:
            state = AppState()
            with dpg.window():
                with dpg.plot():
                    with dpg.plot_axis(dpg.mvYAxis) as yaxis:
                        state.fft_yaxis = yaxis

            with patch("spectrum_app.files.dpg.set_axis_limits") as set_axis_limits:
                apply_project_view(state, {"fft_yaxis": [-70.0, 10.0]})

            set_axis_limits.assert_called_once_with(state.fft_yaxis, -70.0, 10.0)
            self.assertEqual(state.unlock_fft_yaxis_frames, 2)
        finally:
            dpg.destroy_context()

    def test_saved_fft_y_axis_limits_are_unlocked_after_restore(self) -> None:
        dpg.create_context()
        try:
            state = AppState()
            with dpg.window():
                with dpg.plot():
                    with dpg.plot_axis(dpg.mvYAxis) as yaxis:
                        state.fft_yaxis = yaxis
            state.unlock_fft_yaxis_frames = 2

            with patch("spectrum_app.files.dpg.set_axis_limits_auto") as auto_limits:
                unlock_fft_yaxis_if_needed(state)
                auto_limits.assert_not_called()
                unlock_fft_yaxis_if_needed(state)

            auto_limits.assert_called_once_with(state.fft_yaxis)
            self.assertEqual(state.unlock_fft_yaxis_frames, 0)
        finally:
            dpg.destroy_context()

    def test_project_open_fits_fft_y_axis_when_view_has_no_saved_limits(self) -> None:
        dpg.create_context()
        try:
            state = AppState()
            with dpg.window():
                with dpg.plot():
                    with dpg.plot_axis(dpg.mvYAxis) as yaxis:
                        state.fft_yaxis = yaxis
            state.project_progress_waiting_analyzer = True
            state.fit_fft_yaxis_after_project_open = True

            with patch("spectrum_app.files.dpg.fit_axis_data") as fit_axis_data:
                complete_project_open_if_analysis_finished(state)

            fit_axis_data.assert_called_once_with(state.fft_yaxis)
            self.assertFalse(state.fit_fft_yaxis_after_project_open)
        finally:
            dpg.destroy_context()

    def test_project_open_keeps_saved_fft_y_axis_when_view_has_limits(self) -> None:
        dpg.create_context()
        try:
            state = AppState()
            with dpg.window():
                with dpg.plot():
                    with dpg.plot_axis(dpg.mvYAxis) as yaxis:
                        state.fft_yaxis = yaxis
            state.project_progress_waiting_analyzer = True
            state.fit_fft_yaxis_after_project_open = False

            with patch("spectrum_app.files.dpg.fit_axis_data") as fit_axis_data:
                complete_project_open_if_analysis_finished(state)

            fit_axis_data.assert_not_called()
        finally:
            dpg.destroy_context()

    def test_read_project_restores_record_analysis_settings(self) -> None:
        settings = {
            "analyzer_mode": "welch",
            "ref": "none",
            "weighting": "pink",
            "welch_n": 4096,
            "band": [50.0, 12000.0],
            "window_width": 0.4,
            "freq_length": 777,
            "window_func": "cosine",
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "project.bms"
            path.write_text(
                json.dumps(
                    {
                        "format": "bm-spectrum-project",
                        "version": 4,
                        "records": [
                            {
                                "name": "record",
                                "visible": True,
                                "used": True,
                                "settings": settings,
                                "audio": {
                                    "sample_rate": 48000,
                                    "record": encode_project_array(
                                        np.array(
                                            [[0.1, 0.2], [0.3, 0.4]],
                                            dtype=np.float32,
                                        )
                                    ),
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dpg.create_context()
            try:
                state = AppState()
                with dpg.window():
                    with dpg.plot():
                        dpg.add_plot_axis(dpg.mvXAxis)
                        with dpg.plot_axis(dpg.mvYAxis):
                            state.lines.append(dpg.add_line_series([], []))
                    state.lines_table_rows.append(
                        [
                            dpg.add_checkbox(default_value=True),
                            dpg.add_checkbox(default_value=False),
                            dpg.add_input_text(default_value="record 1"),
                        ]
                    )
                    state.analyzer_mode_combo = dpg.add_combo(["welch", "periodogram"])
                    state.ref_combo = dpg.add_combo(["none", "channel b", "generator"])
                    state.weighting_combo = dpg.add_combo(["none", "pink"])
                    state.band_input = dpg.add_input_intx(size=2)
                    state.welch_n_input = dpg.add_input_int()
                    state.window_width_input = dpg.add_input_float()
                    state.freq_length_input = dpg.add_input_int()
                    state.window_func_combo = dpg.add_combo(["cosine", "gaussian"])

                read_project_file(state, path)

                self.assertEqual(state.analyzer.analyzer_mode.value, "welch")
                self.assertEqual(state.analyzer.ref.value, "none")
                self.assertEqual(state.analyzer.weighting.value, "pink")
                self.assertEqual(state.analyzer.welch_n, 4096)
                self.assertEqual(state.analyzer.band, (50.0, 12000.0))
                self.assertEqual(state.analyzer.window_width, 0.4)
                self.assertEqual(state.analyzer.freq_length, 777)
                self.assertEqual(state.analyzer.window_func.value, "cosine")
                self.assertEqual(dpg.get_value(state.analyzer_mode_combo), "welch")
                self.assertEqual(dpg.get_value(state.ref_combo), "none")
                self.assertEqual(dpg.get_value(state.weighting_combo), "pink")
            finally:
                dpg.destroy_context()

    def test_load_project_queues_fft_for_all_raw_records(self) -> None:
        dpg.create_context()
        try:
            state = AppState()
            with dpg.window():
                with dpg.plot():
                    dpg.add_plot_axis(dpg.mvXAxis)
                    with dpg.plot_axis(dpg.mvYAxis):
                        for _ in range(3):
                            state.lines.append(dpg.add_line_series([], []))
                for i in range(3):
                    state.lines_table_rows.append(
                        [
                            dpg.add_checkbox(default_value=i == 0),
                            dpg.add_checkbox(default_value=False),
                            dpg.add_input_text(default_value=f"record {i + 1}"),
                        ]
                    )

            records = []
            for i in range(3):
                records.append(
                    {
                        "name": f"record {i + 1}",
                        "visible": True,
                        "used": i == 0,
                        "settings": self.project_settings(),
                        "audio_record": np.ones((32, 2), np.float32) * (i + 1),
                        "sample_rate": 48000,
                        "generator_signal": np.empty(0, np.float32),
                    }
                )

            load_prepared_project_records(state, records)

            self.assertTrue(state.analyzer.running.is_set())
            self.assertEqual(state.analyzer_line_index, 0)
            self.assertEqual(state.project_reanalysis_queue, [1, 2])
            self.assertTrue(state.project_progress_waiting_analyzer)
        finally:
            dpg.destroy_context()

    def test_project_reanalysis_queue_starts_next_record(self) -> None:
        state = AppState()
        state.records = [
            np.ones((32, 2), np.float32),
            np.ones((32, 2), np.float32) * 2,
        ]
        state.generator_signals = [np.empty(0, np.float32), np.empty(0, np.float32)]
        state.record_sample_rates = [48000, 48000]
        state.record_settings = [self.project_settings(), self.project_settings()]
        state.project_reanalysis_queue = [1]

        started = start_next_project_reanalysis(state)

        self.assertTrue(started)
        self.assertTrue(state.analyzer.running.is_set())
        self.assertEqual(state.analyzer_line_index, 1)
        self.assertEqual(state.project_reanalysis_queue, [])

    def test_read_legacy_project_defers_warning_popup(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.bms"
            path.write_text(
                json.dumps(
                    {
                        "format": "bm-spectrum-project",
                        "version": 1,
                        "records": [],
                    }
                ),
                encoding="utf-8",
            )
            state = AppState()

            read_project_file(state, path)

            self.assertIn("BM Spectrum 0.2.2", state.pending_project_warning or "")
            self.assertEqual(state.pending_project_warning_frames, 2)

    def test_save_project_as_accepts_dialog_current_path_and_file_name(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "new_project.bms"
            app_data = {
                "current_path": directory,
                "file_name": "new_project.bms",
            }

            self.assertEqual(project_path_from_dialog(app_data), path)

            dpg.create_context()
            try:
                state = AppState()
                with dpg.window():
                    with dpg.plot():
                        dpg.add_plot_axis(dpg.mvXAxis)
                        with dpg.plot_axis(dpg.mvYAxis):
                            state.lines.append(dpg.add_line_series([], []))
                    state.lines_table_rows.append(
                        [
                            dpg.add_checkbox(default_value=True),
                            dpg.add_checkbox(default_value=False),
                            dpg.add_input_text(default_value="record 1"),
                        ]
                )
                save_project_as(None, app_data, state)
                drain_project_results(state)
            finally:
                dpg.destroy_context()

            self.assertTrue(path.exists())

    def test_save_project_as_accepts_dialog_selections(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "selected_project.bms"
            app_data = {
                "current_path": directory,
                "selections": {"selected_project.bms": str(path)},
            }

            self.assertEqual(project_path_from_dialog(app_data), path)

            dpg.create_context()
            try:
                state = AppState()
                with dpg.window():
                    with dpg.plot():
                        dpg.add_plot_axis(dpg.mvXAxis)
                        with dpg.plot_axis(dpg.mvYAxis):
                            state.lines.append(dpg.add_line_series([], []))
                    state.lines_table_rows.append(
                        [
                            dpg.add_checkbox(default_value=True),
                            dpg.add_checkbox(default_value=False),
                            dpg.add_input_text(default_value="record 1"),
                        ]
                )
                save_project_as(None, app_data, state)
                drain_project_results(state)
            finally:
                dpg.destroy_context()

            self.assertTrue(path.exists())

    def test_save_project_as_hides_dialog_and_keeps_record_selection_working(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "project.bms"
            dpg.create_context()
            try:
                dpg.create_viewport(title="test", width=800, height=600)
                state = AppState()
                with dpg.window():
                    with dpg.plot():
                        dpg.add_plot_axis(dpg.mvXAxis)
                        with dpg.plot_axis(dpg.mvYAxis):
                            for _ in range(5):
                                state.lines.append(dpg.add_line_series([], []))
                    for i in range(5):
                        state.lines_table_rows.append(
                            [
                                dpg.add_checkbox(default_value=i == 0, user_data=state),
                                dpg.add_checkbox(default_value=False, user_data=state),
                                dpg.add_input_text(
                                    default_value=f"record {i + 1}",
                                    user_data=state,
                                ),
                            ]
                        )
                dialog = dpg.add_file_dialog(
                    show=True,
                    callback=save_project_as,
                    user_data=state,
                    default_filename="project.bms",
                )

                save_project_as(
                    dialog,
                    {"current_path": directory, "file_name": "project.bms"},
                    state,
                )
                drain_project_results(state)

                self.assertTrue(path.exists())
                self.assertFalse(dpg.is_item_shown(dialog))
                used = state.lines_table_rows[1][0]
                dpg.set_value(used, True)
                record_used_click(used, True, dpg.get_item_user_data(used))
                self.assertEqual(
                    [dpg.get_value(row[0]) for row in state.lines_table_rows],
                    [False, True, False, False, False],
                )
                self.assertEqual(state.current_line, 1)
            finally:
                dpg.destroy_context()

    def test_record_selection_sync_repairs_multiple_used_checkmarks(self) -> None:
        dpg.create_context()
        try:
            state = AppState()
            with dpg.window():
                for i in range(5):
                    state.lines_table_rows.append(
                        [
                            dpg.add_checkbox(default_value=i in (0, 2), user_data=state),
                            dpg.add_checkbox(default_value=False, user_data=state),
                            dpg.add_input_text(
                                default_value=f"record {i + 1}",
                                user_data=state,
                            ),
                        ]
                    )
            state.current_line = 0

            sync_record_selection(state)

            self.assertEqual(state.current_line, 2)
            self.assertEqual(
                [dpg.get_value(row[0]) for row in state.lines_table_rows],
                [False, False, True, False, False],
            )
        finally:
            dpg.destroy_context()

    def test_project_path_combines_directory_file_path_name_with_file_name(self) -> None:
        with TemporaryDirectory() as directory:
            app_data = {
                "file_path_name": directory,
                "current_path": directory,
                "file_name": "typed_project.bms",
            }

            self.assertEqual(
                project_path_from_dialog(app_data),
                Path(directory) / "typed_project.bms",
            )

    def test_project_path_uses_dialog_default_filename_for_directory_selection(self) -> None:
        with TemporaryDirectory() as directory:
            dpg.create_context()
            try:
                dialog = dpg.add_file_dialog(
                    show=False,
                    default_filename="project.bms",
                )
                app_data = {
                    "file_path_name": directory,
                    "current_path": directory,
                }

                self.assertEqual(
                    project_path_from_dialog(app_data, dialog),
                    Path(directory) / "project.bms",
                )
            finally:
                dpg.destroy_context()

    def test_sync_ui_shows_deferred_project_warning_popup(self) -> None:
        dpg.create_context()
        try:
            state = AppState()
            with dpg.window(tag=PROJECT_WARNING_DIALOG, show=False):
                dpg.add_text("", tag=PROJECT_WARNING_TEXT)
            state.pending_project_warning = "legacy warning"
            state.pending_project_warning_frames = 2

            flush_project_warning(state)
            self.assertFalse(dpg.is_item_shown(PROJECT_WARNING_DIALOG))
            flush_project_warning(state)
            self.assertFalse(dpg.is_item_shown(PROJECT_WARNING_DIALOG))
            flush_project_warning(state)

            self.assertTrue(dpg.is_item_shown(PROJECT_WARNING_DIALOG))
        finally:
            dpg.destroy_context()

    def test_project_progress_modal_is_centered(self) -> None:
        dpg.create_context()
        try:
            dpg.create_viewport(width=800, height=600)
            state = AppState()
            with dpg.window(tag=PROJECT_PROGRESS_DIALOG, show=False, width=520, height=135):
                dpg.add_text("", tag=PROJECT_PROGRESS_TEXT)
                dpg.add_progress_bar(tag=PROJECT_PROGRESS_BAR)
            state.project_progress_active = True
            state.project_progress_title = "Saving project"
            state.project_progress_status = "Writing project"

            update_project_progress(state)

            x, y = dpg.get_item_pos(PROJECT_PROGRESS_DIALOG)
            self.assertGreaterEqual(x, 100)
            self.assertGreaterEqual(y, 100)
        finally:
            dpg.destroy_context()


if __name__ == "__main__":
    unittest.main()
