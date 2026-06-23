import unittest
from unittest.mock import patch

import dearpygui.dearpygui as dpg
import numpy as np

from spectrum_app.runtime import run_analyzer
from spectrum_app.state import AppState


class RuntimeTests(unittest.TestCase):
    def test_reanalysis_does_not_fit_fft_y_axis(self) -> None:
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
            state.analyzer.result = np.array([[1.0, 2.0], [-10.0, -20.0]])
            state.analyzer.completed.set()
            state.analyzer_line_index = 0

            with patch("spectrum_app.runtime.dpg.fit_axis_data") as fit_axis_data:
                run_analyzer(state)

            fit_axis_data.assert_not_called()
        finally:
            dpg.destroy_context()

    def test_new_completed_record_fits_fft_y_axis(self) -> None:
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
            state.analyzer.result = np.array([[1.0, 2.0], [-10.0, -20.0]])
            state.analyzer.completed.set()
            state.analyzer_line_index = 0
            state.completed_audio_record = np.ones((4, 2), np.float32)
            state.completed_audio_sample_rate = 48000

            with patch("spectrum_app.runtime.dpg.fit_axis_data") as fit_axis_data:
                run_analyzer(state)

            fit_axis_data.assert_called_once_with(state.fft_yaxis)
        finally:
            dpg.destroy_context()


if __name__ == "__main__":
    unittest.main()
