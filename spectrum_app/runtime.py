import dearpygui.dearpygui as dpg

from .models import AnalyzerMode

from .analysis import (
    current_generator_signal,
    process_pending_reanalysis,
    raw_audio_record,
    save_active_audio_record,
)
from .state import AppState


def reenable_io_updater(state: AppState) -> None:
    if state.io_reenable_timer is None:
        return
    if (
        not state.audio_io.running.is_set()
        and not state.meter.enable.is_set()
        and not state.pending_audio_start
        and not state.pending_meter_start
        and not state.io_upd.enable.is_set()
    ):
        state.io_reenable_timer.enabled.set()
    else:
        state.io_reenable_timer.enabled.clear()


def reenable_io_udater(state: AppState) -> None:
    reenable_io_updater(state)


def process_pending_requests(state: AppState) -> None:
    if state.pending_audio_start and state.io_upd.paused.is_set():
        state.pending_audio_start = False
        state.audio_io.running.set()

    if state.pending_meter_start and state.io_upd.paused.is_set():
        state.pending_meter_start = False
        state.meter.enable.set()


def run_analyzer(state: AppState) -> None:
    if not state.analyzer.running.is_set():
        if state.audio_io.running.is_set() and state.audio_io.record_updated.is_set():
            state.analyzer.analyzer_mode = AnalyzerMode.WELCH
            record_ready = True
        elif state.audio_io.record_completed.is_set():
            state.audio_io.record_completed.clear()
            state.analyzer.analyzer_mode = AnalyzerMode.PERIODIOGRAM
            record_ready = True
        else:
            record_ready = False

        if record_ready:
            state.analyzer_line_index = state.current_line
            state.analyzer.sample_rate = state.audio_io.in_fs
            state.analyzer.record = state.audio_io.get_record()
            if state.analyzer.analyzer_mode == AnalyzerMode.PERIODIOGRAM:
                state.completed_audio_record = raw_audio_record(state)
                state.completed_generator_signal = current_generator_signal(
                    state,
                    len(state.completed_audio_record),
                )
                state.completed_audio_sample_rate = int(state.audio_io.in_fs)
            state.analyzer.running.set()

    if state.analyzer.completed.is_set():
        state.analyzer.completed.clear()
        fft_data = state.analyzer.result.copy()
        line_index = state.analyzer_line_index
        if state.completed_audio_record is not None:
            save_active_audio_record(state, state.completed_audio_record)
            state.completed_audio_record = None
            state.completed_generator_signal = None
            state.completed_audio_sample_rate = 0
        if line_index >= len(state.lines):
            return
        line = state.lines[line_index]
        dpg.set_value(line, [fft_data[0].tolist(), fft_data[1].tolist()])
        dpg.show_item(line)
        dpg.set_value(state.lines_table_rows[line_index][1], True)
        if state.fft_yaxis is not None:
            dpg.fit_axis_data(state.fft_yaxis)
        process_pending_reanalysis(state)
    else:
        process_pending_reanalysis(state)

    if state.audio_io.levels_updated.is_set():
        ts, levels = state.audio_io.get_levels(0.01)
        if state.levels_l and state.levels_r:
            dpg.set_value(state.levels_l, [list(ts), list(levels[0])])
            dpg.set_value(state.levels_r, [list(ts), list(levels[1])])
