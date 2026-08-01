from time import monotonic

import dearpygui.dearpygui as dpg

from ..analysis import sync_band_limits, sync_reference_options, sync_welch_limit
from ..callbacks import select_record_line, upd_io
from ..files import (
    complete_project_open_if_analysis_finished,
    process_project_results,
    show_project_warning,
    unlock_fft_yaxis_if_needed,
    update_project_progress,
)
from ..runtime import process_pending_requests, reenable_io_updater, run_analyzer
from ..state import AppState
from .refs import UiRefs


def sync_io_lists(state: AppState, refs: UiRefs) -> None:
    now = monotonic()
    if now - refs.last_io_update >= 0.5:
        upd_io(state, refs.inputs_combo, refs.outputs_combo)
        sync_reference_options(state, refs.ref_combo)
        sync_band_limits(state, refs.band_input)
        sync_welch_limit(state, refs.welch_n_input)
        dpg.set_value(refs.block_size_input, state.settings.audio.block_size)
        refs.last_io_update = now


def sync_run_button(state: AppState, refs: UiRefs) -> None:
    from ..themes import green_theme, red_theme

    if state.audio_io.running.is_set() or state.pending_audio_start:
        dpg.set_item_label(refs.run_btn, "ON")
        dpg.bind_item_theme(refs.run_btn, red_theme)
    else:
        dpg.set_item_label(refs.run_btn, "OFF")
        dpg.bind_item_theme(refs.run_btn, green_theme)


def sync_ui(state: AppState, refs: UiRefs) -> None:
    process_project_results(state)
    sync_io_lists(state, refs)

    flush_project_warning(state)
    update_project_progress(state)
    unlock_fft_yaxis_if_needed(state)
    sync_record_selection(state)
    sync_run_button(state, refs)
    process_pending_requests(state)
    reenable_io_updater(state)

    max_t = max(dpg.get_value(refs.rec_len) + 2, active_record_duration(state))
    dpg.set_axis_limits(refs.levels_xaxis, 0, max_t)
    run_analyzer(state)
    complete_project_open_if_analysis_finished(state)


def sync_record_selection(state: AppState) -> None:
    checked = [
        i for i, row in enumerate(state.lines_table_rows) if bool(dpg.get_value(row[0]))
    ]
    if len(checked) == 1 and checked[0] == state.current_line:
        return
    if checked:
        if state.current_line in checked and len(checked) > 1:
            index = next(i for i in checked if i != state.current_line)
        else:
            index = checked[0]
    else:
        index = state.current_line
    select_record_line(state, index)


def flush_project_warning(state: AppState) -> None:
    if state.pending_project_warning is None:
        return
    if state.pending_project_warning_frames > 0:
        state.pending_project_warning_frames -= 1
        return
    message = state.pending_project_warning
    state.pending_project_warning = None
    show_project_warning(message)


def active_record_duration(state: AppState) -> float:
    if (
        state.current_line >= len(state.records)
        or state.current_line >= len(state.record_sample_rates)
    ):
        return 0.0
    sample_rate = state.record_sample_rates[state.current_line]
    if sample_rate <= 0:
        return 0.0
    return len(state.records[state.current_line]) / sample_rate
