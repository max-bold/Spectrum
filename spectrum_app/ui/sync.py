from time import monotonic

import dearpygui.dearpygui as dpg

from ..analysis import sync_band_limits, sync_reference_options, sync_welch_limit
from ..callbacks import upd_io, upd_level_monitor
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
    sync_io_lists(state, refs)

    if dpg.get_value(refs.mon_cb):
        upd_level_monitor(state, (refs.left_level, refs.right_level))

    sync_run_button(state, refs)
    process_pending_requests(state)
    reenable_io_updater(state)

    max_t = dpg.get_value(refs.rec_len) + 2
    dpg.set_axis_limits(refs.levels_xaxis, 0, max_t)
    run_analyzer(state)
    dpg.set_value(refs.mon_cb, state.meter.enable.is_set() or state.pending_meter_start)
