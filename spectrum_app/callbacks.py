import dearpygui.dearpygui as dpg

from .models import AnalyzerMode, GenMode, RefMode, WeightingMode
from utils.windows import Windows

from .analysis import (
    available_ref_modes,
    clamp_band,
    clamp_freq_length,
    clamp_length,
    clamp_welch_n,
    clamp_window_width,
    ensure_current_reference_is_available,
    request_current_record_reanalysis,
    sync_welch_limit,
)
from .state import AppState
from .settings import DEFAULT_INPUT, DEFAULT_OUTPUT, resolve_device


def run_btn(sender=None, app_data=None, user_data: AppState | None = None):
    state = user_data
    if state is None:
        return
    if state.pending_audio_start:
        state.pending_audio_start = False
        state.io_upd.enable.set()
        return
    if not state.audio_io.running.is_set():
        state.io_upd.enable.clear()
        state.meter.enable.clear()
        state.pending_meter_start = False
        state.pending_audio_start = True
    else:
        state.pending_audio_start = False
        state.audio_io.stop_audio()
        state.io_upd.enable.set()


def set_genmode(source: int, mode: str, state: AppState) -> None:
    state.audio_io.gen_mode = GenMode(mode)


def set_band(source, band: list[int], state: AppState) -> None:
    low, high = clamp_band(state, float(band[0]), float(band[1]))
    state.audio_io.band = (low, high)
    state.analyzer.band = (low, high)
    if [low, high] != list(map(float, band[:2])):
        dpg.set_value(source, [int(low), int(high), 0, 0])
    request_current_record_reanalysis(state)


def commit_band(sender, app_data, user_data) -> None:
    item, state = user_data
    set_band(item, dpg.get_value(item), state)


def set_length(source: int, length: float, state: AppState) -> None:
    state.audio_io.length = clamp_length(length)
    if state.audio_io.length != length:
        dpg.set_value(source, state.audio_io.length)
    if state.welch_n_input is not None:
        sync_welch_limit(state, state.welch_n_input)


def commit_length(sender, app_data, user_data) -> None:
    item, state = user_data
    set_length(item, dpg.get_value(item), state)


def set_input_meter(source: int, enabled: bool, state: AppState) -> None:
    if enabled:
        state.io_upd.enable.clear()
        state.pending_meter_start = True
    else:
        state.pending_meter_start = False
        state.meter.enable.clear()
        state.io_upd.enable.set()


def upd_level_monitor(state: AppState, bars) -> None:
    levels = state.meter.get_levels()
    dpg.set_value(bars[0], levels[0])
    dpg.set_value(bars[1], levels[1])


def upd_io(state: AppState, inputs_combo: int | str, outputs_combo: int | str) -> None:
    inputs = ["default input", *state.io_upd.inputs]
    outputs = ["default output", *state.io_upd.outputs]
    dpg.configure_item(inputs_combo, items=inputs)
    dpg.configure_item(outputs_combo, items=outputs)
    sync_selected_audio_devices(state, inputs_combo, outputs_combo, inputs, outputs)


def sync_selected_audio_devices(
    state: AppState,
    inputs_combo: int | str,
    outputs_combo: int | str,
    inputs: list[str],
    outputs: list[str],
) -> None:
    available_inputs = inputs[1:]
    available_outputs = outputs[1:]
    input_name, input_device = resolve_device(
        state.settings.audio.input_device,
        available_inputs,
        DEFAULT_INPUT,
    )
    output_name, output_device = resolve_device(
        state.settings.audio.output_device,
        available_outputs,
        DEFAULT_OUTPUT,
    )
    changed = (
        input_name != state.settings.audio.input_device
        or output_name != state.settings.audio.output_device
    )
    if input_device is None:
        state.meter.device = None
    else:
        state.meter.device = input_device
    state.settings.audio.input_device = input_name
    state.settings.audio.output_device = output_name
    state.audio_io.device = (input_device, output_device)
    dpg.set_value(inputs_combo, input_name)
    dpg.set_value(outputs_combo, output_name)
    if changed:
        state.settings.save()


def selected_device_is_available(
    device: int | None,
    selected_name: str,
    devices: list[str],
    default_name: str,
) -> bool:
    if device is None:
        return selected_name in ("", default_name)
    return selected_name in devices and device_is_available(device, devices)


def device_is_available(device: int | None, devices: list[str]) -> bool:
    if device is None:
        return True
    return any(device == device_index_from_name(name) for name in devices)


def device_index_from_name(name: str) -> int | None:
    try:
        return int(name.split(":", 1)[0])
    except (TypeError, ValueError):
        return None


def set_input(s, name: str, state: AppState) -> None:
    idx = device_index_from_name(name)
    state.audio_io.device = (idx, state.audio_io.device[1])
    state.meter.device = idx
    state.settings.audio.input_device = name or DEFAULT_INPUT
    state.settings.save()


def set_output(s, name: str, state: AppState) -> None:
    idx = device_index_from_name(name)
    state.audio_io.device = (state.audio_io.device[0], idx)
    state.settings.audio.output_device = name or DEFAULT_OUTPUT
    state.settings.save()


def set_analyzer_mode(s, mode: str, state: AppState) -> None:
    state.analyzer.analyzer_mode = AnalyzerMode(mode)
    request_current_record_reanalysis(state)


def set_analyzer_ref(s, ref: str, state: AppState) -> None:
    ref_mode = RefMode(ref)
    if ref_mode not in available_ref_modes(state):
        ref_mode = RefMode.NONE
        if s is not None:
            dpg.set_value(s, ref_mode.value)
    state.analyzer.ref = ref_mode
    state.audio_io.ref = ref_mode
    request_current_record_reanalysis(state)


def set_analyzer_weighting(s, weighting: str, state: AppState) -> None:
    state.analyzer.weighting = WeightingMode(weighting)
    request_current_record_reanalysis(state)


def set_bucket_size(s, size: int, state: AppState) -> None:
    state.analyzer.welch_n = clamp_welch_n(state, size)
    if state.analyzer.welch_n != size:
        dpg.set_value(s, state.analyzer.welch_n)
    request_current_record_reanalysis(state)


def commit_bucket_size(sender, app_data, user_data) -> None:
    item, state = user_data
    set_bucket_size(item, dpg.get_value(item), state)


def set_window_width(s, width: float, state: AppState) -> None:
    state.analyzer.window_width = clamp_window_width(width)
    if state.analyzer.window_width != width:
        dpg.set_value(s, state.analyzer.window_width)
    request_current_record_reanalysis(state)


def commit_window_width(sender, app_data, user_data) -> None:
    item, state = user_data
    set_window_width(item, dpg.get_value(item), state)


def set_freq_length(s, length: int, state: AppState) -> None:
    state.analyzer.freq_length = clamp_freq_length(length)
    if state.analyzer.freq_length != length:
        dpg.set_value(s, state.analyzer.freq_length)
    request_current_record_reanalysis(state)


def commit_freq_length(sender, app_data, user_data) -> None:
    item, state = user_data
    set_freq_length(item, dpg.get_value(item), state)


def bind_commit_handler(item: int | str, callback, state: AppState) -> None:
    with dpg.item_handler_registry() as handler:
        dpg.add_item_deactivated_after_edit_handler(
            callback=callback,
            user_data=(item, state),
        )
    dpg.bind_item_handler_registry(item, handler)


def bind_input_commit_handlers(state: AppState, refs) -> None:
    bind_commit_handler(refs.band_input, commit_band, state)
    bind_commit_handler(refs.rec_len, commit_length, state)
    bind_commit_handler(refs.welch_n_input, commit_bucket_size, state)
    bind_commit_handler(refs.window_width_input, commit_window_width, state)
    bind_commit_handler(refs.freq_length_input, commit_freq_length, state)


def record_used_click(sender, state, rows) -> None:
    if state == False:
        dpg.set_value(sender, True)
    else:
        app_state: AppState = rows
        for i, row in enumerate(app_state.lines_table_rows):
            if not row[0] == sender:
                dpg.set_value(row[0], False)
            else:
                app_state.current_line = i
        ensure_current_reference_is_available(app_state)
        request_current_record_reanalysis(app_state)


def record_visible_clicked(sender, state, data) -> None:
    app_state: AppState = data
    for row, line in zip(app_state.lines_table_rows, app_state.lines):
        if sender == row[1]:
            if state:
                dpg.show_item(line)
            else:
                dpg.hide_item(line)


def record_set_name(sender, name, data) -> None:
    app_state: AppState = data
    for row, line in zip(app_state.lines_table_rows, app_state.lines):
        if sender == row[2]:
            dpg.set_item_label(line, name)


def set_filter_window_func(sender, func: str, state: AppState) -> None:
    state.analyzer.window_func = Windows(func)
    request_current_record_reanalysis(state)
