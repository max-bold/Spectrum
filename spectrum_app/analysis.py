import dearpygui.dearpygui as dpg
import numpy as np
import sounddevice as sd

from .models import RefMode

from .state import AppState


def ensure_record_storage(state: AppState) -> None:
    while len(state.records) < len(state.lines):
        state.records.append(np.empty((0, 2), np.float32))
    while len(state.generator_signals) < len(state.lines):
        state.generator_signals.append(np.empty(0, np.float32))
    while len(state.record_sample_rates) < len(state.lines):
        state.record_sample_rates.append(0)


def save_active_audio_record(state: AppState, record: np.ndarray) -> None:
    ensure_record_storage(state)
    record_index = state.analyzer_line_index
    if record_index >= len(state.records):
        return
    state.records[record_index] = record.astype(np.float32, copy=True)
    if state.completed_generator_signal is not None:
        state.generator_signals[record_index] = state.completed_generator_signal.astype(
            np.float32,
            copy=True,
        )
    state.record_sample_rates[record_index] = int(state.completed_audio_sample_rate)


def save_audio_record(
    state: AppState,
    record_index: int,
    record: np.ndarray,
    sample_rate: int,
    generator_signal: np.ndarray | None = None,
) -> None:
    ensure_record_storage(state)
    if record_index >= len(state.records):
        return
    state.records[record_index] = record.astype(np.float32, copy=True)
    if generator_signal is None:
        generator_signal = np.empty(0, np.float32)
    state.generator_signals[record_index] = generator_signal.astype(np.float32, copy=True)
    state.record_sample_rates[record_index] = int(sample_rate)


def normalize_record_channels(record: np.ndarray) -> np.ndarray:
    if record.ndim == 1:
        record = record.reshape(-1, 1)
    if record.shape[1] == 1:
        record = np.repeat(record, 2, axis=1)
    elif record.shape[1] > 2:
        record = record[:, :2]
    return record.astype(np.float32, copy=False)


def set_record_name(state: AppState, record_index: int, name: str) -> None:
    if record_index >= len(state.lines_table_rows) or record_index >= len(state.lines):
        return
    dpg.set_value(state.lines_table_rows[record_index][2], name)
    dpg.set_item_label(state.lines[record_index], name)


def analyze_record(
    state: AppState,
    record_index: int,
) -> None:
    ensure_record_storage(state)
    if record_index >= len(state.records):
        return
    ensure_reference_is_available(state, record_index)
    record = analyzer_record_for_line(state, record_index)
    sample_rate = state.record_sample_rates[record_index]
    if len(record) == 0 or sample_rate <= 0:
        return
    state.analyzer_line_index = record_index
    state.analyzer.completed.clear()
    state.completed_audio_record = None
    state.completed_generator_signal = None
    state.completed_audio_sample_rate = 0
    state.analyzer.sample_rate = int(sample_rate)
    state.analyzer.record = record
    state.analyzer.running.set()


def analyzer_record_for_line(state: AppState, record_index: int) -> np.ndarray:
    record = state.records[record_index].copy()
    if len(record) == 0:
        return record
    if state.analyzer.ref == RefMode.GENERATOR:
        generator_signal = state.generator_signals[record_index]
        if len(generator_signal) > 0:
            signal_length = min(len(record), len(generator_signal))
            record[:, 1].fill(0)
            record[:signal_length, 1] = generator_signal[:signal_length]
    return record


def request_current_record_reanalysis(state: AppState) -> None:
    if state.audio_io.running.is_set() or state.pending_audio_start:
        return
    ensure_record_storage(state)
    if state.current_line >= len(state.records):
        return
    if len(state.records[state.current_line]) == 0:
        return
    if state.analyzer.running.is_set():
        state.pending_reanalysis = True
        return
    analyze_record(state, state.current_line)


def process_pending_reanalysis(state: AppState) -> None:
    if not state.pending_reanalysis:
        return
    if state.analyzer.running.is_set() or state.audio_io.running.is_set():
        return
    state.pending_reanalysis = False
    request_current_record_reanalysis(state)


def raw_audio_record(state: AppState) -> np.ndarray:
    with state.audio_io.record_lock:
        record = state.audio_io.record[: state.audio_io.in_position].copy()
    return normalize_record_channels(record)


def current_generator_signal(state: AppState, length: int) -> np.ndarray:
    signal_length = min(length, len(state.audio_io.signal), state.audio_io.out_position)
    generator_signal = np.zeros(length, np.float32)
    if signal_length > 0:
        generator_signal[:signal_length] = state.audio_io.signal[:signal_length, 0]
    return generator_signal


def input_channel_count(state: AppState) -> int:
    try:
        input_info = sd.query_devices(state.audio_io.device[0], "input")
    except (sd.PortAudioError, ValueError):
        return 0
    if not isinstance(input_info, dict):
        return 0
    return int(input_info["max_input_channels"])


def device_sample_rate(device: int | None, kind: str) -> float | None:
    try:
        device_info = sd.query_devices(device, kind)
    except (sd.PortAudioError, ValueError):
        return None
    if not isinstance(device_info, dict):
        return None
    return float(device_info["default_samplerate"])


def band_sample_rate(state: AppState) -> float:
    sample_rates = [
        device_sample_rate(state.audio_io.device[0], "input"),
        device_sample_rate(state.audio_io.device[1], "output"),
    ]
    valid_sample_rates = [fs for fs in sample_rates if fs is not None and fs > 0]
    if valid_sample_rates:
        return min(valid_sample_rates)
    if state.analyzer.sample_rate > 0:
        return float(state.analyzer.sample_rate)
    return 96000.0


def clamp_length(length: float) -> float:
    return min(100.0, max(1.0, float(length)))


def clamp_band(state: AppState, low: float, high: float) -> tuple[float, float]:
    nyquist = max(2.0, band_sample_rate(state) / 2.0)
    low = min(max(1.0, low), nyquist - 1.0)
    high = min(max(low + 1.0, high), nyquist)
    return low, high


def nearest_power_of_two(value: int) -> int:
    value = max(1, int(value))
    lower = 1 << (value.bit_length() - 1)
    upper = lower << 1
    if value - lower <= upper - value:
        return lower
    return upper


def clamp_welch_n(state: AppState, size: int) -> int:
    max_size = max(16, int(state.audio_io.length * band_sample_rate(state)))
    size = min(max(16, int(size)), max_size)
    size = nearest_power_of_two(size)
    if size > max_size:
        size >>= 1
    return max(16, size)


def clamp_window_width(width: float) -> float:
    return min(3.0, max(0.1, float(width)))


def clamp_freq_length(length: int) -> int:
    return min(3000, max(100, int(length)))


def sync_band_limits(state: AppState, band_input: int | str) -> None:
    low, high = clamp_band(state, state.audio_io.band[0], state.audio_io.band[1])
    if (low, high) != state.audio_io.band:
        state.audio_io.band = (low, high)
        state.analyzer.band = (low, high)
        dpg.set_value(band_input, [int(low), int(high), 0, 0])


def sync_welch_limit(state: AppState, welch_input: int | str) -> None:
    welch_n = clamp_welch_n(state, state.analyzer.welch_n)
    if welch_n != state.analyzer.welch_n:
        state.analyzer.welch_n = welch_n
        dpg.set_value(welch_input, welch_n)


def available_ref_modes(state: AppState) -> list[RefMode]:
    refs = [RefMode.NONE]
    has_active_record = active_record_has_data(state)
    if input_channel_count(state) >= 2 or active_record_is_stereo(state):
        refs.append(RefMode.CHANNEL_B)
    if not has_active_record or active_record_has_generator(state):
        refs.append(RefMode.GENERATOR)
    return refs


def sync_reference_options(state: AppState, ref_combo: int | str) -> None:
    state.ref_combo = ref_combo
    refs = available_ref_modes(state)
    items = [ref.value for ref in refs]
    dpg.configure_item(ref_combo, items=items)

    if state.analyzer.ref not in refs:
        state.analyzer.ref = RefMode.NONE
        state.audio_io.ref = RefMode.NONE
        dpg.set_value(ref_combo, RefMode.NONE.value)


def active_record_has_data(state: AppState) -> bool:
    ensure_record_storage(state)
    return state.current_line < len(state.records) and len(state.records[state.current_line]) > 0


def active_record_is_stereo(state: AppState) -> bool:
    ensure_record_storage(state)
    if state.current_line >= len(state.records):
        return False
    record = state.records[state.current_line]
    return record.ndim == 2 and record.shape[1] >= 2 and len(record) > 0


def active_record_has_generator(state: AppState) -> bool:
    ensure_record_storage(state)
    if state.current_line >= len(state.generator_signals):
        return False
    return len(state.generator_signals[state.current_line]) > 0


def ensure_current_reference_is_available(state: AppState) -> None:
    ensure_reference_is_available(state, state.current_line)


def ensure_reference_is_available(state: AppState, record_index: int) -> None:
    ensure_record_storage(state)
    if (
        state.analyzer.ref == RefMode.GENERATOR
        and record_index < len(state.records)
        and len(state.records[record_index]) > 0
        and (
            record_index >= len(state.generator_signals)
            or len(state.generator_signals[record_index]) == 0
        )
    ):
        state.analyzer.ref = RefMode.NONE
        state.audio_io.ref = RefMode.NONE
        if state.ref_combo is not None:
            dpg.set_value(state.ref_combo, RefMode.NONE.value)
