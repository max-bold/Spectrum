from dataclasses import dataclass, field
from threading import Event, Thread
from time import sleep, time
import dearpygui.dearpygui as dpg
import sounddevice as sd

from utils.analyzer import Analyzer
from utils.audio import io_list_updater, InputMeter, AudioIO
from utils.classes import AnalyzerMode, GenMode, RefMode, WeightingMode
from utils.windows import Windows


@dataclass
class AppState:
    analyzer: Analyzer = field(default_factory=Analyzer)
    meter: InputMeter = field(default_factory=InputMeter)
    io_upd: io_list_updater = field(default_factory=io_list_updater)
    audio_io: AudioIO = field(default_factory=AudioIO)
    lines: list[int | str] = field(default_factory=list)
    fft_xaxis: int | str | None = None
    fft_yaxis: int | str | None = None
    current_line: int = 0
    levels_l: int | str | None = None
    levels_r: int | str | None = None
    welch_n_input: int | str | None = None
    lines_table_rows: list[list[int | str]] = field(default_factory=list)
    io_reenable_timer: "Timer | None" = None
    pending_audio_start: bool = False
    pending_meter_start: bool = False
    services_started: bool = False

    def start_services(self) -> None:
        if self.services_started:
            return
        self.audio_io.ref = self.analyzer.ref
        self.analyzer.start()
        self.meter.start()
        self.io_upd.start()
        self.io_upd.enable.set()
        self.audio_io.start()
        self.io_reenable_timer = Timer(5, self.io_upd.enable.set)
        self.io_reenable_timer.start()
        self.services_started = True

    def stop_services(self) -> None:
        self.pending_audio_start = False
        self.pending_meter_start = False
        self.audio_io.kill()
        self.meter.enable.clear()
        self.io_upd.enable.clear()
        if self.io_reenable_timer:
            self.io_reenable_timer.enabled.clear()
        if self.audio_io.is_alive():
            self.audio_io.join(timeout=2.0)


def create_app_state() -> AppState:
    return AppState()


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
    dpg.configure_item(inputs_combo, items=state.io_upd.inputs)
    dpg.configure_item(outputs_combo, items=state.io_upd.outputs)


def set_input(s, name: str, state: AppState) -> None:
    idx = state.io_upd.get_device_indx(name)
    state.audio_io.device = (idx, state.audio_io.device[1])
    state.meter.device = idx


def set_output(s, name: str, state: AppState) -> None:
    idx = state.io_upd.get_device_indx(name)
    state.audio_io.device = (state.audio_io.device[0], idx)


def set_analyzer_mode(s, mode: str, state: AppState) -> None:
    state.analyzer.analyzer_mode = AnalyzerMode(mode)


def set_analyzer_ref(s, ref: str, state: AppState) -> None:
    ref_mode = RefMode(ref)
    if ref_mode not in available_ref_modes(state):
        ref_mode = RefMode.GENERATOR
        dpg.set_value(s, ref_mode.value)
    state.analyzer.ref = ref_mode
    state.audio_io.ref = ref_mode


def set_analyzer_weighting(s, weighting: str, state: AppState) -> None:
    state.analyzer.weighting = WeightingMode(weighting)


def set_bucket_size(s, size: int, state: AppState) -> None:
    state.analyzer.welch_n = clamp_welch_n(state, size)
    if state.analyzer.welch_n != size:
        dpg.set_value(s, state.analyzer.welch_n)


def commit_bucket_size(sender, app_data, user_data) -> None:
    item, state = user_data
    set_bucket_size(item, dpg.get_value(item), state)


def set_window_width(s, width: float, state: AppState) -> None:
    state.analyzer.window_width = clamp_window_width(width)
    if state.analyzer.window_width != width:
        dpg.set_value(s, state.analyzer.window_width)


def commit_window_width(sender, app_data, user_data) -> None:
    item, state = user_data
    set_window_width(item, dpg.get_value(item), state)


def set_freq_length(s, length: int, state: AppState) -> None:
    state.analyzer.freq_length = clamp_freq_length(length)
    if state.analyzer.freq_length != length:
        dpg.set_value(s, state.analyzer.freq_length)


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


def input_channel_count(state: AppState) -> int:
    try:
        input_info = sd.query_devices(state.audio_io.device[0], "input")
    except sd.PortAudioError:
        return 0
    if not isinstance(input_info, dict):
        return 0
    return int(input_info["max_input_channels"])


def device_sample_rate(device: int | None, kind: str) -> float | None:
    try:
        device_info = sd.query_devices(device, kind)
    except sd.PortAudioError:
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
    refs = [RefMode.NONE, RefMode.GENERATOR]
    if input_channel_count(state) >= 2:
        refs.insert(1, RefMode.CHANNEL_B)
    return refs


def sync_reference_options(state: AppState, ref_combo: int | str) -> None:
    refs = available_ref_modes(state)
    items = [ref.value for ref in refs]
    dpg.configure_item(ref_combo, items=items)

    if state.analyzer.ref not in refs:
        state.analyzer.ref = RefMode.GENERATOR
        state.audio_io.ref = RefMode.GENERATOR
        dpg.set_value(ref_combo, RefMode.GENERATOR.value)


class Timer(Thread):
    def __init__(self, delay: float, function, args=(), kwargs=None) -> None:
        super().__init__()
        self.delay = delay
        self.enabled = Event()
        self.function = function
        self.start_time: float = 0.0
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = True

    def run(self) -> None:
        while True:
            self.enabled.wait()
            self.start_time = time()
            while self.enabled.is_set():
                if time() - self.start_time >= self.delay:
                    self.function(*self.args, **self.kwargs)
                    self.enabled.clear()
                else:
                    sleep(0.01)


def reenable_io_udater(state: AppState) -> None:
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
            state.analyzer.sample_rate = state.audio_io.in_fs
            state.analyzer.record = state.audio_io.get_record()
            state.analyzer.running.set()

    if state.analyzer.completed.is_set():
        state.analyzer.completed.clear()
        fft_data = state.analyzer.result.copy()
        line = state.lines[state.current_line]
        dpg.set_value(line, [fft_data[0].tolist(), fft_data[1].tolist()])
        dpg.show_item(line)
        dpg.set_value(state.lines_table_rows[state.current_line][1], True)
        if state.fft_yaxis is not None:
            dpg.fit_axis_data(state.fft_yaxis)

    if state.audio_io.levels_updated.is_set():
        ts, levels = state.audio_io.get_levels(0.01)
        if state.levels_l and state.levels_r:
            dpg.set_value(state.levels_l, [list(ts), list(levels[0])])
            dpg.set_value(state.levels_r, [list(ts), list(levels[1])])
