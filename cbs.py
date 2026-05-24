from dataclasses import dataclass, field
import json
from pathlib import Path
from threading import Event, Thread
from time import sleep, time
import dearpygui.dearpygui as dpg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import sounddevice as sd
import soundfile as sf

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
    ref_combo: int | str | None = None
    welch_n_input: int | str | None = None
    lines_table_rows: list[list[int | str]] = field(default_factory=list)
    records: list[np.ndarray] = field(default_factory=list)
    generator_signals: list[np.ndarray] = field(default_factory=list)
    record_sample_rates: list[int] = field(default_factory=list)
    analyzer_line_index: int = 0
    completed_audio_record: np.ndarray | None = None
    completed_generator_signal: np.ndarray | None = None
    completed_audio_sample_rate: int = 0
    pending_reanalysis: bool = False
    project_path: Path | None = None
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


def show_export_dialog(sender, app_data, dialog: int | str) -> None:
    dpg.show_item(dialog)


def show_import_dialog(sender, app_data, dialog: int | str) -> None:
    dpg.show_item(dialog)


def show_wav_export_dialog(sender, app_data, user_data) -> None:
    state, dialog = user_data
    record_name = current_record_name(state)
    dpg.configure_item(dialog, default_filename=record_name)
    dpg.show_item(dialog)


def show_save_project_dialog(sender, app_data, user_data) -> None:
    state, dialog = user_data
    if state.project_path is not None:
        dpg.configure_item(
            dialog,
            default_path=str(state.project_path.parent),
            default_filename=state.project_path.name,
        )
    dpg.show_item(dialog)


def show_open_project_dialog(sender, app_data, dialog: int | str) -> None:
    dpg.show_item(dialog)


def save_project(sender, app_data, user_data) -> None:
    state, save_as_dialog = user_data
    if state.project_path is None:
        show_save_project_dialog(sender, app_data, (state, save_as_dialog))
        return
    write_project_file(state, state.project_path)


def save_project_as(sender, app_data: dict, state: AppState) -> None:
    path = project_path_from_dialog(app_data)
    if path is None:
        return
    write_project_file(state, path)


def open_project(sender, app_data: dict, state: AppState) -> None:
    path = project_path_from_dialog(app_data)
    if path is None:
        return
    read_project_file(state, path)


def import_wav(sender, app_data: dict, state: AppState) -> None:
    path = export_path_from_dialog(app_data)
    if path is None:
        return
    if state.analyzer.running.is_set():
        print("Analyzer is busy")
        return

    try:
        record, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"WAV import failed: {exc}")
        return

    record = normalize_record_channels(record)
    if len(record) == 0:
        print("WAV import failed: empty file")
        return

    ensure_record_storage(state)
    record_index = state.current_line
    save_audio_record(state, record_index, record, int(sample_rate))
    set_record_name(state, record_index, path.stem)
    ensure_current_reference_is_available(state)
    analyze_record(state, record_index)


def project_path_from_dialog(app_data: dict) -> Path | None:
    file_path = app_data.get("file_path_name")
    if not file_path:
        return None
    return ensure_project_extension(Path(file_path))


def ensure_project_extension(path: Path) -> Path:
    if path.suffix.lower() == ".bms":
        return path
    return path.with_suffix(".bms")


def write_project_file(state: AppState, path: Path) -> None:
    project = {
        "format": "bm-spectrum-project",
        "version": 1,
        "records": collect_project_records(state),
    }
    try:
        path.write_text(json.dumps(project, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"Project save failed: {exc}")
        return
    state.project_path = path
    update_viewport_title(state)


def collect_project_records(state: AppState) -> list[dict]:
    records = []
    for i, (line, row) in enumerate(zip(state.lines, state.lines_table_rows)):
        x_values, y_values = line_series_values(line)
        name = dpg.get_value(row[2]) or dpg.get_item_label(line) or f"record {i + 1}"
        records.append(
            {
                "name": name,
                "visible": bool(dpg.get_value(row[1])),
                "used": bool(dpg.get_value(row[0])),
                "x": list(x_values or []),
                "y": list(y_values or []),
            }
        )
    return records


def line_series_values(line: int | str) -> tuple[list, list]:
    value = dpg.get_value(line)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return list(value[0] or []), list(value[1] or [])
    return [], []


def read_project_file(state: AppState, path: Path) -> None:
    try:
        raw_project = json.loads(path.read_text(encoding="utf-8"))
        records = validate_project(raw_project)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Project open failed: {exc}")
        return

    load_project_records(state, records)
    state.project_path = path
    update_viewport_title(state)


def update_viewport_title(state: AppState) -> None:
    title = "BM Spectrum"
    if state.project_path is not None:
        title = f"{title} {state.project_path}"
    dpg.set_viewport_title(title)


def validate_project(project: object) -> list[dict]:
    if not isinstance(project, dict):
        raise ValueError("invalid project file")
    if project.get("format") != "bm-spectrum-project":
        raise ValueError("unsupported project format")
    if project.get("version") != 1:
        raise ValueError("unsupported project version")
    records = project.get("records")
    if not isinstance(records, list):
        raise ValueError("project has no records")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid record data")
        x_values = record.get("x", [])
        y_values = record.get("y", [])
        if not isinstance(x_values, list) or not isinstance(y_values, list):
            raise ValueError("invalid record samples")
        if len(x_values) != len(y_values):
            raise ValueError("record sample length mismatch")
    return records


def load_project_records(state: AppState, records: list[dict]) -> None:
    ensure_record_storage(state)
    used_index = next(
        (i for i, record in enumerate(records[: len(state.lines)]) if record.get("used")),
        0,
    )
    for i, (line, row) in enumerate(zip(state.lines, state.lines_table_rows)):
        record = records[i] if i < len(records) else {}
        name = str(record.get("name") or f"record {i + 1}")
        visible = bool(record.get("visible", False))
        x_values = record.get("x", [])
        y_values = record.get("y", [])

        dpg.set_value(line, [x_values, y_values])
        dpg.set_item_label(line, name)
        dpg.set_value(row[0], i == used_index)
        dpg.set_value(row[1], visible)
        dpg.set_value(row[2], name)
        if visible:
            dpg.show_item(line)
        else:
            dpg.hide_item(line)

    state.current_line = used_index
    for i in range(len(state.records)):
        state.records[i] = np.empty((0, 2), np.float32)
        state.generator_signals[i] = np.empty(0, np.float32)
        state.record_sample_rates[i] = 0
    if state.fft_xaxis is not None:
        dpg.fit_axis_data(state.fft_xaxis)
    if state.fft_yaxis is not None:
        dpg.fit_axis_data(state.fft_yaxis)


def save_fft_plot(sender, app_data: dict, state: AppState) -> None:
    path = export_path_from_dialog(app_data)
    if path is None:
        return

    series = []
    for line, row in zip(state.lines, state.lines_table_rows):
        if not dpg.is_item_shown(line):
            continue
        x_values, y_values = line_series_values(line)
        if not x_values or not y_values:
            continue
        label = dpg.get_value(row[2]) or dpg.get_item_label(line)
        series.append((x_values, y_values, label))

    if not series:
        print("No visible FFT records to save")
        return

    path = ensure_supported_image_extension(path)
    fig = Figure(figsize=(10, 6), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for x_values, y_values, label in series:
        ax.semilogx(x_values, y_values, label=label)
    ax.set_title("FFT")
    ax.set_xlabel("Hz")
    ax.set_ylabel("PSD [dB]")
    ax.grid(True, which="both")
    ax.legend()
    ax.text(
        0.015,
        0.985,
        "BM Sepctrum",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="#222222",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
    )
    save_figure(fig, path, state)


def export_path_from_dialog(app_data: dict) -> Path | None:
    file_path = app_data.get("file_path_name")
    if not file_path:
        return None
    return Path(file_path)


def ensure_supported_image_extension(path: Path) -> Path:
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        return path
    return path.with_suffix(".png")


def save_current_record_wav(sender, app_data: dict, state: AppState) -> None:
    path = export_path_from_dialog(app_data)
    if path is None:
        return

    ensure_record_storage(state)
    record_index = state.current_line
    if record_index >= len(state.records):
        print("No active record to export")
        return

    record = state.records[record_index]
    sample_rate = state.record_sample_rates[record_index]
    if len(record) == 0 or sample_rate <= 0:
        print("No WAV record to export")
        return

    path = ensure_wav_extension(path)
    try:
        sf.write(path, record, sample_rate)
        print(f"Saved: {path}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"WAV export failed: {exc}")


def ensure_wav_extension(path: Path) -> Path:
    if path.suffix.lower() == ".wav":
        return path
    return path.with_suffix(".wav")


def current_record_name(state: AppState) -> str:
    if state.current_line < len(state.lines_table_rows):
        name = dpg.get_value(state.lines_table_rows[state.current_line][2])
        if name:
            return sanitize_filename(str(name))
    return f"record {state.current_line + 1}"


def sanitize_filename(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    sanitized = "".join("_" if char in invalid_chars else char for char in name).strip()
    return sanitized or "record"


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


def save_figure(fig, path: Path, state: AppState) -> None:
    try:
        fig.tight_layout()
        fig.savefig(path)
        print(f"Saved: {path}")
    except OSError as exc:
        print(f"Save failed: {exc}")


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
