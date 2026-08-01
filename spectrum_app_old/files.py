import base64
import binascii
import json
from pathlib import Path
from threading import Thread
import zlib

import dearpygui.dearpygui as dpg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import soundfile as sf

from .analysis import (
    analyze_record,
    apply_current_record_analysis_settings,
    current_analysis_settings,
    ensure_current_reference_is_available,
    ensure_record_storage,
    normalize_record_channels,
    save_audio_record,
    set_record_name,
    sync_analysis_controls,
    update_current_record_level_plot,
    update_record_level_plot,
)
from .models import AnalyzerMode, RefMode, WeightingMode
from .state import AppState
from utils.windows import Windows


PROJECT_WARNING_DIALOG = "project_warning_dialog"
PROJECT_WARNING_TEXT = "project_warning_text"
PROJECT_PROGRESS_DIALOG = "project_progress_dialog"
PROJECT_PROGRESS_TEXT = "project_progress_text"
PROJECT_PROGRESS_BAR = "project_progress_bar"
LEGACY_PROJECT_MESSAGE = (
    "This .bms project uses an old format. "
    "These projects are supported up to BM Spectrum 0.2.2."
)
AUDIO_ARRAY_ENCODING = "float32-le-zlib-base64"


class LegacyProjectError(ValueError):
    pass


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
    start_project_save(state, state.project_path)


def save_project_as(sender, app_data: dict, state: AppState) -> None:
    path = project_path_from_dialog(app_data, sender)
    if path is None:
        print(f"Project save failed: cannot resolve path from dialog data: {app_data}")
        return
    if dpg.does_item_exist(sender):
        dpg.hide_item(sender)
    print(f"Saving project: {path}")
    start_project_save(state, path)


def open_project(sender, app_data: dict, state: AppState) -> None:
    path = project_path_from_dialog(app_data, sender)
    if path is None:
        return
    if dpg.does_item_exist(sender):
        dpg.hide_item(sender)
    start_project_open(state, path)


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
    update_record_level_plot(state, record, int(sample_rate))
    ensure_current_reference_is_available(state)
    analyze_record(state, record_index)


def project_path_from_dialog(
    app_data: dict,
    dialog: int | str | None = None,
) -> Path | None:
    file_path = path_from_dialog(app_data, dialog)
    if file_path is None:
        return None
    return ensure_project_extension(file_path)


def path_from_dialog(
    app_data: dict,
    dialog: int | str | None = None,
) -> Path | None:
    file_name = app_data.get("file_name")
    current_path = app_data.get("current_path")
    default_filename = dialog_default_filename(dialog)

    file_path = app_data.get("file_path_name")
    if file_path:
        path = Path(file_path)
        if dialog_path_is_directory(path):
            name = file_name or default_filename
            if name:
                return path / name
        else:
            return path

    selections = app_data.get("selections")
    if isinstance(selections, dict):
        for selected_path in selections.values():
            if selected_path:
                path = Path(selected_path)
                if dialog_path_is_directory(path):
                    name = file_name or default_filename
                    if name:
                        return path / name
                else:
                    return path
        for selected_name in selections.keys():
            if selected_name and current_path:
                return Path(current_path) / selected_name

    if file_name and current_path:
        return Path(current_path) / file_name
    if default_filename and current_path:
        return Path(current_path) / default_filename

    return None


def dialog_path_is_directory(path: Path) -> bool:
    if str(path).endswith(("\\", "/")):
        return True
    try:
        return path.exists() and path.is_dir()
    except OSError:
        return False


def dialog_default_filename(dialog: int | str | None) -> str:
    if dialog is None or not dpg.does_item_exist(dialog):
        return ""
    try:
        return str(dpg.get_item_configuration(dialog).get("default_filename") or "")
    except Exception:
        return ""


def ensure_project_extension(path: Path) -> Path:
    if path.suffix.lower() == ".bms":
        return path
    if not path.name:
        return path / "project.bms"
    return path.with_suffix(".bms")


def write_project_file(state: AppState, path: Path) -> None:
    try:
        project = {
            "format": "bm-spectrum-project",
            "version": 4,
            "view": collect_project_view(state),
            "records": collect_project_records(state),
        }
        path.write_text(json.dumps(project, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Project save failed: {exc}")
        return
    state.project_path = path
    update_viewport_title(state)


def start_project_save(state: AppState, path: Path) -> None:
    if state.project_progress_active:
        print("Project operation already in progress")
        return
    try:
        records = collect_project_records(state, encode_audio=False)
        view = collect_project_view(state)
    except (TypeError, ValueError, RuntimeError) as exc:
        print(f"Project save failed: {exc}")
        return
    start_project_progress(state, "Saving project", f"Writing {path}")
    Thread(
        target=project_save_worker,
        args=(state, path, records, view),
        daemon=True,
    ).start()


def project_save_worker(
    state: AppState,
    path: Path,
    records: list[dict],
    view: dict,
) -> None:
    try:
        project = {
            "format": "bm-spectrum-project",
            "version": 4,
            "view": view,
            "records": encode_project_records(records),
        }
        path.write_text(json.dumps(project, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        state.project_results.put({"kind": "save", "path": path, "error": str(exc)})
        return
    state.project_results.put({"kind": "save", "path": path, "error": None})


def start_project_open(state: AppState, path: Path) -> None:
    if state.project_progress_active:
        print("Project operation already in progress")
        return
    start_project_progress(state, "Opening project", f"Reading {path}")
    Thread(target=project_open_worker, args=(state, path), daemon=True).start()


def project_open_worker(state: AppState, path: Path) -> None:
    try:
        raw_project = json.loads(path.read_text(encoding="utf-8"))
        records = validate_project(raw_project)
        prepared = prepare_project_records(records)
        view = validate_project_view(raw_project.get("view", {}))
    except LegacyProjectError as exc:
        state.project_results.put(
            {"kind": "open", "path": path, "legacy": str(exc), "error": None}
        )
        return
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        state.project_results.put({"kind": "open", "path": path, "error": str(exc)})
        return
    state.project_results.put(
        {
            "kind": "open",
            "path": path,
            "records": prepared,
            "view": view,
            "error": None,
        }
    )


def start_project_progress(state: AppState, title: str, status: str) -> None:
    state.project_progress_active = True
    state.project_progress_title = title
    state.project_progress_status = status
    state.project_progress_value = 0.0
    state.project_progress_phase = 0.0
    state.project_progress_waiting_analyzer = False


def process_project_results(state: AppState) -> None:
    while not state.project_results.empty():
        result = state.project_results.get()
        if result["kind"] == "save":
            finish_project_progress(state)
            if result["error"]:
                print(f"Project save failed: {result['error']}")
                continue
            state.project_path = result["path"]
            update_viewport_title(state)
            print(f"Saved project: {result['path']}")
        elif result["kind"] == "open":
            if result.get("legacy"):
                finish_project_progress(state)
                state.pending_project_warning = result["legacy"]
                state.pending_project_warning_frames = 2
                continue
            if result["error"]:
                finish_project_progress(state)
                print(f"Project open failed: {result['error']}")
                continue
            state.project_progress_status = "Calculating FFT..."
            view = result.get("view", {})
            state.fit_fft_yaxis_after_project_open = "fft_yaxis" not in view
            load_prepared_project_records(state, result["records"])
            apply_project_view(state, view)
            state.project_path = result["path"]
            update_viewport_title(state)


def finish_project_progress(state: AppState) -> None:
    state.project_progress_active = False
    state.project_progress_waiting_analyzer = False
    state.project_reanalysis_queue.clear()
    state.fit_fft_yaxis_after_project_open = False


def update_project_progress(state: AppState) -> None:
    if not dpg.does_item_exist(PROJECT_PROGRESS_DIALOG):
        return
    if state.project_progress_active:
        state.project_progress_phase = (state.project_progress_phase + 0.02) % 1.0
        dpg.configure_item(PROJECT_PROGRESS_DIALOG, label=state.project_progress_title)
        dpg.set_value(PROJECT_PROGRESS_TEXT, state.project_progress_status)
        dpg.set_value(PROJECT_PROGRESS_BAR, state.project_progress_phase)
        center_dialog(PROJECT_PROGRESS_DIALOG, 520, 135)
        dpg.configure_item(PROJECT_PROGRESS_DIALOG, show=True)
        return
    dpg.hide_item(PROJECT_PROGRESS_DIALOG)


def complete_project_open_if_analysis_finished(state: AppState) -> None:
    if not state.project_progress_waiting_analyzer:
        return
    if state.analyzer.running.is_set() or state.analyzer.completed.is_set():
        return
    if start_next_project_reanalysis(state):
        return
    fit_fft_yaxis_after_project_open(state)
    apply_current_record_analysis_settings(state)
    sync_analysis_controls(state)
    finish_project_progress(state)


def fit_fft_yaxis_after_project_open(state: AppState) -> None:
    if state.fit_fft_yaxis_after_project_open and state.fft_yaxis is not None:
        dpg.fit_axis_data(state.fft_yaxis)


def start_next_project_reanalysis(state: AppState) -> bool:
    while state.project_reanalysis_queue:
        record_index = state.project_reanalysis_queue.pop(0)
        analyze_record(state, record_index)
        if state.analyzer.running.is_set():
            state.project_progress_status = (
                f"Calculating FFT {record_index + 1}/"
                f"{len(state.lines_table_rows)}..."
            )
            return True
    return False


def collect_project_records(state: AppState, encode_audio: bool = True) -> list[dict]:
    ensure_record_storage(state)
    records = []
    for i, (line, row) in enumerate(zip(state.lines, state.lines_table_rows)):
        name = dpg.get_value(row[2]) or dpg.get_item_label(line) or f"record {i + 1}"
        project_record = {
            "name": name,
            "visible": bool(dpg.get_value(row[1])),
            "used": bool(dpg.get_value(row[0])),
            "settings": project_analysis_settings(state, i),
        }
        audio = collect_project_audio(state, i, encode=encode_audio)
        if audio is not None:
            project_record["audio"] = audio
        records.append(project_record)
    return records


def collect_project_view(state: AppState) -> dict:
    view = {}
    if state.fft_yaxis is not None:
        try:
            lower, upper = dpg.get_axis_limits(state.fft_yaxis)
        except Exception:
            lower, upper = None, None
        if valid_axis_limits(lower, upper):
            view["fft_yaxis"] = [float(lower), float(upper)]
    return view


def validate_project_view(view: object) -> dict:
    if not isinstance(view, dict):
        return {}
    limits = view.get("fft_yaxis")
    if (
        isinstance(limits, list)
        and len(limits) == 2
        and valid_axis_limits(limits[0], limits[1])
    ):
        return {"fft_yaxis": [float(limits[0]), float(limits[1])]}
    return {}


def apply_project_view(state: AppState, view: dict) -> None:
    limits = view.get("fft_yaxis")
    if state.fft_yaxis is None or limits is None:
        return
    dpg.set_axis_limits(state.fft_yaxis, float(limits[0]), float(limits[1]))
    state.unlock_fft_yaxis_frames = 2


def unlock_fft_yaxis_if_needed(state: AppState) -> None:
    if state.unlock_fft_yaxis_frames <= 0:
        return
    state.unlock_fft_yaxis_frames -= 1
    if state.unlock_fft_yaxis_frames > 0 or state.fft_yaxis is None:
        return
    dpg.set_axis_limits_auto(state.fft_yaxis)


def valid_axis_limits(lower: object, upper: object) -> bool:
    try:
        lower = float(lower)
        upper = float(upper)
    except (TypeError, ValueError):
        return False
    return np.isfinite(lower) and np.isfinite(upper) and lower < upper


def encode_project_records(records: list[dict]) -> list[dict]:
    encoded = []
    for record in records:
        encoded_record = {
            "name": record["name"],
            "visible": record["visible"],
            "used": record["used"],
            "settings": record["settings"],
        }
        audio = record.get("audio")
        if audio is not None:
            encoded_audio = {
                "sample_rate": audio["sample_rate"],
                "record": encode_project_array(audio["record"]),
            }
            generator_signal = audio.get("generator_signal")
            if generator_signal is not None and len(generator_signal) > 0:
                encoded_audio["generator_signal"] = encode_project_array(
                    generator_signal
                )
            encoded_record["audio"] = encoded_audio
        encoded.append(encoded_record)
    return encoded


def project_analysis_settings(state: AppState, record_index: int) -> dict:
    if record_index < len(state.record_settings):
        return dict(state.record_settings[record_index])
    return current_analysis_settings(state)


def collect_project_audio(
    state: AppState,
    record_index: int,
    encode: bool = True,
) -> dict | None:
    if record_index >= len(state.records) or record_index >= len(state.record_sample_rates):
        return None
    record = state.records[record_index]
    sample_rate = state.record_sample_rates[record_index]
    if len(record) == 0 or sample_rate <= 0:
        return None

    normalized = normalize_record_channels(record).copy()
    audio = {"sample_rate": int(sample_rate)}
    if encode:
        audio["record"] = encode_project_array(normalized)
    else:
        audio["record"] = normalized
    if record_index < len(state.generator_signals):
        generator_signal = state.generator_signals[record_index]
        if len(generator_signal) > 0:
            signal = generator_signal.astype(np.float32, copy=True)
            if encode:
                audio["generator_signal"] = encode_project_array(signal)
            else:
                audio["generator_signal"] = signal
    return audio


def encode_project_array(array: np.ndarray) -> dict:
    data = np.ascontiguousarray(array, dtype="<f4")
    compressed = zlib.compress(data.tobytes())
    return {
        "encoding": AUDIO_ARRAY_ENCODING,
        "shape": list(data.shape),
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def decode_project_array(value: object) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError("invalid project array")
    if value.get("encoding") != AUDIO_ARRAY_ENCODING:
        raise ValueError("unsupported project array encoding")
    shape = value.get("shape")
    data = value.get("data")
    if (
        not isinstance(shape, list)
        or not shape
        or not all(isinstance(size, int) and size >= 0 for size in shape)
        or not isinstance(data, str)
    ):
        raise ValueError("invalid project array")
    try:
        raw = zlib.decompress(base64.b64decode(data.encode("ascii")))
    except (binascii.Error, zlib.error) as exc:
        raise ValueError("invalid project array data") from exc
    array = np.frombuffer(raw, dtype="<f4")
    expected_size = int(np.prod(shape))
    if array.size != expected_size:
        raise ValueError("project array size mismatch")
    return array.reshape(shape).astype(np.float32, copy=True)


def line_series_values(line: int | str) -> tuple[list, list]:
    value = dpg.get_value(line)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return list(value[0] or []), list(value[1] or [])
    return [], []


def read_project_file(state: AppState, path: Path) -> None:
    try:
        raw_project = json.loads(path.read_text(encoding="utf-8"))
        records = validate_project(raw_project)
    except LegacyProjectError as exc:
        state.pending_project_warning = str(exc)
        state.pending_project_warning_frames = 2
        return
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
    try:
        dpg.set_viewport_title(title)
    except Exception:
        pass


def show_project_warning(message: str) -> None:
    if not dpg.does_item_exist(PROJECT_WARNING_DIALOG):
        print(message)
        return
    dpg.set_value(PROJECT_WARNING_TEXT, message)
    center_dialog(PROJECT_WARNING_DIALOG, 520, 150)
    dpg.configure_item(PROJECT_WARNING_DIALOG, show=True)
    dpg.focus_item(PROJECT_WARNING_DIALOG)


def center_dialog(dialog: int | str, fallback_width: int, fallback_height: int) -> None:
    try:
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
    except Exception:
        return
    width = dpg.get_item_width(dialog) or fallback_width
    height = dpg.get_item_height(dialog) or fallback_height
    x = max(0, int((viewport_width - width) / 2))
    y = max(0, int((viewport_height - height) / 2))
    dpg.set_item_pos(dialog, (x, y))


def close_project_warning(sender, app_data, user_data) -> None:
    dpg.hide_item(PROJECT_WARNING_DIALOG)


def validate_project(project: object) -> list[dict]:
    if not isinstance(project, dict):
        raise ValueError("invalid project file")
    if project.get("format") != "bm-spectrum-project":
        raise ValueError("unsupported project format")
    if project.get("version") == 1:
        raise LegacyProjectError(LEGACY_PROJECT_MESSAGE)
    if project.get("version") != 4:
        raise ValueError("unsupported project version")
    records = project.get("records")
    if not isinstance(records, list):
        raise ValueError("project has no records")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid record data")
        validate_project_analysis_settings(record.get("settings"))
        audio = record.get("audio")
        if audio is not None:
            validate_project_audio(audio)
    return records


def validate_project_analysis_settings(settings: object) -> None:
    if not isinstance(settings, dict):
        raise ValueError("invalid record analysis settings")
    try:
        AnalyzerMode(settings["analyzer_mode"])
        RefMode(settings["ref"])
        WeightingMode(settings["weighting"])
        int(settings["welch_n"])
        band = settings["band"]
        float(band[0])
        float(band[1])
        float(settings["window_width"])
        int(settings["freq_length"])
        Windows(settings["window_func"])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("invalid record analysis settings") from exc


def validate_project_audio(audio: object) -> None:
    if not isinstance(audio, dict):
        raise ValueError("invalid record audio data")

    sample_rate = audio.get("sample_rate")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("invalid record audio sample rate")

    try:
        record = normalize_record_channels(decode_project_array(audio.get("record")))
    except (ValueError, IndexError) as exc:
        raise ValueError("invalid record audio samples") from exc
    if record.ndim != 2 or record.shape[1] != 2 or len(record) == 0:
        raise ValueError("invalid record audio samples")
    if not np.isfinite(record).all():
        raise ValueError("invalid record audio samples")

    if "generator_signal" in audio:
        try:
            generator_signal = decode_project_array(audio["generator_signal"]).reshape(-1)
        except ValueError as exc:
            raise ValueError("invalid record generator signal") from exc
        if not np.isfinite(generator_signal).all():
            raise ValueError("invalid record generator signal")


def load_project_records(state: AppState, records: list[dict]) -> None:
    ensure_record_storage(state)
    prepared = prepare_project_records(records)
    load_prepared_project_records(state, prepared)


def prepare_project_records(records: list[dict]) -> list[dict]:
    prepared = []
    for record in records:
        audio_record, sample_rate, generator_signal = project_audio_from_record(record)
        prepared.append(
            {
                "name": str(record.get("name") or "record"),
                "visible": bool(record.get("visible", False)),
                "used": bool(record.get("used", False)),
                "settings": dict(record["settings"]),
                "audio_record": audio_record,
                "sample_rate": sample_rate,
                "generator_signal": generator_signal,
            }
        )
    return prepared


def load_prepared_project_records(state: AppState, records: list[dict]) -> None:
    ensure_record_storage(state)
    used_index = next(
        (i for i, record in enumerate(records[: len(state.lines)]) if record.get("used")),
        0,
    )
    for i, (line, row) in enumerate(zip(state.lines, state.lines_table_rows)):
        record = records[i] if i < len(records) else {}
        name = str(record.get("name") or f"record {i + 1}")
        visible = bool(record.get("visible", False))

        dpg.set_value(line, [[], []])
        dpg.set_item_label(line, name)
        dpg.set_value(row[0], i == used_index)
        dpg.set_value(row[1], visible)
        dpg.set_value(row[2], name)
        dpg.hide_item(line)

    state.current_line = used_index
    for i in range(len(state.records)):
        project_record = records[i] if i < len(records) else {}
        state.records[i] = project_record.get(
            "audio_record", np.empty((0, 2), np.float32)
        )
        state.generator_signals[i] = project_record.get(
            "generator_signal", np.empty(0, np.float32)
        )
        state.record_sample_rates[i] = int(project_record.get("sample_rate", 0))
        if i < len(state.record_settings) and "settings" in project_record:
            state.record_settings[i] = dict(project_record["settings"])
    apply_current_record_analysis_settings(state)
    sync_analysis_controls(state)
    update_current_record_level_plot(state)
    ensure_current_reference_is_available(state)
    state.project_reanalysis_queue = [
        i
        for i, record in enumerate(state.records[: len(state.lines)])
        if len(record) > 0 and state.record_sample_rates[i] > 0
    ]
    state.project_progress_waiting_analyzer = bool(state.project_reanalysis_queue)
    start_next_project_reanalysis(state)
    if state.project_progress_active and not state.project_progress_waiting_analyzer:
        fit_fft_yaxis_after_project_open(state)
        finish_project_progress(state)


def project_audio_from_record(
    project_record: dict,
) -> tuple[np.ndarray, int, np.ndarray]:
    audio = project_record.get("audio")
    if not isinstance(audio, dict):
        return np.empty((0, 2), np.float32), 0, np.empty(0, np.float32)

    record = normalize_record_channels(decode_project_array(audio["record"]))
    if "generator_signal" in audio:
        generator_signal = decode_project_array(audio["generator_signal"]).reshape(-1)
    else:
        generator_signal = np.empty(0, np.float32)
    return record.copy(), int(audio["sample_rate"]), generator_signal.copy()


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
    return path_from_dialog(app_data)


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


def save_figure(fig, path: Path, state: AppState) -> None:
    try:
        fig.tight_layout()
        fig.savefig(path)
        print(f"Saved: {path}")
    except OSError as exc:
        print(f"Save failed: {exc}")
