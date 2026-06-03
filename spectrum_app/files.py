import json
from pathlib import Path

import dearpygui.dearpygui as dpg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import soundfile as sf

from .analysis import (
    analyze_record,
    ensure_current_reference_is_available,
    ensure_record_storage,
    normalize_record_channels,
    save_audio_record,
    set_record_name,
)
from .state import AppState


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


def save_figure(fig, path: Path, state: AppState) -> None:
    try:
        fig.tight_layout()
        fig.savefig(path)
        print(f"Saved: {path}")
    except OSError as exc:
        print(f"Save failed: {exc}")
