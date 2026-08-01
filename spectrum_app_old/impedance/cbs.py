from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from time import monotonic
from typing import Any

import dearpygui.dearpygui as dpg
from utils.audio import io_list_updater

from spectrum_app_old.settings import (
    AppSettings,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    device_index,
    load_settings,
    resolve_device,
    validate_audio_settings,
)

if __package__:
    from .imp_measure import (
        CalibrationStage,
        ImpedanceAppState,
        MeasurementConfig,
        MeasurementState,
        PhaseDisplayMode,
        WindowFunction,
        export_impedance_plot,
        impedance_axis_limits,
        phase_axis_limits,
        phase_plot_data,
        resolve_sample_rate,
    )
    from .spice_table import SPICE_SECTION_COUNT, SpiceModelTable
    from .project import (
        ensure_project_extension,
        load_impedance_project,
        save_impedance_project,
    )
else:
    from imp_measure import (
        CalibrationStage,
        ImpedanceAppState,
        MeasurementConfig,
        MeasurementState,
        PhaseDisplayMode,
        WindowFunction,
        export_impedance_plot,
        impedance_axis_limits,
        phase_axis_limits,
        phase_plot_data,
        resolve_sample_rate,
    )
    from spice_table import SPICE_SECTION_COUNT, SpiceModelTable
    from project import (
        ensure_project_extension,
        load_impedance_project,
        save_impedance_project,
    )


@dataclass
class ImpedanceUi:
    state: ImpedanceAppState
    settings: AppSettings
    io_updater: io_list_updater
    input_level_meter: Any
    spice_table: SpiceModelTable
    impedance_line: int | str
    phase_line: int | str
    impedance_axis: int | str
    phase_axis: int | str
    calibrate_button: int | str
    test_button: int | str
    measure_button: int | str
    status_text: int | str
    error_dialog: int | str
    error_text: int | str
    error_close_button: int | str
    calibration_dialog: int | str
    calibration_text: int | str
    calibration_continue_button: int | str
    calibration_cancel_button: int | str
    io_menu_item: int | str
    open_project_menu_item: int | str
    save_project_menu_item: int | str
    spice_menu_item: int | str
    phase_angle_menu_item: int | str
    phase_derivative_menu_item: int | str
    io_dialog: int | str
    open_project_dialog: int | str
    save_project_dialog: int | str
    spice_dialog: int | str
    spice_status_text: int | str
    spice_close_button: int | str
    input_combo: int | str
    output_combo: int | str
    block_size_input: int | str
    close_io_button: int | str
    capture_settings: tuple[int | str, ...]
    filter_settings: tuple[int | str, ...]
    phase_mode: PhaseDisplayMode = PhaseDisplayMode.ANGLE
    plot_data_token: int | None = None
    unlock_impedance_axis_frames: int = 0
    unlock_phase_axis_frames: int = 0
    project_path: Path | None = None
    project_worker: Thread | None = None
    project_results: Queue = field(default_factory=Queue)
    project_busy: bool = False
    revision: int = -1
    last_io_update: float = 0.0


def create_state() -> ImpedanceAppState:
    return ImpedanceAppState()


def create_io_settings() -> tuple[AppSettings, io_list_updater]:
    settings = load_settings()
    updater = io_list_updater()
    updater.upd_inputs()
    updater.upd_outputs()
    validate_audio_settings(settings, updater.inputs, updater.outputs)
    updater.start()
    return settings, updater


def bind_ui(ui: ImpedanceUi, export_dialog: int | str) -> None:
    dpg.configure_item(
        ui.calibrate_button,
        callback=show_calibration_setup,
        user_data=ui,
    )
    dpg.configure_item(
        ui.measure_button,
        callback=start_measurement,
        user_data=ui,
    )
    dpg.configure_item(
        ui.test_button,
        callback=toggle_test_signal,
        user_data=ui,
    )
    dpg.configure_item(export_dialog, callback=export_plot, user_data=ui)
    dpg.configure_item(
        ui.open_project_menu_item,
        callback=show_open_project_dialog,
        user_data=ui,
    )
    dpg.configure_item(
        ui.save_project_menu_item,
        callback=save_project_menu,
        user_data=ui,
    )
    dpg.configure_item(
        ui.open_project_dialog,
        callback=open_project,
        user_data=ui,
    )
    dpg.configure_item(
        ui.save_project_dialog,
        callback=save_project_as,
        user_data=ui,
    )
    dpg.configure_item(
        ui.io_menu_item,
        callback=show_io_settings,
        user_data=ui,
    )
    dpg.configure_item(
        ui.spice_menu_item,
        callback=show_spice_model,
        user_data=ui,
    )
    dpg.configure_item(
        ui.spice_close_button,
        callback=close_spice_model,
        user_data=ui,
    )
    dpg.configure_item(
        ui.phase_angle_menu_item,
        callback=set_phase_display,
        user_data=(ui, PhaseDisplayMode.ANGLE),
    )
    dpg.configure_item(
        ui.phase_derivative_menu_item,
        callback=set_phase_display,
        user_data=(ui, PhaseDisplayMode.DERIVATIVE),
    )
    dpg.configure_item(
        ui.input_combo,
        callback=set_input_device,
        user_data=ui,
    )
    dpg.configure_item(
        ui.output_combo,
        callback=set_output_device,
        user_data=ui,
    )
    dpg.configure_item(
        ui.block_size_input,
        callback=set_block_size,
        user_data=ui,
    )
    dpg.configure_item(
        ui.close_io_button,
        callback=close_io_settings,
        user_data=ui,
    )
    dpg.configure_item(
        ui.error_close_button,
        callback=close_error,
        user_data=ui,
    )
    dpg.configure_item(
        ui.calibration_continue_button,
        callback=continue_calibration,
        user_data=ui,
    )
    dpg.configure_item(
        ui.calibration_cancel_button,
        callback=cancel_calibration,
        user_data=ui,
    )
    for item in ui.filter_settings:
        dpg.configure_item(
            item,
            callback=filtering_changed,
            user_data=ui,
        )
    sync_ui(ui)


def build_config(ui: ImpedanceUi) -> MeasurementConfig:
    band = dpg.get_value("band_input")
    input_device = device_index(
        ui.settings.audio.input_device,
        DEFAULT_INPUT,
    )
    output_device = device_index(
        ui.settings.audio.output_device,
        DEFAULT_OUTPUT,
    )
    return MeasurementConfig(
        sample_rate=resolve_sample_rate(input_device, output_device),
        duration=dpg.get_value("impedance_duration_input"),
        reference_resistor=dpg.get_value("reference_resistor_input"),
        calibration_resistor=dpg.get_value("calibration_resistor_input"),
        f_min=float(band[0]),
        f_max=float(band[1]),
        window_width=dpg.get_value("impedance_window_width_input"),
        points=dpg.get_value("impedance_freq_length_input"),
        window_function=WindowFunction(
            dpg.get_value("impedance_window_func_input")
        ),
        input_device=input_device,
        output_device=output_device,
        block_size=ui.settings.audio.block_size,
    )


def show_calibration_setup(sender, app_data, user_data: ImpedanceUi) -> None:
    dpg.set_value(
        user_data.calibration_text,
        "Stage 1 of 2: channel calibration\n\n"
        "Connect CH1 and CH2 to the same audio_out point relative to ground.\n"
        "Both inputs must receive the same electrical multitone signal.\n"
        "Different channel gain and a small delay are allowed.",
    )
    dpg.configure_item(
        user_data.calibration_continue_button,
        label="Start stage 1",
    )
    dpg.show_item(user_data.calibration_dialog)


def continue_calibration(sender, app_data, user_data: ImpedanceUi) -> None:
    try:
        snapshot = user_data.state.snapshot()
        if snapshot.calibration_stage == CalibrationStage.WAITING_REFERENCE:
            started = user_data.state.continue_calibration()
        else:
            if not pause_io_updater(user_data):
                raise ValueError("Audio device scan did not stop")
            started = user_data.state.start_calibration(build_config(user_data))
        if started:
            dpg.hide_item(user_data.calibration_dialog)
    except (TypeError, ValueError) as exc:
        show_error(user_data, exc)


def cancel_calibration(sender, app_data, user_data: ImpedanceUi) -> None:
    user_data.state.cancel_calibration()
    dpg.hide_item(user_data.calibration_dialog)


def start_measurement(sender, app_data, user_data: ImpedanceUi) -> None:
    try:
        if user_data.state.snapshot().state == MeasurementState.MEASURING:
            user_data.state.stop_measurement()
            return
        if not pause_io_updater(user_data):
            raise ValueError("Audio device scan did not stop")
        user_data.state.start_measurement(build_config(user_data))
    except (TypeError, ValueError) as exc:
        show_error(user_data, exc)


def toggle_test_signal(sender, app_data, user_data: ImpedanceUi) -> None:
    try:
        if user_data.state.snapshot().testing:
            user_data.state.stop_test_signal()
            return
        if not pause_io_updater(user_data):
            raise ValueError("Audio device scan did not stop")
        if not user_data.state.start_test_signal(build_config(user_data)):
            raise ValueError("Could not start the test signal now")
    except (TypeError, ValueError) as exc:
        show_error(user_data, exc)


def filtering_changed(sender, app_data, user_data: ImpedanceUi) -> None:
    try:
        user_data.state.request_reprocess(build_config(user_data))
    except (TypeError, ValueError) as exc:
        show_error(user_data, exc)


def set_phase_display(sender, app_data, user_data) -> None:
    ui, mode = user_data
    ui.phase_mode = mode
    dpg.set_value(
        ui.phase_angle_menu_item,
        mode == PhaseDisplayMode.ANGLE,
    )
    dpg.set_value(
        ui.phase_derivative_menu_item,
        mode == PhaseDisplayMode.DERIVATIVE,
    )
    update_plot(ui, ui.state.snapshot())


def configure_phase_axis(
    ui: ImpedanceUi,
    mode: PhaseDisplayMode,
    phase: Any,
) -> None:
    if mode == PhaseDisplayMode.ANGLE:
        dpg.set_axis_limits(ui.phase_axis, -180.0, 180.0)
        ui.unlock_phase_axis_frames = 0
        return
    lower, upper = phase_axis_limits(phase)
    dpg.set_axis_limits(ui.phase_axis, lower, upper)
    ui.unlock_phase_axis_frames = 2


def update_pending_axis_limits(ui: ImpedanceUi) -> None:
    if ui.unlock_impedance_axis_frames > 0:
        ui.unlock_impedance_axis_frames -= 1
        if ui.unlock_impedance_axis_frames == 0:
            dpg.set_axis_limits_auto(ui.impedance_axis)

    if ui.unlock_phase_axis_frames > 0:
        ui.unlock_phase_axis_frames -= 1
        if ui.unlock_phase_axis_frames == 0:
            dpg.set_axis_limits_auto(ui.phase_axis)


def show_export_dialog(sender, app_data, user_data) -> None:
    dpg.show_item(user_data)


def show_open_project_dialog(
    sender,
    app_data,
    user_data: ImpedanceUi,
) -> None:
    dpg.show_item(user_data.open_project_dialog)


def save_project_menu(sender, app_data, user_data: ImpedanceUi) -> None:
    dpg.show_item(user_data.save_project_dialog)


def save_project_as(sender, app_data: dict, user_data: ImpedanceUi) -> None:
    path = project_path_from_dialog(app_data, ensure_extension=True)
    if path is None:
        show_error(user_data, "Cannot resolve the BMI project path")
        return
    dpg.hide_item(user_data.save_project_dialog)
    start_project_save(user_data, path)


def open_project(sender, app_data: dict, user_data: ImpedanceUi) -> None:
    path = project_path_from_dialog(app_data, ensure_extension=False)
    if path is None:
        show_error(user_data, "Cannot resolve the BMI project path")
        return
    dpg.hide_item(user_data.open_project_dialog)
    start_project_open(user_data, path)


def start_project_save(ui: ImpedanceUi, path: Path) -> None:
    if ui.project_busy:
        return
    try:
        project = ui.state.export_project(ui.phase_mode)
    except ValueError as exc:
        show_error(ui, exc)
        return
    ui.project_busy = True
    ui.revision = -1
    dpg.set_value(ui.status_text, f"Saving BMI project: {path}")
    worker = Thread(
        target=_project_save_worker,
        args=(ui, path, project),
        daemon=True,
    )
    ui.project_worker = worker
    worker.start()


def start_project_open(ui: ImpedanceUi, path: Path) -> None:
    if ui.project_busy:
        return
    ui.project_busy = True
    ui.revision = -1
    dpg.set_value(ui.status_text, f"Opening BMI project: {path}")
    worker = Thread(
        target=_project_open_worker,
        args=(ui, path),
        daemon=True,
    )
    ui.project_worker = worker
    worker.start()


def _project_save_worker(ui: ImpedanceUi, path: Path, project) -> None:
    try:
        saved_path = save_impedance_project(path, project)
    except Exception as exc:
        ui.project_results.put(("save", path, None, str(exc)))
        return
    ui.project_results.put(("save", saved_path, None, None))


def _project_open_worker(ui: ImpedanceUi, path: Path) -> None:
    try:
        project = load_impedance_project(path)
    except Exception as exc:
        ui.project_results.put(("open", path, None, str(exc)))
        return
    ui.project_results.put(("open", path, project, None))


def process_project_results(ui: ImpedanceUi) -> None:
    while True:
        try:
            operation, path, project, error = ui.project_results.get_nowait()
        except Empty:
            return
        ui.project_busy = False
        ui.project_worker = None
        ui.revision = -1
        if error is not None:
            show_error(ui, f"BMI project {operation} failed:\n{error}")
            continue
        if operation == "save":
            ui.project_path = path
            dpg.set_value(ui.status_text, f"BMI project saved: {path}")
            continue
        try:
            ui.state.restore_project(project)
            apply_project_controls(ui, project)
        except (TypeError, ValueError) as exc:
            show_error(ui, f"BMI project open failed:\n{exc}")
            continue
        ui.project_path = path
        ui.plot_data_token = None
        ui.revision = -1


def apply_project_controls(ui: ImpedanceUi, project) -> None:
    config = project.result_config or project.calibration_config
    dpg.set_value("band_input", [int(config.f_min), int(config.f_max)])
    dpg.set_value("impedance_duration_input", config.duration)
    dpg.set_value("reference_resistor_input", config.reference_resistor)
    dpg.set_value("calibration_resistor_input", config.calibration_resistor)
    dpg.set_value("impedance_window_width_input", config.window_width)
    dpg.set_value("impedance_freq_length_input", config.points)
    dpg.set_value(
        "impedance_window_func_input",
        config.window_function.value,
    )
    ui.settings.audio.block_size = config.block_size
    ui.settings.save()
    dpg.set_value(ui.block_size_input, config.block_size)
    ui.phase_mode = project.phase_mode
    dpg.set_value(
        ui.phase_angle_menu_item,
        project.phase_mode == PhaseDisplayMode.ANGLE,
    )
    dpg.set_value(
        ui.phase_derivative_menu_item,
        project.phase_mode == PhaseDisplayMode.DERIVATIVE,
    )


def project_path_from_dialog(
    app_data: dict,
    *,
    ensure_extension: bool,
) -> Path | None:
    file_path = app_data.get("file_path_name")
    if file_path:
        path = Path(file_path)
        if (
            str(file_path).endswith(("\\", "/"))
            or (path.exists() and path.is_dir())
        ):
            file_name = app_data.get("file_name")
            if not file_name:
                return None
            path /= file_name
    else:
        current_path = app_data.get("current_path")
        file_name = app_data.get("file_name")
        if not current_path or not file_name:
            return None
        path = Path(current_path) / file_name
    return ensure_project_extension(path) if ensure_extension else path


def show_io_settings(sender, app_data, user_data: ImpedanceUi) -> None:
    user_data.io_updater.enable.set()
    user_data.last_io_update = 0.0
    sync_io_settings(user_data, force=True)
    dpg.show_item(user_data.io_dialog)


def close_io_settings(sender, app_data, user_data: ImpedanceUi) -> None:
    pause_io_updater(user_data)
    dpg.hide_item(user_data.io_dialog)


def show_spice_model(sender, app_data, user_data: ImpedanceUi) -> None:
    snapshot = user_data.state.snapshot()
    if snapshot.frequency is None or snapshot.impedance is None:
        show_error(user_data, "No measurement available for SPICE modeling")
        return
    dpg.show_item(user_data.spice_dialog)
    if snapshot.spice_values is None and not snapshot.modeling:
        user_data.state.request_spice_model()


def close_spice_model(sender, app_data, user_data: ImpedanceUi) -> None:
    dpg.hide_item(user_data.spice_dialog)


def show_error(ui: ImpedanceUi, error: Exception | str) -> None:
    dpg.set_value(ui.error_text, str(error))
    dpg.show_item(ui.error_dialog)


def close_error(sender, app_data, user_data: ImpedanceUi) -> None:
    dpg.hide_item(user_data.error_dialog)


def pause_io_updater(ui: ImpedanceUi) -> bool:
    ui.io_updater.enable.clear()
    return ui.io_updater.paused.wait(timeout=2.0)


def set_input_device(sender, name: str, user_data: ImpedanceUi) -> None:
    name = name or DEFAULT_INPUT
    if name == user_data.settings.audio.input_device:
        return
    user_data.settings.audio.input_device = name
    user_data.settings.save()
    user_data.state.invalidate_calibration("IO settings changed; calibrate again")


def set_output_device(sender, name: str, user_data: ImpedanceUi) -> None:
    name = name or DEFAULT_OUTPUT
    if name == user_data.settings.audio.output_device:
        return
    user_data.settings.audio.output_device = name
    user_data.settings.save()
    user_data.state.invalidate_calibration("IO settings changed; calibrate again")


def set_block_size(sender, value: int, user_data: ImpedanceUi) -> None:
    block_size = max(1, int(value))
    changed = block_size != user_data.settings.audio.block_size
    user_data.settings.audio.block_size = block_size
    if block_size != value:
        dpg.set_value(sender, block_size)
    user_data.settings.save()
    if changed:
        user_data.state.invalidate_calibration(
            "IO settings changed; calibrate again"
        )


def sync_io_settings(ui: ImpedanceUi, force: bool = False) -> None:
    now = monotonic()
    if not force and now - ui.last_io_update < 0.5:
        return
    inputs = ui.io_updater.inputs
    outputs = ui.io_updater.outputs
    input_name, _ = resolve_device(
        ui.settings.audio.input_device,
        inputs,
        DEFAULT_INPUT,
    )
    output_name, _ = resolve_device(
        ui.settings.audio.output_device,
        outputs,
        DEFAULT_OUTPUT,
    )
    changed = (
        input_name != ui.settings.audio.input_device
        or output_name != ui.settings.audio.output_device
    )
    ui.settings.audio.input_device = input_name
    ui.settings.audio.output_device = output_name
    dpg.configure_item(ui.input_combo, items=[DEFAULT_INPUT, *inputs])
    dpg.configure_item(ui.output_combo, items=[DEFAULT_OUTPUT, *outputs])
    dpg.set_value(ui.input_combo, input_name)
    dpg.set_value(ui.output_combo, output_name)
    if changed:
        ui.settings.save()
        ui.state.invalidate_calibration(
            "Audio device is unavailable; using default"
        )
    ui.last_io_update = now


def export_plot(sender, app_data: dict, user_data: ImpedanceUi) -> None:
    snapshot = user_data.state.snapshot()
    if snapshot.frequency is None or snapshot.impedance is None:
        show_error(user_data, "No measurement to export")
        return
    file_path = app_data.get("file_path_name")
    if not file_path:
        return
    try:
        saved_path = export_impedance_plot(
            file_path,
            snapshot.frequency,
            snapshot.impedance,
        )
    except (OSError, ValueError) as exc:
        show_error(user_data, f"Export error: {exc}")
        return
    dpg.set_value(user_data.status_text, f"Plot exported: {saved_path}")


def sync_ui(ui: ImpedanceUi) -> None:
    ui.input_level_meter.resize()
    ui.spice_table.resize()
    update_pending_axis_limits(ui)
    process_project_results(ui)
    if dpg.is_item_shown(ui.io_dialog):
        sync_io_settings(ui)
    elif ui.io_updater.enable.is_set():
        ui.io_updater.enable.clear()

    snapshot = ui.state.snapshot()
    if snapshot.revision == ui.revision:
        return
    ui.revision = snapshot.revision

    if (
        snapshot.calibration_stage == CalibrationStage.WAITING_REFERENCE
        and not dpg.is_item_shown(ui.calibration_dialog)
    ):
        dpg.set_value(
            ui.calibration_text,
            "Stage 2 of 2: resistor calibration\n\n"
            "Connect the circuit:\n"
            "audio_out -- Rref -- p1 -- Rcal -- ground\n\n"
            "CH1: audio_out to ground\n"
            "CH2: p1 to ground",
        )
        dpg.configure_item(
            ui.calibration_continue_button,
            label="Start stage 2",
        )
        dpg.show_item(ui.calibration_dialog)
    elif snapshot.calibration_stage == CalibrationStage.IDLE:
        dpg.hide_item(ui.calibration_dialog)

    acquiring = snapshot.state in (
        MeasurementState.CALIBRATING,
        MeasurementState.MEASURING,
    )
    busy = (
        acquiring
        or snapshot.processing
        or snapshot.testing
        or ui.project_busy
    )
    can_measure = snapshot.state in (
        MeasurementState.CALIBRATED,
        MeasurementState.MEASURING_COMPLETED,
    )
    dpg.configure_item(ui.calibrate_button, enabled=not busy)
    dpg.configure_item(
        ui.measure_button,
        enabled=(
            snapshot.state == MeasurementState.MEASURING
            or (can_measure and not busy)
        ),
    )
    dpg.configure_item(
        ui.test_button,
        enabled=not acquiring and not snapshot.processing,
    )
    dpg.configure_item(ui.io_menu_item, enabled=not busy)
    project_operation_active = (
        snapshot.state == MeasurementState.MEASURING
        or snapshot.calibration_stage in (
            CalibrationStage.CHANNELS,
            CalibrationStage.REFERENCE,
        )
        or snapshot.processing
        or snapshot.testing
        or snapshot.modeling
    )
    dpg.configure_item(
        ui.open_project_menu_item,
        enabled=not project_operation_active and not ui.project_busy,
    )
    dpg.configure_item(
        ui.save_project_menu_item,
        enabled=(
            snapshot.project_available
            and not project_operation_active
            and not ui.project_busy
        ),
    )
    dpg.configure_item(
        ui.spice_menu_item,
        enabled=(
            snapshot.state == MeasurementState.MEASURING_COMPLETED
            and not acquiring
        ),
    )
    dpg.configure_item(
        ui.calibrate_button,
        label=(
            "Calibrating..."
            if snapshot.state == MeasurementState.CALIBRATING
            else "Calibrate"
        ),
    )
    dpg.configure_item(
        ui.measure_button,
        label=(
            "Stop"
            if snapshot.state == MeasurementState.MEASURING
            else "Measure"
        ),
    )
    dpg.configure_item(
        ui.test_button,
        label=("Stop test" if snapshot.testing else "Test"),
    )
    for item in ui.capture_settings:
        dpg.configure_item(item, enabled=not busy)
    for item in ui.filter_settings:
        dpg.configure_item(item, enabled=not acquiring)

    status = f"State: {snapshot.state.value} | {snapshot.status}"
    dpg.set_value(ui.status_text, status)
    if snapshot.error:
        show_error(ui, snapshot.error)
    ui.input_level_meter.set_levels(*snapshot.levels)

    if snapshot.frequency is not None and snapshot.impedance is not None:
        plot_data_token = hash(snapshot.frequency.tobytes()) ^ hash(
            snapshot.impedance.tobytes()
        )
        if plot_data_token != ui.plot_data_token:
            update_plot(ui, snapshot)
            ui.plot_data_token = plot_data_token
    elif snapshot.state in (
        MeasurementState.UNCALIBRATED,
        MeasurementState.CALIBRATING,
        MeasurementState.MEASURING,
    ):
        dpg.set_value(ui.impedance_line, [[], []])
        dpg.set_value(ui.phase_line, [[], []])
        ui.plot_data_token = None

    if snapshot.spice_values is not None:
        ui.spice_table.set_values(
            snapshot.spice_values.l1,
            snapshot.spice_values.sections,
            snapshot.spice_values.r1,
        )
    else:
        ui.spice_table.set_values(
            "",
            tuple(("", "", "") for _ in range(SPICE_SECTION_COUNT)),
            "",
        )
    if snapshot.modeling:
        dpg.set_value(ui.spice_status_text, "Calculating SPICE model...")
    elif snapshot.spice_values is not None:
        dpg.set_value(ui.spice_status_text, "SPICE model ready")
    else:
        dpg.set_value(
            ui.spice_status_text,
            "No model calculated for the current measurement",
        )


def update_plot(ui: ImpedanceUi, snapshot) -> None:
    if snapshot.frequency is None or snapshot.impedance is None:
        return
    frequency = snapshot.frequency
    magnitude = abs(snapshot.impedance)
    impedance_min, impedance_max = impedance_axis_limits(magnitude)
    phase, axis_label, series_label = phase_plot_data(
        frequency,
        snapshot.impedance,
        ui.phase_mode,
    )
    x_values = frequency.tolist()
    dpg.set_value(ui.impedance_line, [x_values, magnitude.tolist()])
    dpg.set_value(ui.phase_line, [x_values, phase.tolist()])
    dpg.configure_item(ui.phase_axis, label=axis_label)
    dpg.configure_item(ui.phase_line, label=series_label)
    dpg.set_axis_limits(
        ui.impedance_axis,
        impedance_min,
        impedance_max,
    )
    ui.unlock_impedance_axis_frames = 2
    configure_phase_axis(ui, ui.phase_mode, phase)
