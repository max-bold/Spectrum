from dataclasses import dataclass
from time import monotonic
from typing import Any

import dearpygui.dearpygui as dpg
from utils.audio import io_list_updater

from spectrum_app.settings import (
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
        phase_plot_data,
        resolve_sample_rate,
    )
    from .spice_table import SPICE_SECTION_COUNT, SpiceModelTable
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
        phase_plot_data,
        resolve_sample_rate,
    )
    from spice_table import SPICE_SECTION_COUNT, SpiceModelTable


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
    phase_angle_menu_item: int | str
    phase_derivative_menu_item: int | str
    io_dialog: int | str
    input_combo: int | str
    output_combo: int | str
    block_size_input: int | str
    close_io_button: int | str
    capture_settings: tuple[int | str, ...]
    filter_settings: tuple[int | str, ...]
    phase_mode: PhaseDisplayMode = PhaseDisplayMode.ANGLE
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
    dpg.configure_item(export_dialog, callback=export_plot, user_data=ui)
    dpg.configure_item(
        ui.io_menu_item,
        callback=show_io_settings,
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
        if not pause_io_updater(user_data):
            raise ValueError("Audio device scan did not stop")
        user_data.state.start_measurement(build_config(user_data))
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


def show_export_dialog(sender, app_data, user_data) -> None:
    dpg.show_item(user_data)


def show_io_settings(sender, app_data, user_data: ImpedanceUi) -> None:
    user_data.io_updater.enable.set()
    user_data.last_io_update = 0.0
    sync_io_settings(user_data, force=True)
    dpg.show_item(user_data.io_dialog)


def close_io_settings(sender, app_data, user_data: ImpedanceUi) -> None:
    pause_io_updater(user_data)
    dpg.hide_item(user_data.io_dialog)


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
    busy = acquiring or snapshot.processing
    can_measure = snapshot.state in (
        MeasurementState.CALIBRATED,
        MeasurementState.MEASURING_COMPLETED,
    )
    dpg.configure_item(ui.calibrate_button, enabled=not busy)
    dpg.configure_item(ui.measure_button, enabled=can_measure and not busy)
    dpg.configure_item(ui.io_menu_item, enabled=not busy)
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
            "Measuring..."
            if snapshot.state == MeasurementState.MEASURING
            else "Measure"
        ),
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
        update_plot(ui, snapshot)
    elif snapshot.state in (
        MeasurementState.UNCALIBRATED,
        MeasurementState.CALIBRATING,
        MeasurementState.MEASURING,
    ):
        dpg.set_value(ui.impedance_line, [[], []])
        dpg.set_value(ui.phase_line, [[], []])

    if snapshot.spice_values is not None:
        ui.spice_table.set_values(
            snapshot.spice_values.l1,
            snapshot.spice_values.sections,
            snapshot.spice_values.r1,
        )
    elif snapshot.state in (
        MeasurementState.UNCALIBRATED,
        MeasurementState.CALIBRATING,
        MeasurementState.MEASURING,
    ):
        ui.spice_table.set_values(
            "",
            tuple(("", "", "") for _ in range(SPICE_SECTION_COUNT)),
            "",
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
    dpg.set_axis_limits_auto(ui.phase_axis)
    dpg.fit_axis_data(ui.phase_axis)
