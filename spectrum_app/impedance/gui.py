import sys
from pathlib import Path

import dearpygui.dearpygui as dpg

if __package__:
    from . import cbs
    from .ilm import METER_WIDTH, add_input_level_meter
    from .spice_table import (
        SPICE_TABLE_HEIGHT,
        add_spice_model_table,
    )
else:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from spectrum_app.impedance import cbs
    from spectrum_app.impedance.ilm import METER_WIDTH, add_input_level_meter
    from spectrum_app.impedance.spice_table import (
        SPICE_TABLE_HEIGHT,
        add_spice_model_table,
    )

from spectrum_app.fonts import bind_app_font

IMPEDANCE_WINDOW = "Impedance window"
GRAPH_TABLE_SPACING = 4
STATUS_BAR_HEIGHT = 28
STATUS_BAR_RESERVE = STATUS_BAR_HEIGHT + GRAPH_TABLE_SPACING
WINDOW_FUNCTIONS = ("flat", "cosine", "gaussian", "triangular")
CONTROL_PANEL_WIDTH = 175
IO_SETTINGS_WIDTH = 520
IO_SETTINGS_CONTENT_WIDTH = 503


def impedance_ui(
    state: cbs.ImpedanceAppState | None = None,
) -> cbs.ImpedanceUi:
    state = state or cbs.create_state()
    settings, io_updater = cbs.create_io_settings()
    with dpg.theme() as window_layout_theme:
        with dpg.theme_component(dpg.mvWindowAppItem):
            dpg.add_theme_style(
                dpg.mvStyleVar_WindowPadding,
                8,
                GRAPH_TABLE_SPACING,
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_ItemSpacing,
                8,
                GRAPH_TABLE_SPACING,
            )
    with dpg.window(
        label=IMPEDANCE_WINDOW,
        tag=IMPEDANCE_WINDOW,
        no_scrollbar=True,
        no_scroll_with_mouse=True,
    ) as impedance_window:
        dpg.bind_item_theme(impedance_window, window_layout_theme)
        export_dialog = dpg.add_file_dialog(
            show=False,
            modal=True,
            width=700,
            height=400,
            callback=cbs.export_plot,
            default_filename="impedance.png",
            tag="impedance_export_dialog",
        )
        dpg.add_file_extension(".png", parent=export_dialog)
        dpg.add_file_extension(".jpg", parent=export_dialog)
        dpg.add_file_extension(".jpeg", parent=export_dialog)
        with dpg.window(
            label="IO Settings",
            show=False,
            popup=True,
            no_resize=True,
            no_scrollbar=True,
            width=IO_SETTINGS_WIDTH,
            height=235,
            tag="impedance_io_settings",
        ) as io_dialog:
            dpg.add_text("Input")
            input_combo = dpg.add_combo(
                items=[cbs.DEFAULT_INPUT],
                default_value=settings.audio.input_device,
                width=IO_SETTINGS_CONTENT_WIDTH,
                tag="impedance_input_device",
            )
            dpg.add_text("Output")
            output_combo = dpg.add_combo(
                items=[cbs.DEFAULT_OUTPUT],
                default_value=settings.audio.output_device,
                width=IO_SETTINGS_CONTENT_WIDTH,
                tag="impedance_output_device",
            )
            dpg.add_text("Block size, samples")
            block_size_input = dpg.add_input_int(
                default_value=settings.audio.block_size,
                min_value=1,
                min_clamped=True,
                step=0,
                width=IO_SETTINGS_CONTENT_WIDTH,
                tag="impedance_block_size",
            )
            close_io_button = dpg.add_button(
                label="Close",
                width=IO_SETTINGS_CONTENT_WIDTH,
                tag="impedance_io_close",
            )
        with dpg.window(
            label="Error",
            show=False,
            modal=True,
            no_resize=True,
            no_close=True,
            no_scrollbar=True,
            width=680,
            height=230,
            tag="impedance_error_dialog",
        ) as error_dialog:
            error_text = dpg.add_input_text(
                multiline=True,
                readonly=True,
                no_horizontal_scroll=True,
                width=-1,
                height=155,
                tag="impedance_error_text",
            )
            error_close_button = dpg.add_button(
                label="OK",
                width=-1,
                tag="impedance_error_close",
            )
        with dpg.window(
            label="Impedance calibration",
            show=False,
            modal=True,
            no_resize=True,
            no_close=True,
            no_scrollbar=True,
            width=560,
            height=265,
            tag="impedance_calibration_dialog",
        ) as calibration_dialog:
            calibration_text = dpg.add_input_text(
                multiline=True,
                readonly=True,
                no_horizontal_scroll=True,
                width=-1,
                height=180,
                tag="impedance_calibration_text",
            )
            with dpg.group(horizontal=True):
                calibration_continue_button = dpg.add_button(
                    label="Start",
                    width=268,
                    tag="impedance_calibration_continue",
                )
                calibration_cancel_button = dpg.add_button(
                    label="Cancel",
                    width=268,
                    tag="impedance_calibration_cancel",
                )

        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(
                    label="Export Plot",
                    callback=cbs.show_export_dialog,
                    user_data=export_dialog,
                    tag="impedance_export_plot_menu",
                )
            with dpg.menu(label="Settings"):
                io_menu_item = dpg.add_menu_item(
                    label="IO",
                    tag="impedance_io_settings_menu",
                )
                with dpg.menu(label="Phaze"):
                    phase_angle_menu_item = dpg.add_menu_item(
                        label="Angle",
                        check=True,
                        default_value=True,
                        tag="impedance_phase_angle_menu",
                    )
                    phase_derivative_menu_item = dpg.add_menu_item(
                        label="Dir",
                        check=True,
                        default_value=False,
                        tag="impedance_phase_derivative_menu",
                    )

        layout_group = dpg.add_group(
            horizontal=False,
            tag="impedance_layout",
        )
        content_panel = dpg.add_child_window(
            parent=layout_group,
            width=-1,
            height=-STATUS_BAR_RESERVE,
            border=False,
            no_scrollbar=True,
            no_scroll_with_mouse=True,
            tag="impedance_content",
        )
        with dpg.group(
            parent=content_panel,
            horizontal=True,
            tag="graph&table",
        ):
            with dpg.child_window(
                width=-(METER_WIDTH + CONTROL_PANEL_WIDTH + 16),
                height=-1,
                border=False,
                no_scrollbar=True,
                no_scroll_with_mouse=True,
            ) as graph_cell:
                with dpg.plot(
                    label="Impedance` Plot",
                    width=-1,
                    height=-(SPICE_TABLE_HEIGHT + GRAPH_TABLE_SPACING),
                ) as impedance_plot:
                    with dpg.plot_axis(
                        dpg.mvXAxis,
                        label="Hz",
                        scale=dpg.mvPlotScale_Log10,
                    ) as xaxis:
                        dpg.set_axis_limits(xaxis, 20, 20e3)
                    with dpg.plot_axis(
                        dpg.mvYAxis,
                        label="IMP (OHM)",
                    ) as yaxis:
                        dpg.set_axis_limits(yaxis, 0, 100)
                        imp_graph = dpg.add_line_series(
                            [],
                            [],
                            label="Impedance",
                        )
                    with dpg.plot_axis(
                        dpg.mvYAxis,
                        label="Phase (deg)",
                        tag="phase_axis",
                    ) as phase_axis:
                        dpg.set_axis_limits(phase_axis, -180, 180)
                        phase_graph = dpg.add_line_series(
                            [],
                            [],
                            label="Phase",
                            parent=phase_axis,
                        )
                spice_table = add_spice_model_table(
                    graph_cell,
                    impedance_plot,
                )

            with dpg.child_window(
                width=METER_WIDTH,
                height=-1,
                border=False,
                no_scrollbar=True,
                no_scroll_with_mouse=True,
            ) as meter_cell:
                input_level_meter = add_input_level_meter(
                    meter_cell,
                    impedance_plot,
                    SPICE_TABLE_HEIGHT + GRAPH_TABLE_SPACING,
                )

            with dpg.child_window(
                width=CONTROL_PANEL_WIDTH,
                height=-1,
                border=False,
                no_scrollbar=True,
                no_scroll_with_mouse=True,
            ):
                calibrate_button = dpg.add_button(
                    label="Calibrate",
                    tag="calibrate_btn",
                    width=-1,
                    height=30,
                )
                measure_button = dpg.add_button(
                    label="Measure",
                    tag="measure_btn",
                    width=-1,
                    height=50,
                    enabled=False,
                )
                with dpg.collapsing_header(
                    label="Generator",
                    default_open=True,
                ):
                    dpg.add_text("Band, Hz")
                    band_input = dpg.add_input_intx(
                        default_value=[int(20), int(20e3)],
                        size=2,
                        width=-1,
                        tag="band_input",
                    )
                    dpg.add_text("Duration, s")
                    duration_input = dpg.add_input_float(
                        default_value=5.0,
                        min_value=0.1,
                        min_clamped=True,
                        step=0,
                        width=-1,
                        tag="impedance_duration_input",
                    )
                with dpg.collapsing_header(
                    label="Measurement",
                    default_open=True,
                ):
                    dpg.add_text("Reference resistor, Ohm")
                    reference_resistor_input = dpg.add_input_float(
                        default_value=3.25,
                        min_value=0.001,
                        min_clamped=True,
                        step=0,
                        width=-1,
                        tag="reference_resistor_input",
                    )
                    dpg.add_text("Calibration resistor, Ohm")
                    calibration_resistor_input = dpg.add_input_float(
                        default_value=10.4,
                        min_value=0.001,
                        min_clamped=True,
                        step=0,
                        width=-1,
                        tag="calibration_resistor_input",
                    )
                with dpg.collapsing_header(
                    label="Filtering",
                    default_open=True,
                ):
                    dpg.add_text("Window")
                    window_function_input = dpg.add_combo(
                        WINDOW_FUNCTIONS,
                        default_value="gaussian",
                        width=-1,
                        tag="impedance_window_func_input",
                    )
                    dpg.add_text("Window width, octaves")
                    window_width_input = dpg.add_input_float(
                        default_value=0.1,
                        step=0.1,
                        width=-1,
                        tag="impedance_window_width_input",
                    )
                    dpg.add_text("Samples")
                    points_input = dpg.add_input_int(
                        default_value=1024,
                        step=0,
                        width=-1,
                        tag="impedance_freq_length_input",
                    )

        status_text = dpg.add_input_text(
            parent=layout_group,
            default_value="State: uncalibrated | Calibration required",
            readonly=True,
            width=-1,
            height=STATUS_BAR_HEIGHT,
            tag="impedance_status_bar",
        )

    ui = cbs.ImpedanceUi(
        state=state,
        settings=settings,
        io_updater=io_updater,
        input_level_meter=input_level_meter,
        spice_table=spice_table,
        impedance_line=imp_graph,
        phase_line=phase_graph,
        impedance_axis=yaxis,
        phase_axis=phase_axis,
        calibrate_button=calibrate_button,
        measure_button=measure_button,
        status_text=status_text,
        error_dialog=error_dialog,
        error_text=error_text,
        error_close_button=error_close_button,
        calibration_dialog=calibration_dialog,
        calibration_text=calibration_text,
        calibration_continue_button=calibration_continue_button,
        calibration_cancel_button=calibration_cancel_button,
        io_menu_item=io_menu_item,
        phase_angle_menu_item=phase_angle_menu_item,
        phase_derivative_menu_item=phase_derivative_menu_item,
        io_dialog=io_dialog,
        input_combo=input_combo,
        output_combo=output_combo,
        block_size_input=block_size_input,
        close_io_button=close_io_button,
        capture_settings=(
            reference_resistor_input,
            calibration_resistor_input,
            duration_input,
            band_input,
        ),
        filter_settings=(
            window_width_input,
            points_input,
            window_function_input,
        ),
    )
    cbs.bind_ui(ui, export_dialog)
    return ui


if __name__ == "__main__":
    dpg.create_context()
    bind_app_font()
    ui = impedance_ui()
    dpg.create_viewport(title="Impedance Measurement", width=1024, height=768)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(IMPEDANCE_WINDOW, True)
    try:
        while dpg.is_dearpygui_running():
            cbs.sync_ui(ui)
            dpg.render_dearpygui_frame()
    finally:
        ui.io_updater.enable.clear()
        dpg.destroy_context()
