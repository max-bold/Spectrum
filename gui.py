from time import time
import dearpygui.dearpygui as dpg
import cbs
from utils.analyzer import AnalyserPipeline

default_pipeline = AnalyserPipeline()
REC_NUMBER = 5
dpg.create_context()

with dpg.theme(tag="green_theme"):
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(
            dpg.mvThemeCol_Button, (0, 200, 0, 255), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ButtonHovered, (0, 220, 0, 255), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ButtonActive, (0, 160, 0, 255), category=dpg.mvThemeCat_Core
        )

# создаём красную тему
with dpg.theme(tag="red_theme"):
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(
            dpg.mvThemeCol_Button, (200, 0, 0, 255), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ButtonHovered, (220, 0, 0, 255), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ButtonActive, (160, 0, 0, 255), category=dpg.mvThemeCat_Core
        )


with dpg.window(tag="Primary Window") as main_window:
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Open project")
            dpg.add_menu_item(label="Open record")
            dpg.add_separator()
            dpg.add_menu_item(label="Save project")
            dpg.add_menu_item(label="Save record")
            dpg.add_separator()
            dpg.add_menu_item(label="Exit")
        with dpg.menu(label="Settings"):
            # dpg.add_menu_item(label="Audio IO", callback=cbs.audioset_open)
            dpg.add_menu_item(label="Analyzer", callback=lambda: dpg.show_item("AnSet"))
    with dpg.group(horizontal=True):
        with dpg.group(width=-200):
            with dpg.plot(height=-200, label="FFT") as fft_plot:
                dpg.add_plot_legend()
                xaxis = dpg.add_plot_axis(
                    dpg.mvXAxis, label="Hz", scale=dpg.mvPlotScale_Log10
                )
                dpg.set_axis_limits(xaxis, 20, 20e3)
                with dpg.plot_axis(dpg.mvYAxis, label="PSD [db]") as yaxis:
                    lines = []
                    for i in range(REC_NUMBER):
                        line = dpg.add_line_series(
                            [], [], label=f"record {i+1}", show=False
                        )
                        lines.append(line)
            with dpg.plot(height=-1, label="Time plot"):
                levels_xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="s")
                with dpg.plot_axis(dpg.mvYAxis) as levels_yaxis:
                    dpg.set_axis_limits(levels_yaxis, 0, 1)
                    levels_line_l = dpg.add_line_series([], [])
                    levels_line_r = dpg.add_line_series([], [])
        with dpg.group():
            run_btn = dpg.add_button(
                label="OFF",
                width=-1,
                height=50,
                callback=cbs.run_btn,
                tag="run btn",
            )
            dpg.bind_item_theme(run_btn, "green_theme")
            with dpg.child_window(height=-142):

                with dpg.collapsing_header(label="generator", default_open=True):
                    dpg.add_combo(
                        ["log sweep", "pink noise"],
                        width=-1,
                        default_value=default_pipeline.gen_mode,
                        callback=cbs.set_genmode,
                    )
                    dpg.add_text("band, Hz")
                    dpg.add_input_intx(
                        default_value=[
                            int(default_pipeline.band[0]),
                            int(default_pipeline.band[1]),
                        ],
                        size=2,
                        width=-1,
                        callback=cbs.set_band,
                    )
                    dpg.add_text("length, s")
                    dpg.add_input_float(
                        width=-1,
                        step=0,
                        step_fast=0,
                        default_value=default_pipeline.length,
                        format="%.1f",
                        tag="length input",
                        callback=cbs.set_length,
                    )

                with dpg.collapsing_header(label="audio io"):
                    with dpg.group(width=-1):
                        dpg.add_text("input")
                        inputs_combo = dpg.add_combo(
                            default_value="default input", callback=cbs.set_input
                        )
                        with dpg.group(horizontal=True):
                            mon_cb = dpg.add_checkbox(
                                callback=cbs.set_input_meter, tag="meter_cb"
                            )
                            with dpg.group():
                                left_level = dpg.add_progress_bar(height=7)
                                right_level = dpg.add_progress_bar(height=7)
                        dpg.add_text("output")
                        outputs_combo = dpg.add_combo(
                            default_value="default output", callback=cbs.set_output
                        )

                with dpg.collapsing_header(label="analyzer"):
                    with dpg.group(width=-1):
                        dpg.add_text("mode")
                        analyzer_mode = dpg.add_combo(
                            ["rta", "recording"],
                            default_value="recording",
                            callback=cbs.set_analyzer_mode,
                        )
                        with dpg.group() as rta_bucket_size_group:
                            dpg.add_text("rta bucket size, samples")
                            dpg.add_input_int(
                                default_value=default_pipeline.rta_bucket_size,
                                callback=cbs.set_bucket_size,
                                step=0,
                            )
                        dpg.add_text("reference")
                        dpg.add_combo(
                            ["none", "channel b", "generator"],
                            default_value=default_pipeline.ref
                            callback=cbs.set_analyzer_ref,
                        )
                        dpg.add_text("weighting")
                        dpg.add_combo(
                            ["none", "pink"],
                            default_value="none",
                            callback=cbs.set_analyzer_weighting,
                        )
                with dpg.collapsing_header(label="filtering"):
                    with dpg.group(width=-1):
                        dpg.add_text("window width, octaves")
                        dpg.add_input_float(
                            default_value=default_pipeline.window_width,
                            callback=cbs.set_window_width,
                            step=0.1,
                        )
                        dpg.add_text("pints number")
                        dpg.add_input_int(
                            default_value=default_pipeline.freq_length,
                            callback=cbs.set_freq_length,
                            step=0,
                        )
                        dpg.add_text("window function")
                        dpg.add_combo(
                            ["blackman", "boxcar"],
                            default_value=default_pipeline.filter_window_func,
                            callback=cbs.set_filter_window_func,
                        )
            dpg.add_separator(label="records")
            with dpg.table(header_row=False, policy=dpg.mvTable_SizingFixedFit):
                for i in range(3):
                    dpg.add_table_column()
                rows = []
                for i in range(REC_NUMBER):
                    with dpg.table_row():
                        used = dpg.add_checkbox(
                            user_data=rows,
                            callback=cbs.record_used_click,
                        )
                        visible = dpg.add_checkbox(
                            user_data=(rows, lines),
                            callback=cbs.record_visible_clicked,
                        )
                        name = dpg.add_input_text(
                            width=140,
                            user_data=(rows, lines),
                            callback=cbs.record_set_name,
                            default_value=f"record {i+1}",
                        )
                        rows.append([used, visible, name])
                dpg.set_value(rows[0][0], True)
                cbs.current_rec = 0


with dpg.window(
    label="Audio IO settings",
    width=300,
    height=200,
    show=False,
    tag="AIO",
    modal=True,
    pos=[100, 100],
    no_resize=True,
) as AIO_stt:
    dpg.add_text("input:")
    dpg.add_combo(
        ["System default input"],
        width=-1,
        tag="input combo",
        default_value="System default input",
    )
    dpg.add_text("output:")
    dpg.add_combo(
        ["System default outputput"],
        width=-1,
        tag="output combo",
        default_value="System default outputput",
    )


with dpg.window(
    label="Analyzer settings",
    width=300,
    height=200,
    show=False,
    tag="AnSet",
    modal=True,
    pos=[100, 100],
):
    dpg.add_text("Reference:")
    dpg.add_combo(["generator", "input B"], width=-1, default_value="generator")

with dpg.window(modal=True, show=False, tag="rec progress", no_resize=True):
    dpg.add_text("Measuring !!!")
    dpg.add_progress_bar(width=300, height=50, tag="Measure prog bar")


dpg.create_viewport(title="BM Spectrum", width=1024, height=768)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)

t1_enbl = False
t1_val = 0
t1_setpoint = 5  # seconds

while dpg.is_dearpygui_running():

    cbs.upd_io(inputs_combo, outputs_combo)

    if dpg.get_value(mon_cb):
        cbs.upd_level_monitor((left_level, right_level))

    if dpg.get_value(analyzer_mode) == "rta":
        dpg.show_item(rta_bucket_size_group)
    else:
        dpg.hide_item(rta_bucket_size_group)

    if cbs.pipe.run_flag.is_set():
        dpg.set_item_label(run_btn, "ON")
        dpg.bind_item_theme(run_btn, "red_theme")
    else:
        dpg.set_item_label(run_btn, "OFF")
        dpg.bind_item_theme(run_btn, "green_theme")

    # Reenable cbs.meter after t1_setpoint after all sd users have stopped
    if (
        not cbs.pipe.run_flag.is_set()
        and not cbs.meter.enable.is_set()
        and not cbs.io_upd.enable.is_set()
    ):
        if not t1_enbl:
            t1_val = time()
            t1_enbl = True
        elif time() > t1_val + t1_setpoint:
            cbs.io_upd.enable.set()
            t1_enbl = False
    elif t1_enbl:
        t1_enbl = False

    if cbs.pipe.run_flag.is_set() or cbs.pipe.final_fft_ready.is_set():
        data = cbs.pipe.get_fft()
        if data.shape[1]:
            dpg.set_value(lines[cbs.current_rec], list(data))
        ts, levels = cbs.pipe.get_levels()
        dpg.set_value(levels_line_l, [list(ts), list(levels[0])])
        dpg.set_value(levels_line_r, [list(ts), list(levels[1])])
        max_t = cbs.pipe.length + cbs.pipe.end_padding + 0.5
        dpg.set_axis_limits(levels_xaxis, 0, max_t)
        if not dpg.get_value(rows[cbs.current_rec][1]):
            dpg.set_value(rows[cbs.current_rec][1], True)
            dpg.show_item(lines[cbs.current_rec])

    dpg.set_value(mon_cb, cbs.meter.enable.is_set())

    dpg.render_dearpygui_frame()

dpg.destroy_context()
