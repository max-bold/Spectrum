import dearpygui.dearpygui as dpg

dpg.create_context()
import cbs

def_analyzer = cbs.Analyzer()
def_audio_io = cbs.AudioIO()

REC_NUMBER = 5

from utils.themes import green_theme, red_theme

with dpg.window(tag="Primary Window") as main_window:
    # with dpg.menu_bar():
    #     with dpg.menu(label="File"):
    #         dpg.add_menu_item(label="Open project")
    #         dpg.add_menu_item(label="Open record")
    #         dpg.add_separator()
    #         dpg.add_menu_item(label="Save project")
    #         dpg.add_menu_item(label="Save record")
    #         dpg.add_separator()
    #         dpg.add_menu_item(label="Exit")
    #     with dpg.menu(label="Settings"):
    #         # dpg.add_menu_item(label="Audio IO", callback=cbs.audioset_open)
    #         dpg.add_menu_item(label="Analyzer", callback=lambda: dpg.show_item("AnSet"))
    with dpg.group(horizontal=True):
        with dpg.group(width=-200):
            with dpg.plot(height=-200, label="FFT") as fft_plot:
                dpg.add_plot_legend()
                xaxis = dpg.add_plot_axis(
                    dpg.mvXAxis, label="Hz", scale=dpg.mvPlotScale_Log10
                )
                dpg.set_axis_limits(xaxis, 20, 20e3)
                with dpg.plot_axis(dpg.mvYAxis, label="PSD [db]") as yaxis:
                    for i in range(REC_NUMBER):
                        line = dpg.add_line_series(
                            [], [], label=f"record {i+1}", show=False
                        )
                        cbs.lines.append(line)
            with dpg.plot(height=-1, label="Time plot"):
                levels_xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="s")
                with dpg.plot_axis(dpg.mvYAxis) as levels_yaxis:
                    dpg.set_axis_limits(levels_yaxis, 0, 1)
                    cbs.levels_l = dpg.add_line_series([], [])
                    cbs.levels_r = dpg.add_line_series([], [])
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
                        cbs.GenMode.list(),
                        width=-1,
                        default_value=def_audio_io.gen_mode.value,
                        callback=cbs.set_genmode,
                    )
                    dpg.add_text("band, Hz")
                    dpg.add_input_intx(
                        default_value=[
                            int(def_audio_io.band[0]),
                            int(def_audio_io.band[1]),
                        ],
                        size=2,
                        width=-1,
                        callback=cbs.set_band,
                    )
                    dpg.add_text("length, s")
                    rec_len = dpg.add_input_float(
                        width=-1,
                        step=0,
                        step_fast=0,
                        default_value=def_audio_io.length,
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
                            cbs.AnalyzerMode.list(),
                            default_value=def_analyzer.analyzer_mode.value,
                            callback=cbs.set_analyzer_mode,
                        )
                        with dpg.group() as rta_bucket_size_group:
                            dpg.add_text("welch bucket size, samples")
                            dpg.add_input_int(
                                default_value=def_analyzer.welch_n,
                                callback=cbs.set_bucket_size,
                                step=0,
                            )
                        dpg.add_text("reference")
                        dpg.add_combo(
                            cbs.RefMode.list(),
                            default_value=def_audio_io.ref.value,
                            callback=cbs.set_analyzer_ref,
                        )
                        dpg.add_text("weighting")
                        dpg.add_combo(
                            cbs.WeightingMode.list(),
                            default_value=def_analyzer.weighting.value,
                            callback=cbs.set_analyzer_weighting,
                        )
                with dpg.collapsing_header(label="filtering"):
                    with dpg.group(width=-1):
                        dpg.add_text("window width, octaves")
                        dpg.add_input_float(
                            default_value=def_analyzer.window_width,
                            callback=cbs.set_window_width,
                            step=0.1,
                        )
                        dpg.add_text("pints number")
                        dpg.add_input_int(
                            default_value=def_analyzer.freq_length,
                            callback=cbs.set_freq_length,
                            step=0,
                        )
                        dpg.add_text("window function")
                        dpg.add_combo(
                            cbs.Windows.list(),
                            default_value=def_analyzer.window_func.value,
                            callback=cbs.set_filter_window_func,
                        )
            dpg.add_separator(label="records")
            with dpg.table(header_row=False, policy=dpg.mvTable_SizingFixedFit):
                for i in range(3):
                    dpg.add_table_column()
                # rows = []
                for i in range(REC_NUMBER):
                    with dpg.table_row():
                        used = dpg.add_checkbox(
                            user_data=cbs.lines_table_rows,
                            callback=cbs.record_used_click,
                        )
                        visible = dpg.add_checkbox(
                            user_data=(cbs.lines_table_rows, cbs.lines),
                            callback=cbs.record_visible_clicked,
                        )
                        name = dpg.add_input_text(
                            width=140,
                            user_data=(cbs.lines_table_rows, cbs.lines),
                            callback=cbs.record_set_name,
                            default_value=f"record {i+1}",
                        )
                        cbs.lines_table_rows.append([used, visible, name])
                dpg.set_value(cbs.lines_table_rows[0][0], True)
                cbs.current_line = 0


# with dpg.window(
#     label="Audio IO settings",
#     width=300,
#     height=200,
#     show=False,
#     tag="AIO",
#     modal=True,
#     pos=[100, 100],
#     no_resize=True,
# ) as AIO_stt:
#     dpg.add_text("input:")
#     dpg.add_combo(
#         ["System default input"],
#         width=-1,
#         tag="input combo",
#         default_value="System default input",
#     )
#     dpg.add_text("output:")
#     dpg.add_combo(
#         ["System default outputput"],
#         width=-1,
#         tag="output combo",
#         default_value="System default outputput",
#     )


# with dpg.window(
#     label="Analyzer settings",
#     width=300,
#     height=200,
#     show=False,
#     tag="AnSet",
#     modal=True,
#     pos=[100, 100],
# ):
#     dpg.add_text("Reference:")
#     dpg.add_combo(["generator", "input B"], width=-1, default_value="generator")

# with dpg.window(modal=True, show=False, tag="rec progress", no_resize=True):
#     dpg.add_text("Measuring !!!")
#     dpg.add_progress_bar(width=300, height=50, tag="Measure prog bar")


dpg.create_viewport(title="BM Spectrum", width=1024, height=768)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)

while dpg.is_dearpygui_running():

    cbs.upd_io(inputs_combo, outputs_combo)

    if dpg.get_value(mon_cb):
        cbs.upd_level_monitor((left_level, right_level))

    if cbs.audio_io.running.is_set():
        dpg.set_item_label(run_btn, "ON")
        dpg.bind_item_theme(run_btn, red_theme)
    else:
        dpg.set_item_label(run_btn, "OFF")
        dpg.bind_item_theme(run_btn, green_theme)
    cbs.reenable_io_udater()
    max_t = dpg.get_value(rec_len)+2
    dpg.set_axis_limits(levels_xaxis, 0, max_t)
    cbs.run_analyzer()
    dpg.set_value(mon_cb, cbs.meter.enable.is_set())
    dpg.render_dearpygui_frame()

dpg.destroy_context()
cbs.audio_io.kill()