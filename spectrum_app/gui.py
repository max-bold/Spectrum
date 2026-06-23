import dearpygui.dearpygui as dpg

from . import cbs
from .ui.refs import REC_NUMBER, UiRefs


CONTROL_PANEL_WIDTH = 200
IO_SETTINGS_WIDTH = 520
IO_SETTINGS_CONTENT_WIDTH = 503


def build_ui(state: cbs.AppState) -> UiRefs:
    from .themes import green_theme

    with dpg.window(tag="Primary Window"):
        fft_dialog = dpg.add_file_dialog(
            show=False,
            modal=True,
            width=700,
            height=400,
            callback=cbs.save_fft_plot,
            user_data=state,
            default_filename="fft.png",
        )
        dpg.add_file_extension(".png", parent=fft_dialog)
        dpg.add_file_extension(".jpg", parent=fft_dialog)
        dpg.add_file_extension(".jpeg", parent=fft_dialog)
        wav_dialog = dpg.add_file_dialog(
            show=False,
            modal=True,
            width=700,
            height=400,
            callback=cbs.save_current_record_wav,
            user_data=state,
            default_filename="record",
        )
        dpg.add_file_extension(".wav", parent=wav_dialog)
        wav_import_dialog = dpg.add_file_dialog(
            show=False,
            modal=True,
            width=700,
            height=400,
            callback=cbs.import_wav,
            user_data=state,
        )
        dpg.add_file_extension(".wav", parent=wav_import_dialog)
        project_save_dialog = dpg.add_file_dialog(
            show=False,
            modal=True,
            width=700,
            height=400,
            callback=cbs.save_project_as,
            user_data=state,
            default_filename="project.bms",
        )
        dpg.add_file_extension(".bms", parent=project_save_dialog)
        project_open_dialog = dpg.add_file_dialog(
            show=False,
            modal=True,
            width=700,
            height=400,
            callback=cbs.open_project,
            user_data=state,
        )
        dpg.add_file_extension(".bms", parent=project_open_dialog)
        with dpg.window(
            label="Warning",
            show=False,
            modal=True,
            no_resize=True,
            no_close=True,
            no_scrollbar=True,
            width=520,
            height=150,
            tag=cbs.PROJECT_WARNING_DIALOG,
        ):
            dpg.add_text("", wrap=490, tag=cbs.PROJECT_WARNING_TEXT)
            dpg.add_button(
                label="OK",
                width=-1,
                callback=cbs.close_project_warning,
            )
        with dpg.window(
            label="Project",
            show=False,
            modal=True,
            no_resize=True,
            no_close=True,
            no_scrollbar=True,
            width=520,
            height=135,
            tag=cbs.PROJECT_PROGRESS_DIALOG,
        ):
            dpg.add_text("", wrap=490, tag=cbs.PROJECT_PROGRESS_TEXT)
            dpg.add_progress_bar(width=-1, tag=cbs.PROJECT_PROGRESS_BAR)
        with dpg.window(
            label="IO Settings",
            show=False,
            popup=True,
            no_resize=True,
            no_scrollbar=True,
            width=IO_SETTINGS_WIDTH,
            height=235,
            tag="main_io_settings",
        ) as io_dialog:
            dpg.add_text("Input")
            inputs_combo = dpg.add_combo(
                items=[cbs.DEFAULT_INPUT],
                default_value=state.settings.audio.input_device,
                width=IO_SETTINGS_CONTENT_WIDTH,
                callback=cbs.set_input,
                user_data=state,
                tag="main_input_device",
            )
            dpg.add_text("Output")
            outputs_combo = dpg.add_combo(
                items=[cbs.DEFAULT_OUTPUT],
                default_value=state.settings.audio.output_device,
                width=IO_SETTINGS_CONTENT_WIDTH,
                callback=cbs.set_output,
                user_data=state,
                tag="main_output_device",
            )
            dpg.add_text("Block size, samples")
            block_size_input = dpg.add_input_int(
                default_value=state.settings.audio.block_size,
                min_value=1,
                min_clamped=True,
                step=0,
                width=IO_SETTINGS_CONTENT_WIDTH,
                callback=cbs.set_block_size,
                user_data=state,
                tag="main_block_size",
            )
            close_io_button = dpg.add_button(
                label="Close",
                width=IO_SETTINGS_CONTENT_WIDTH,
                callback=cbs.close_io_settings,
                user_data=io_dialog,
                tag="main_io_close",
            )

        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(
                    label="Open",
                    callback=cbs.show_open_project_dialog,
                    user_data=project_open_dialog,
                )
                dpg.add_menu_item(
                    label="Save",
                    callback=cbs.save_project,
                    user_data=(state, project_save_dialog),
                )
                dpg.add_menu_item(
                    label="Save As",
                    callback=cbs.show_save_project_dialog,
                    user_data=(state, project_save_dialog),
                )
                with dpg.menu(label="Import"):
                    dpg.add_menu_item(
                        label="WAV",
                        callback=cbs.show_import_dialog,
                        user_data=wav_import_dialog,
                    )
                with dpg.menu(label="Export"):
                    dpg.add_menu_item(
                        label="Plot",
                        callback=cbs.show_export_dialog,
                        user_data=fft_dialog,
                    )
                    dpg.add_menu_item(
                        label="WAV",
                        callback=cbs.show_wav_export_dialog,
                        user_data=(state, wav_dialog),
                    )
            with dpg.menu(label="Settings"):
                io_menu_item = dpg.add_menu_item(
                    label="IO",
                    callback=cbs.show_io_settings,
                    user_data=io_dialog,
                    tag="main_io_settings_menu",
                )

        with dpg.group(horizontal=True):
            with dpg.group(width=-CONTROL_PANEL_WIDTH-7):
                with dpg.plot(height=-200, label="FFT"):
                    dpg.add_plot_legend()
                    xaxis = dpg.add_plot_axis(
                        dpg.mvXAxis, label="Hz", scale=dpg.mvPlotScale_Log10
                    )
                    state.fft_xaxis = xaxis
                    dpg.set_axis_limits(xaxis, 20, 20e3)
                    with dpg.plot_axis(dpg.mvYAxis, label="PSD [dB]") as yaxis:
                        state.fft_yaxis = yaxis
                        for i in range(REC_NUMBER):
                            line = dpg.add_line_series(
                                [], [], label=f"record {i + 1}", show=False
                            )
                            state.lines.append(line)

                with dpg.plot(height=-1, label="Time plot"):
                    levels_xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="s")
                    with dpg.plot_axis(dpg.mvYAxis) as levels_yaxis:
                        dpg.set_axis_limits(levels_yaxis, 0, 1)
                        state.levels_l = dpg.add_line_series([], [])
                        state.levels_r = dpg.add_line_series([], [])

            with dpg.group(width=CONTROL_PANEL_WIDTH):
                run_btn = dpg.add_button(
                    label="OFF",
                    width=-1,
                    height=50,
                    callback=cbs.run_btn,
                    user_data=state,
                    tag="run btn",
                )
                dpg.bind_item_theme(run_btn, green_theme)

                with dpg.child_window(height=-142):
                    with dpg.collapsing_header(label="Generator", default_open=True):
                        dpg.add_combo(
                            cbs.GenMode.list(),
                            width=-1,
                            default_value=state.audio_io.gen_mode.value,
                            callback=cbs.set_genmode,
                            user_data=state,
                        )
                        dpg.add_text("Band, Hz")
                        band_input = dpg.add_input_intx(
                            default_value=[
                                int(state.audio_io.band[0]),
                                int(state.audio_io.band[1]),
                            ],
                            size=2,
                            width=-1,
                            callback=cbs.set_band,
                            user_data=state,
                        )
                        state.band_input = band_input
                        dpg.add_text("Length, s")
                        rec_len = dpg.add_input_float(
                            width=-1,
                            step=0,
                            step_fast=0,
                            default_value=state.audio_io.length,
                            format="%.1f",
                            tag="length input",
                            callback=cbs.set_length,
                            user_data=state,
                        )

                    with dpg.collapsing_header(label="Analyzer"):
                        with dpg.group(width=-1):
                            dpg.add_text("Mode")
                            analyzer_mode_combo = dpg.add_combo(
                                cbs.AnalyzerMode.list(),
                                default_value=state.analyzer.analyzer_mode.value,
                                callback=cbs.set_analyzer_mode,
                                user_data=state,
                            )
                            state.analyzer_mode_combo = analyzer_mode_combo
                            with dpg.group():
                                dpg.add_text("Welch bucket size, samples")
                                welch_n_input = dpg.add_input_int(
                                    default_value=state.analyzer.welch_n,
                                    callback=cbs.set_bucket_size,
                                    user_data=state,
                                    step=0,
                                )
                                state.welch_n_input = welch_n_input
                            dpg.add_text("Reference")
                            ref_combo = dpg.add_combo(
                                cbs.RefMode.list(),
                                default_value=state.audio_io.ref.value,
                                callback=cbs.set_analyzer_ref,
                                user_data=state,
                            )
                            state.ref_combo = ref_combo
                            dpg.add_text("Weighting")
                            weighting_combo = dpg.add_combo(
                                cbs.WeightingMode.list(),
                                default_value=state.analyzer.weighting.value,
                                callback=cbs.set_analyzer_weighting,
                                user_data=state,
                            )
                            state.weighting_combo = weighting_combo

                    with dpg.collapsing_header(label="Filtering"):
                        with dpg.group(width=-1):
                            dpg.add_text("Window width, octaves")
                            window_width_input = dpg.add_input_float(
                                default_value=state.analyzer.window_width,
                                callback=cbs.set_window_width,
                                user_data=state,
                                step=0.1,
                            )
                            dpg.add_text("Points number")
                            freq_length_input = dpg.add_input_int(
                                default_value=state.analyzer.freq_length,
                                callback=cbs.set_freq_length,
                                user_data=state,
                                step=0,
                            )
                            dpg.add_text("Window function")
                            window_func_combo = dpg.add_combo(
                                cbs.Windows.list(),
                                default_value=state.analyzer.window_func.value,
                                callback=cbs.set_filter_window_func,
                                user_data=state,
                            )
                            state.window_width_input = window_width_input
                            state.freq_length_input = freq_length_input
                            state.window_func_combo = window_func_combo

                dpg.add_separator(label="Records")
                with dpg.table(header_row=False, policy=dpg.mvTable_SizingFixedFit):
                    for _ in range(3):
                        dpg.add_table_column()
                    for i in range(REC_NUMBER):
                        with dpg.table_row():
                            used = dpg.add_checkbox(
                                user_data=state,
                                callback=cbs.record_used_click,
                            )
                            visible = dpg.add_checkbox(
                                user_data=state,
                                callback=cbs.record_visible_clicked,
                            )
                            name = dpg.add_input_text(
                                width=140,
                                user_data=state,
                                callback=cbs.record_set_name,
                                default_value=f"record {i + 1}",
                            )
                            state.lines_table_rows.append([used, visible, name])
                    dpg.set_value(state.lines_table_rows[0][0], True)
                    state.current_line = 0

    return UiRefs(
        run_btn=run_btn,
        band_input=band_input,
        rec_len=rec_len,
        io_menu_item=io_menu_item,
        io_dialog=io_dialog,
        inputs_combo=inputs_combo,
        outputs_combo=outputs_combo,
        block_size_input=block_size_input,
        close_io_button=close_io_button,
        ref_combo=ref_combo,
        welch_n_input=welch_n_input,
        window_width_input=window_width_input,
        freq_length_input=freq_length_input,
        levels_xaxis=levels_xaxis,
    )


