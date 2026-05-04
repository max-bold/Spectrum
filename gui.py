from dataclasses import dataclass
from time import monotonic

import dearpygui.dearpygui as dpg

import cbs


REC_NUMBER = 5


@dataclass
class UiRefs:
    run_btn: int | str
    band_input: int | str
    rec_len: int | str
    mon_cb: int | str
    left_level: int | str
    right_level: int | str
    inputs_combo: int | str
    outputs_combo: int | str
    ref_combo: int | str
    welch_n_input: int | str
    window_width_input: int | str
    freq_length_input: int | str
    levels_xaxis: int | str
    last_io_update: float = 0.0


def build_ui(state: cbs.AppState) -> UiRefs:
    from utils.themes import green_theme

    with dpg.window(tag="Primary Window"):
        with dpg.group(horizontal=True):
            with dpg.group(width=-200):
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

            with dpg.group():
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

                    with dpg.collapsing_header(label="Audio I/O"):
                        with dpg.group(width=-1):
                            dpg.add_text("Input")
                            inputs_combo = dpg.add_combo(
                                default_value="default input",
                                callback=cbs.set_input,
                                user_data=state,
                            )
                            with dpg.group(horizontal=True):
                                mon_cb = dpg.add_checkbox(
                                    callback=cbs.set_input_meter,
                                    user_data=state,
                                    tag="meter_cb",
                                )
                                with dpg.group():
                                    left_level = dpg.add_progress_bar(height=7)
                                    right_level = dpg.add_progress_bar(height=7)
                            dpg.add_text("Output")
                            outputs_combo = dpg.add_combo(
                                default_value="default output",
                                callback=cbs.set_output,
                                user_data=state,
                            )

                    with dpg.collapsing_header(label="Analyzer"):
                        with dpg.group(width=-1):
                            dpg.add_text("Mode")
                            dpg.add_combo(
                                cbs.AnalyzerMode.list(),
                                default_value=state.analyzer.analyzer_mode.value,
                                callback=cbs.set_analyzer_mode,
                                user_data=state,
                            )
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
                            dpg.add_text("Weighting")
                            dpg.add_combo(
                                cbs.WeightingMode.list(),
                                default_value=state.analyzer.weighting.value,
                                callback=cbs.set_analyzer_weighting,
                                user_data=state,
                            )

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
                            dpg.add_combo(
                                cbs.Windows.list(),
                                default_value=state.analyzer.window_func.value,
                                callback=cbs.set_filter_window_func,
                                user_data=state,
                            )

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
        mon_cb=mon_cb,
        left_level=left_level,
        right_level=right_level,
        inputs_combo=inputs_combo,
        outputs_combo=outputs_combo,
        ref_combo=ref_combo,
        welch_n_input=welch_n_input,
        window_width_input=window_width_input,
        freq_length_input=freq_length_input,
        levels_xaxis=levels_xaxis,
    )


def sync_io_lists(state: cbs.AppState, refs: UiRefs) -> None:
    now = monotonic()
    if now - refs.last_io_update >= 0.5:
        cbs.upd_io(state, refs.inputs_combo, refs.outputs_combo)
        cbs.sync_reference_options(state, refs.ref_combo)
        cbs.sync_band_limits(state, refs.band_input)
        cbs.sync_welch_limit(state, refs.welch_n_input)
        refs.last_io_update = now


def sync_run_button(state: cbs.AppState, refs: UiRefs) -> None:
    from utils.themes import green_theme, red_theme

    if state.audio_io.running.is_set() or state.pending_audio_start:
        dpg.set_item_label(refs.run_btn, "ON")
        dpg.bind_item_theme(refs.run_btn, red_theme)
    else:
        dpg.set_item_label(refs.run_btn, "OFF")
        dpg.bind_item_theme(refs.run_btn, green_theme)


def sync_ui(state: cbs.AppState, refs: UiRefs) -> None:
    sync_io_lists(state, refs)

    if dpg.get_value(refs.mon_cb):
        cbs.upd_level_monitor(state, (refs.left_level, refs.right_level))

    sync_run_button(state, refs)
    cbs.process_pending_requests(state)
    cbs.reenable_io_udater(state)

    max_t = dpg.get_value(refs.rec_len) + 2
    dpg.set_axis_limits(refs.levels_xaxis, 0, max_t)
    cbs.run_analyzer(state)
    dpg.set_value(refs.mon_cb, state.meter.enable.is_set() or state.pending_meter_start)


def main() -> None:
    dpg.create_context()
    state = cbs.create_app_state()
    refs = build_ui(state)
    cbs.bind_input_commit_handlers(state, refs)
    state.start_services()

    dpg.create_viewport(title="BM Spectrum", width=1024, height=768)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)

    try:
        while dpg.is_dearpygui_running():
            sync_ui(state, refs)
            dpg.render_dearpygui_frame()
    finally:
        state.stop_services()
        dpg.destroy_context()


if __name__ == "__main__":
    main()
