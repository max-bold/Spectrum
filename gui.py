import dearpygui.dearpygui as dpg
import cbs

dpg.create_context()

with dpg.window(tag="Primary Window"):
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
            dpg.add_menu_item(label="Audio IO", callback=cbs.audioset_open)
            dpg.add_menu_item(label="Analyzer", callback=lambda: dpg.show_item("AnSet"))
    with dpg.group(horizontal=True):
        with dpg.group(width=-200):
            with dpg.plot(height=-200, label="FFT"):
                xaxis = dpg.add_plot_axis(
                    dpg.mvXAxis, label="Hz", scale=dpg.mvPlotScale_Log10
                )
                dpg.set_axis_limits(xaxis, 20, 20e3)
                with dpg.plot_axis(dpg.mvYAxis, label="PSD [db]") as yaxis:
                    pass
            with dpg.plot(height=-1, label="Time plot"):
                pass
        with dpg.group():
            dpg.add_button(label="RUN", width=-1, height=50, callback=cbs.run_btn, tag="run btn")
            dpg.add_separator(label="generator")
            dpg.add_combo(
                ["linear sweep", "log sweep", "pink noise"],
                width=-1,
                default_value="log sweep",
            )
            # dpg.add_text("band, Hz")
            dpg.add_checkbox(label="band, Hz", default_value=True)
            dpg.add_input_intx(default_value=[20, 20000], size=2, width=-1)
            dpg.add_text("length, s")
            dpg.add_input_float(
                width=-1,
                step=0,
                step_fast=0,
                default_value=30.0,
                format="%.1f",
                tag="length input",
            )
            dpg.add_separator(label="analyzer")
            dpg.add_text("input level monitor")
            with dpg.group(horizontal=True):
                dpg.add_checkbox(tag="ilm checkbox", callback=cbs.ilm_act, user_data=cbs.measurethread)
                dpg.add_progress_bar(width=-1, default_value=0.3, tag="ilm pbar")
            dpg.add_text("smoothing, octaves")
            with dpg.group(horizontal=True):
                dpg.add_text("1:")
                dpg.add_slider_int(min_value=1, max_value=10, width=-1, default_value=5)
            dpg.add_separator(label="records")
            dpg.add_child_window(tag= "rec group",label="records")
            # dpg.add_text("new name:")
            # dpg.add_input_text(width=-1)
            # dpg.add_listbox(["rec1", "rec2", "rec3"], width=-1, num_items=20)


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

dpg.create_viewport(title="BM Spectrum", width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()
