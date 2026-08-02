import dearpygui.dearpygui as dpg
import numpy as np

dpg.create_context()

# creating data
x= np.linspace(0, 2*np.pi, 100)
y1= np.sin(x)
y2= 0.5 + 0.5 * np.sin(x)

with dpg.window(label="Tutorial") as pm: #pyright: ignore[reportGeneralTypeIssues]
    with dpg.group(width=-1, height=-1): #pyright: ignore[reportGeneralTypeIssues]
    # create plot
        with dpg.plot(label="Line Series"): #pyright: ignore[reportGeneralTypeIssues]
            # optionally create legend
            dpg.add_plot_legend()

            # REQUIRED: create x and y axes
            dpg.add_plot_axis(dpg.mvXAxis, label="x")
            with dpg.plot_axis(dpg.mvYAxis, label="y1", tag="y_axis"): #pyright: ignore[reportGeneralTypeIssues]
                dpg.add_line_series(x, y1, label="sin(x)")
            with dpg.plot_axis(dpg.mvYAxis, label="y2", tag="y_axis2"): #pyright: ignore[reportGeneralTypeIssues]
                dpg.add_line_series(x, y2, label="0.5 + 0.5 * sin(x)")
            with dpg.plot_axis(dpg.mvYAxis, label="y3", tag="y_axis3"): #pyright: ignore[reportGeneralTypeIssues]
                dpg.add_line_series(x, y2, label="0.5 + 0.5 * sin(x)")

dpg.set_primary_window(pm, True)
dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()