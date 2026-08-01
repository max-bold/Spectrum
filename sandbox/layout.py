from tkinter import BOTTOM, RIGHT

import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title='Custom Title', width=1025, height=768)
BOTTOM_PANE_HEIGHT = 200
RIGHT_PANE_WIDTH = 240

with dpg.window(label="Example Window") as pm: #pyright: ignore[reportGeneralTypeIssues]
    with dpg.menu_bar(): #pyright: ignore[reportGeneralTypeIssues]
        with dpg.menu(label="File"): #pyright: ignore[reportGeneralTypeIssues]
            pass
    with dpg.group(): #pyright: ignore[reportGeneralTypeIssues]
        with dpg.group(horizontal=True,height=-BOTTOM_PANE_HEIGHT-27): #pyright: ignore[reportGeneralTypeIssues]
            with dpg.child_window( #pyright: ignore[reportGeneralTypeIssues]
                width=-RIGHT_PANE_WIDTH-8,
            ):
                dpg.add_text("TOP LEFT")
            with dpg.child_window( #pyright: ignore[reportGeneralTypeIssues]
                width=RIGHT_PANE_WIDTH,
            ):
                dpg.add_text("TOP RIGHT")
        with dpg.group(horizontal=True, height=BOTTOM_PANE_HEIGHT): #pyright: ignore[reportGeneralTypeIssues]
            with dpg.child_window( #pyright: ignore[reportGeneralTypeIssues]
                width=-RIGHT_PANE_WIDTH-8,
            ):
                dpg.add_text("BOTTOM LEFT")
            with dpg.child_window( #pyright: ignore[reportGeneralTypeIssues]
                width=RIGHT_PANE_WIDTH,
            ):
                dpg.add_text("BOTTOM RIGHT")
        dpg.add_text("BOTTOM")

dpg.set_primary_window(pm, True)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()