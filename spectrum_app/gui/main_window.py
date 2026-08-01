from typing import TYPE_CHECKING

import dearpygui.dearpygui as dpg

from spectrum_app.gui.app_state import AppStatePanel
from spectrum_app.gui.measurement import MeasurementPanel

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class MainWindow:
    TITLE = "BM Spectrum"
    TAG = "app::main_window"
    WIDTH = 1024
    HEIGHT = 768
    BOTTOM_PANE_HEIGHT = 200
    SIDE_PANE_WIDTH = 240

    def __init__(self, app: "SpectrumApplication") -> None:
        self.app = app
        self.file_menu = "app::file_menu"
        self.tools_menu = "app::tools_menu"
        self.settings_menu = "app::settings_menu"
        self.plot_host = "app::plot_host"
        self.bottom_host = "app::bottom_host"
        self.control_panel_host = "app::control_panel_host"
        self.measurement_panel = MeasurementPanel(app)
        self.module_gui_host = self.measurement_panel.module_gui_host
        self.appstate_host = "app::appstate_host"
        self.app_state_panel = AppStatePanel(app)
        self.status = "app::status_bar"

    @property
    def tag(self) -> str:
        return self.TAG

    def build(self) -> None:
        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            tag=self.TAG,
        ):
            with dpg.menu_bar():  # pyright: ignore[reportGeneralTypeIssues]
                dpg.add_menu(label="File", tag=self.file_menu)
                dpg.add_menu(label="Tools", tag=self.tools_menu)
                dpg.add_menu(label="Settings", tag=self.settings_menu)

            with dpg.group(): #pyright: ignore[reportGeneralTypeIssues]
                    with dpg.group(horizontal=True,height=-self.BOTTOM_PANE_HEIGHT-27): #pyright: ignore[reportGeneralTypeIssues]
                        with dpg.plot( #pyright: ignore[reportGeneralTypeIssues]
                            width=-self.SIDE_PANE_WIDTH-8,
                            height=-1,
                            tag=self.plot_host,
                        ):
                            pass
                        with dpg.child_window( #pyright: ignore[reportGeneralTypeIssues]
                            width=self.SIDE_PANE_WIDTH,
                            tag=self.control_panel_host,
                        ):
                            self.measurement_panel.build()
                    with dpg.group(horizontal=True, height=self.BOTTOM_PANE_HEIGHT): #pyright: ignore[reportGeneralTypeIssues]
                        with dpg.child_window( #pyright: ignore[reportGeneralTypeIssues]
                            width=-self.SIDE_PANE_WIDTH-8,
                            tag=self.bottom_host,
                        ):
                            pass
                        with dpg.child_window( #pyright: ignore[reportGeneralTypeIssues]
                            width=self.SIDE_PANE_WIDTH,
                            tag=self.appstate_host,
                        ):
                            self.app_state_panel.build()
                    dpg.add_text("",tag=self.status)

    def update(self) -> None:
        self.measurement_panel.update()

    def set_status_text(self, text: str) -> None:
        dpg.set_value(self.status, text)
