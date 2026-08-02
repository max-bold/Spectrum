from typing import TYPE_CHECKING

import dearpygui.dearpygui as dpg

from spectrum_app.gui.app_state import AppStatePanel
from spectrum_app.gui.measurement import MeasurementPanel
from spectrum_app.gui.measurement_io import MeasurementDialogs
from spectrum_app.gui.plot import Plot
from spectrum_app.gui.project import ProjectDialogs
from spectrum_app.gui.settings import SettingsWindow

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
        self.import_menu = "app::import_menu"
        self.export_menu = "app::export_menu"
        self.tools_menu = "app::tools_menu"
        self.settings_menu = "app::settings_menu"
        self.settings_window = SettingsWindow(app)
        self.project_dialogs = ProjectDialogs(app)
        self.measurement_dialogs = MeasurementDialogs(app)
        self.plot = Plot(app)
        self.plot_host = self.plot.tag
        self.bottom_host = "app::bottom_host"
        self.control_panel_host = "app::control_panel_host"
        self.measurement_panel = MeasurementPanel(app)
        self.module_gui_host = self.measurement_panel.module_gui_host
        self.appstate_host = "app::appstate_host"
        self.app_state_panel = AppStatePanel(app)
        self.status = "app::status_bar"
        self._built = False

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
                dpg.add_menu_item(
                    label="Application",
                    parent=self.settings_menu,
                    callback=self.settings_window.show,
                )

            with dpg.group(): #pyright: ignore[reportGeneralTypeIssues]
                    with dpg.group(horizontal=True,height=-self.BOTTOM_PANE_HEIGHT-27): #pyright: ignore[reportGeneralTypeIssues]
                        self.plot.build(
                            width=-self.SIDE_PANE_WIDTH-8,
                            height=-1,
                        )
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

        self.project_dialogs.build(self.file_menu)
        dpg.add_menu(
            label="Import",
            tag=self.import_menu,
            parent=self.file_menu,
        )
        dpg.add_menu(
            label="Export",
            tag=self.export_menu,
            parent=self.file_menu,
        )
        self.plot.build_export(self.export_menu)
        self.measurement_dialogs.build(self.import_menu, self.export_menu)
        self.settings_window.build()
        self._built = True

    def update(self) -> None:
        self.measurement_dialogs.update()
        self.plot.update()
        self.measurement_panel.update()
        self.settings_window.update()
        audio_error = self.app._audio_service.consume_error()
        if audio_error is not None:
            self.set_status_text(f"Audio error: {audio_error}")

    def set_status_text(self, text: str) -> None:
        dpg.set_value(self.status, text)

    def project_loaded(self) -> None:
        if not self._built:
            return
        self.app_state_panel.rebuild()
        self.measurement_panel.update(force=True)

    def measurement_added(self) -> None:
        if not self._built:
            return
        self.app_state_panel.rebuild()
        self.measurement_panel.update(force=True)
