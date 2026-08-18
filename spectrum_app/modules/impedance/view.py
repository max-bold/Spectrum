from typing import TYPE_CHECKING, Any

import dearpygui.dearpygui as dpg

from audioanalysis import SmoothingWindow, SpiceTableValues
from spectrum_app.gui.controls import LevelMeter, add_level_meter

if TYPE_CHECKING:
    from spectrum_app.modules.impedance.module import ImpedanceModule


class ImpedanceView:
    ROOT = "module::impedance::controls"
    BOTTOM = "module::impedance::bottom"
    LEVEL_METER = "module::impedance::level_meter"
    WINDOW_WIDTH = "module::impedance::window_width"
    WINDOW_WIDTH_HANDLERS = "module::impedance::window_width::handlers"
    POINTS = "module::impedance::points"
    POINTS_HANDLERS = "module::impedance::points::handlers"
    CALIBRATION_DIALOG = "module::impedance::calibration"
    CALIBRATION_TEXT = "module::impedance::calibration::text"
    CALIBRATION_RESISTORS = "module::impedance::calibration::resistors"
    REFERENCE_RESISTOR = "module::impedance::calibration::reference_resistor"
    CALIBRATION_RESISTOR = "module::impedance::calibration::calibration_resistor"
    CALIBRATION_CONTINUE = "module::impedance::calibration::continue"
    CALIBRATE_ITEM = "module::impedance::calibrate_tool"
    TEST_TONE_ITEM = "module::impedance::test_tone_tool"
    TOOLS_ITEM = "module::impedance::spice_fit_tool"
    SPICE_WINDOW = "module::impedance::spice_fit"
    SPICE_TEXT = "module::impedance::spice_fit::text"
    CALIBRATION_WIDTH = 560
    CALIBRATION_HEIGHT = 340

    def __init__(self, module: "ImpedanceModule") -> None:
        self.module = module
        self.status = "module::impedance::status"
        self.level_meter: LevelMeter | None = None

    def build(
        self,
        controls_parent: int | str,
        bottom_parent: int | str,
        state: dict[str, Any],
    ) -> None:
        with dpg.group(  # pyright: ignore[reportGeneralTypeIssues]
            parent=controls_parent,
            tag=self.ROOT,
        ):
            with dpg.collapsing_header(  # pyright: ignore[reportGeneralTypeIssues]
                label="Generator",
                default_open=True,
            ):
                dpg.add_text("Band, Hz")
                dpg.add_input_intx(
                    size=2,
                    default_value=list(state["band"]),
                    width=-1,
                    callback=self._set_band,
                )
                dpg.add_text("Duration, s")
                dpg.add_input_float(
                    default_value=state["duration"],
                    min_value=0.1,
                    min_clamped=True,
                    step=0,
                    width=-1,
                    callback=self._set_duration,
                )
            with dpg.collapsing_header(  # pyright: ignore[reportGeneralTypeIssues]
                label="Filtering",
                default_open=True,
            ):
                dpg.add_text("Window")
                dpg.add_combo(
                    SmoothingWindow.list(),
                    default_value=state["window"],
                    width=-1,
                    callback=self._set_window,
                )
                dpg.add_text("Window width, octaves")
                dpg.add_input_float(
                    tag=self.WINDOW_WIDTH,
                    default_value=state["window_width"],
                    step=0.1,
                    width=-1,
                    callback=self._set_window_width,
                    on_enter=True,
                )
                dpg.add_text("Points")
                dpg.add_input_int(
                    tag=self.POINTS,
                    default_value=state["points"],
                    step=0,
                    width=-1,
                    callback=self._set_points,
                    on_enter=True,
                )

        with dpg.item_handler_registry(  # pyright: ignore[reportGeneralTypeIssues]
            tag=self.WINDOW_WIDTH_HANDLERS,
        ):
            dpg.add_item_deactivated_after_edit_handler(
                callback=self._commit_window_width,
            )
        dpg.bind_item_handler_registry(
            self.WINDOW_WIDTH,
            self.WINDOW_WIDTH_HANDLERS,
        )
        with dpg.item_handler_registry(  # pyright: ignore[reportGeneralTypeIssues]
            tag=self.POINTS_HANDLERS,
        ):
            dpg.add_item_deactivated_after_edit_handler(
                callback=self._commit_points,
            )
        dpg.bind_item_handler_registry(self.POINTS, self.POINTS_HANDLERS)

        with dpg.group(  # pyright: ignore[reportGeneralTypeIssues]
            parent=bottom_parent,
            tag=self.BOTTOM,
            horizontal=True,
        ):
            self.level_meter = add_level_meter(
                self.BOTTOM,
                self.LEVEL_METER,
                bottom_parent,
                labels=("A", "B"),
                height_offset=-16,
            )
            dpg.add_text("Calibration required", tag=self.status, wrap=-1)

        dpg.add_menu_item(
            label="Calibrate",
            tag=self.CALIBRATE_ITEM,
            parent=self.module.app.main_window.tools_menu,
            callback=self.module.request_calibration,
        )
        dpg.add_menu_item(
            label="Test tone",
            tag=self.TEST_TONE_ITEM,
            parent=self.module.app.main_window.tools_menu,
            callback=self.module.toggle_test_signal,
        )
        dpg.add_menu_item(
            label="SPICE Fit",
            tag=self.TOOLS_ITEM,
            parent=self.module.app.main_window.tools_menu,
            callback=self.module.request_spice_fit,
        )
        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            label="SPICE Fit — needs testing",
            tag=self.SPICE_WINDOW,
            width=520,
            height=420,
            show=False,
            modal=True,
            on_close=self.hide_spice,
        ):
            dpg.add_text("No model calculated", tag=self.SPICE_TEXT)

        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            label="Impedance calibration",
            tag=self.CALIBRATION_DIALOG,
            width=self.CALIBRATION_WIDTH,
            height=self.CALIBRATION_HEIGHT,
            show=False,
            modal=True,
            no_resize=True,
            on_close=self.module.cancel_calibration,
        ):
            dpg.add_text("", tag=self.CALIBRATION_TEXT, wrap=520)
            with dpg.group(  # pyright: ignore[reportGeneralTypeIssues]
                tag=self.CALIBRATION_RESISTORS,
            ):
                dpg.add_text("Reference resistor, Ohm")
                dpg.add_input_float(
                    tag=self.REFERENCE_RESISTOR,
                    default_value=state["reference_resistor"],
                    min_value=0.001,
                    min_clamped=True,
                    step=0,
                    width=-1,
                    callback=self._set_reference_resistor,
                )
                dpg.add_text("Calibration resistor, Ohm")
                dpg.add_input_float(
                    tag=self.CALIBRATION_RESISTOR,
                    default_value=state["calibration_resistor"],
                    min_value=0.001,
                    min_clamped=True,
                    step=0,
                    width=-1,
                    callback=self._set_calibration_resistor,
                )
            dpg.add_button(
                label="Continue",
                tag=self.CALIBRATION_CONTINUE,
                width=-1,
                callback=self.module.continue_calibration,
            )
            dpg.add_button(
                label="Cancel",
                width=-1,
                callback=self.module.cancel_calibration,
            )

    def destroy(self) -> None:
        for item in (
            self.ROOT,
            self.BOTTOM,
            self.CALIBRATION_DIALOG,
            self.SPICE_WINDOW,
            self.CALIBRATE_ITEM,
            self.TEST_TONE_ITEM,
            self.TOOLS_ITEM,
            self.WINDOW_WIDTH_HANDLERS,
            self.POINTS_HANDLERS,
        ):
            if dpg.does_item_exist(item):
                dpg.delete_item(item)
        self.level_meter = None

    def update(self) -> None:
        if self.level_meter is not None:
            self.level_meter.resize()

    def set_enabled(self, enabled: bool) -> None:
        if dpg.does_item_exist(self.ROOT):
            dpg.configure_item(self.ROOT, enabled=enabled)

    def update_status(
        self,
        status: str,
        levels: tuple[float, float],
    ) -> None:
        dpg.set_value(self.status, status)
        if self.level_meter is not None:
            self.level_meter.set_levels(*levels)

    def show_calibration_stage(self, stage: int) -> None:
        if stage == 1:
            text = (
                "Stage 1 of 2: channel calibration\n\n"
                "Connect inputs A and B to the same audio_out point relative to "
                "ground. Both inputs must receive the same electrical signal."
            )
            label = "Start stage 1"
            state = self.module.measurement.module_state
            dpg.set_value(self.REFERENCE_RESISTOR, state["reference_resistor"])
            dpg.set_value(
                self.CALIBRATION_RESISTOR,
                state["calibration_resistor"],
            )
        else:
            text = (
                "Stage 2 of 2: resistor calibration\n\n"
                "Connect the entered reference and calibration resistors to "
                "the measurement input, then continue."
            )
            label = "Start stage 2"
        dpg.configure_item(self.CALIBRATION_RESISTORS, show=stage == 1)
        dpg.set_value(self.CALIBRATION_TEXT, text)
        dpg.set_item_label(self.CALIBRATION_CONTINUE, label)
        main_position = dpg.get_item_pos(self.module.app.main_window.tag)
        main_size = dpg.get_item_rect_size(self.module.app.main_window.tag)
        dpg.set_item_pos(
            self.CALIBRATION_DIALOG,
            [
                main_position[0]
                + (main_size[0] - self.CALIBRATION_WIDTH) / 2,
                main_position[1]
                + (main_size[1] - self.CALIBRATION_HEIGHT) / 2,
            ],
        )
        dpg.configure_item(self.CALIBRATION_DIALOG, show=True)

    def hide_calibration(self) -> None:
        dpg.configure_item(self.CALIBRATION_DIALOG, show=False)

    def show_spice(self, status: str, values: SpiceTableValues | None) -> None:
        if values is None:
            text = status
        else:
            lines = [f"R1 = {values.r1} Ohm", f"L1 = {values.l1} mH"]
            for index, (inductance, capacitance, resistance) in enumerate(
                values.sections,
                start=1,
            ):
                lines.append(
                    f"Section {index}: L={inductance} mH, "
                    f"C={capacitance} uF, R={resistance} Ohm"
                )
            text = "\n".join(lines)
        dpg.set_value(self.SPICE_TEXT, text)
        dpg.configure_item(self.SPICE_WINDOW, show=True)

    def hide_spice(self, sender=None, app_data=None, user_data=None) -> None:
        dpg.configure_item(self.SPICE_WINDOW, show=False)

    def _set_band(self, sender: int | str, value: list[int], user_data=None) -> None:
        band = self.module.set_setting("band", (value[0], value[1]))
        dpg.set_value(sender, [*band, 0, 0])

    def _set_duration(self, sender: int | str, value: float, user_data=None) -> None:
        value = self.module.set_setting("duration", value)
        dpg.set_value(sender, value)

    def _set_reference_resistor(
        self, sender: int | str, value: float, user_data=None
    ) -> None:
        value = self.module.set_setting("reference_resistor", value)
        dpg.set_value(sender, value)

    def _set_calibration_resistor(
        self, sender: int | str, value: float, user_data=None
    ) -> None:
        value = self.module.set_setting("calibration_resistor", value)
        dpg.set_value(sender, value)

    def _set_window(self, sender: int | str, value: str, user_data=None) -> None:
        self.module.set_setting("window", value)

    def _set_window_width(
        self, sender: int | str, value: float, user_data=None
    ) -> None:
        value = self.module.set_setting("window_width", value)
        dpg.set_value(sender, value)

    def _set_points(self, sender: int | str, value: int, user_data=None) -> None:
        value = self.module.set_setting("points", value)
        dpg.set_value(sender, value)

    def _commit_window_width(self, sender=None, app_data=None, user_data=None) -> None:
        self._set_window_width(
            self.WINDOW_WIDTH,
            dpg.get_value(self.WINDOW_WIDTH),
        )

    def _commit_points(self, sender=None, app_data=None, user_data=None) -> None:
        self._set_points(self.POINTS, dpg.get_value(self.POINTS))
