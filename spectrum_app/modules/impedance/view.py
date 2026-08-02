from typing import TYPE_CHECKING, Any

import dearpygui.dearpygui as dpg

from audioanalysis import SmoothingWindow, SpiceTableValues

if TYPE_CHECKING:
    from spectrum_app.modules.impedance.module import ImpedanceModule


class ImpedanceView:
    ROOT = "module::impedance::controls"
    BOTTOM = "module::impedance::bottom"
    CALIBRATION_DIALOG = "module::impedance::calibration"
    CALIBRATION_TEXT = "module::impedance::calibration::text"
    CALIBRATION_CONTINUE = "module::impedance::calibration::continue"
    TEST_TONE_ITEM = "module::impedance::test_tone_tool"
    TOOLS_ITEM = "module::impedance::spice_fit_tool"
    SPICE_WINDOW = "module::impedance::spice_fit"
    SPICE_TEXT = "module::impedance::spice_fit::text"

    def __init__(self, module: "ImpedanceModule") -> None:
        self.module = module
        self.status = "module::impedance::status"
        self.levels = "module::impedance::levels"

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
                label="Measurement",
                default_open=True,
            ):
                dpg.add_text("Reference resistor, Ohm")
                dpg.add_input_float(
                    default_value=state["reference_resistor"],
                    min_value=0.001,
                    min_clamped=True,
                    step=0,
                    width=-1,
                    callback=self._set_reference_resistor,
                )
                dpg.add_text("Calibration resistor, Ohm")
                dpg.add_input_float(
                    default_value=state["calibration_resistor"],
                    min_value=0.001,
                    min_clamped=True,
                    step=0,
                    width=-1,
                    callback=self._set_calibration_resistor,
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
                    default_value=state["window_width"],
                    step=0.1,
                    width=-1,
                    callback=self._set_window_width,
                )
                dpg.add_text("Points")
                dpg.add_input_int(
                    default_value=state["points"],
                    step=0,
                    width=-1,
                    callback=self._set_points,
                )

        with dpg.group(  # pyright: ignore[reportGeneralTypeIssues]
            parent=bottom_parent,
            tag=self.BOTTOM,
        ):
            dpg.add_text("CH1: 0.000   CH2: 0.000", tag=self.levels)
            dpg.add_text("Calibration required", tag=self.status, wrap=-1)

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
            width=560,
            height=260,
            show=False,
            modal=True,
            no_resize=True,
            on_close=self.module.cancel_calibration,
        ):
            dpg.add_text("", tag=self.CALIBRATION_TEXT, wrap=520)
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
            self.TEST_TONE_ITEM,
            self.TOOLS_ITEM,
        ):
            if dpg.does_item_exist(item):
                dpg.delete_item(item)

    def set_enabled(self, enabled: bool) -> None:
        if dpg.does_item_exist(self.ROOT):
            dpg.configure_item(self.ROOT, enabled=enabled)

    def update_status(
        self,
        status: str,
        levels: tuple[float, float],
    ) -> None:
        dpg.set_value(self.status, status)
        dpg.set_value(
            self.levels,
            f"CH1: {levels[0]:.3f}   CH2: {levels[1]:.3f}",
        )
    def show_calibration_stage(self, stage: int) -> None:
        if stage == 1:
            text = (
                "Stage 1 of 2: channel calibration\n\n"
                "Connect CH1 and CH2 to the same audio_out point relative to "
                "ground. Both inputs must receive the same electrical signal."
            )
            label = "Start stage 1"
        else:
            text = (
                "Stage 2 of 2: resistor calibration\n\n"
                "Connect the entered reference and calibration resistors to "
                "the measurement input, then continue."
            )
            label = "Start stage 2"
        dpg.set_value(self.CALIBRATION_TEXT, text)
        dpg.set_item_label(self.CALIBRATION_CONTINUE, label)
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
