from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import dearpygui.dearpygui as dpg

from spectrum_app.core.settings import AppSettings
from spectrum_app.modules.rta.types import (
    PERIODIC_IFFT_GENERATOR,
    RTA_NOISE_GENERATORS,
    RTANoiseGeneratorType,
)

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication

RTAMode = Literal["mono", "stereo"]
RTAPlotMode = Literal["auto", "bars", "line"]
RTAWindowFunction = Literal["hann", "blackman", "boxcar"]
SettingsChanged = Callable[[str], None]


class RTASettings:
    MODULE_ID = "rta"
    DEFAULT_MODE: RTAMode = "mono"
    DEFAULT_PLOT_TYPE: RTAPlotMode = "auto"
    DEFAULT_WINDOW_FUNCTION: RTAWindowFunction = "boxcar"
    DEFAULT_GENERATOR: RTANoiseGeneratorType = PERIODIC_IFFT_GENERATOR
    DEFAULT_PRE_SILENCE = 0.1
    DEFAULT_FADE_IN = 0.5
    DEFAULT_FADE_OUT = 0.5

    def __init__(
        self,
        app_settings: AppSettings,
        on_change: SettingsChanged | None = None,
    ) -> None:
        self._app_settings = app_settings
        self._on_change = on_change

    @property
    def mode(self) -> RTAMode:
        return cast(
            RTAMode,
            self._choice("mode", self.DEFAULT_MODE, ("mono", "stereo")),
        )

    @mode.setter
    def mode(self, value: RTAMode) -> None:
        self._set_choice("mode", value, ("mono", "stereo"))

    @property
    def plot_type(self) -> RTAPlotMode:
        return cast(
            RTAPlotMode,
            self._choice(
                "plot_type",
                self.DEFAULT_PLOT_TYPE,
                ("auto", "bars", "line"),
            ),
        )

    @plot_type.setter
    def plot_type(self, value: RTAPlotMode) -> None:
        self._set_choice("plot_type", value, ("auto", "bars", "line"))

    @property
    def window_function(self) -> RTAWindowFunction:
        return cast(
            RTAWindowFunction,
            self._choice(
                "window_function",
                self.DEFAULT_WINDOW_FUNCTION,
                ("hann", "blackman", "boxcar"),
            ),
        )

    @window_function.setter
    def window_function(self, value: RTAWindowFunction) -> None:
        self._set_choice(
            "window_function",
            value,
            ("hann", "blackman", "boxcar"),
        )

    @property
    def generator(self) -> RTANoiseGeneratorType:
        return cast(
            RTANoiseGeneratorType,
            self._choice(
                "generator",
                self.DEFAULT_GENERATOR,
                RTA_NOISE_GENERATORS,
            ),
        )

    @generator.setter
    def generator(self, value: RTANoiseGeneratorType) -> None:
        self._set_choice("generator", value, RTA_NOISE_GENERATORS)

    @property
    def pre_silence(self) -> float:
        return self._float("pre_silence", self.DEFAULT_PRE_SILENCE)

    @pre_silence.setter
    def pre_silence(self, value: float) -> None:
        self._set_float("pre_silence", value)

    @property
    def fade_in(self) -> float:
        return self._float("fade_in", self.DEFAULT_FADE_IN)

    @fade_in.setter
    def fade_in(self, value: float) -> None:
        self._set_float("fade_in", value)

    @property
    def fade_out(self) -> float:
        return self._float("fade_out", self.DEFAULT_FADE_OUT)

    @fade_out.setter
    def fade_out(self, value: float) -> None:
        self._set_float("fade_out", value)

    def _choice(self, key: str, default: str, choices: tuple[str, ...]) -> str:
        value = self._app_settings.module_setting(self.MODULE_ID, key, default)
        return value if isinstance(value, str) and value in choices else default

    def _set_choice(self, key: str, value: str, choices: tuple[str, ...]) -> None:
        if value not in choices:
            raise ValueError(f"Unknown RTA {key}: {value}")
        self._set(key, value)

    def _float(self, key: str, default: float) -> float:
        value = self._app_settings.module_setting(self.MODULE_ID, key, default)
        try:
            return min(10.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return default

    def _set_float(self, key: str, value: Any) -> None:
        self._set(key, min(10.0, max(0.0, float(value))))

    def _set(self, key: str, value: str | float) -> None:
        if self._app_settings.module_setting(self.MODULE_ID, key) == value:
            return
        self._app_settings.set_module_setting(self.MODULE_ID, key, value)
        if self._on_change is not None:
            self._on_change(key)


class RTASettingsWindow:
    WIDTH = 480
    HEIGHT = 430
    TAG = "module::rta::settings"
    MENU_ITEM = "module::rta::settings_menu_item"

    def __init__(self, app: "SpectrumApplication", settings: RTASettings) -> None:
        self.app = app
        self.settings = settings
        self.mode = f"{self.TAG}::mode"
        self.plot_type = f"{self.TAG}::plot_type"
        self.window_function = f"{self.TAG}::window_function"
        self.generator = f"{self.TAG}::generator"
        self.pre_silence = f"{self.TAG}::pre_silence"
        self.fade_in = f"{self.TAG}::fade_in"
        self.fade_out = f"{self.TAG}::fade_out"

    def build(self) -> None:
        dpg.add_menu_item(
            label="RTA",
            tag=self.MENU_ITEM,
            parent=self.app.main_window.settings_menu,
            callback=self.show,
        )
        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            label="RTA settings",
            tag=self.TAG,
            width=self.WIDTH,
            height=self.HEIGHT,
            show=False,
            modal=True,
            no_resize=True,
            no_collapse=True,
            on_close=self.hide,
        ):
            self._add_combo("Mode", self.mode, ["mono", "stereo"], self._set_mode)
            self._add_combo(
                "Plot type",
                self.plot_type,
                self._plot_types(),
                self._set_plot_type,
            )
            self._add_combo(
                "Window function",
                self.window_function,
                ["hann", "blackman", "boxcar"],
                self._set_window_function,
            )
            self._add_combo(
                "Noise generator",
                self.generator,
                list(RTA_NOISE_GENERATORS),
                self._set_generator,
            )
            dpg.add_separator()
            self._add_float(
                "Pre-silence, s",
                self.pre_silence,
                self.settings.pre_silence,
                self._set_pre_silence,
            )
            self._add_float(
                "Fade in, s",
                self.fade_in,
                self.settings.fade_in,
                self._set_fade_in,
            )
            self._add_float(
                "Fade out, s",
                self.fade_out,
                self.settings.fade_out,
                self._set_fade_out,
            )

    def destroy(self) -> None:
        for item in (self.TAG, self.MENU_ITEM):
            if dpg.does_item_exist(item):
                dpg.delete_item(item)

    def show(self, sender=None, app_data=None, user_data=None) -> None:
        self._sync_controls()
        main_position = dpg.get_item_pos(self.app.main_window.tag)
        main_size = dpg.get_item_rect_size(self.app.main_window.tag)
        if main_size == [100, 100]:
            main_size = [
                dpg.get_viewport_client_width(),
                dpg.get_viewport_client_height(),
            ]
        dpg.set_item_pos(
            self.TAG,
            [
                main_position[0] + (main_size[0] - self.WIDTH) / 2,
                main_position[1] + (main_size[1] - self.HEIGHT) / 2,
            ],
        )
        dpg.configure_item(self.TAG, show=True)

    def hide(self, sender=None, app_data=None, user_data=None) -> None:
        dpg.configure_item(self.TAG, show=False)

    def _add_combo(self, label: str, tag: str, items: list[str], callback) -> None:
        dpg.add_text(label)
        dpg.add_combo(items, tag=tag, width=-1, callback=callback)

    def _add_float(self, label: str, tag: str, value: float, callback) -> None:
        dpg.add_text(label)
        dpg.add_input_float(
            tag=tag,
            default_value=value,
            min_value=0.0,
            min_clamped=True,
            step=0,
            width=-1,
            callback=callback,
        )

    def _plot_types(self) -> list[str]:
        if self.settings.mode == "stereo":
            return ["auto", "line"]
        return ["auto", "bars", "line"]

    def _sync_controls(self) -> None:
        dpg.set_value(self.mode, self.settings.mode)
        dpg.configure_item(self.plot_type, items=self._plot_types())
        plot_type = self.settings.plot_type
        if self.settings.mode == "stereo" and plot_type == "bars":
            plot_type = "line"
        dpg.set_value(self.plot_type, plot_type)
        dpg.set_value(self.window_function, self.settings.window_function)
        dpg.set_value(self.generator, self.settings.generator)
        dpg.set_value(self.pre_silence, self.settings.pre_silence)
        dpg.set_value(self.fade_in, self.settings.fade_in)
        dpg.set_value(self.fade_out, self.settings.fade_out)

    def _change_allowed(self) -> bool:
        if not self.app.app_state.measuring:
            return True
        self.app.main_window.set_status_text(
            "Stop the active measurement before changing RTA settings"
        )
        self._sync_controls()
        return False

    def _set_mode(self, sender, value: str, user_data=None) -> None:
        if not self._change_allowed():
            return
        self.settings.mode = cast(RTAMode, value)
        if value == "stereo" and self.settings.plot_type == "bars":
            self.settings.plot_type = "line"
        self._sync_controls()

    def _set_plot_type(self, sender, value: str, user_data=None) -> None:
        if self._change_allowed():
            self.settings.plot_type = cast(RTAPlotMode, value)

    def _set_window_function(self, sender, value: str, user_data=None) -> None:
        if self._change_allowed():
            self.settings.window_function = cast(RTAWindowFunction, value)

    def _set_generator(self, sender, value: str, user_data=None) -> None:
        if self._change_allowed():
            self.settings.generator = cast(RTANoiseGeneratorType, value)
            self._sync_controls()

    def _set_pre_silence(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.pre_silence = value
            dpg.set_value(sender, self.settings.pre_silence)

    def _set_fade_in(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.fade_in = value
            dpg.set_value(sender, self.settings.fade_in)

    def _set_fade_out(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.fade_out = value
            dpg.set_value(sender, self.settings.fade_out)
