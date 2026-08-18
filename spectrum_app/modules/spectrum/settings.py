from typing import TYPE_CHECKING, Any, Literal, cast

import dearpygui.dearpygui as dpg

from spectrum_app.core.settings import AppSettings

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


GeneratorMode = Literal["log chirp", "pink noise"]


class SpectrumSettings:
    MODULE_ID = "spectrum"
    DEFAULT_GENERATOR_MODE: GeneratorMode = "log chirp"
    DEFAULT_WELCH_SAMPLES = 8192
    DEFAULT_ONLINE_WELCH = True
    DEFAULT_PRE_SILENCE = 0.2
    DEFAULT_POST_SILENCE = 1.0
    DEFAULT_FADE_IN = 0.5
    DEFAULT_FADE_OUT = 0.5

    def __init__(self, app_settings: AppSettings) -> None:
        self._app_settings = app_settings

    @property
    def generator_mode(self) -> GeneratorMode:
        value = self._app_settings.module_setting(
            self.MODULE_ID,
            "generator_mode",
            self.DEFAULT_GENERATOR_MODE,
        )
        if value not in ("log chirp", "pink noise"):
            return self.DEFAULT_GENERATOR_MODE
        return cast(GeneratorMode, value)

    @generator_mode.setter
    def generator_mode(self, value: GeneratorMode) -> None:
        if value not in ("log chirp", "pink noise"):
            raise ValueError(f"Unknown generator mode: {value}")
        self._app_settings.set_module_setting(
            self.MODULE_ID,
            "generator_mode",
            value,
        )

    @property
    def welch_samples(self) -> int:
        value = self._app_settings.module_setting(
            self.MODULE_ID,
            "welch_samples",
            self.DEFAULT_WELCH_SAMPLES,
        )
        try:
            return self._normalize_welch_samples(value)
        except (TypeError, ValueError):
            return self.DEFAULT_WELCH_SAMPLES

    @welch_samples.setter
    def welch_samples(self, value: int) -> None:
        self._app_settings.set_module_setting(
            self.MODULE_ID,
            "welch_samples",
            self._normalize_welch_samples(value),
        )

    @property
    def online_welch(self) -> bool:
        value = self._app_settings.module_setting(
            self.MODULE_ID,
            "online_welch",
            self.DEFAULT_ONLINE_WELCH,
        )
        return value if isinstance(value, bool) else self.DEFAULT_ONLINE_WELCH

    @online_welch.setter
    def online_welch(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("Online Welch setting must be boolean")
        self._app_settings.set_module_setting(
            self.MODULE_ID,
            "online_welch",
            value,
        )

    @property
    def pre_silence(self) -> float:
        return self._time_setting("pre_silence", self.DEFAULT_PRE_SILENCE)

    @pre_silence.setter
    def pre_silence(self, value: float) -> None:
        self._set_time_setting("pre_silence", value)

    @property
    def post_silence(self) -> float:
        return self._time_setting("post_silence", self.DEFAULT_POST_SILENCE)

    @post_silence.setter
    def post_silence(self, value: float) -> None:
        self._set_time_setting("post_silence", value)

    @property
    def fade_in(self) -> float:
        return self._time_setting("fade_in", self.DEFAULT_FADE_IN)

    @fade_in.setter
    def fade_in(self, value: float) -> None:
        self._set_time_setting("fade_in", value)

    @property
    def fade_out(self) -> float:
        return self._time_setting("fade_out", self.DEFAULT_FADE_OUT)

    @fade_out.setter
    def fade_out(self, value: float) -> None:
        self._set_time_setting("fade_out", value)

    def _time_setting(self, key: str, default: float) -> float:
        value = self._app_settings.module_setting(self.MODULE_ID, key, default)
        try:
            return self._normalize_time(value)
        except (TypeError, ValueError):
            return default

    def _set_time_setting(self, key: str, value: float) -> None:
        self._app_settings.set_module_setting(
            self.MODULE_ID,
            key,
            self._normalize_time(value),
        )

    @staticmethod
    def _normalize_time(value: Any) -> float:
        return min(10.0, max(0.0, float(value)))

    @staticmethod
    def _normalize_welch_samples(value: Any) -> int:
        samples = max(16, int(value))
        lower = 1 << (samples.bit_length() - 1)
        upper = lower << 1
        return lower if samples - lower <= upper - samples else upper


class SpectrumSettingsWindow:
    WIDTH = 480
    HEIGHT = 390
    TAG = "module::spectrum::settings"
    MENU_ITEM = "module::spectrum::settings_menu_item"

    def __init__(
        self,
        app: "SpectrumApplication",
        settings: SpectrumSettings,
    ) -> None:
        self.app = app
        self.settings = settings
        self.generator_mode = "module::spectrum::settings::generator_mode"
        self.welch_samples = "module::spectrum::settings::welch_samples"
        self.online_welch = "module::spectrum::settings::online_welch"
        self.pre_silence = "module::spectrum::settings::pre_silence"
        self.post_silence = "module::spectrum::settings::post_silence"
        self.fade_in = "module::spectrum::settings::fade_in"
        self.fade_out = "module::spectrum::settings::fade_out"

    def build(self) -> None:
        dpg.add_menu_item(
            label="Spectrum",
            tag=self.MENU_ITEM,
            parent=self.app.main_window.settings_menu,
            callback=self.show,
        )
        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            label="Spectrum settings",
            tag=self.TAG,
            width=self.WIDTH,
            height=self.HEIGHT,
            show=False,
            modal=True,
            no_resize=True,
            no_collapse=True,
            on_close=self.hide,
        ):
            dpg.add_text("Generator")
            dpg.add_combo(
                ["log chirp", "pink noise"],
                tag=self.generator_mode,
                default_value=self.settings.generator_mode,
                width=-1,
                callback=self._set_generator_mode,
            )
            dpg.add_separator()
            self._add_time(
                "Pre silence, s",
                self.pre_silence,
                self.settings.pre_silence,
                self._set_pre_silence,
            )
            self._add_time(
                "Fade in, s",
                self.fade_in,
                self.settings.fade_in,
                self._set_fade_in,
            )
            self._add_time(
                "Fade out, s",
                self.fade_out,
                self.settings.fade_out,
                self._set_fade_out,
            )
            self._add_time(
                "Post silence, s",
                self.post_silence,
                self.settings.post_silence,
                self._set_post_silence,
            )
            dpg.add_separator()
            dpg.add_text("Welch bucket size, samples")
            dpg.add_input_int(
                tag=self.welch_samples,
                default_value=self.settings.welch_samples,
                width=-1,
                step=0,
                callback=self._set_welch_samples,
            )
            dpg.add_checkbox(
                label="Online Welch",
                tag=self.online_welch,
                default_value=self.settings.online_welch,
                callback=self._set_online_welch,
            )

    def destroy(self) -> None:
        if dpg.does_item_exist(self.TAG):
            dpg.delete_item(self.TAG)
        if dpg.does_item_exist(self.MENU_ITEM):
            dpg.delete_item(self.MENU_ITEM)

    def show(self, sender=None, app_data=None, user_data=None) -> None:
        self._sync_controls()
        main_position = dpg.get_item_pos(self.app.main_window.tag)
        main_size = dpg.get_item_rect_size(self.app.main_window.tag)
        if main_size == [100, 100]:
            main_size = [
                dpg.get_viewport_client_width(),
                dpg.get_viewport_client_height(),
            ]
        position = [
            main_position[0] + (main_size[0] - self.WIDTH) / 2,
            main_position[1] + (main_size[1] - self.HEIGHT) / 2,
        ]
        dpg.set_item_pos(self.TAG, position)
        dpg.configure_item(self.TAG, show=True)

    def hide(self, sender=None, app_data=None, user_data=None) -> None:
        dpg.configure_item(self.TAG, show=False)

    def _sync_controls(self) -> None:
        dpg.set_value(self.generator_mode, self.settings.generator_mode)
        dpg.set_value(self.welch_samples, self.settings.welch_samples)
        dpg.set_value(self.online_welch, self.settings.online_welch)
        dpg.set_value(self.pre_silence, self.settings.pre_silence)
        dpg.set_value(self.post_silence, self.settings.post_silence)
        dpg.set_value(self.fade_in, self.settings.fade_in)
        dpg.set_value(self.fade_out, self.settings.fade_out)

    def _add_time(self, label: str, tag: str, value: float, callback) -> None:
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

    def _change_allowed(self) -> bool:
        if not self.app.app_state.measuring:
            return True
        self.app.main_window.set_status_text(
            "Stop the active measurement before changing Spectrum settings"
        )
        self._sync_controls()
        return False

    def _set_generator_mode(
        self,
        sender: int | str,
        value: str,
        user_data=None,
    ) -> None:
        if self._change_allowed():
            self.settings.generator_mode = cast(GeneratorMode, value)

    def _set_welch_samples(
        self,
        sender: int | str,
        value: int,
        user_data=None,
    ) -> None:
        if self._change_allowed():
            self.settings.welch_samples = value
            dpg.set_value(sender, self.settings.welch_samples)

    def _set_online_welch(
        self,
        sender: int | str,
        value: bool,
        user_data=None,
    ) -> None:
        if self._change_allowed():
            self.settings.online_welch = value

    def _set_pre_silence(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.pre_silence = value
            dpg.set_value(sender, self.settings.pre_silence)

    def _set_post_silence(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.post_silence = value
            dpg.set_value(sender, self.settings.post_silence)

    def _set_fade_in(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.fade_in = value
            dpg.set_value(sender, self.settings.fade_in)

    def _set_fade_out(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.fade_out = value
            dpg.set_value(sender, self.settings.fade_out)
