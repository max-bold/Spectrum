from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import dearpygui.dearpygui as dpg

from spectrum_app.core.settings import AppSettings

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


SettingsChanged = Callable[[str], None]


class THDSettings:
    MODULE_ID = "thd"
    DEFAULT_SEGMENT_SECONDS = 1.0
    DEFAULT_OVERLAP_PERCENT = 90.0
    DEFAULT_FADE_IN_SECONDS = 0.5
    DEFAULT_FADE_OUT_SECONDS = 0.5
    DEFAULT_NOTCH_RATIO = 1.5
    DEFAULT_POINTS = 1_200

    def __init__(
        self,
        app_settings: AppSettings,
        on_change: SettingsChanged | None = None,
    ) -> None:
        self._app_settings = app_settings
        self._on_change = on_change

    @property
    def segment_seconds(self) -> float:
        return self._float_setting(
            "segment_seconds",
            self.DEFAULT_SEGMENT_SECONDS,
            minimum=0.05,
            maximum=10.0,
        )

    @segment_seconds.setter
    def segment_seconds(self, value: float) -> None:
        self._set(
            "segment_seconds",
            self._normalize_float(value, 0.05, 10.0),
        )

    @property
    def overlap_percent(self) -> float:
        return self._float_setting(
            "overlap_percent",
            self.DEFAULT_OVERLAP_PERCENT,
            minimum=0.0,
            maximum=99.0,
        )

    @overlap_percent.setter
    def overlap_percent(self, value: float) -> None:
        self._set(
            "overlap_percent",
            self._normalize_float(value, 0.0, 99.0),
        )

    @property
    def fade_in_seconds(self) -> float:
        return self._float_setting(
            "fade_in_seconds",
            self.DEFAULT_FADE_IN_SECONDS,
            minimum=0.0,
            maximum=10.0,
        )

    @fade_in_seconds.setter
    def fade_in_seconds(self, value: float) -> None:
        self._set(
            "fade_in_seconds",
            self._normalize_float(value, 0.0, 10.0),
        )

    @property
    def fade_out_seconds(self) -> float:
        return self._float_setting(
            "fade_out_seconds",
            self.DEFAULT_FADE_OUT_SECONDS,
            minimum=0.0,
            maximum=10.0,
        )

    @fade_out_seconds.setter
    def fade_out_seconds(self, value: float) -> None:
        self._set(
            "fade_out_seconds",
            self._normalize_float(value, 0.0, 10.0),
        )

    @property
    def notch_ratio(self) -> float:
        return self._float_setting(
            "notch_ratio",
            self.DEFAULT_NOTCH_RATIO,
            minimum=1.01,
            maximum=4.0,
        )

    @notch_ratio.setter
    def notch_ratio(self, value: float) -> None:
        self._set(
            "notch_ratio",
            self._normalize_float(value, 1.01, 4.0),
        )

    @property
    def points(self) -> int:
        value = self._app_settings.module_setting(
            self.MODULE_ID,
            "points",
            self.DEFAULT_POINTS,
        )
        try:
            return min(10_000, max(32, int(value)))
        except (TypeError, ValueError):
            return self.DEFAULT_POINTS

    @points.setter
    def points(self, value: int) -> None:
        self._set("points", min(10_000, max(32, int(value))))

    def _float_setting(
        self,
        key: str,
        default: float,
        *,
        minimum: float,
        maximum: float,
    ) -> float:
        value = self._app_settings.module_setting(self.MODULE_ID, key, default)
        try:
            return self._normalize_float(value, minimum, maximum)
        except (TypeError, ValueError):
            return default

    def _set(self, key: str, value: int | float) -> None:
        if self._app_settings.module_setting(self.MODULE_ID, key) == value:
            return
        self._app_settings.set_module_setting(self.MODULE_ID, key, value)
        if self._on_change is not None:
            self._on_change(key)

    @staticmethod
    def _normalize_float(value: Any, minimum: float, maximum: float) -> float:
        return min(maximum, max(minimum, float(value)))


class THDSettingsWindow:
    WIDTH = 480
    HEIGHT = 400
    TAG = "module::thd::settings"
    MENU_ITEM = "module::thd::settings_menu_item"

    def __init__(
        self,
        app: "SpectrumApplication",
        settings: THDSettings,
    ) -> None:
        self.app = app
        self.settings = settings
        self.segment_seconds = "module::thd::settings::segment_seconds"
        self.overlap_percent = "module::thd::settings::overlap_percent"
        self.fade_in_seconds = "module::thd::settings::fade_in_seconds"
        self.fade_out_seconds = "module::thd::settings::fade_out_seconds"
        self.notch_ratio = "module::thd::settings::notch_ratio"
        self.points = "module::thd::settings::points"

    def build(self) -> None:
        dpg.add_menu_item(
            label="THD",
            tag=self.MENU_ITEM,
            parent=self.app.main_window.settings_menu,
            callback=self.show,
        )
        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            label="THD settings",
            tag=self.TAG,
            width=self.WIDTH,
            height=self.HEIGHT,
            show=False,
            modal=True,
            no_resize=True,
            no_collapse=True,
            on_close=self.hide,
        ):
            dpg.add_text("STFT window, s")
            dpg.add_input_float(
                tag=self.segment_seconds,
                default_value=self.settings.segment_seconds,
                width=-1,
                step=0.05,
                callback=self._set_segment_seconds,
            )
            dpg.add_text("STFT overlap, %")
            dpg.add_input_float(
                tag=self.overlap_percent,
                default_value=self.settings.overlap_percent,
                width=-1,
                step=1.0,
                callback=self._set_overlap_percent,
            )
            dpg.add_separator()
            dpg.add_text("Fade in, s")
            dpg.add_input_float(
                tag=self.fade_in_seconds,
                default_value=self.settings.fade_in_seconds,
                width=-1,
                step=0.1,
                callback=self._set_fade_in_seconds,
            )
            dpg.add_text("Fade out, s")
            dpg.add_input_float(
                tag=self.fade_out_seconds,
                default_value=self.settings.fade_out_seconds,
                width=-1,
                step=0.1,
                callback=self._set_fade_out_seconds,
            )
            dpg.add_text("Rejection window ratio")
            dpg.add_input_float(
                tag=self.notch_ratio,
                default_value=self.settings.notch_ratio,
                width=-1,
                step=0.1,
                callback=self._set_notch_ratio,
            )
            dpg.add_text("Result points")
            dpg.add_input_int(
                tag=self.points,
                default_value=self.settings.points,
                width=-1,
                step=0,
                callback=self._set_points,
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

    def _sync_controls(self) -> None:
        dpg.set_value(self.segment_seconds, self.settings.segment_seconds)
        dpg.set_value(self.overlap_percent, self.settings.overlap_percent)
        dpg.set_value(self.fade_in_seconds, self.settings.fade_in_seconds)
        dpg.set_value(self.fade_out_seconds, self.settings.fade_out_seconds)
        dpg.set_value(self.notch_ratio, self.settings.notch_ratio)
        dpg.set_value(self.points, self.settings.points)

    def _change_allowed(self) -> bool:
        if not self.app.app_state.measuring:
            return True
        self.app.main_window.set_status_text(
            "Stop the active measurement before changing THD settings"
        )
        self._sync_controls()
        return False

    def _set_segment_seconds(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.segment_seconds = value
            dpg.set_value(sender, self.settings.segment_seconds)

    def _set_overlap_percent(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.overlap_percent = value
            dpg.set_value(sender, self.settings.overlap_percent)

    def _set_fade_in_seconds(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.fade_in_seconds = value
            dpg.set_value(sender, self.settings.fade_in_seconds)

    def _set_fade_out_seconds(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.fade_out_seconds = value
            dpg.set_value(sender, self.settings.fade_out_seconds)

    def _set_notch_ratio(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.notch_ratio = value
            dpg.set_value(sender, self.settings.notch_ratio)

    def _set_points(self, sender, value: int, user_data=None) -> None:
        if self._change_allowed():
            self.settings.points = value
            dpg.set_value(sender, self.settings.points)
