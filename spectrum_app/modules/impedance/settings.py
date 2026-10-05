import math

from spectrum_app.core.settings import AppSettings


class ImpedanceSettings:
    """Application-wide SPICE fitting preferences, persisted with settings."""

    DEFAULT_SPICE_ACCURACY_PERCENT = 2.0

    def __init__(self, app_settings: AppSettings) -> None:
        self._app_settings = app_settings

    @property
    def spice_accuracy_percent(self) -> float:
        value = self._app_settings.module_setting(
            "impedance", "spice_accuracy_percent", self.DEFAULT_SPICE_ACCURACY_PERCENT
        )
        try:
            return self._normalize(value)
        except (TypeError, ValueError):
            return self.DEFAULT_SPICE_ACCURACY_PERCENT

    @spice_accuracy_percent.setter
    def spice_accuracy_percent(self, value: float) -> None:
        self._app_settings.set_module_setting(
            "impedance", "spice_accuracy_percent", self._normalize(value)
        )

    @staticmethod
    def _normalize(value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("SPICE accuracy must be finite")
        return min(20.0, max(0.1, value))
