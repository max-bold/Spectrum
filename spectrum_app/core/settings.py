from collections.abc import Callable
import json
from pathlib import Path
from typing import Literal


AxisScale = Literal["linear", "log"]
PhaseUnit = Literal["deg", "deg/dec"]


class AppSettings:
    DEFAULT_PATH = Path(__file__).parents[1] / "settings.json"
    DEFAULT_FREQUENCY_RANGE = (20.0, 20_000.0)

    def __init__(self, on_change: Callable[[], None] | None = None) -> None:
        self._path: Path | None = None
        self._on_change = on_change
        self._loading = False
        self._frequency_range = self.DEFAULT_FREQUENCY_RANGE
        self._impedance_scale: AxisScale = "linear"
        self._thd_scale: AxisScale = "linear"
        self._phase_unit: PhaseUnit = "deg"

    @property
    def frequency_range(self) -> tuple[float, float]:
        return self._frequency_range

    @frequency_range.setter
    def frequency_range(self, value: tuple[float, float]) -> None:
        low, high = map(float, value)
        if low <= 0 or high <= low:
            raise ValueError("Frequency range must be positive and increasing")
        self._set("_frequency_range", (low, high))

    @property
    def impedance_scale(self) -> AxisScale:
        return self._impedance_scale

    @impedance_scale.setter
    def impedance_scale(self, value: AxisScale) -> None:
        self._set("_impedance_scale", self._validate_scale(value))

    @property
    def thd_scale(self) -> AxisScale:
        return self._thd_scale

    @thd_scale.setter
    def thd_scale(self, value: AxisScale) -> None:
        self._set("_thd_scale", self._validate_scale(value))

    @property
    def phase_unit(self) -> PhaseUnit:
        return self._phase_unit

    @phase_unit.setter
    def phase_unit(self, value: PhaseUnit) -> None:
        if value not in ("deg", "deg/dec"):
            raise ValueError(f"Unknown phase unit: {value}")
        self._set("_phase_unit", value)

    def load(self, path: str | Path | None = None) -> bool:
        self._path = Path(path) if path is not None else self.DEFAULT_PATH
        if not self._path.exists():
            return False

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            plot = data["plot"]
            self._loading = True
            self.frequency_range = tuple(plot["frequency_range"])
            self.impedance_scale = plot["impedance_scale"]
            self.thd_scale = plot["thd_scale"]
            self.phase_unit = plot["phase_unit"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._reset_defaults()
            return False
        finally:
            self._loading = False
        return True

    def save(self) -> None:
        if self._path is None:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self._path)

    def to_dict(self) -> dict[str, object]:
        return {
            "plot": {
                "frequency_range": list(self.frequency_range),
                "impedance_scale": self.impedance_scale,
                "thd_scale": self.thd_scale,
                "phase_unit": self.phase_unit,
            }
        }

    def _set(self, attribute: str, value: object) -> None:
        if getattr(self, attribute) == value:
            return
        setattr(self, attribute, value)
        if self._loading:
            return
        self.save()
        if self._on_change is not None:
            self._on_change()

    @staticmethod
    def _validate_scale(value: str) -> AxisScale:
        if value not in ("linear", "log"):
            raise ValueError(f"Unknown axis scale: {value}")
        return value

    def _reset_defaults(self) -> None:
        self._frequency_range = self.DEFAULT_FREQUENCY_RANGE
        self._impedance_scale = "linear"
        self._thd_scale = "linear"
        self._phase_unit = "deg"
