from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from spectrum_app.core.model import Measurement

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class BaseModule(ABC):
    """Common lifecycle implemented by every measurement module."""

    id: str
    name: str

    def __init__(self) -> None:
        self._app: SpectrumApplication | None = None
        self._measurement: Measurement | None = None

    @property
    def app(self) -> "SpectrumApplication":
        if self._app is None:
            raise RuntimeError("Module is not initialized")
        return self._app

    @property
    def measurement(self) -> Measurement:
        if self._measurement is None:
            raise RuntimeError("Module is not active")
        return self._measurement

    @property
    def measurement_button_label(self) -> str:
        """Label for the main action button while the module is idle."""
        return "MEASURE"

    @abstractmethod
    def initialize(self, app: "SpectrumApplication") -> None:
        """Initialize application-wide resources and menu items."""
        self._app = app

    @abstractmethod
    def activate(self, measurement: Measurement) -> None:
        """Load one measurement state and create its controls."""
        self._measurement = measurement

    @abstractmethod
    def start_measurement(self) -> None:
        """Start a new measurement worker."""

    @abstractmethod
    def stop_measurement(self) -> None:
        """Request interruption of the current measurement."""

    @abstractmethod
    def deactivate(self) -> None:
        """Destroy controls and release the active measurement."""
        self._measurement = None

    def update(self) -> None:
        """Process worker results from the application's main thread."""

    def shutdown(self) -> None:
        """Release application-wide resources owned by the module."""
        self._app = None
