from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .application import SpectrumApplication

__all__ = ["SpectrumApplication"]


def __getattr__(name: str) -> Any:
    """Keep the public application import without eagerly loading its module."""
    if name == "SpectrumApplication":
        from .application import SpectrumApplication

        return SpectrumApplication
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
