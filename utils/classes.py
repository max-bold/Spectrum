from enum import Enum
from typing import Any

class ListableEnum(Enum):
    @classmethod
    def list(cls) -> list[str]:
        return list(str(value.value) for value in cls)


class GenMode(ListableEnum):
    LOG_SWEEP = "log sweep"
    PINK_NOISE = "pink noise"


class RefMode(ListableEnum):
    NONE = "none"
    CHANNEL_B = "channel b"
    GENERATOR = "generator"


class AnalyzerMode(ListableEnum):
    WELCH = "welch"
    PERIODIOGRAM = "periodiogram"


class WeightingMode(ListableEnum):
    NONE = "none"
    PINK = "pink"


