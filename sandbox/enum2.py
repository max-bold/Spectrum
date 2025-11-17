from enum import Enum


class ListableEnum(Enum):
    @classmethod
    def list(cls) -> list[Any]:
        return list(value.value for value in cls)

class GenMode(ListableEnum):
    LOG_SWEEP = "log sweep"
    PINK_NOISE = "pink noise"

print(GenMode.list())