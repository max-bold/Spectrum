from enum import Enum

class GenMode(Enum):
    LOG_SWEEP = "log sweep"
    PINK_NOISE = "pink noise"

class RefMode(Enum):
    NONE = "none"
    CHANNEL_B = "channel b"
    GENERATOR = "generator"