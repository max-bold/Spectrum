import sounddevice as sd


def get_default_io() -> tuple[int, int]:
    dev = []
    for k in ["input", "output"]:
        info = sd.query_devices(kind=k)
        if isinstance(info, dict) and info["index"]:
            dev.append(info["index"])
    return tuple(dev)

print(sd.query_devices())
print(get_default_io())