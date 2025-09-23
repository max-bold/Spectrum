import sounddevice as sd
import numpy as np
from typing import Literal

def list_devices(io:Literal["input", "output"]|None=None)->list[str]:
    hostapi_names = []
    for hostapi in sd.query_hostapis():
        if isinstance(hostapi, dict):
            hostapi_names.append(hostapi["name"])

    devices = []
    for info in sd.query_devices():
        if isinstance(info, dict):
            idx = info["index"]
            name = info["name"]
            ha = hostapi_names[info["hostapi"]]
            ins = info["max_input_channels"]
            outs = info["max_output_channels"]
            fs = info["default_samplerate"]
            text = f"{idx}: {name}, {ha}, {ins}>>{outs}, {fs/1000:.1f} kHz"
            if io is None:
                devices.append(text)
            elif io == "input" and ins:
                devices.append(text)
            elif io == "output" and outs:
                devices.append(text)
    return devices

def get_device_indx(name:str)->int:
    return int(dev.split(":")[0])

for dev in list_devices():
    print(dev)
