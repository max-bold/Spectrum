from collections.abc import Callable
from os import name
from tkinter import NO
import sounddevice as sd
from threading import Thread, Lock, Event
from time import sleep
from typing import Any, Iterable, Literal, Mapping
import numpy as np


class io_list_updater(Thread):
    def __init__(self) -> None:
        self.inputs_list: list[str] = []
        self.outputs_list: list[str] = []
        self.list_lock = Lock()
        self.enable = Event()
        self.paused = Event()
        return super().__init__(daemon=True)

    def run(self) -> None:
        while True:
            if self.enable.is_set():
                self.paused.clear()
                self.upd_inputs()
                self.upd_outputs()
                sd._terminate()
                sd._initialize()
            else:
                self.paused.set()
            sleep(1)

    def upd_inputs(self):
        dev_names = self.list_devices("input")
        with self.list_lock:
            self.inputs_list = dev_names[:]

    def upd_outputs(self):
        dev_names = self.list_devices("output")
        with self.list_lock:
            self.outputs_list = dev_names[:]

    @staticmethod
    def list_devices(io: Literal["input", "output"] | None = None) -> list[str]:
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
                if not ha == "Windows WDM-KS":
                    if io is None:
                        devices.append(text)
                    elif io == "input" and ins:
                        devices.append(text)
                    elif io == "output" and outs:
                        devices.append(text)
        return devices

    @staticmethod
    def get_device_indx(name: str) -> int:
        return int(name.split(":")[0])

    @property
    def inputs(self) -> list[str]:
        with self.list_lock:
            return self.inputs_list[:]

    @property
    def outputs(self) -> list[str]:
        with self.list_lock:
            return self.outputs_list[:]


class InputMeter(Thread):
    def __init__(self) -> None:
        self.level = np.zeros(2)
        self.level_lock = Lock()
        self.enable = Event()
        self.device: int | None = None
        return super().__init__(daemon=True)

    def run(self):
        while True:
            self.enable.wait()
            stream = sd.InputStream(device=self.device, channels=2)
            stream.start()
            while self.enable.is_set():
                chunk = stream.read(1024)[0]
                levels: np.ndarray = np.max(np.abs(chunk), axis=0)
                with self.level_lock:
                    self.level = levels.copy()
            stream.close()

    def get_levels(self) -> np.ndarray:
        with self.level_lock:
            return self.level.copy()


if __name__ == "__main__":
    # updater = io_list_updater()
    # updater.start()
    # updater.join()
    meter = InputMeter()
    meter.start()
    meter.enable.set()

    while True:
        print(meter.get_levels())
