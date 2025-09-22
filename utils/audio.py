from os import name
from tkinter import NO
import sounddevice as sd
from threading import Thread, Lock
from time import sleep


class io_list_updater(Thread):
    def __init__(self) -> None:
        self.inputs_list: list[str] = []
        self.outputs_list: list[str] = []
        self.list_lock = Lock()
        return super().__init__(daemon=True)

    def run(self) -> None:
        while True:

            self.upd_inputs()
            self.upd_outputs()
            sd._terminate()
            sd._initialize()
                
            sleep(1)

    def upd_inputs(self):
        dev_names = []
        for dev in sd.query_devices():
            if isinstance(dev, dict):
                if dev["max_input_channels"]:
                    dev_names.append(dev["name"])
        with self.list_lock:
            self.inputs_list = dev_names[:]

    def upd_outputs(self):
        dev_names = []
        for dev in sd.query_devices():
            if isinstance(dev, dict):
                if dev["max_output_channels"]:
                    dev_names.append(dev["name"])
        with self.list_lock:
            self.outputs_list = dev_names[:]

    @property
    def inputs(self) -> list[str]:
        with self.list_lock:
            return self.inputs_list[:]

    @property
    def outputs(self) -> list[str]:
        with self.list_lock:
            return self.outputs_list[:]

def restarter():
    while True:
        print(sd.query_devices(), end="\n---------------\n")
        sd._terminate()
        sd._initialize()
        sleep(1)

if __name__ == "__main__":
    # updater = io_list_updater()
    # updater.start()
    # updater.join()
    res = Thread(None, restarter)
    res.start()