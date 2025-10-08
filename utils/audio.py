import sounddevice as sd
from threading import Thread, Lock, Event
from time import sleep
from typing import Literal, NamedTuple
import numpy as np
from generators import log_sweep, pink_noise
import sys

WIN32 = sys.platform == "win32"

if WIN32:
    import pythoncom


class PaStreamCallbackTimeInfo(NamedTuple):
    inputBufferAdcTime: float
    currentTime: float
    outputBufferDacTime: float


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
            try:
                stream = sd.InputStream(device=self.device, channels=2)
                stream.start()
                while self.enable.is_set():
                    chunk = stream.read(1024)[0]
                    levels: np.ndarray = np.max(np.abs(chunk), axis=0)
                    with self.level_lock:
                        self.level = levels.copy()
                stream.close()
            except sd.PortAudioError as e:
                self.enable.clear()
                print(e)

    def get_levels(self) -> np.ndarray:
        with self.level_lock:
            return self.level.copy()


class AudioIO(Thread):
    def __init__(
        self,
        length: float,
        device: tuple[int, int] | None = None,
        gen_mode: Literal["pink noise", "log sweep"] = "log sweep",
        band: tuple[float, float] = (20, 20000),
        # rec_len: int = 0,
        ref: Literal["none", "channel b", "generator"] = "channel b",
    ) -> None:
        self.length: float = length
        self.device: tuple[int, int] | None = device
        self.gen_mode: Literal["pink noise", "log sweep"] = gen_mode
        self.band: tuple[float, float] = band
        # self.rec_len = rec_len
        # self.signal_pos: int = 0
        # self.record_pos: int = 0
        self.fs: int = 0
        self.n: int = 0
        self.record: np.ndarray = np.empty((0, 2), np.float32)
        self.signal: np.ndarray = np.empty((0, 2), np.float32)
        # self.levels: np.ndarray | None = None
        self.ref: Literal["none", "channel b", "generator"] = ref
        self.position: int = 0

        self.record_lock = Lock()
        # self.levels_lock = Lock()
        self.running = Event()
        self.record_updated = Event()

        return super().__init__()

    def run(self):
        if WIN32:
            pythoncom.CoInitializeEx(0)  # type: ignore
        try:
            stream = sd.Stream(device=self.device, callback=self.callback)

            self.fs = stream.samplerate
            self.n = int(self.length * self.fs) if self.fs else 0

            # rec_n = self.rec_len if self.rec_len else n
            self.record = np.zeros((self.n, 2), np.float32)

            self.running.set()

            if self.gen_mode == "log sweep":
                self.signal = log_sweep(self.n, self.fs, self.band)
            if self.gen_mode == "pink noise":
                self.signal = pink_noise(self.n, self.fs, self.band)

            stream.start()
            while self.running.is_set():
                with self.record_lock:
                    if self.position >= self.n:
                        self.running.clear()
                sleep(0)
            stream.close()
        except Exception as e:
            print(f"AudioIO exception: {e}")
        finally:
            if WIN32:
                pythoncom.CoUninitialize()  # type: ignore
            self.running.clear()

    def callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        n: int,
        time: tuple[PaStreamCallbackTimeInfo],
        status,
    ) -> None:
        # with self.signal_lock:
        s = self.position
        e = min(s + n, self.n)
        chunk = np.zeros((n, 2), np.float32)
        chunk[: e - s] = self.signal[s:e]
        outdata[:, :2] = chunk

        with self.record_lock:
            # s = self.record_pos
            # e = min(s + n, len(self.record))
            # trim = s + n - len(self.record)
            # if trim > 0:
            #     self.record[:-trim] = self.record[trim:]
            #     s -= trim
            # e = s + n
            # if self.gentorec:
            #     indata[:, 1] = chunk[:, 1]
            self.record[s:e] = indata[: e - s]

            
            self.position = e
        # with self.levels_lock:
        #     res = np.zeros((1, 3), np.float32)
        #     res[0, 1:] = np.max(np.abs(indata), axis=0)
        #     res[0, 0] = self.sample_count / self.fs
        #     self.sample_count += n
        #     self.levels = np.append(self.levels, res, axis=0)
        # print(self.levels.T.shape)
        self.record_updated.set()

    def get_record(self, full: bool = False):
        with self.record_lock:
            if full:
                data = self.record.copy()
            else:
                data = self.record[: self.position].copy()
            # e = len(self.record) if full else self.record_pos
        self.record_updated.clear()
        if self.ref == "generator":
            data[:, 1] = self.signal[: len(data), 0]
        return data

    def get_levels(self):
        with self.record_lock:
            e = self.position
            rec_copy = self.record[:e].copy()
        chunk_size = int(0.1 // self.fs)
        ss = np.arange(0, e, chunk_size)
        es = ss + chunk_size
        levels = np.zeros((len(ss), 2))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.signal import welch, periodogram
    from scipy.signal.windows import gaussian

    pass
    # # updater = io_list_updater()
    # # updater.start()
    # # updater.join()
    # meter = InputMeter()
    # meter.start()
    # meter.enable.set()

    # while True:
    #     print(meter.get_levels())

    for dev in io_list_updater.list_devices():
        print(dev)

    # stream = sd.Stream()
    io = AudioIO(30, gen_mode="log sweep", device=(20, 16))
    io.start()
    io.join()
    x = io.get_record()[:, 0]

    # plt.plot(x)
    # print()
    f, p = periodogram(x, io.fs)
    plt.semilogx(f, 10 * np.log10((f * p).clip(1e-20)), linewidth=0.2)

    log_f = np.geomspace(20, 20e3, 10000)
    log_p = np.interp(log_f, f, p)
    ww = 1000
    w = gaussian(ww, ww / 8)
    log_p = np.convolve(log_p, w / np.sum(w), "same")
    plt.semilogx(log_f, 10 * np.log10((log_f * log_p).clip(1e-20)))

    # plt.plot(10 * np.log10(w))
    plt.grid(True, "both")
    plt.show()
