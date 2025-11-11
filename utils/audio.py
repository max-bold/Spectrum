import sounddevice as sd
from threading import Thread, Lock, Event
from time import sleep, time, monotonic, monotonic_ns
from typing import Literal, NamedTuple
import numpy as np
from generators import log_sweep, pink_noise
import sys
from classes import GenMode, RefMode

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
        gen_mode: GenMode = GenMode.LOG_SWEEP,
        band: tuple[float, float] = (20, 20000),
        ref: RefMode = RefMode.CHANNEL_B,
    ) -> None:
        self.length: float = length
        self.device: tuple[int, int] | None = device
        self.gen_mode: GenMode = gen_mode
        self.band: tuple[float, float] = band
        self.fs: int = 0
        self.n: int = 0
        self.record: np.ndarray = np.empty((0, 2), np.float32)
        self.signal: np.ndarray = np.empty((0, 2), np.float32)
        self.ref: RefMode = ref
        self.position: int = 0
        self.record_lock = Lock()
        self.running = Event()
        self.record_updated = Event()
        self.levels_updated = Event()
        self.exit = Event()
        self.stop = Event()
        self.cb_ts = np.zeros(1000, int)
        self.cb_ns = np.zeros(1000, int)
        self.cb_i = 0
        self.cb_t0 = 0.0

        self.start_time: float = 0

        return super().__init__(daemon=True)

    def run(self):
        if WIN32:
            pythoncom.CoInitializeEx(0)  # type: ignore
        try:
            while not self.exit.is_set():
                self.running.wait()
                self.stop.clear()
                if not self.exit.is_set():
                    stream = sd.Stream(device=self.device, callback=self.callback)

                    self.fs = stream.samplerate
                    self.n = int(self.length * self.fs) if self.fs else 0

                    self.record = np.zeros((self.n, 2), np.float32)
                    self.position = 0

                    if self.gen_mode == GenMode.LOG_SWEEP:
                        self.signal = log_sweep(self.n, self.fs, self.band)
                    if self.gen_mode == GenMode.PINK_NOISE:
                        self.signal = pink_noise(self.n, self.fs, self.band)

                    self.first_cb = True
                    stream.start()
                    self.stop.wait()
                    stream.close()
                self.running.clear()
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
        time: PaStreamCallbackTimeInfo,
        status: sd.CallbackFlags,
    ) -> None:

        if self.cb_i < 1000:
            self.cb_ts[self.cb_i] = monotonic_ns()
            self.cb_ns[self.cb_i] = n
            self.cb_i += 1

        if not self.cb_t0:
            self.cb_t0 = time.currentTime

        if status.output_underflow:
            print("Output underflow:", status, time.currentTime - self.cb_t0)

        if status.input_overflow:
            print("Input overflow:", status, time.currentTime - self.cb_t0)

        s = self.position
        e = min(s + n, self.n)
        # outdata[:] = np.zeros((n, 2), np.float32)
        outdata[: e - s, :2] = self.signal[s:e]
        # outdata[:, :2] = self.signal[s:s+n]

        # with self.record_lock:
        # self.record[s:e] = indata[: e - s, :2]
        self.position = e
        # self.position += n

        # self.record_updated.set()
        # self.levels_updated.set()

        # if self.position >= self.n or not self.running.is_set():
        if self.position >= self.n:
            self.stop.set()

    def get_record(self):
        with self.record_lock:
            data = self.record[: self.position].copy()
        self.record_updated.clear()
        if self.ref == RefMode.GENERATOR:
            data[:, 1] = self.signal[: len(data), 0]
        return data

    def get_levels(self, time_step: float = 0.1):
        with self.record_lock:
            e = self.position
            data = self.record[:e].copy()
        chunk_size = int(time_step * self.fs)
        ss = np.arange(0, e, chunk_size)
        es = ss + chunk_size
        ts = np.arange(len(ss)) * time_step
        levels = np.zeros((len(ss), 2))
        for i, (s, e) in enumerate(zip(ss, es)):
            chunk = data[s:e]
            levels[i] = np.max(np.abs(chunk), axis=0)
        self.levels_updated.clear()
        return ts, levels.T

    def kill(self):
        self.exit.set()
        self.running.set()
        self.stop.set()

    def run_once(self):
        self.running.set()
        self.stop.wait()


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
    io = AudioIO(10, gen_mode=GenMode.LOG_SWEEP, band=(100, 20000))
    io.start()
    io.run_once()
    # ts, levels = io.get_levels(0.1)
    # # plt.plot(ts, 20*np.log10(levels[0].clip(1e-12)), label="Channel 1")
    # # plt.plot(ts, 20*np.log10(levels[1].clip(1e-12)), label="Channel 2")

    # a, b = io.get_record().T
    # ts = np.arange(len(a)) / io.fs
    # plt.plot(ts, a, label="Channel 1")
    # plt.plot(ts, b, label="Channel 2")
    ts = np.diff(io.cb_ts[: io.cb_i])
    ns = np.cumsum(io.cb_ns[: len(ts)]) / io.fs * 1e9 - np.cumsum(ts)
    plt.plot(np.cumsum(ts) / 1e9, ts)
    plt.plot(np.cumsum(ts) / 1e9, ns)

    # # plt.plot(x)
    # # print()
    # f, p = periodogram(x, io.fs)
    # plt.semilogx(f, 10 * np.log10((f * p).clip(1e-20)), linewidth=0.2)

    # log_f = np.geomspace(20, 20e3, 10000)
    # log_p = np.interp(log_f, f, p)
    # ww = 1000
    # w = gaussian(ww, ww / 8)
    # log_p = np.convolve(log_p, w / np.sum(w), "same")
    # plt.semilogx(log_f, 10 * np.log10((log_f * log_p).clip(1e-20)))

    # # plt.plot(10 * np.log10(w))
    # plt.grid(True, "both")
    plt.show()
