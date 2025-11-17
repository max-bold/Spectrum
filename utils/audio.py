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
        length: float = 10.0,
        device: tuple[int, int] | tuple[None, None] = (None, None),
        gen_mode: GenMode = GenMode.LOG_SWEEP,
        band: tuple[float, float] = (20, 20000),
        ref: RefMode = RefMode.CHANNEL_B,
        daemon=False,
    ) -> None:
        self.length = length
        self.device: tuple[int | None, int | None] = device
        self.gen_mode: GenMode = gen_mode
        self.band = band
        self.in_fs = 0
        self.in_n = 0
        self.out_fs = 0
        self.out_n = 0
        self.record = np.empty((0, 2), np.float32)
        self.signal = np.empty((0, 2), np.float32)
        self.ref: RefMode = ref
        self.out_position = 0
        self.in_position = 0
        self.record_lock = Lock()
        self.running = Event()
        self.record_updated = Event()
        self.levels_updated = Event()
        self.exit = Event()
        self.out_stop = Event()
        self.in_stop = Event()
        self.padding_time = 0.2

        self.start_time = 0

        return super().__init__(daemon=daemon)

    def run(self) -> None:
        if WIN32:
            pythoncom.CoInitializeEx(0)  # type: ignore
        try:
            while not self.exit.is_set():
                self.running.wait()
                self.in_stop.clear()
                self.out_stop.clear()
                if not self.exit.is_set():
                    in_stream = sd.InputStream(
                        device=self.device[0],
                        callback=self.input_callback,
                    )
                    self.in_fs = in_stream.samplerate
                    self.in_n = int((self.length + self.padding_time) * self.in_fs)

                    self.record = np.zeros((self.in_n, 2), np.float32)
                    self.in_position = 0

                    out_stream = sd.OutputStream(
                        device=self.device[1],
                        callback=self.output_callback,
                    )
                    self.out_fs = out_stream.samplerate
                    self.out_n = int(self.length * self.out_fs)
                    self.out_position = 0
                    if self.gen_mode == GenMode.LOG_SWEEP:
                        self.signal = log_sweep(self.out_n, self.out_fs, self.band)
                    if self.gen_mode == GenMode.PINK_NOISE:
                        self.signal = pink_noise(self.out_n, self.out_fs, self.band)
                    out_padding = int(self.out_fs * self.padding_time)
                    self.signal = np.pad(self.signal, ((out_padding, 0), (0, 0)))
                    self.out_n += out_padding

                    in_stream.start()
                    out_stream.start()
                    self.out_stop.wait()
                    self.in_stop.wait()
                    in_stream.stop()
                    out_stream.stop()
                    in_stream.close()
                    out_stream.close()
                self.running.clear()
        except Exception as e:
            print(f"AudioIO exception: {e}")
        finally:
            if WIN32:
                pythoncom.CoUninitialize()  # type: ignore
            self.running.clear()

    def output_callback(
        self,
        outdata: np.ndarray,
        n: int,
        time: PaStreamCallbackTimeInfo,
        status: sd.CallbackFlags,
    ) -> None:
        s = self.out_position
        e = min(s + n, self.out_n)
        outdata.fill(0)
        outdata[: e - s, :2] = self.signal[s:e]
        self.out_position = e

        if e >= self.out_n:
            self.out_stop.set()

    def input_callback(
        self,
        indata: np.ndarray,
        n: int,
        time: PaStreamCallbackTimeInfo,
        status: sd.CallbackFlags,
    ) -> None:
        with self.record_lock:
            s = self.in_position
            e = min(s + n, self.in_n)
            if indata.shape[1] < 2:
                indata = np.repeat(indata, 2, axis=1)
            self.record[s:e] = indata[: e - s, :2]
            self.in_position = e

        self.levels_updated.set()
        self.record_updated.set()

        if e >= self.in_n:
            self.in_stop.set()

    def get_record(self) -> np.ndarray:
        with self.record_lock:
            data = self.record[: self.in_position].copy()
        self.record_updated.clear()
        if self.ref == RefMode.GENERATOR:
            data[:, 1] = self.signal[: len(data), 0]
        return data

    def get_levels(self, time_step: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            ts (np.ndarray): 1D array of time stamps (shape: (N,)), where N is the number of chunks.
            levels (np.ndarray): 2D array of shape (2, N), where levels[0] is the max level for channel 1,
                                 and levels[1] is the max level for channel 2, for each time chunk.
        """
        with self.record_lock:
            e = self.in_position
            data = self.record[:e].copy()
        chunk_size = int(time_step * self.in_fs)
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
        self.in_stop.set()
        self.out_stop.set()

    def run_once(self):
        self.running.set()
        self.in_stop.wait()
        self.out_stop.wait()

    def stop_audio(self):
        self.running.clear()
        self.in_stop.set()
        self.out_stop.set()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
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
    io = AudioIO(10, gen_mode=GenMode.LOG_SWEEP, band=(100, 20000), device=(14, 12), daemon=True)
    io.start()
    io.run_once()
    # while True:
    #     io.run_once()
    figure, plots = plt.subplots(2, 1, figsize=(8, 6))
    plots: list[Axes]
    ts, levels = io.get_levels(0.1)
    plots[0].plot(ts, 20 * np.log10(levels[0].clip(1e-12)), label="Channel 1")
    plots[0].plot(ts, 20 * np.log10(levels[1].clip(1e-12)), label="Channel 2")

    rec = io.get_record()
    fs, Pxx = periodogram(rec, fs=io.in_fs, axis=0)
    plots[1].semilogx(
        fs, 10 * np.log10((fs * Pxx[:, 0]).clip(1e-20)), label="Channel 1"
    )
    plots[1].semilogx(
        fs, 10 * np.log10((fs * Pxx[:, 1]).clip(1e-20)), label="Channel 2"
    )
    # a, b = io.get_record().T
    # ts = np.arange(len(a)) / io.fs
    # plt.plot(ts, a, label="Channel 1")
    # plt.plot(ts, b, label="Channel 2")
    # ts = np.diff(io.cb_ts[: io.cb_i])
    # ns = np.cumsum(io.cb_ns[: len(ts)]) / io.fs * 1e9 - np.cumsum(ts)
    # plt.plot(np.cumsum(ts) / 1e9, ts)
    # plt.plot(np.cumsum(ts) / 1e9, ns)

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
    plt.legend()
    plots[0].grid(True, "both")
    plots[1].grid(True, "both")
    plt.show()
