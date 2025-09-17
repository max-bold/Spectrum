import numpy as np

# import sounddevice as sd
from numpy.typing import NDArray
from threading import Thread, Event, Lock
from queue import Queue, Empty  # , Full
from typing import Literal, Callable, Any
from scipy.fft import rfft
from abc import ABC, abstractmethod
from time import sleep
from plugins import Pluggable
from scipy.signal.windows import hamming
from generators import LogSweepGeneratorPluggable

# class RTA(Thread):
#     def __init__(
#         self,
#         mode: Literal["open", "closed", "external"] = "open",
#         device: tuple[int, int] | None = None,
#         band: tuple[float, float] = (20, 20e3),
#         callback: Callable | None = None,
#     ) -> None:
#         self.channels = 2
#         self.mode = mode
#         self.device = device
#         self.stop_event = Event()
#         self.boost = 3.0
#         self.band = band
#         self.blocksize = 1024 * 4
#         self.bandfilter: np.ndarray | None = None
#         self.callback = callback
#         self.window = 1 / 3
#         self.res_length = 1024
#         self.output_queue: Queue[NDArray] = Queue(10)
#         self.input_queue: Queue[NDArray] = Queue(10)
#         self.results_queue: Queue[tuple[NDArray, NDArray]] = Queue(1)
#         self.samplerate: int | None = None
#         self.stream: sd.Stream | None = None
#         return super().__init__()

#     def generate_noise(self):
#         b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
#         a = [1, -2.494956002, 2.017265875, -0.522189400]
#         if self.samplerate:
#             bandfilter = butter(
#                 4,
#                 self.band,
#                 btype="band",
#                 fs=self.samplerate,
#                 output="sos",
#             )
#             while not self.stop_event.is_set():
#                 if self.stop_event.is_set():
#                     pass
#                 noise = np.random.uniform(-1, 1, self.blocksize) * self.boost
#                 noise = lfilter(b, a, noise)
#                 noise = sosfilt(bandfilter, noise)
#                 noise = np.column_stack((noise, noise))
#                 noise = noise.astype(np.float32).clip(-1, 1)
#                 self.output_queue.put(noise)
#             pass

#     def read_write(self):
#         if self.stream:
#             # Wait until generator will fill output queue
#             while not self.output_queue.full():
#                 pass
#             self.stream.start()
#             while not self.stop_event.is_set():
#                 outdata = self.output_queue.get_nowait()
#                 self.stream.write(outdata)
#                 indata = self.stream.read(self.blocksize)[0]
#                 self.input_queue.put_nowait(indata)
#             self.stream.stop()
#             self.stream.close()

#     def run(self) -> None:
#         self.stream = sd.Stream(
#             device=self.device,
#             channels=self.channels,
#             blocksize=self.blocksize,
#         )
#         self.samplerate = self.stream.samplerate
#         generator = Thread(target=self.generate_noise)
#         generator.start()
#         interface = Thread(target=self.read_write)
#         interface.start()

#         if self.samplerate:
#             fft_step = self.samplerate / self.blocksize
#             log_f = np.logspace(
#                 np.log10(self.band[0]),
#                 np.log10(self.band[1]),
#                 num=self.res_length,
#             )
#             window_start = (log_f / 2 ** (self.window / 2) / fft_step).astype(int)
#             window_end = (log_f * 2 ** (self.window / 2) / fft_step).astype(int)

#             while not self.stop_event.is_set():
#                 indata = self.input_queue.get()

#                 # Calc FFTs
#                 f_y = np.abs(rfft(indata, axis=0))
#                 fft = (f_y[:, 0] / f_y[:, 1]).clip(1e-12, None)

#                 # Filter FFT on log-scale
#                 log_y = np.zeros(len(log_f))
#                 for i, start, end in zip(range(len(log_f)), window_start, window_end):
#                     if end <= start:
#                         log_y[i] = fft[start]
#                     else:
#                         log_y[i] = np.median(fft[start:end])
#                 try:
#                     self.results_queue.put_nowait((log_f, log_y))
#                 except Full:
#                     pass

#             generator.join()
#             interface.join()
#             # Callback with results in db scale
#             # if self.callback:
#             #     self.callback(log_f, 20 * np.log10(log_y))

#     def stop(self) -> None:
#         self.stop_event.set()
#         # self.join()


# def sweep() -> NDArray[np.float64]:
#     pass


# class Analyzer(Thread, ABC):
#     def __init__(self) -> None:
#         self.input_queue: Queue[NDArray[np.float64]] = Queue(10)
#         self.freq_queue: Queue[NDArray[np.float64]] = Queue(10)
#         self.levels_queue: Queue[NDArray[np.float64]] = Queue(10)
#         self._stop_signal = Event()
#         return super().__init__()

#     @abstractmethod
#     def run(self) -> None:
#         pass

#     def stop(self) -> None:
#         self._stop_signal.set()
#         try:
#             self.freq_queue.get_nowait()
#         except Empty:
#             pass
#         try:
#             self.levels_queue.get_nowait()
#         except Empty:
#             pass
#         return self.join()

#     def getresults(self) -> NDArray[np.float64]:
#         freq_data = self.freq_queue.get()
#         self.freq_queue.task_done()
#         # levels_data = self.levels_queue.get()
#         # self.levels_queue.task_done()
#         return freq_data


# class RecordingAnalyzer(Analyzer):

#     class Recorder(Thread):
#         def __init__(self, queue: Queue[NDArray[np.float64]]):
#             self.queue = queue
#             self.record: NDArray[np.float64] = np.empty((0, 2))
#             self.stop_event = Event()
#             self.record_lock = Lock()
#             return super().__init__()

#         def run(self) -> None:
#             while not self.stop_event.is_set():
#                 try:
#                     chunk = self.queue.get_nowait()
#                     with self.record_lock:
#                         self.record = np.concatenate((self.record, chunk))
#                     self.queue.task_done()
#                 except Empty:
#                     sleep(0.01)
#             print("Recorder stoped.")

#     def __init__(self, rate):
#         self.rate = rate
#         self.freq_length = 1024
#         self.window_width = 1 / 3
#         self.ref = "Chanel B"
#         self.weighting: None | Literal["A"] = None  # "A", "C" or None
#         super().__init__()

#     def run(self) -> None:
#         recorder = self.Recorder(self.input_queue)
#         recorder.start()
#         # levels = []
#         while not self._stop_signal.is_set():
#             # print(record.shape)
#             with recorder.record_lock:
#                 record = recorder.record.copy()
#             if len(record):
#                 yf = np.array(rfft(record, axis=0))
#                 yf = 2 * np.abs(yf) / len(record)
#                 if self.ref == "None":
#                     fft = yf[:, 0].clip(1e-12, None)
#                 else:
#                     fft = (yf[:, 0] / yf[:, 1]).clip(1e-12, None)
#                 log_f = np.logspace(
#                     np.log10(20),
#                     np.log10(20e3),
#                     num=self.freq_length,
#                 )
#                 fft_step = self.rate / len(record)
#                 window_start = np.astype(
#                     log_f / 2 ** (self.window_width / 2) / fft_step, int
#                 )
#                 window_end = np.astype(
#                     log_f * 2 ** (self.window_width / 2) / fft_step, int
#                 )
#                 log_y = np.zeros(len(log_f))
#                 for i, start, end in zip(range(len(log_f)), window_start, window_end):
#                     if end <= start:
#                         log_y[i] = fft[start]
#                     else:
#                         log_y[i] = np.median(fft[start:end])
#                 if self.weighting == "A":
#                     # A-weighting approximation
#                     log_y = log_y * np.sqrt(log_f)
#                 freq_data = np.vstack((log_f, 20 * np.log10(log_y)))
#                 self.freq_queue.put(freq_data)
#                 # levels.append(np.max(np.abs(chunk)))
#                 # self.levels_queue.put(np.array(levels))
#                 # chunk = self.input_queue.get()
#                 # record = np.concatenate((record, chunk))
#         recorder.stop_event.set()
#         recorder.join()
#         with recorder.record_lock:
#             print(f"Analyzer stoped after {len(recorder.record)} samples.")


class AnalyzerPluggable(Pluggable):
    def __init__(self, rate: int, source: Pluggable):
        self.rate = rate
        self.freq_length = 1024
        self.band: tuple[float, float] = (20, 20e3)
        self.window_width = 1 / 3
        self.ref: Literal["none", "channel B", "generator"] = "channel B"
        self.weighting: None | Literal["pink"] = None  # "A", "C" or None
        self.record = np.empty((0, 2), dtype=np.float64)
        self.record_lock = Lock()
        self.mode: Literal["recording", "rta"] = "recording"
        self.rta_bucket_size = 1024 * 4
        self.recorder_sleep: float = 0.01
        self.fft_queue: Queue[NDArray[np.float64]] = Queue(1)
        self.levels = np.empty((0, 2), dtype=np.float64)
        self.levels_lock = Lock()
        return super().__init__(source)

    def recorder(self):
        if self.source:
            while not self.stop_event.is_set():
                input = self.source.output()
                with self.record_lock:
                    self.record = np.concatenate((self.record, input))
                sleep(self.recorder_sleep)
                try:
                    self.output_queue.put_nowait(input)
                except:
                    pass
                with self.levels_lock:
                    self.levels = np.append(self.levels, np.max(input, axis=0))
        print("Recorder stoped.")

    def run(self) -> None:
        recorder = Thread(target=self.recorder, daemon=True)
        recorder.start()
        freq_data = None
        while not self.stop_event.is_set():
            record = None
            with self.record_lock:
                if self.mode == "recording":
                    if len(self.record) > 0:
                        record = self.record.copy()
                elif self.mode == "rta":
                    if len(self.record) > self.rta_bucket_size:
                        record = self.record[: self.rta_bucket_size]
                        self.record = self.record[self.rta_bucket_size :]
                    elif len(self.record) == self.rta_bucket_size:
                        record = self.record.copy()
                        self.record = np.empty((0, 2), dtype=np.float64)
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

            if record is None:
                sleep(0.01)
                continue

            if len(record) > 100000:
                pass

            if self.ref == "none":
                record = record[:, 0]
                n = record.shape[0]
                yf = np.array(rfft(record))
                fft = 2 * np.abs(yf) / n
            elif self.ref == "channel B":
                yf = np.array(rfft(record, axis=0))
                yf = 2 * np.abs(yf) / len(record)
                fft = yf[:, 0] / yf[:, 1].clip(1e-20)
            elif self.ref == "generator":

                def getsource(plugin: Pluggable) -> LogSweepGeneratorPluggable | None:
                    if plugin.source:
                        if isinstance(plugin.source, LogSweepGeneratorPluggable):
                            return plugin.source
                        else:
                            return getsource(plugin.source)
                    else:
                        return None

                gen = getsource(self)

                if gen:
                    nk = np.sqrt(
                        gen.length * gen.rate / np.log10(gen.band[1] / gen.band[0]) / 10
                    ) * 10 ** (0.35726 / 20)
                    record = record[:, 0]
                    n = record.shape[0]
                    yf = np.array(rfft(record))
                    fft = np.abs(yf) / nk
                else:
                    raise ValueError(
                        "Only LogSweepGeneratorPluggable generator is supported"
                    )

            else:
                raise ValueError(f"Unknown ref: {self.ref}")
            log_f, log_y = self.log_filter(fft)
            if self.weighting == "pink":
                # 3 db/oct weighting approximation
                log_y = log_y * np.sqrt(log_f)
            freq_data = np.vstack((log_f, 20 * np.log10(log_y.clip(1e-20))))
            self.fft_queue.put(freq_data)
            # print(f"Analyzer put freq_data with shape {freq_data.shape} to output_queue.")
        print("Analyzer stop_event set. Waiting output queue to be empty...")
        if freq_data is not None:
            self.fft_queue.put(freq_data)
        self.fft_queue.join()
        print(f"Analyzer stoped.")

    def get_fft(self):
        res = self.fft_queue.get()
        self.fft_queue.task_done()
        return res

    def get_levels(self):
        with self.levels_lock:
            res = self.levels.copy()
        return res

    def log_filter(
        self, yf: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        points = int(
            self.freq_length / np.log(20e3 / 20) * np.log(self.band[1] / self.band[0])
        )
        # print(points)
        log_f = np.logspace(
            np.log10(self.band[0]),
            np.log10(self.band[1]),
            num=points,
            dtype=np.float64,
        )
        fft_step = self.rate / len(yf) / 2
        half_w = 2 ** (self.window_width / 2)
        window_start = np.astype(log_f / half_w / fft_step, int)
        window_end = np.astype(log_f * half_w / fft_step, int)
        log_y = np.zeros_like(log_f)
        for i, start, end in zip(range(len(log_f)), window_start, window_end):
            if end <= start:
                log_y[i] = yf[start]
            else:
                log_y[i] = np.median(yf[start:end])
        return log_f, log_y


if __name__ == "__main__":
    pass

    # import dearpygui.dearpygui as dpg

    # dpg.create_context()

    # with dpg.window(label="Tutorial", tag="pw"):
    #     with dpg.plot(label="Sectrogram", width=-1, height=-70):
    #         x_axis = dpg.add_plot_axis(
    #             dpg.mvXAxis, label="Hz", scale=dpg.mvPlotScale_Log10
    #         )
    #         dpg.set_axis_limits(x_axis, 20, 20e3)
    #         with dpg.plot_axis(dpg.mvYAxis, label="db") as y_axis:
    #             dpg.add_line_series([], [], tag="fft")

    # def upd_plot(x, y):
    #     dpg.set_value("fft", [x, y])

    # rta = RTA(mode="open", callback=upd_plot)
    # rta.start()

    # dpg.create_viewport(title="Custom Title", width=800, height=600)
    # dpg.setup_dearpygui()
    # dpg.set_primary_window("pw", True)
    # dpg.show_viewport()
    # try:
    #     while dpg.is_dearpygui_running():
    #         try:
    #             x, y = (
    #                 rta.results_queue.get_nowait()
    #             )  # non-blocking, если данных нет — queue.Empty
    #         except Empty:
    #             pass
    #         else:
    #             dpg.set_value("fft", [x, y])
    #         dpg.render_dearpygui_frame()
    # finally:
    #     dpg.destroy_context()
    #     rta.stop()
