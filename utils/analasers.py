import numpy as np
import sounddevice as sd
from numpy.typing import NDArray
from threading import Thread, Event
from queue import Queue, Empty, Full
from typing import Literal, Callable
from scipy.signal import lfilter, sosfilt, butter
from scipy.fft import rfft, rfftfreq


class RTA(Thread):
    def __init__(
        self,
        mode: Literal["open", "closed", "external"] = "open",
        device: tuple[int, int] | None = None,
        band: tuple[float, float] = (20, 20e3),
        callback: Callable | None = None,
    ) -> None:
        self.channels = 2
        self.mode = mode
        self.device = device
        self.stop_event = Event()
        self.boost = 3.0
        self.band = band
        self.blocksize = 1024 * 4
        self.bandfilter: np.ndarray | None = None
        self.callback = callback
        self.window = 1 / 3
        self.res_length = 1024
        self.output_queue: Queue[NDArray] = Queue(10)
        self.input_queue: Queue[NDArray] = Queue(10)
        self.results_queue: Queue[tuple[NDArray, NDArray]] = Queue(1)
        self.samplerate: int | None = None
        self.stream: sd.Stream | None = None
        return super().__init__()

    def generate_noise(self):
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        if self.samplerate:
            bandfilter = butter(
                4,
                self.band,
                btype="band",
                fs=self.samplerate,
                output="sos",
            )
            while not self.stop_event.is_set():
                if self.stop_event.is_set():
                    pass
                noise = np.random.uniform(-1, 1, self.blocksize) * self.boost
                noise = lfilter(b, a, noise)
                noise = sosfilt(bandfilter, noise)
                noise = np.column_stack((noise, noise))
                noise = noise.astype(np.float32).clip(-1, 1)
                self.output_queue.put(noise)
            pass

    def read_write(self):
        if self.stream:
            # Wait until generator will fill output queue
            while not self.output_queue.full():
                pass
            self.stream.start()
            while not self.stop_event.is_set():
                outdata = self.output_queue.get_nowait()
                self.stream.write(outdata)
                indata = self.stream.read(self.blocksize)[0]
                self.input_queue.put_nowait(indata)
            self.stream.stop()
            self.stream.close()

    def run(self) -> None:
        self.stream = sd.Stream(
            device=self.device,
            channels=self.channels,
            blocksize=self.blocksize,
        )
        self.samplerate = self.stream.samplerate
        generator = Thread(target=self.generate_noise)
        generator.start()
        interface = Thread(target=self.read_write)
        interface.start()

        if self.samplerate:
            fft_step = self.samplerate / self.blocksize
            log_f = np.logspace(
                np.log10(self.band[0]),
                np.log10(self.band[1]),
                num=self.res_length,
            )
            window_start = (log_f / 2 ** (self.window / 2) / fft_step).astype(int)
            window_end = (log_f * 2 ** (self.window / 2) / fft_step).astype(int)

            while not self.stop_event.is_set():
                indata = self.input_queue.get()

                # Calc FFTs
                f_y = np.abs(rfft(indata, axis=0))
                fft = (f_y[:, 0] / f_y[:, 1]).clip(1e-12, None)

                # Filter FFT on log-scale
                log_y = np.zeros(len(log_f))
                for i, start, end in zip(range(len(log_f)), window_start, window_end):
                    if end <= start:
                        log_y[i] = fft[start]
                    else:
                        log_y[i] = np.median(fft[start:end])
                try:
                    self.results_queue.put_nowait((log_f, log_y))
                except Full:
                    pass

            generator.join()
            interface.join()
            # Callback with results in db scale
            # if self.callback:
            #     self.callback(log_f, 20 * np.log10(log_y))

    def stop(self) -> None:
        self.stop_event.set()
        # self.join()


def sweep() -> NDArray[np.float64]:
    pass


if __name__ == "__main__":
    import dearpygui.dearpygui as dpg

    dpg.create_context()

    with dpg.window(label="Tutorial", tag="pw"):
        with dpg.plot(label="Sectrogram", width=-1, height=-70):
            x_axis = dpg.add_plot_axis(
                dpg.mvXAxis, label="Hz", scale=dpg.mvPlotScale_Log10
            )
            dpg.set_axis_limits(x_axis, 20, 20e3)
            with dpg.plot_axis(dpg.mvYAxis, label="db") as y_axis:
                dpg.add_line_series([], [], tag="fft")

    def upd_plot(x, y):
        dpg.set_value("fft", [x, y])

    rta = RTA(mode="open", callback=upd_plot)
    rta.start()

    dpg.create_viewport(title="Custom Title", width=800, height=600)
    dpg.setup_dearpygui()
    dpg.set_primary_window("pw", True)
    dpg.show_viewport()
    try:
        while dpg.is_dearpygui_running():
            try:
                x, y = (
                    rta.results_queue.get_nowait()
                )  # non-blocking, если данных нет — queue.Empty
            except Empty:
                pass
            else:
                dpg.set_value("fft", [x, y])
            dpg.render_dearpygui_frame()
    finally:
        dpg.destroy_context()
        rta.stop()
