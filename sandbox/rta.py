import numpy as np
import sounddevice as sd
import dearpygui.dearpygui as dpg
from threading import Thread, Event
from scipy.fft import rfft, rfftfreq
import time



class Analyser(Thread):
    def __init__(self):
        super().__init__()
        self.stream = sd.InputStream(channels=2)
        self.running_event = Event()
        self.window = None
        self.length = None

    def run(self) -> None:
        self.stream.start()
        self.window = dpg.get_value("ww_slider")
        self.length = dpg.get_value("fft_window")
        log_f = np.logspace(np.log10(20), np.log10(20000), num=1000)
        while True:
            rec_len = int(self.length * self.stream.samplerate)
            fft_step = 1 / self.length
            window_start = (log_f / 2 ** (self.window / 2) / fft_step).astype(int)
            window_end = (log_f * 2 ** (self.window / 2) / fft_step).astype(int)
            data = self.stream.read(rec_len)[0]
            fft = (2 * np.abs(rfft(data, axis=0)) / rec_len).clip(1e-12, None)
            log_y = np.zeros((len(log_f), 2))
            for i, start, end in zip(range(len(log_f)), window_start, window_end):
                if end <= start:
                    log_y[i] = fft[start]
                else:
                    log_y[i] = np.median(fft[start:end], axis=0)
            dpg.set_value("fft", [log_f, 20 * np.log10(log_y[:, 0] / log_y[:, 1])])
            # dpg.set_value("fft_r", [log_f, 20 * np.log10(log_y[:, 1])])

    def setwindow(self, tag, window: float):
        self.window = 1 / window

    def setlength(self, tag, length: float):
        self.length = length


analyser = Analyser()


def start_analyser():
    analyser.start()
    dpg.disable_item("start_btn")


dpg.create_context()

with dpg.window(label="Tutorial", tag="pw"):
    with dpg.plot(label="Sectrogram", width=-1, height=-70):
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Hz", scale=dpg.mvPlotScale_Log10)
        dpg.set_axis_limits(x_axis, 20, 20e3)
        with dpg.plot_axis(dpg.mvYAxis, label="db") as y_axis:
            dpg.add_line_series([], [], tag="fft")
            # # dpg.add_line_series([], [], label="Right Channel", tag="fft_r")
            # dpg.set_axis_limits(y_axis, -120, 0)
            # # dpg.set_value(y_axis, [-120, 0])
            # # dpg.set_axis_limits_auto(y_axis)
    dpg.add_button(label="Start Analysis", callback=start_analyser, tag="start_btn")
    dpg.add_slider_float(
        label="window width",
        default_value=3,
        min_value=1,
        max_value=30,
        tag="ww_slider",
        format="1/%.0f",
        callback=analyser.setwindow,
    )
    dpg.add_slider_float(
        label="FFT width, seconds",
        default_value=0.5,
        min_value=0.1,
        max_value=5.0,
        callback=analyser.setlength,
        format="%.1f s",
        tag="fft_window",
    )

dpg.create_viewport(title="Custom Title", width=800, height=600)
dpg.setup_dearpygui()
dpg.set_primary_window("pw", True)
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
