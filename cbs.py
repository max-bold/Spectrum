from threading import Event, Thread
from time import sleep, time
import dearpygui.dearpygui as dpg

from utils.analyzer import Analyzer
from utils.audio import io_list_updater, InputMeter, AudioIO, GenMode, RefMode
from utils.classes import AnalyzerMode, WeightingMode, GenMode, RefMode
from utils.windows import Windows


analyzer = Analyzer()
analyzer.start()


meter = InputMeter()
meter.start()

io_upd = io_list_updater()
io_upd.start()
io_upd.enable.set()

audio_io = AudioIO()
audio_io.ref = analyzer.ref
audio_io.start()

lines = []
current_line = 0

levels_l = None
levels_r = None

lines_table_rows = []


def run_btn():
    if not audio_io.running.is_set():
        io_upd.enable.clear()
        io_upd.paused.wait()
        meter.enable.clear()
        # dpg.set_value("meter_cb", False)
        audio_io.running.set()
    else:
        audio_io.stop_audio()
        io_upd.enable.set()


def set_genmode(source: int, mode: str):
    audio_io.gen_mode = GenMode(mode)


def set_band(source, band: list[int]) -> None:
    audio_io.band = (float(band[0]), float(band[1]))
    analyzer.band = (float(band[0]), float(band[1]))


def set_length(source: int, length: float) -> None:
    audio_io.length = length


def set_input_meter(source: int, state: bool) -> None:
    if state:
        io_upd.enable.clear()
        io_upd.paused.wait()
        meter.enable.set()
    else:
        meter.enable.clear()
        io_upd.enable.set()


def upd_level_monitor(bars) -> None:
    levels = meter.get_levels()
    dpg.set_value(bars[0], levels[0])
    dpg.set_value(bars[1], levels[1])


def upd_io(inputs_combo: int | str, outputs_combo: int | str) -> None:
    dpg.configure_item(inputs_combo, items=io_upd.inputs)
    dpg.configure_item(outputs_combo, items=io_upd.outputs)


def set_input(s, name: str):
    idx = io_upd.get_device_indx(name)
    audio_io.device = (idx, audio_io.device[1])
    meter.device = idx


def set_output(s, name: str) -> None:
    idx = io_upd.get_device_indx(name)
    audio_io.device = (audio_io.device[0], idx)


def set_analyzer_mode(s, mode: str) -> None:
    analyzer.analyzer_mode = AnalyzerMode(mode)


def set_analyzer_ref(s, ref: str) -> None:
    analyzer.ref = RefMode(ref)
    audio_io.ref = RefMode(ref)


def set_analyzer_weighting(s, weighting: str) -> None:
    analyzer.weighting = WeightingMode(weighting)

def set_bucket_size(s, size: int) -> None:
    analyzer.welch_n = size


def set_window_width(s, width: float) -> None:
    analyzer.window_width = width


def set_freq_length(s, length: int) -> None:
    analyzer.freq_length = length


def record_used_click(sender, state, rows) -> None:
    global current_line
    if state == False:
        dpg.set_value(sender, True)
    else:
        for i, row in enumerate(rows):
            if not row[0] == sender:
                dpg.set_value(row[0], False)
            else:
                current_line = i


def record_visible_clicked(sender, state, data) -> None:
    for row, line in zip(*data):
        if sender == row[1]:
            if state:
                dpg.show_item(line)
            else:
                dpg.hide_item(line)


def record_set_name(sender, name, data) -> None:
    for row, line in zip(*data):
        if sender == row[2]:
            dpg.set_item_label(line, name)


def set_filter_window_func(sender, func: Windows) -> None:
    analyzer.window_func = func


class Timer(Thread):
    def __init__(self, delay: float, function, args=(), kwargs={}) -> None:
        super().__init__()
        self.delay = delay
        self.enabled = Event()
        self.function = function
        self.start_time: float = 0.0
        self.args = args
        self.kwargs = kwargs
        self.daemon = True

    def run(self) -> None:
        while True:
            self.enabled.wait()
            self.start_time = time()
            while self.enabled.is_set():
                if time() - self.start_time >= self.delay:
                    self.function(*self.args, **self.kwargs)
                    self.enabled.clear()
                else:
                    sleep(0.01)


t1 = Timer(5, io_upd.enable.set)


def reenable_io_udater():
    if (
        not audio_io.running.is_set()
        and not meter.enable.is_set()
        and not io_upd.enable.is_set()
    ):
        t1.enabled.set()
    else:
        t1.enabled.clear()


def run_analyzer():
    if not analyzer.running.is_set():
        
        if audio_io.running.is_set() and audio_io.record_updated.is_set():
            analyzer.analyzer_mode = AnalyzerMode.WELCH
            record_ready = True
        elif audio_io.record_completed.is_set():
            audio_io.record_completed.clear()
            analyzer.analyzer_mode = AnalyzerMode.PERIODIOGRAM
            record_ready = True
        else:
            record_ready = False

        if record_ready:
            analyzer.sample_rate = audio_io.in_fs
            analyzer.record = audio_io.get_record()
            analyzer.running.set()

    if analyzer.completed.is_set():
        analyzer.completed.clear()
        fft_data = analyzer.result.copy()
        dpg.set_value(lines[current_line], list(fft_data))
        dpg.show_item(lines[current_line])
        dpg.set_value(lines_table_rows[current_line][1], True)

    if audio_io.levels_updated.is_set():
        ts, levels = audio_io.get_levels(0.01)
        if levels_l and levels_r:
            dpg.set_value(levels_l, [list(ts), list(levels[0])])
            dpg.set_value(levels_r, [list(ts), list(levels[1])])
