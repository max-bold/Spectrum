from tkinter import NO
import dearpygui.dearpygui as dpg

from utils.analyzer import AnalyserPipeline
from typing import Literal
from utils.audio import io_list_updater, InputMeter

pipe = AnalyserPipeline()
pipe.start()

run_state = False

meter = InputMeter()
meter.start()

io_upd = io_list_updater()
io_upd.start()
io_upd.enable.set()


def run_btn():
    if not pipe.run_flag.is_set():
        io_upd.enable.clear()
        io_upd.paused.wait()
        meter.enable.clear()
        # dpg.set_value("meter_cb", False)
        pipe.run_flag.set()
    else:
        pipe.run_flag.clear()
        io_upd.enable.set()


def set_genmode(source: int, mode: Literal["pink noise", "log sweep"]):
    pipe.gen_mode = mode


def set_band(source, band: list[int]) -> None:
    pipe.band = (float(band[0]), float(band[1]))


def set_length(source: int, length: float) -> None:
    pipe.length = length


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
    pipe.device = (idx, pipe.device[1])
    meter.device = idx


def set_output(s, name: str) -> None:
    idx = io_upd.get_device_indx(name)
    pipe.device = (pipe.device[0], idx)


def set_analyzer_mode(s, mode: Literal["rta", "recording"]) -> None:
    pipe.analyzer_mode = mode


def set_analyzer_ref(s, ref: Literal["none", "channel B", "generator"]) -> None:
    pipe.ref = ref


def set_analyzer_weighting(s, weighting: Literal["none", "pink"]) -> None:
    pipe.weighting = weighting


def set_bucket_size(s, size: int) -> None:
    pipe.rta_bucket_size = size


def set_window_width(s, width: float) -> None:
    pipe.window_width = width


def set_freq_length(s, length: int) -> None:
    pipe.freq_length = length


current_rec = 0


def record_used_click(sender, state, rows) -> None:
    global current_rec
    if state == False:
        dpg.set_value(sender, True)
    else:
        for i, row in enumerate(rows):
            if not row[0] == sender:
                dpg.set_value(row[0], False)
            else:
                current_rec = i


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

def set_filter_window_func(sender, func:Literal["blackman","boxcar"]) -> None:
        pipe.filter_window_func = func

