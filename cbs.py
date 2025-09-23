from tkinter import NO
import dearpygui.dearpygui as dpg

from time import time, sleep
from threading import Thread
from utils.analyzer import AnalyserPipeline
from typing import Literal
from sounddevice import query_devices
from utils.audio import io_list_updater, InputMeter
# from utils.themes import red_theme, green_theme

pipe = AnalyserPipeline()

run_state = False





def run_btn(sender, _,userdata):
    print(userdata)
    global run_state
    run_state = not run_state
    if run_state:
        dpg.set_item_label(sender, "ON")
        dpg.bind_item_theme(sender, "red_theme")
    else:
        dpg.set_item_label(sender, "OFF")
        dpg.bind_item_theme(sender, "green_theme")

    # dpg.show_item("rec progress")
    # wh = dpg.get_item_height("Primary Window")
    # ww = dpg.get_item_width("Primary Window")
    # h = dpg.get_item_height("rec progress")
    # w = dpg.get_item_width("rec progress")
    # dpg.set_item_pos("rec progress", [(ww - w) / 2, (wh - h) / 2])

    # def upd_pb():
    #     dpg.disable_item("run btn")
    #     st = time()
    #     l = dpg.get_value("length input")
    #     while time() - st < l:
    #         dpg.set_value("Measure prog bar", (time() - st) / l)
    #         sleep(0.1)
    #     dpg.hide_item("rec progress")
    #     dpg.enable_item("run btn")
    #     with dpg.group(horizontal=True, parent="rec group"):
    #         dpg.add_checkbox(default_value=True)
    #         dpg.add_input_text(default_value="new record", width=-1)

    # t = Thread(target=upd_pb)
    # t.start()


# def audioset_open():
#     dpg.configure_item("input combo", items=list(listinputs().keys()))
#     dpg.configure_item("output combo", items=list(listoutputs().keys()))
#     dpg.show_item("AIO")


# def measureinp():
#     selinput = dpg.get_value("input combo")
#     inputs = listinputs()
#     if selinput in inputs:
#         input = listinputs[selinput]
#     else:
#         input = None
#     m = measure_input(input)
#     while True:
#         dpg.set_value("ilm pbar", next(m))


# class Measurethread(Thread):
#     def __init__(self):
#         self.runflag = True
#         super().__init__()

#     def run(self):
#         selinput = dpg.get_value("input combo")
#         inputs = listinputs()
#         if selinput in inputs:
#             input = listinputs()[selinput]
#         else:
#             input = None
#         m = measure_input(input)
#         while self.runflag:
#             dpg.set_value("ilm pbar", next(m))
#         # m.send(True)

#     def stop(self):
#         self.runflag = False


# measurethread = Measurethread()


# def ilm_act(sender, cheked, thread: Measurethread):
#     if dpg.get_value("ilm checkbox"):
#         thread.start()
#     else:
#         thread.stop()


def set_genmode(source: int, mode: Literal["pink noise", "log sweep"]):
    pipe.gen_mode = mode


def set_band(source, band: list[int]) -> None:
    pipe.band = (float(band[0]), float(band[1]))


def set_length(source: int, length: float) -> None:
    print(length, type(length))


meter = InputMeter()
meter.start()

io_upd = io_list_updater()
io_upd.start()
io_upd.enable.set()


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
    # print(pipe.device)


def set_output(s, name: str):
    idx = io_upd.get_device_indx(name)
    pipe.device = (pipe.device[0], idx)
    # print(pipe.device)
