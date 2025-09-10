import dearpygui.dearpygui as dpg
from utils.audio import listinputs, listoutputs
from time import time, sleep
from threading import Thread


def run_btn():
    dpg.show_item("rec progress")
    wh = dpg.get_item_height("Primary Window")
    ww = dpg.get_item_width("Primary Window")
    h = dpg.get_item_height("rec progress")
    w = dpg.get_item_width("rec progress")
    dpg.set_item_pos("rec progress", [(ww - w) / 2, (wh - h) / 2])

    def upd_pb():
        dpg.disable_item("run btn")
        st = time()
        l = dpg.get_value("length input")
        while time() - st < l:
            dpg.set_value("Measure prog bar", (time() - st) / l)
            sleep(0.1)
        dpg.hide_item("rec progress")
        dpg.enable_item("run btn")
        with dpg.group(horizontal=True, parent="rec group"):
            dpg.add_checkbox(default_value=True)
            dpg.add_input_text(default_value="new record", width=-1)

    t = Thread(target=upd_pb)
    t.start()


def audioset_open():
    dpg.configure_item("input combo", items=list(listinputs().keys()))
    dpg.configure_item("output combo", items=list(listoutputs().keys()))
    dpg.show_item("AIO")

def measureinp():
    selinput = dpg.get_value("input combo")
    inputs = listinputs()
    if selinput in inputs:
        input = listinputs[selinput]
    else:
        input = None
    m = measure_input(input)
    while True:
        dpg.set_value("ilm pbar",next(m))

class Measurethread(Thread):
    def __init__(self):
        self.runflag = True
        super().__init__()
    def run(self):
        selinput = dpg.get_value("input combo")
        inputs = listinputs()
        if selinput in inputs:
            input = listinputs()[selinput]
        else:
            input = None
        m = measure_input(input)
        while self.runflag:
            dpg.set_value("ilm pbar",next(m))
        # m.send(True)
    def stop(self):
        self.runflag = False
    
measurethread = Measurethread()

def ilm_act(sender,cheked,thread:Measurethread):
    if dpg.get_value("ilm checkbox"):
        thread.start()
    else:
        thread.stop()
