import time
import random
import threading
import queue
import dearpygui.dearpygui as dpg

q = queue.Queue()

def producer():
    while True:
        x = list(range(100))
        y = [random.random() for _ in x]
        q.put((x, y))
        time.sleep(0.05)  # имитация источника данных

# GUI
dpg.create_context()

with dpg.window(label="Plot window"):
    with dpg.plot(label="Live Plot", height=400, width=600):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="X-Axis", tag="xaxis")
        dpg.add_plot_axis(dpg.mvYAxis, label="Y-Axis", tag="yaxis")
        # создаём серию с тегом 'series'
        dpg.add_line_series([], [], label="series", parent="yaxis", tag="series")

dpg.create_viewport(title="Live plot example", width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()

# стартуем продьюсера в фоне
threading.Thread(target=producer, daemon=True).start()

try:
    while dpg.is_dearpygui_running():
        try:
            x, y = q.get_nowait()  # non-blocking, если данных нет — queue.Empty
        except queue.Empty:
            pass
        else:
            # обновляем серию: удаляем старую и добавляем новую (всё в главном потоке)
            if dpg.does_item_exist("series"):
                dpg.delete_item("series")
            dpg.add_line_series(x, y, label="series", parent="yaxis", tag="series")
        # рендерим один кадр DPG
        dpg.render_dearpygui_frame()
finally:
    dpg.destroy_context()