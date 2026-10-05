"""Capture the actual GUI with synthetic demo data; no audio capture is used."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dearpygui.dearpygui as dpg
import numpy as np

from spectrum_app.application import SpectrumApplication
from spectrum_app.core.model import AxisSpec, GraphData


def main() -> None:
    app = SpectrumApplication()
    measurement = app.create_measurement("spectrum")
    measurement.name = "Loudspeaker (demo)"
    time = np.linspace(0, 10, 256)
    envelope = np.minimum(np.minimum(time * 4, (10 - time) * 4), 1).clip(0)
    measurement.module_state["level_time"] = time
    measurement.module_state["level_values"] = np.column_stack((
        envelope * (0.56 + 0.035 * np.sin(time * 2.3)),
        envelope * (0.43 + 0.025 * np.sin(time * 2.3 + 0.3)),
    ))
    f = np.geomspace(20, 20_000, 1024)
    level = -12 + 12 / (1 + (80/f)**4) - 4 * (f/14000)**2
    level += 1.6 * np.sin(np.log(f) * 6) * np.exp(-((np.log(f/1800))/2.5)**2)
    measurement.graphs = [GraphData("Frequency response", f, level, AxisSpec.FREQ, AxisSpec.LEVEL,
                                   color=(87, 218, 239, 255))]
    app.app_state.visible_graph_ids = [measurement.graphs[0].id]
    # Avoid changing user preferences or accessing physical audio during capture.
    app.settings.load = lambda: False
    app._audio_service.start = lambda: None
    app._audio_service.shutdown = lambda: None
    app.settings.frequency_range = (20, 20000)
    app._initialize()
    try:
        for _ in range(8):
            app.main_window.update()
            app.dpg.render_frame()
        path = ROOT / "docs/assets/spectrum-v0.3.2.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        dpg.output_frame_buffer(str(path))
        for _ in range(4):
            app.dpg.render_frame()
            app.dpg.process_callbacks()
        print(path)
    finally:
        app._shutdown()


if __name__ == "__main__":
    main()
