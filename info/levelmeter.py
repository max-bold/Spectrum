import numpy as np
import sounddevice as sd
import dearpygui.dearpygui as dpg
from threading import Thread, Event
import time

# dB scaling helpers
MIN_DB = -60.0  # floor for display


def amp_to_db(amp: float, min_db: float = MIN_DB) -> float:
    # convert linear amplitude to dB, clip to min_db
    a = max(amp, 1e-12)
    db = 20.0 * np.log10(a)
    return max(db, min_db)


def db_to_norm(db: float, min_db: float = MIN_DB) -> float:
    # map dB in [min_db, 0] to [0, 1]
    return float(max(0.0, min(1.0, (db - min_db) / (-min_db))))


class level_update(Thread):
    def __init__(self):
        super().__init__()
        self.stream = sd.InputStream(channels=2, blocksize=1024)
        # simple single-threaded access model: this thread controls the stream
        # directly. If you later need cross-thread access, reintroduce locking.
        self.running_event = Event()
        # self._should_run = True

        # keep last displayed dB to avoid over-updating the GUI
        self._last_db_l = MIN_DB - 10.0
        self._last_db_r = MIN_DB - 10.0

    def run(self) -> None:
        level_l = 0
        level_r = 0
        W = 30
        # while self._should_run:
        while True:
            # pass
            if self.running_event.is_set():
                # read a block (helper handles start/errors)
                data = self._read_block(1024)
                if data is not None:
                    # compute per-channel maxima efficiently
                    abs_data = np.abs(data)
                    max_vals = np.max(abs_data, axis=0)
                    level_l = (level_l * (W - 1) + max_vals[0]) / W
                    level_r = (level_r * (W - 1) + max_vals[1]) / W

                    # convert to dB and normalized progress value
                    db_l = amp_to_db(level_l)
                    db_r = amp_to_db(level_r)
                    norm_l = db_to_norm(db_l)
                    norm_r = db_to_norm(db_r)

                    # small hysteresis: only update GUI when dB changes enough
                    if abs(db_l - self._last_db_l) >= 0.5:
                        dpg.set_value("left_db", f"L: {db_l:.1f} dB")
                        self._last_db_l = db_l
                        dpg.set_value("left", norm_l)
                    if abs(db_r - self._last_db_r) >= 0.5:
                        dpg.set_value("right_db", f"R: {db_r:.1f} dB")
                        self._last_db_r = db_r
                        dpg.set_value("right", norm_r)
                # keep loop stable; errors already handled inside helper
            else:
                # ensure the stream is stopped when not running
                self._stop_stream()
                # dpg.set_value("left", 0)
                # dpg.set_value("right", 0)
                time.sleep(0.05)

    def _read_block(self, frames=1024):
        """Ensure stream is running and read a block.
        Returns the data array or None on error."""
        if not self.stream.active:
            try:
                self.stream.start()
            except Exception:
                return None
        try:
            data, _ = self.stream.read(frames)
        except Exception:
            return None
        return data

    def _stop_stream(self):
        """Stop the stream if active."""
        if self.stream.active:
            try:
                self.stream.stop()
            except Exception:
                pass


updater = level_update()
updater.daemon = True
updater.start()


def startstream():
    updater.running_event.set()


def stopstream():
    updater.running_event.clear()


dpg.create_context()
dpg.create_viewport(title="Try1", width=800, height=600)

with dpg.window(tag="pw"):
    dpg.add_progress_bar(tag="left")
    dpg.add_text("L: -- dB", tag="left_db")
    dpg.add_progress_bar(tag="right")
    dpg.add_text("R: -- dB", tag="right_db")
    dpg.add_button(label="Start", callback=startstream)
    dpg.add_button(label="Stop", callback=stopstream)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("pw", True)
dpg.start_dearpygui()
dpg.destroy_context()
