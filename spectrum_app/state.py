from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
from time import sleep, time

import numpy as np

from utils.analyzer import Analyzer
from utils.audio import io_list_updater, InputMeter, AudioIO
from .settings import AppSettings, load_settings, validate_audio_settings


class Timer(Thread):
    def __init__(self, delay: float, function, args=(), kwargs=None) -> None:
        super().__init__()
        self.delay = delay
        self.enabled = Event()
        self.function = function
        self.start_time: float = 0.0
        self.args = args
        self.kwargs = kwargs or {}
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


@dataclass
class AppState:
    settings: AppSettings = field(default_factory=load_settings)
    analyzer: Analyzer = field(default_factory=Analyzer)
    meter: InputMeter = field(default_factory=InputMeter)
    io_upd: io_list_updater = field(default_factory=io_list_updater)
    audio_io: AudioIO = field(default_factory=AudioIO)
    lines: list[int | str] = field(default_factory=list)
    fft_xaxis: int | str | None = None
    fft_yaxis: int | str | None = None
    current_line: int = 0
    levels_l: int | str | None = None
    levels_r: int | str | None = None
    ref_combo: int | str | None = None
    welch_n_input: int | str | None = None
    lines_table_rows: list[list[int | str]] = field(default_factory=list)
    records: list[np.ndarray] = field(default_factory=list)
    generator_signals: list[np.ndarray] = field(default_factory=list)
    record_sample_rates: list[int] = field(default_factory=list)
    analyzer_line_index: int = 0
    completed_audio_record: np.ndarray | None = None
    completed_generator_signal: np.ndarray | None = None
    completed_audio_sample_rate: int = 0
    pending_reanalysis: bool = False
    project_path: Path | None = None
    io_reenable_timer: "Timer | None" = None
    pending_audio_start: bool = False
    pending_meter_start: bool = False
    services_started: bool = False

    def start_services(self) -> None:
        if self.services_started:
            return
        self.audio_io.ref = self.analyzer.ref
        self.analyzer.start()
        self.meter.start()
        self.io_upd.start()
        self.io_upd.enable.set()
        self.audio_io.start()
        self.io_reenable_timer = Timer(5, self.io_upd.enable.set)
        self.io_reenable_timer.start()
        self.services_started = True

    def stop_services(self) -> None:
        self.pending_audio_start = False
        self.pending_meter_start = False
        self.audio_io.kill()
        self.meter.enable.clear()
        self.io_upd.enable.clear()
        if self.io_reenable_timer:
            self.io_reenable_timer.enabled.clear()
        if self.audio_io.is_alive():
            self.audio_io.join(timeout=2.0)


def create_app_state() -> AppState:
    settings = load_settings()
    input_device, output_device = validate_audio_settings(settings)
    state = AppState(
        settings=settings,
        meter=InputMeter(block_size=settings.audio.block_size),
        audio_io=AudioIO(
            device=(input_device, output_device),
            block_size=settings.audio.block_size,
        ),
    )
    state.meter.device = input_device
    return state
