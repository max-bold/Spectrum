from __future__ import annotations

import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROJECT_PATH = Path(__file__).with_name("test1.bms")
OUTPUT_PATH = Path(__file__).with_name("spectrum_waveform_channel_a.png")
CHANNEL_INDEX = 0
INITIAL_RANGE = (0.8, 1.2)
MAX_DISPLAY_POINTS = 150_000


def minmax_indices(first: int, last: int) -> np.ndarray:
    count = last - first
    if count <= MAX_DISPLAY_POINTS:
        return np.arange(first, last)

    block_size = int(np.ceil(count / (MAX_DISPLAY_POINTS // 2)))
    indices: list[int] = []
    for start in range(first, last, block_size):
        stop = min(start + block_size, last)
        block = samples[start:stop]
        minimum = start + int(np.argmin(block))
        maximum = start + int(np.argmax(block))
        indices.extend(sorted((minimum, maximum)))
    return np.asarray(indices, dtype=np.int64)


with PROJECT_PATH.open("rb") as project_file:
    app_state = pickle.load(project_file)

measurement = next(
    item for item in app_state.measurements if item.module_id == "spectrum"
)
recording = measurement.module_state["recording"]
sample_rate = float(recording.sample_rate)
samples = recording.as_array(np.float64)[:, CHANNEL_INDEX]
duration = len(samples) / sample_rate

figure, axis = plt.subplots(num="Spectrum raw waveform — Channel A")
line, = axis.plot([], [], linewidth=0.8, color="tab:cyan")
axis.set_title("Spectrum recording — raw Channel A")
axis.set_xlabel("Recording time, s")
axis.set_ylabel("Amplitude")
axis.grid(True, alpha=0.25)
axis.set_ylim(-1.05, 1.05)

for boundary in np.arange(0.0, duration + 0.05, 0.1):
    axis.axvline(boundary, color="tab:red", linewidth=0.55, alpha=0.22)


def refresh_visible_data(_axis=None) -> None:
    left, right = axis.get_xlim()
    first = max(0, int(np.floor(left * sample_rate)))
    last = min(len(samples), int(np.ceil(right * sample_rate)) + 1)
    if last <= first:
        return
    indices = minmax_indices(first, last)
    line.set_data(indices / sample_rate, samples[indices])
    figure.canvas.draw_idle()


axis.callbacks.connect("xlim_changed", refresh_visible_data)
axis.set_xlim(*INITIAL_RANGE)
refresh_visible_data()
figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
plt.show()
