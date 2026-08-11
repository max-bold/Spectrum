"""Benchmark the current and legacy logarithmic window generators."""

from __future__ import annotations

import argparse
import gc
import importlib.util
from pathlib import Path
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CENTERS = 1024
BAND = (20.0, 20_000.0)
DURATION_SECONDS = 20.0
SAMPLE_RATE = 192_000.0
OLD_WIDTH = 0.33
NEW_WIDTH = 0.1


def load_smoothing_module():
    module_path = ROOT / "audioanalysis" / "smoothing.py"
    spec = importlib.util.spec_from_file_location("smoothing_benchmark", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_batch(function, window, grid, frequency_step, width, repetitions):
    coefficient_count = 0
    started = perf_counter()
    for _ in range(repetitions):
        for center in grid:
            weights, _, _ = function(
                window,
                float(center),
                frequency_step,
                width,
            )
            coefficient_count += len(weights)
    elapsed = perf_counter() - started
    return elapsed, coefficient_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", nargs="+", type=int, default=[1, 10])
    args = parser.parse_args()
    if any(repetitions <= 0 for repetitions in args.repetitions):
        parser.error("Repetition counts must be positive")

    smoothing = load_smoothing_module()
    grid = np.geomspace(*BAND, CENTERS)
    width = float(np.log2(grid[-1] / grid[0]) / (len(grid) - 1))
    fft_size = int(DURATION_SECONDS * SAMPLE_RATE)
    frequency_step = SAMPLE_RATE / fft_size
    functions = (
        ("old", smoothing.log_window_old, OLD_WIDTH),
        ("new", smoothing.log_window, NEW_WIDTH),
    )

    print(f"centers={CENTERS}")
    print(f"band={BAND[0]:g}..{BAND[1]:g} Hz")
    print(f"duration={DURATION_SECONDS:g} s")
    print(f"sample_rate={SAMPLE_RATE:g} Hz")
    print(f"fft_size={fft_size}")
    print(f"frequency_step={frequency_step:.9f} Hz")
    print(f"grid_step={width:.12f} octave")
    print(f"old_width={OLD_WIDTH:.12f} octave")
    print(f"new_width={NEW_WIDTH:.12f} octave")
    print()
    print(
        f"{'window':<12} {'function':<8} {'width':>8} {'repeats':>8} "
        f"{'time, s':>11} {'us/window':>12} {'coefficients':>14} "
        f"{'Mcoeff/s':>10}"
    )

    gc.collect()
    gc.disable()
    try:
        for repetitions in args.repetitions:
            for window in smoothing.SmoothingWindow:
                for name, function, function_width in functions:
                    elapsed, coefficient_count = run_batch(
                        function,
                        window,
                        grid,
                        frequency_step,
                        function_width,
                        repetitions,
                    )
                    window_calls = repetitions * len(grid)
                    microseconds = elapsed / window_calls * 1e6
                    throughput = coefficient_count / elapsed / 1e6
                    print(
                        f"{window.value:<12} {name:<8} {function_width:>8.3f} "
                        f"{repetitions:>8} {elapsed:>11.6f} "
                        f"{microseconds:>12.3f} {coefficient_count:>14} "
                        f"{throughput:>10.3f}"
                    )
    finally:
        gc.enable()


if __name__ == "__main__":
    main()
