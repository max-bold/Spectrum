from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from threading import Lock, Thread
from time import monotonic

# Running a file directly from ``sandbox`` puts this directory first on
# ``sys.path``.  Its legacy ``platform.py`` would otherwise shadow Python's
# standard-library module imported by sounddevice.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if not entry or Path(entry).resolve() != SCRIPT_DIR
]

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


SAMPLE_RATE = 192_000
CHUNK_SIZE = 19_200
STREAM_BLOCK_SIZE = 19_200
FREQUENCY = 100.0
AMPLITUDE = 0.9
PRE_SILENCE_SECONDS = 0.2
SIGNAL_SECONDS = 2.0
RECORDING_TAIL_SECONDS = 1.0

OUTPUT_DIR = SCRIPT_DIR
DATA_PATH = OUTPUT_DIR / "sr_rw_bs19200_recording.npz"
PLOT_PATH = OUTPUT_DIR / "sr_rw_bs19200.png"
INFO_PATH = OUTPUT_DIR / "sr_rw_bs19200.json"


def find_device(direction: str) -> tuple[int, dict]:
    host_apis = sd.query_hostapis()
    candidates: list[tuple[int, dict]] = []
    for index, raw_device in enumerate(sd.query_devices()):
        device = dict(raw_device)
        host_api = host_apis[int(device["hostapi"])]
        if host_api["name"] != "Windows WASAPI":
            continue
        name = str(device["name"])
        channels = int(device[f"max_{direction}_channels"])
        prefix = "IN 1-2" if direction == "input" else "OUT 1-2"
        if channels >= 2 and prefix in name and "BEHRINGER" in name:
            candidates.append((index, device))
    if len(candidates) != 1:
        names = [f"[{index}] {device['name']}" for index, device in candidates]
        raise RuntimeError(
            f"Expected one Behringer WASAPI {direction} device, found: {names}"
        )
    return candidates[0]


def make_playback() -> np.ndarray:
    silence_samples = int(round(PRE_SILENCE_SECONDS * SAMPLE_RATE))
    signal_samples = int(round(SIGNAL_SECONDS * SAMPLE_RATE))
    time = np.arange(signal_samples, dtype=np.float64) / SAMPLE_RATE
    signal = AMPLITUDE * np.sin(2.0 * np.pi * FREQUENCY * time)
    mono = np.concatenate(
        (np.zeros(silence_samples, dtype=np.float64), signal)
    ).astype(np.float32)
    playback = np.zeros((len(mono), 2), dtype=np.float32)
    playback[:, 0] = mono
    playback[:, 1] = mono
    return playback


def run_test() -> tuple[np.ndarray, np.ndarray, dict]:
    input_index, input_device = find_device("input")
    output_index, output_device = find_device("output")
    playback = make_playback()
    target_samples = len(playback) + int(
        round(RECORDING_TAIL_SECONDS * SAMPLE_RATE)
    )
    chunks: list[np.ndarray] = []
    read_log: list[dict] = []
    write_log: list[dict] = []
    writer_error: BaseException | None = None
    writer_error_lock = Lock()

    input_stream = sd.InputStream(
        device=input_index,
        samplerate=SAMPLE_RATE,
        channels=2,
        dtype="float32",
        blocksize=STREAM_BLOCK_SIZE,
    )
    output_stream = sd.OutputStream(
        device=output_index,
        samplerate=SAMPLE_RATE,
        channels=2,
        dtype="float32",
        blocksize=STREAM_BLOCK_SIZE,
    )

    def write_signal() -> None:
        nonlocal writer_error
        try:
            position = 0
            while position < len(playback):
                end = min(position + CHUNK_SIZE, len(playback))
                started = monotonic()
                stream_time_before = float(output_stream.time)
                underflowed = output_stream.write(playback[position:end])
                finished = monotonic()
                write_log.append(
                    {
                        "start": position,
                        "end": end,
                        "frames": end - position,
                        "wall_seconds": finished - started,
                        "stream_time_before": stream_time_before,
                        "stream_time_after": float(output_stream.time),
                        "underflowed": bool(underflowed),
                    }
                )
                if underflowed:
                    raise RuntimeError(f"Output underflow at frame {position}")
                position = end
        except BaseException as error:
            with writer_error_lock:
                writer_error = error

    started_at = monotonic()
    writer: Thread | None = None
    try:
        input_stream.start()
        output_stream.start()
        writer = Thread(target=write_signal, name="sr-rw-output", daemon=True)
        writer.start()

        recorded = 0
        while recorded < target_samples:
            with writer_error_lock:
                error = writer_error
            if error is not None:
                raise error
            frames = min(CHUNK_SIZE, target_samples - recorded)
            started = monotonic()
            stream_time_before = float(input_stream.time)
            data, overflowed = input_stream.read(frames)
            finished = monotonic()
            block = np.asarray(data, dtype=np.float32).copy()
            read_log.append(
                {
                    "start": recorded,
                    "end": recorded + len(block),
                    "requested_frames": frames,
                    "returned_frames": len(block),
                    "wall_seconds": finished - started,
                    "stream_time_before": stream_time_before,
                    "stream_time_after": float(input_stream.time),
                    "overflowed": bool(overflowed),
                }
            )
            if overflowed:
                raise RuntimeError(f"Input overflow at frame {recorded}")
            chunks.append(block)
            recorded += len(block)

        if writer is not None:
            writer.join(timeout=5.0)
            if writer.is_alive():
                raise RuntimeError("Output writer did not finish")
        with writer_error_lock:
            error = writer_error
        if error is not None:
            raise error
    finally:
        for stream in (output_stream, input_stream):
            try:
                stream.stop()
            finally:
                stream.close()

    recording = np.concatenate(chunks, axis=0)
    metadata = {
        "input_device": f"[{input_index}] {input_device['name']}",
        "output_device": f"[{output_index}] {output_device['name']}",
        "requested_sample_rate": SAMPLE_RATE,
        "input_stream_sample_rate": float(input_stream.samplerate),
        "output_stream_sample_rate": float(output_stream.samplerate),
        "stream_blocksize": STREAM_BLOCK_SIZE,
        "chunk_size": CHUNK_SIZE,
        "elapsed_seconds": monotonic() - started_at,
        "read_log": read_log,
        "write_log": write_log,
    }
    return playback, recording, metadata


def save_results(
    playback: np.ndarray,
    recording: np.ndarray,
    metadata: dict,
) -> None:
    np.savez_compressed(
        DATA_PATH,
        playback=playback,
        recording=recording,
        sample_rate=np.int64(SAMPLE_RATE),
    )
    INFO_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def plot_recording(recording: np.ndarray, *, show: bool) -> None:
    time = np.arange(len(recording), dtype=np.float64) / SAMPLE_RATE
    channel = recording[:, 0]
    figure, (overview, zoom) = plt.subplots(2, 1, figsize=(14, 8))
    overview.plot(time, channel, linewidth=0.45, color="tab:cyan")
    overview.set_xlim(0.0, len(recording) / SAMPLE_RATE)
    overview.set_title("Blocking WASAPI recording: OUT 1 → IN 1")
    overview.set_ylabel("Amplitude")
    overview.grid(True, alpha=0.25)

    zoom.set_xlim(0.8, 1.2)
    zoom.set_ylim(-1.05, 1.05)
    zoom.plot(time, channel, linewidth=0.8, color="tab:cyan")
    for boundary in np.arange(0.8, 1.201, CHUNK_SIZE / SAMPLE_RATE):
        zoom.axvline(boundary, color="tab:red", linewidth=0.7, alpha=0.35)
    zoom.set_title("Raw IN 1; red lines = 19200-frame read boundaries")
    zoom.set_xlabel("Recording time, s")
    zoom.set_ylabel("Amplitude")
    zoom.grid(True, alpha=0.25)

    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    playback, recording, metadata = run_test()
    save_results(playback, recording, metadata)
    plot_recording(recording, show=not args.no_show)
    print(f"Input:  {metadata['input_device']}")
    print(f"Output: {metadata['output_device']}")
    print(f"Recorded {len(recording)} frames in {metadata['elapsed_seconds']:.3f} s")
    print(f"Data: {DATA_PATH}")
    print(f"Plot: {PLOT_PATH}")
    print(f"Log:  {INFO_PATH}")


if __name__ == "__main__":
    main()
