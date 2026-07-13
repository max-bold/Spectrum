from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    host_api: str
    input_channels: int
    output_channels: int
    default_sample_rate: float


def list_devices(kind: Literal["input", "output"] | None = None) -> list[AudioDevice]:
    """List PortAudio devices as reusable data objects."""
    hostapis = sd.query_hostapis()
    devices = sd.query_devices()
    hostapi_names = [str(item["name"]) for item in hostapis if isinstance(item, dict)]
    result: list[AudioDevice] = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        host_api_index = int(device["hostapi"])
        if host_api_index >= len(hostapi_names):
            continue
        audio_device = AudioDevice(
            index=int(device["index"]),
            name=str(device["name"]),
            host_api=hostapi_names[host_api_index],
            input_channels=int(device["max_input_channels"]),
            output_channels=int(device["max_output_channels"]),
            default_sample_rate=float(device["default_samplerate"]),
        )
        if kind == "input" and audio_device.input_channels <= 0:
            continue
        if kind == "output" and audio_device.output_channels <= 0:
            continue
        result.append(audio_device)
    return result


def play_and_record(
    signal: NDArray[np.floating],
    *,
    sample_rate: int,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    channels: int = 2,
    blocking: bool = True,
) -> NDArray[np.float32]:
    """Play a signal and record synchronized input through sounddevice."""
    recording = sd.playrec(
        np.asarray(signal, dtype=np.float32),
        samplerate=sample_rate,
        channels=channels,
        device=(input_device, output_device),
        blocking=blocking,
    )
    return np.asarray(recording, dtype=np.float32)
