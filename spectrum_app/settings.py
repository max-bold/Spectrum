from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from utils.audio import io_list_updater

DEFAULT_INPUT = "default input"
DEFAULT_OUTPUT = "default output"
DEFAULT_BLOCK_SIZE = 1024


@dataclass
class AudioSettings:
    input_device: str = DEFAULT_INPUT
    output_device: str = DEFAULT_OUTPUT
    block_size: int = DEFAULT_BLOCK_SIZE


@dataclass
class AppSettings:
    audio: AudioSettings = field(default_factory=AudioSettings)
    path: Path = field(default_factory=lambda: settings_path(), repr=False)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"audio": asdict(self.audio)}, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)


def settings_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "BM Spectrum" / "settings.json"


def load_settings(path: Path | None = None) -> AppSettings:
    path = path or settings_path()
    settings = AppSettings(path=path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        audio = raw.get("audio", {})
        settings.audio = AudioSettings(
            input_device=str(audio.get("input_device", DEFAULT_INPUT)),
            output_device=str(audio.get("output_device", DEFAULT_OUTPUT)),
            block_size=max(1, int(audio.get("block_size", DEFAULT_BLOCK_SIZE))),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return settings


def device_index(device_name: str, default_name: str) -> int | None:
    if not device_name or device_name == default_name:
        return None
    try:
        return int(device_name.split(":", 1)[0])
    except (TypeError, ValueError):
        return None


def resolve_device(
    saved_name: str,
    devices: list[str],
    default_name: str,
) -> tuple[str, int | None]:
    if not saved_name or saved_name == default_name:
        return default_name, None
    if saved_name in devices:
        return saved_name, device_index(saved_name, default_name)

    identity = _device_identity(saved_name)
    for device_name in devices:
        if _device_identity(device_name) == identity:
            return device_name, device_index(device_name, default_name)
    return default_name, None


def validate_audio_settings(
    settings: AppSettings,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
) -> tuple[int | None, int | None]:
    inputs = io_list_updater.list_devices("input") if inputs is None else inputs
    outputs = io_list_updater.list_devices("output") if outputs is None else outputs
    input_name, input_index = resolve_device(
        settings.audio.input_device,
        inputs,
        DEFAULT_INPUT,
    )
    output_name, output_index = resolve_device(
        settings.audio.output_device,
        outputs,
        DEFAULT_OUTPUT,
    )
    changed = (
        input_name != settings.audio.input_device
        or output_name != settings.audio.output_device
    )
    settings.audio.input_device = input_name
    settings.audio.output_device = output_name
    if changed:
        settings.save()
    return input_index, output_index


def _device_identity(device_name: str) -> str:
    return device_name.split(":", 1)[-1].strip()
