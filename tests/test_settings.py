import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from spectrum_app.settings import (
    AppSettings,
    AudioSettings,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    load_settings,
    resolve_device,
    validate_audio_settings,
)


class SettingsTests(unittest.TestCase):
    def test_settings_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings = AppSettings(
                audio=AudioSettings(
                    input_device="3: input",
                    output_device="5: output",
                    block_size=2048,
                ),
                path=path,
            )
            settings.save()

            loaded = load_settings(path)

            self.assertEqual(loaded.audio, settings.audio)

    def test_missing_devices_are_reset_to_default_and_saved(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings = AppSettings(
                audio=AudioSettings(
                    input_device="3: missing input",
                    output_device="5: missing output",
                ),
                path=path,
            )

            devices = validate_audio_settings(settings, [], [])

            self.assertEqual(devices, (None, None))
            self.assertEqual(settings.audio.input_device, DEFAULT_INPUT)
            self.assertEqual(settings.audio.output_device, DEFAULT_OUTPUT)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["audio"]["input_device"], DEFAULT_INPUT)

    def test_device_is_recovered_when_portaudio_index_changes(self) -> None:
        saved = "3: USB Audio, Windows WASAPI, 2>>0, 48.0 kHz"
        current = "7: USB Audio, Windows WASAPI, 2>>0, 48.0 kHz"

        name, index = resolve_device(saved, [current], DEFAULT_INPUT)

        self.assertEqual(name, current)
        self.assertEqual(index, 7)

    def test_invalid_settings_file_uses_defaults(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text("{broken", encoding="utf-8")

            settings = load_settings(path)

            self.assertEqual(settings.audio, AudioSettings())


if __name__ == "__main__":
    unittest.main()
