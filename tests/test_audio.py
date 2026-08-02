from types import SimpleNamespace
import unittest

import numpy as np

from spectrum_app.core.audio import AudioError, AudioInput, AudioOutput, AudioService
from spectrum_app.core.settings import AppSettings


class FakeStream:
    def __init__(self, **kwargs) -> None:
        self.config = kwargs
        self.started = False
        self.stopped = False
        self.closed = False
        self.writes: list[np.ndarray] = []
        self.overflow = False
        self.underflow = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def read(self, samples: int) -> tuple[np.ndarray, bool]:
        channels = self.config["channels"]
        return np.zeros((samples, channels), dtype=np.float32), self.overflow

    def write(self, data: np.ndarray) -> bool:
        self.writes.append(data)
        return self.underflow


class FakeSoundDevice:
    def __init__(self) -> None:
        self.devices = [
            {
                "name": "Microphone",
                "index": 0,
                "hostapi": 0,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48_000.0,
            },
            {
                "name": "Interface",
                "index": 1,
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 2,
                "default_samplerate": 96_000.0,
            },
        ]
        self.host_apis: list[dict[str, object]] = [
            {"name": "Windows WASAPI"}
        ]
        self.default = SimpleNamespace(device=(0, 1))
        self.terminate_calls = 0
        self.initialize_calls = 0
        self.input_streams: list[FakeStream] = []
        self.output_streams: list[FakeStream] = []

    def _terminate(self) -> None:
        self.terminate_calls += 1

    def _initialize(self) -> None:
        self.initialize_calls += 1

    def query_hostapis(self) -> list[dict[str, object]]:
        return self.host_apis

    def query_devices(self, device=None, kind=None):
        if kind is None:
            return self.devices
        index = self.default.device[0 if kind == "input" else 1]
        return self.devices[index]

    def InputStream(self, **kwargs) -> FakeStream:
        stream = FakeStream(**kwargs)
        self.input_streams.append(stream)
        return stream

    def OutputStream(self, **kwargs) -> FakeStream:
        stream = FakeStream(**kwargs)
        self.output_streams.append(stream)
        return stream


class AudioServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AppSettings()
        self.backend = FakeSoundDevice()
        self.service = AudioService(
            self.settings,
            backend=self.backend,
            update_interval=60.0,
        )
        self.audio_input = AudioInput(self.service)
        self.audio_output = AudioOutput(self.service)
        self.service.start()

    def tearDown(self) -> None:
        self.service.shutdown()

    def test_device_defaults_are_available_before_open(self) -> None:
        self.assertEqual(self.audio_input.sample_rate, 48_000)
        self.assertEqual(self.audio_input.channels, 2)
        self.assertEqual(self.audio_output.sample_rate, 96_000)
        self.assertEqual(self.audio_output.channels, 2)

        second_input = self.service.input_devices[1]
        self.settings.input_device = second_input.id

        self.assertEqual(self.audio_input.sample_rate, 96_000)
        self.assertEqual(self.audio_input.channels, 1)

    def test_stream_uses_device_defaults_and_recommended_block_is_metadata(self) -> None:
        self.settings.input_block_size = 4096

        self.assertTrue(self.audio_input.open())

        stream = self.backend.input_streams[-1]
        self.assertEqual(stream.config["device"], 0)
        self.assertEqual(stream.config["samplerate"], 48_000)
        self.assertEqual(stream.config["channels"], 2)
        self.assertEqual(stream.config["blocksize"], 0)
        self.assertEqual(self.audio_input.block_size, 4096)

        data = self.audio_input.read(37)
        self.assertEqual(data.shape, (37, 2))

    def test_device_refresh_is_paused_while_a_stream_is_open(self) -> None:
        refreshes = self.backend.terminate_calls
        self.assertTrue(self.audio_input.open())

        self.service._refresh_devices()
        self.assertEqual(self.backend.terminate_calls, refreshes)

        self.assertTrue(self.audio_input.close())
        self.service._refresh_devices()
        self.assertEqual(self.backend.terminate_calls, refreshes + 1)

    def test_output_accepts_arbitrary_array_lengths(self) -> None:
        self.assertTrue(self.audio_output.open())

        self.audio_output.write(np.ones((13, 2), dtype=np.float64))

        written = self.backend.output_streams[-1].writes[-1]
        self.assertEqual(written.shape, (13, 2))
        self.assertEqual(written.dtype, np.float32)

    def test_stream_error_closes_the_session_and_raises_audio_error(self) -> None:
        self.assertTrue(self.audio_input.open())
        stream = self.backend.input_streams[-1]
        stream.overflow = True

        with self.assertRaisesRegex(AudioError, "overflow"):
            self.audio_input.read(16)

        self.assertTrue(stream.closed)
        self.assertTrue(self.audio_input.close())

    def test_shutdown_closes_every_open_stream(self) -> None:
        self.assertTrue(self.audio_input.open())
        self.assertTrue(self.audio_output.open())

        self.service.shutdown()

        self.assertTrue(self.backend.input_streams[-1].closed)
        self.assertTrue(self.backend.output_streams[-1].closed)


if __name__ == "__main__":
    unittest.main()
