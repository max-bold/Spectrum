import unittest
from threading import Event
from unittest.mock import patch

import numpy as np

import audioanalysis.generators as generators_module
from audioanalysis import (
    ASignal,
    FrequencyBand,
    PinkNoiseThread,
    SpectrumConfig,
    SmoothingWindow,
    analyze_spectrum,
    as_channels,
    grid_smooth,
    log_chirp,
    log_smooth,
    normalize_peak,
    peak_levels,
    pink_noise,
    pink_noise_zi,
    white_noise,
)
from audioanalysis.impedance import calculate_impedance_from_transfer
from scipy.signal import sosfreqz


class AudioAnalysisPackageTests(unittest.TestCase):
    def test_asignal_wraps_arrays_as_samples_by_channels(self) -> None:
        mono = ASignal(np.array([0.1, -0.5, 0.2], dtype=np.float32), 48_000)
        tuple_mono = ASignal((0.1, -0.5, 0.2), 48_000)
        stereo = ASignal(
            np.array(
                [
                    [0.1, -0.2],
                    [-0.5, 0.4],
                    [0.2, -0.8],
                ],
                dtype=np.float32,
            ),
            48_000,
        )

        self.assertEqual(mono.as_array().shape, (3, 1))
        self.assertEqual(tuple_mono.as_array().shape, (3, 1))
        self.assertEqual(stereo.as_array().shape, (3, 2))
        self.assertEqual(mono.sample_count, 3)
        self.assertEqual(mono.sample_rate, 48_000)
        self.assertEqual(stereo.channel_count, 2)

    def test_asignal_accepts_any_positive_channel_count(self) -> None:
        capture = ASignal(np.zeros((128, 4), dtype=np.float32), 48_000)
        rec_ref_gen = ASignal(np.zeros((128, 3), dtype=np.float32), 48_000)

        self.assertEqual(capture.as_array().shape, (128, 4))
        self.assertEqual(capture.channel_count, 4)
        self.assertEqual(rec_ref_gen.as_array().shape, (128, 3))
        self.assertEqual(rec_ref_gen.channel_count, 3)

    def test_asignal_combines_signals_by_channels(self) -> None:
        left = ASignal(np.array([0.1, -0.5, 0.2], dtype=np.float32), 48_000)
        right = ASignal(np.array([[0.2], [0.4], [-0.8]], dtype=np.float32), 48_000)

        combined = ASignal((left, right))

        self.assertEqual(combined.as_array().shape, (3, 2))
        self.assertEqual(combined.sample_rate, 48_000)
        np.testing.assert_allclose(combined.as_array()[:, 0], [0.1, -0.5, 0.2])
        np.testing.assert_allclose(combined.as_array()[:, 1], [0.2, 0.4, -0.8])

    def test_asignal_rejects_combining_different_sample_rates(self) -> None:
        left = ASignal(np.array([0.1, -0.5], dtype=np.float32), 48_000)
        right = ASignal(np.array([0.2, 0.4], dtype=np.float32), 44_100)

        with self.assertRaisesRegex(ValueError, "sample rates"):
            ASignal((left, right))

    def test_asignal_returns_channel_as_mono_signal(self) -> None:
        signal = ASignal(np.array([[0.1, -0.2], [-0.5, 0.4]], dtype=np.float32), 48_000)

        channel = signal[1]

        self.assertEqual(channel.as_array().shape, (2, 1))
        self.assertEqual(channel.sample_rate, 48_000)
        np.testing.assert_allclose(channel.as_array()[:, 0], [-0.2, 0.4])

    def test_asignal_to_channels_requires_mono_signal(self) -> None:
        mono = ASignal(np.array([0.1, -0.5], dtype=np.float32), 48_000)
        stereo = ASignal(np.array([[0.1, -0.2], [-0.5, 0.4]], dtype=np.float32), 48_000)

        duplicated = mono.to_channels(3)

        self.assertEqual(duplicated.as_array().shape, (2, 3))
        self.assertEqual(duplicated.sample_rate, 48_000)
        np.testing.assert_allclose(duplicated.as_array()[0], [0.1, 0.1, 0.1])
        with self.assertRaisesRegex(ValueError, "mono"):
            stereo.to_channels(3)

    def test_asignal_normalizes_globally_and_per_channel(self) -> None:
        signal = ASignal(
            np.array(
                [
                    [0.5, 2.0, 0.0],
                    [-1.0, -4.0, 0.0],
                ],
                dtype=np.float32,
            ),
            48_000,
        )

        shared = signal.normalize(1.0)
        per_channel = signal.normalize(1.0, per_channel=True)

        np.testing.assert_allclose(shared.max(), [0.25, 1.0, 0.0])
        np.testing.assert_allclose(per_channel.max(), [1.0, 1.0, 0.0])

    def test_asignal_as_array_returns_copy_with_requested_dtype(self) -> None:
        signal = ASignal(np.array([0.1, -0.5], dtype=np.float32), 48_000)

        data = signal.as_array(np.float64)
        data[0, 0] = 10.0

        self.assertEqual(data.dtype, np.float64)
        np.testing.assert_allclose(signal.as_array()[:, 0], [0.1, -0.5])

    def test_asignal_peak_levels_returns_chunks_by_channels(self) -> None:
        signal = ASignal(
            np.array(
                [
                    [0.1, -0.2],
                    [-0.4, 0.3],
                    [0.2, -0.8],
                    [-0.7, 0.1],
                    [0.6, -0.5],
                ],
                dtype=np.float32,
            ),
            48_000,
        )

        levels = signal.peak_levels(2)

        self.assertEqual(levels.shape, (3, 2))
        np.testing.assert_allclose(levels, [[0.4, 0.3], [0.7, 0.8], [0.6, 0.5]])

    def test_asignal_trims_by_sample_range(self) -> None:
        signal = ASignal(
            np.array(
                [
                    [0.0, 0.1],
                    [1.0, 1.1],
                    [2.0, 2.1],
                    [3.0, 3.1],
                ],
                dtype=np.float32,
            ),
            48_000,
        )

        trimmed = signal.trim(2, start=1)

        self.assertEqual(trimmed.sample_rate, 48_000)
        np.testing.assert_allclose(trimmed.as_array(), [[1.0, 1.1], [2.0, 2.1]])

    def test_asignal_applies_linear_fades(self) -> None:
        signal = ASignal(np.ones((5, 2), dtype=np.float32), 48_000)

        faded = signal.fade(3, 2)

        np.testing.assert_allclose(
            faded.as_array(),
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [1.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
        )

    def test_asignal_pads_with_zero_samples(self) -> None:
        signal = ASignal(np.array([[0.5, -0.5], [1.0, -1.0]], dtype=np.float32), 48_000)

        padded = signal.pad(1, 2)

        self.assertEqual(padded.sample_rate, 48_000)
        np.testing.assert_allclose(
            padded.as_array(),
            [
                [0.0, 0.0],
                [0.5, -0.5],
                [1.0, -1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        )

    def test_generators_return_requested_channels(self) -> None:
        chirp = log_chirp(
            480,
            48_000,
            FrequencyBand(100.0, 10_000.0),
            channels=2,
        )
        noise = pink_noise(
            480,
            48_000,
            FrequencyBand(100.0, 10_000.0),
            channels=1,
            rng=np.random.default_rng(1),
        )
        white = white_noise(
            480,
            48_000,
            channels=3,
            rng=np.random.default_rng(2),
        )

        self.assertIsInstance(chirp, ASignal)
        self.assertIsInstance(noise, ASignal)
        self.assertIsInstance(white, ASignal)
        self.assertEqual(chirp.as_array().shape, (2528, 2))
        self.assertEqual(noise.as_array().shape, (2528, 1))
        self.assertEqual(white.as_array().shape, (2528, 3))
        self.assertEqual(chirp.sample_rate, 48_000)
        self.assertEqual(noise.sample_rate, 48_000)
        self.assertEqual(white.sample_rate, 48_000)
        self.assertLessEqual(float(np.max(np.abs(chirp.as_array()))), 0.900001)
        self.assertLessEqual(float(np.max(np.abs(noise.as_array()))), 0.900001)
        self.assertLessEqual(float(np.max(np.abs(white.as_array()))), 0.900001)

    def test_generators_can_disable_padding_and_fade(self) -> None:
        chirp = log_chirp(
            480,
            48_000,
            FrequencyBand(100.0, 10_000.0),
            pad=0,
            fade=0,
        )

        self.assertEqual(chirp.as_array().shape, (480, 1))

    def test_generators_apply_zero_padding(self) -> None:
        signal = white_noise(
            16,
            48_000,
            channels=1,
            rng=np.random.default_rng(3),
            pad=4,
            fade=0,
        )

        self.assertEqual(signal.as_array().shape, (24, 1))
        np.testing.assert_allclose(signal.as_array()[:4], 0.0)
        np.testing.assert_allclose(signal.as_array()[-4:], 0.0)

    def test_pink_noise_can_return_filter_state(self) -> None:
        zi = pink_noise_zi(48_000, FrequencyBand(100.0, 10_000.0))

        signal, zf = pink_noise(
            32,
            48_000,
            FrequencyBand(100.0, 10_000.0),
            rng=np.random.default_rng(5),
            pad=0,
            fade=0,
            zi=zi,
        )

        self.assertIsInstance(signal, ASignal)
        self.assertEqual(signal.as_array().shape, (32, 1))
        self.assertEqual(zf.shape, zi.shape)

    def test_pink_noise_does_not_normalize_each_generated_block(self) -> None:
        short = pink_noise(
            480,
            48_000,
            FrequencyBand(100.0, 10_000.0),
            amplitude=0.9,
            rng=np.random.default_rng(5),
            pad=0,
            fade=0,
        )
        long = pink_noise(
            960,
            48_000,
            FrequencyBand(100.0, 10_000.0),
            amplitude=0.9,
            rng=np.random.default_rng(5),
            pad=0,
            fade=0,
        )

        np.testing.assert_allclose(short.as_array(), long.as_array()[:480])

    def test_pinking_filter_tracks_physical_frequencies_across_sample_rates(
        self,
    ) -> None:
        frequency = np.geomspace(30.0, 15_000.0, 512)
        _, reference = sosfreqz(
            generators_module.PINKING_SOS,
            worN=frequency,
            fs=generators_module.PINKING_REFERENCE_SAMPLE_RATE,
        )

        for sample_rate in (48_000, 96_000, 192_000):
            with self.subTest(sample_rate=sample_rate):
                _, actual = sosfreqz(
                    generators_module._pinking_sos(sample_rate),
                    worN=frequency,
                    fs=sample_rate,
                )
                difference_db = 20.0 * np.log10(np.abs(actual / reference))
                self.assertLess(float(np.max(np.abs(difference_db))), 0.4)

    def test_pinking_filter_is_unchanged_at_reference_sample_rate(self) -> None:
        actual = generators_module._pinking_sos(
            generators_module.PINKING_REFERENCE_SAMPLE_RATE
        )

        np.testing.assert_array_equal(actual, generators_module.PINKING_SOS)

    def test_pink_noise_thread_does_not_normalize_filtered_blocks(self) -> None:
        raw_peaks = iter((0.5, 1.0))
        amplitudes: list[float] = []

        def fake_pink_noise(*args: object, **kwargs: object) -> tuple[ASignal, np.ndarray]:
            samples = int(args[0])
            sample_rate = int(args[1])
            channels = int(kwargs["channels"])
            amplitudes.append(float(kwargs["amplitude"]))
            data = np.full((samples, channels), next(raw_peaks), dtype=np.float32)
            return ASignal(data, sample_rate), np.asarray(kwargs["zi"])

        with (
            patch.object(
                generators_module.sd,
                "query_devices",
                return_value={
                    "default_samplerate": 1_000.0,
                    "max_output_channels": 2,
                },
            ),
            patch.object(generators_module, "pink_noise", side_effect=fake_pink_noise),
        ):
            thread = PinkNoiseThread(
                band=FrequencyBand(20.0, 400.0),
                amplitude=0.25,
                block_size=100,
                pad=0.0,
                fade=0.0,
            )
            first = thread._next_block(100)
            second = thread._next_block(100)

        np.testing.assert_allclose(first, 0.5)
        np.testing.assert_allclose(second, 1.0)
        self.assertEqual(amplitudes, [0.25, 0.25])

    def test_pink_noise_thread_writes_blocks_to_output_stream(self) -> None:
        instances: list[object] = []
        holder: dict[str, PinkNoiseThread] = {}
        write_count = 0
        first_stop = Event()
        second_stop = Event()

        class FakeOutputStream:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.writes: list[np.ndarray] = []
                self.stopped = False
                instances.append(self)

            def __enter__(self) -> "FakeOutputStream":
                return self

            def __exit__(
                self,
                exc_type: object,
                exc: object,
                traceback: object,
            ) -> None:
                return None

            def write(self, data: np.ndarray) -> None:
                nonlocal write_count
                self.writes.append(data.copy())
                write_count += 1
                if write_count == 3:
                    holder["thread"].stop()
                    first_stop.set()
                if write_count == 6:
                    holder["thread"].stop()
                    second_stop.set()

            def stop(self) -> None:
                self.stopped = True

        with (
            patch.object(
                generators_module.sd,
                "query_devices",
                return_value={
                    "default_samplerate": 48_000.0,
                    "max_output_channels": 2,
                },
            ),
            patch.object(generators_module.sd, "OutputStream", FakeOutputStream),
        ):
            thread = PinkNoiseThread(
                device=7,
                band=FrequencyBand(100.0, 10_000.0),
                amplitude=0.25,
                block_size=32,
                pad=0.0,
                fade=0.0,
            )
            holder["thread"] = thread

            thread.start()
            self.assertTrue(first_stop.wait(timeout=2.0))
            self.assertTrue(thread.is_alive())
            thread.start()
            self.assertTrue(second_stop.wait(timeout=2.0))
            thread.close(timeout=2.0)
            thread.raise_if_failed()

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(instances), 2)
        stream = instances[0]
        self.assertIsInstance(stream, FakeOutputStream)
        assert isinstance(stream, FakeOutputStream)
        self.assertEqual(stream.kwargs["samplerate"], 48_000)
        self.assertEqual(stream.kwargs["device"], 7)
        self.assertEqual(stream.kwargs["channels"], 2)
        self.assertEqual(stream.kwargs["blocksize"], 32)
        self.assertEqual(stream.kwargs["dtype"], "float32")
        self.assertGreaterEqual(write_count, 6)
        for instance in instances:
            self.assertIsInstance(instance, FakeOutputStream)
            assert isinstance(instance, FakeOutputStream)
            self.assertTrue(instance.stopped)
            self.assertGreaterEqual(len(instance.writes), 3)
            for block in instance.writes:
                self.assertEqual(block.shape, (32, 2))
                self.assertEqual(block.dtype, np.float32)
                self.assertLessEqual(float(np.max(np.abs(block))), 0.250001)

    def test_pink_noise_thread_refreshes_device_defaults_on_restart(self) -> None:
        instances: list[object] = []
        holder: dict[str, PinkNoiseThread] = {}
        write_count = 0
        first_stop = Event()
        second_stop = Event()
        device_defaults = [
            {"default_samplerate": 48_000.0, "max_output_channels": 2},
            {"default_samplerate": 48_000.0, "max_output_channels": 2},
            {"default_samplerate": 44_100.0, "max_output_channels": 1},
        ]

        class FakeOutputStream:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.writes: list[np.ndarray] = []
                instances.append(self)

            def __enter__(self) -> "FakeOutputStream":
                return self

            def __exit__(
                self,
                exc_type: object,
                exc: object,
                traceback: object,
            ) -> None:
                return None

            def write(self, data: np.ndarray) -> None:
                nonlocal write_count
                self.writes.append(data.copy())
                write_count += 1
                if write_count == 1:
                    holder["thread"].stop()
                    first_stop.set()
                if write_count == 2:
                    holder["thread"].stop()
                    second_stop.set()

        with (
            patch.object(
                generators_module.sd,
                "query_devices",
                side_effect=device_defaults,
            ),
            patch.object(generators_module.sd, "OutputStream", FakeOutputStream),
        ):
            thread = PinkNoiseThread(
                band=FrequencyBand(100.0, 10_000.0),
                block_size=32,
                pad=0.0,
                fade=0.0,
            )
            holder["thread"] = thread

            thread.start()
            self.assertTrue(first_stop.wait(timeout=2.0))
            thread.start()
            self.assertTrue(second_stop.wait(timeout=2.0))
            thread.close(timeout=2.0)
            thread.raise_if_failed()

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(instances), 2)
        first_stream = instances[0]
        second_stream = instances[1]
        self.assertIsInstance(first_stream, FakeOutputStream)
        self.assertIsInstance(second_stream, FakeOutputStream)
        assert isinstance(first_stream, FakeOutputStream)
        assert isinstance(second_stream, FakeOutputStream)
        self.assertEqual(first_stream.kwargs["samplerate"], 48_000)
        self.assertEqual(first_stream.kwargs["channels"], 2)
        self.assertEqual(first_stream.writes[0].shape, (32, 2))
        self.assertEqual(second_stream.kwargs["samplerate"], 44_100)
        self.assertEqual(second_stream.kwargs["channels"], 1)
        self.assertEqual(second_stream.writes[0].shape, (32, 1))

    def test_pink_noise_thread_rejects_start_after_close(self) -> None:
        with patch.object(
            generators_module.sd,
            "query_devices",
            return_value={
                "default_samplerate": 48_000.0,
                "max_output_channels": 2,
            },
        ):
            thread = PinkNoiseThread(block_size=32, pad=0.0, fade=0.0)

        thread.close(timeout=2.0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            thread.start()

    def test_pink_noise_thread_reports_missing_device_in_init(self) -> None:
        errors = [
            generators_module.sd.PortAudioError("Error querying device 999999"),
            ValueError("No output device matching 'missing'"),
        ]

        for error in errors:
            with self.subTest(error=type(error).__name__):
                with patch.object(
                    generators_module.sd,
                    "query_devices",
                    side_effect=error,
                ):
                    with self.assertRaisesRegex(ValueError, "No such device"):
                        PinkNoiseThread(device=999999)

    def test_pink_noise_thread_reports_missing_device_on_restart(self) -> None:
        instances: list[object] = []
        holder: dict[str, PinkNoiseThread] = {}
        first_stop = Event()

        class FakeOutputStream:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.writes: list[np.ndarray] = []
                instances.append(self)

            def __enter__(self) -> "FakeOutputStream":
                return self

            def __exit__(
                self,
                exc_type: object,
                exc: object,
                traceback: object,
            ) -> None:
                return None

            def write(self, data: np.ndarray) -> None:
                self.writes.append(data.copy())
                holder["thread"].stop()
                first_stop.set()

        with (
            patch.object(
                generators_module.sd,
                "query_devices",
                side_effect=[
                    {"default_samplerate": 48_000.0, "max_output_channels": 2},
                    {"default_samplerate": 48_000.0, "max_output_channels": 2},
                    generators_module.sd.PortAudioError("Error querying device 7"),
                ],
            ),
            patch.object(generators_module.sd, "OutputStream", FakeOutputStream),
        ):
            thread = PinkNoiseThread(device=7, block_size=32, pad=0.0, fade=0.0)
            holder["thread"] = thread

            thread.start()
            self.assertTrue(first_stop.wait(timeout=2.0))
            thread.start()
            thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(instances), 1)
        with self.assertRaisesRegex(ValueError, "No such device"):
            thread.raise_if_failed()

    def test_pink_noise_thread_validates_output_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "Block size"):
            PinkNoiseThread(block_size=0)
        with patch.object(
            generators_module.sd,
            "query_devices",
            return_value={
                "default_samplerate": 48_000.0,
                "max_output_channels": 0,
            },
        ):
            with self.assertRaisesRegex(ValueError, "Channel count"):
                PinkNoiseThread()

    def test_pink_noise_thread_uses_output_device_defaults(self) -> None:
        with patch.object(
            generators_module.sd,
            "query_devices",
            return_value={
                "default_samplerate": 48_000.0,
                "max_output_channels": 4,
            },
        ):
            thread = PinkNoiseThread(
                device=3,
                band=FrequencyBand(100.0, 10_000.0),
                block_size=32,
                pad=0.0,
                fade=0.0,
            )

        self.assertEqual(thread.sample_rate, 48_000)
        self.assertEqual(thread.channels, 4)
        self.assertEqual(thread.pad_samples, 0)
        self.assertEqual(thread.fade_samples, 0)

    def test_pink_noise_thread_builds_fade_and_silence_stop_tail(self) -> None:
        with patch.object(
            generators_module.sd,
            "query_devices",
            return_value={
                "default_samplerate": 44_100.0,
                "max_output_channels": 2,
            },
        ):
            thread = PinkNoiseThread(block_size=40_000, fade=1.5, pad=0.2)

        tail = thread._stop_tail()

        self.assertEqual(tail.shape, (74_970, 2))
        np.testing.assert_allclose(tail[-8_820:], 0.0)
        self.assertLess(abs(float(tail[-8_821, 0])), 1e-6)

    def test_pink_noise_thread_waits_until_buffer_is_half_block(self) -> None:
        sleeps: list[float] = []
        with patch.object(
            generators_module.sd,
            "query_devices",
            return_value={
                "default_samplerate": 1_000.0,
                "max_output_channels": 2,
            },
        ):
            thread = PinkNoiseThread(
                band=FrequencyBand(20.0, 400.0),
                block_size=100,
                fade=0.0,
                pad=0.0,
            )

        with (
            patch.object(
                generators_module.time,
                "monotonic",
                side_effect=[0.0, 0.05],
            ),
            patch.object(generators_module.time, "sleep", side_effect=sleeps.append),
        ):
            ready = thread._wait_for_buffer_room(0.0, 100)

        self.assertTrue(ready)
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 0.01)

    def test_normalize_peak_can_normalize_each_channel(self) -> None:
        samples_by_channels = np.array(
            [
                [0.5, 2.0, 0.0],
                [-1.0, -4.0, 0.0],
            ],
            dtype=np.float32,
        )
        channels_by_samples = samples_by_channels.T

        shared = normalize_peak(samples_by_channels, peak=1.0)
        per_channel = normalize_peak(
            samples_by_channels,
            peak=1.0,
            per_channel=True,
        )
        channel_first = normalize_peak(
            channels_by_samples,
            peak=1.0,
            per_channel=True,
            axis=1,
        )

        np.testing.assert_allclose(np.max(np.abs(shared), axis=0), [0.25, 1.0, 0.0])
        np.testing.assert_allclose(
            np.max(np.abs(per_channel), axis=0),
            [1.0, 1.0, 0.0],
        )
        np.testing.assert_allclose(
            np.max(np.abs(channel_first), axis=1),
            [1.0, 1.0, 0.0],
        )

    def test_normalize_peak_rejects_invalid_per_channel_axis(self) -> None:
        signal = np.zeros((2, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "out of bounds"):
            normalize_peak(signal, per_channel=True, axis=2)

    def test_normalize_peak_ignores_per_channel_for_mono_signal(self) -> None:
        signal = np.array([0.5, -2.0, 1.0], dtype=np.float32)

        shared = normalize_peak(signal, peak=1.0)
        per_channel = normalize_peak(signal, peak=1.0, per_channel=True)

        np.testing.assert_allclose(per_channel, shared)

    def test_levels_match_existing_shape_contract(self) -> None:
        record = np.array([[0.1], [-0.5], [0.2]], dtype=np.float32)

        timestamps, levels = peak_levels(record, sample_rate=3, time_step=1.0)

        np.testing.assert_allclose(timestamps, [0.0])
        np.testing.assert_allclose(levels, [[0.5], [0.5]])
        self.assertEqual(as_channels(record).shape, (3, 2))

    def test_as_channels_returns_mono_as_samples_by_channels(self) -> None:
        record = np.array([[0.1], [-0.5], [0.2]], dtype=np.float32)

        mono = as_channels(record, channels=1)

        self.assertEqual(mono.shape, (3, 1))
        np.testing.assert_allclose(mono[:, 0], [0.1, -0.5, 0.2])

    def test_grid_smooth_preserves_complex_values(self) -> None:
        frequency = np.linspace(0.0, 1000.0, 513)
        values = np.linspace(1.0, 2.0, 513) + 1j * np.linspace(-1.0, 1.0, 513)
        grid = np.geomspace(20.0, 900.0, 32)

        smoothed = grid_smooth(
            frequency,
            values,
            grid,
            window=SmoothingWindow.GAUSSIAN,
        )

        self.assertTrue(np.iscomplexobj(smoothed))
        self.assertEqual(smoothed.shape, grid.shape)

    def test_log_smooth_preserves_channel_axis(self) -> None:
        frequency = np.linspace(0.0, 24_000.0, 1025)
        values = np.column_stack(
            [
                np.linspace(1.0, 2.0, 1025),
                np.linspace(2.0, 4.0, 1025),
            ],
        )

        grid, smoothed = log_smooth(
            frequency,
            values,
            band=(20.0, 20_000.0),
            width=1 / 10,
            points=128,
        )

        self.assertEqual(grid.shape, (128,))
        self.assertEqual(smoothed.shape, (128, 2))

    def test_spectrum_analyzer_returns_log_grid(self) -> None:
        sample_rate = 48_000
        time = np.arange(sample_rate) / sample_rate
        signal = np.sin(2 * np.pi * 1000.0 * time).astype(np.float32)[:, None]

        result = analyze_spectrum(
            signal,
            SpectrumConfig(
                sample_rate=sample_rate,
                band=FrequencyBand(100.0, 10_000.0),
                points=64,
            ),
        )

        self.assertEqual(result.frequency.size, 64)
        self.assertEqual(result.values.size, 64)

    def test_spectrum_analyzer_requires_samples_by_channels_record(self) -> None:
        signal = np.zeros(48_000, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "samples, channels"):
            analyze_spectrum(signal, SpectrumConfig(sample_rate=48_000))

    def test_impedance_from_transfer(self) -> None:
        impedance = calculate_impedance_from_transfer(np.array([0.5]), 8.0)

        np.testing.assert_allclose(impedance, [8.0 + 0.0j])


if __name__ == "__main__":
    unittest.main()
