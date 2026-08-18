from threading import Event, current_thread
import time
import unittest

import numpy as np

from audioanalysis import ASignal, FrequencyBand, SemiAnalogTHDConfig
from spectrum_app.modules.impedance.jobs import ImpedanceCapture
from spectrum_app.modules.phase.jobs import PhaseAcquisition
from spectrum_app.modules.rta.jobs import RTAIOWorker, RTARuntimeConfig
from spectrum_app.modules.spectrum.jobs import SpectrumAcquisition
from spectrum_app.modules.thd.jobs import THDAcquisition


class BlockingAudioInput:
    sample_rate = 8_000
    blocksize = 64

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.active = False
        self.closed = False
        self.closed_during_io = False
        self.close_thread: str | None = None

    def open(self) -> bool:
        return True

    def read(self, samples: int) -> np.ndarray:
        self.active = True
        self.started.set()
        if not self.release.wait(2.0):
            raise TimeoutError("test input was not released")
        self.active = False
        return np.zeros((samples, 2), dtype=np.float32)

    def close(self) -> bool:
        self.closed_during_io = self.closed_during_io or self.active
        self.closed = True
        self.close_thread = current_thread().name
        return True


class BlockingAudioOutput:
    sample_rate = 8_000
    blocksize = 64

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.active = False
        self.closed = False
        self.closed_during_io = False
        self.close_thread: str | None = None

    def open(self) -> bool:
        return True

    def write(self, data: np.ndarray) -> None:
        self.active = True
        self.started.set()
        if not self.release.wait(2.0):
            raise TimeoutError("test output was not released")
        self.active = False

    def close(self) -> bool:
        self.closed_during_io = self.closed_during_io or self.active
        self.closed = True
        self.close_thread = current_thread().name
        return True


class ClippingAudioInput(BlockingAudioInput):
    def __init__(self) -> None:
        super().__init__()
        self.read_count = 0

    def read(self, samples: int) -> np.ndarray:
        self.read_count += 1
        block = np.zeros((samples, 2), dtype=np.float32)
        block[:, 0] = 1.0
        return block


class ModuleStopTests(unittest.TestCase):
    def test_normal_stop_waits_for_blocking_io_before_stream_close(self) -> None:
        cases = (
            ("spectrum-acquisition", self._spectrum_worker, True),
            ("thd-acquisition", self._thd_worker, True),
            ("impedance-capture", self._impedance_worker, True),
            ("phase-acquisition", self._phase_worker, True),
            ("rta-io", self._rta_worker, False),
        )

        for worker_name, factory, uses_output in cases:
            with self.subTest(module=worker_name):
                audio_input = BlockingAudioInput()
                audio_output = BlockingAudioOutput()
                worker = factory(audio_input, audio_output)
                worker.start()
                try:
                    self.assertTrue(audio_input.started.wait(1.0))
                    if uses_output:
                        self.assertTrue(audio_output.started.wait(1.0))

                    worker.request_stop()
                    time.sleep(0.01)
                    self.assertFalse(audio_input.closed)
                    self.assertFalse(audio_output.closed)

                    audio_input.release.set()
                    audio_output.release.set()
                    worker.join(timeout=2.0)
                    self.assertFalse(worker.is_alive())
                    self.assertTrue(audio_input.closed)
                    self.assertTrue(audio_output.closed)
                    self.assertFalse(audio_input.closed_during_io)
                    self.assertFalse(audio_output.closed_during_io)
                    self.assertEqual(audio_input.close_thread, worker_name)
                    self.assertEqual(audio_output.close_thread, worker_name)
                finally:
                    audio_input.release.set()
                    audio_output.release.set()
                    if worker.is_alive():
                        worker.join(timeout=2.0)

    def test_input_clipping_stops_every_audio_worker_on_the_current_block(self) -> None:
        cases = (
            self._spectrum_worker,
            self._thd_worker,
            self._impedance_worker,
            self._phase_worker,
            self._rta_worker,
        )

        for factory in cases:
            with self.subTest(worker=factory.__name__):
                audio_input = ClippingAudioInput()
                audio_output = BlockingAudioOutput()
                audio_output.release.set()
                worker = factory(audio_input, audio_output)
                worker.start()
                worker.join(timeout=2.0)

                self.assertFalse(worker.is_alive())
                self.assertEqual(audio_input.read_count, 1)
                self.assertTrue(audio_input.closed)
                self.assertTrue(audio_output.closed)

    @staticmethod
    def _spectrum_worker(audio_input, audio_output) -> SpectrumAcquisition:
        return SpectrumAcquisition(
            audio_input,
            audio_output,
            generator_mode="log chirp",
            band=FrequencyBand(20.0, 3_000.0),
            duration=1.0,
            pre_silence=0.0,
            post_silence=0.0,
            fade_in=0.0,
            fade_out=0.0,
            online_interval=None,
            on_level=lambda *args: None,
            on_snapshot=lambda *args: None,
            on_complete=lambda *args: None,
        )

    @staticmethod
    def _thd_worker(audio_input, audio_output) -> THDAcquisition:
        return THDAcquisition(
            audio_input,
            audio_output,
            SemiAnalogTHDConfig(
                sample_rate=8_000,
                duration=1.0,
                band=FrequencyBand(50.0, 3_000.0),
                fade_in_seconds=0.01,
                fade_out_seconds=0.01,
            ),
            on_level=lambda *args: None,
            on_complete=lambda *args: None,
        )

    @staticmethod
    def _impedance_worker(audio_input, audio_output) -> ImpedanceCapture:
        return ImpedanceCapture(
            audio_input,
            audio_output,
            ASignal(np.ones((8_000, 1), dtype=np.float32), 8_000),
            on_level=lambda *args: None,
            on_complete=lambda *args: None,
        )

    @staticmethod
    def _phase_worker(audio_input, audio_output) -> PhaseAcquisition:
        return PhaseAcquisition(
            audio_input,
            audio_output,
            band=FrequencyBand(20.0, 3_000.0),
            duration=1.0,
            pre_silence=0.0,
            post_silence=0.0,
            fade=0.01,
            on_level=lambda *args: None,
            on_complete=lambda *args: None,
        )

    @staticmethod
    def _rta_worker(audio_input, audio_output) -> RTAIOWorker:
        return RTAIOWorker(
            audio_input,
            audio_output,
            RTARuntimeConfig(
                band=FrequencyBand(20.0, 3_000.0),
                noise=False,
                level_db=0.0,
                window_seconds=1.0,
                hop_seconds=0.1,
                pre_silence=0.1,
                fade_in=0.5,
                fade_out=0.5,
            ),
            on_level=lambda *args: None,
            on_snapshot=lambda *args: None,
            on_complete=lambda *args: None,
            on_clipping=lambda message: None,
        )


if __name__ == "__main__":
    unittest.main()
