import pyaudio
import numpy as np
import time
import gen
from pprint import pp
import matplotlib.pyplot as plt
from threading import Thread
from typing import Callable


def playaudio(waveform: np.ndarray[np.int16], rate: int):
    pa = pyaudio.PyAudio()
    chunk_size = 1024
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        output=True,
        frames_per_buffer=chunk_size,
    )
    for i in range(0, len(waveform), chunk_size):
        chunk = waveform[i : i + chunk_size].tobytes()
        stream.write(chunk)
    stream.stop_stream()
    stream.close()
    pa.terminate()


def playpinknoise(length: float, band=None) -> None:
    pa = pyaudio.PyAudio()
    chunk_size = 1024
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        output=True,
        frames_per_buffer=chunk_size,
    )
    st = time.time()
    for chunk in gen.pink_noise_gen(length, 1024, 44100, band):
        if np.max(np.abs(chunk)) > 1:
            print("Overload detected!!!")
        chunk = chunk * 2**15 * 0.9
        chunk = chunk.astype(np.int16).tobytes()
        stream.write(chunk)
    stream.stop_stream()
    stream.close()
    pa.terminate()


def listinputs():
    names = {"System default input": None}
    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"]:
            names[info["name"]] = info["index"]
    return names


def listoutputs():
    names = {"System default outputput": None}
    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxOutputChannels"]:
            names[info["name"]] = info["index"]
    return names


class imputindicator(Thread):
    def __init__(self, cb: Callable):
        self.runflag = True
        self.input = None
        self.cb = cb
        super().__init__()

    def setinput(self, input):
        self.input = input

    def run(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(
            44100,
            1,
            pyaudio.paInt16,
            input=True,
            input_device_index=self.input,
        )

        while self.runflag:
            bytes = stream.read(1024)
            na = np.frombuffer(bytes, np.int16).astype(np.float64)
            rms = np.max(na) / 2**15
            self.cb(rms)
        stream.stop_stream()
        stream.close()
        pa.terminate()

    def stop(self):
        self.runflag = False


if __name__ == "__main__":
    # Example usage
    # RATE = 44100
    # from gen import pink_noise

    # waveform = pink_noise(30, RATE, (20, 200))  # Example waveform
    # waveform = gen.tone(30, 44100, 1000)
    # waveform = (waveform * 2**15 * 0.9).astype(np.int16)  # Scale to int16 range
    # playaudio(waveform, 44100)

    # playpinknoise(
    #     30, (20, 200)
    # )  # Play pink noise for 30 seconds in the range 100-10k Hz

    # pp("outputs:")
    # pp(listoutputs())
    # pp("inputs:")
    # pp(listinputs())

    generator = measure_input()
    st = time.time()
    res = []
    while time.time() - st < 3:
        res.append(next(generator))

    generator.send(True)

    plt.plot(res)
    plt.show()
