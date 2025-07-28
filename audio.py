import pyaudio
import numpy as np
import time


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


if __name__ == "__main__":
    # Example usage
    RATE = 44100
    from gen import pink_noise

    waveform = pink_noise(100, 10000, 30, RATE)
    playaudio(waveform, RATE)
