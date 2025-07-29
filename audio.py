import pyaudio
import numpy as np
import time
import gen


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


if __name__ == "__main__":
    # Example usage
    RATE = 44100
    from gen import pink_noise

    waveform = pink_noise(30, RATE, (20, 200))  # Example waveform
    waveform = (waveform * 2**15 * 0.9).astype(np.int16)  # Scale to int16 range
    playaudio(waveform, RATE)

    # playpinknoise(
    #     30, (20, 200)
    # )  # Play pink noise for 30 seconds in the range 100-10k Hz
