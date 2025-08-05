from gen import pink_noise, logsweep
from analyse import calc_psd
import pyaudio
import numpy as np
from pprint import pp
from matplotlib import pyplot as plt

RATE = 96000
BUFFER = 1024 * 4


def float64_to_24bit_pcm(data: np.ndarray) -> bytes:
    """
    Преобразует float64 [-1.0, 1.0] в 24-бит PCM little-endian.
    """
    assert data.dtype == np.float64
    # Масштабируем
    data /= np.max(np.abs(data))  # Нормализация
    scaled = data * (2**23 - 1) * 0.9
    int_data = scaled.astype(np.int32)

    # Берем младшие 3 байта каждого int32 (little endian)
    byte_array = bytearray()
    for sample in int_data:
        byte_array.extend(int(sample).to_bytes(4, byteorder="little", signed=True)[:3])
        # res.extend(sample.tobytes()[:3])  # Берем только первые 3 байта
    return bytes(byte_array)


# waveform = pink_noise(30, RATE, (20, 200))
# wf = (waveform * 2**15 * 0.7).astype(np.int16)  # Scale to int16 range

# pa = pyaudio.PyAudio()

# # for i in range(pa.get_device_count()):
# #     info = pa.get_device_info_by_index(i)
# #     if "Sound Blaster" in info["name"]:
# #         pp(info)


# output = pa.open(
#     format=pyaudio.paInt16,
#     channels=1,
#     rate=RATE,
#     output=True,
#     frames_per_buffer=BUFFER,
#     start=False,
#     # output_device_index=16,
# )

# input = pa.open(
#     format=pyaudio.paInt16,
#     channels=1,
#     rate=RATE,
#     input=True,
#     frames_per_buffer=BUFFER,
#     start=False,
#     # input_device_index=16,
# )

# # buf = float64_to_24bit_pcm(waveform[:BUFFER])
# buf = wf[:BUFFER].tobytes()
# output.start_stream()
# input.start_stream()
# output.write(buf)
# frames = []
# for i in range(BUFFER, len(wf), BUFFER):
#     output.write(wf[i : i + BUFFER].tobytes())
#     rec = input.read(BUFFER)
#     frames.append(rec)

# output.stop_stream()
# output.close()
# pa.terminate()

pa = pyaudio.PyAudio()
stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    input=True,
    output=True,
    rate=RATE,
    frames_per_buffer=BUFFER,
)

rec = bytearray()
# noise = pink_noise(30, RATE)
noise = logsweep(30, RATE, (20, 20000))
noise_bytes = (noise * 2**15 * 0.9).astype(np.int16).tobytes()
for i in range(0, len(noise_bytes), BUFFER):
    stream.write(noise_bytes[i : i + BUFFER])
    rec.extend(stream.read(int(BUFFER / 2), exception_on_overflow=False))

stream.stop_stream()
stream.close()
pa.terminate()

rec = np.frombuffer(rec, dtype=np.int16)
rec = rec.astype(np.float64) / (2**15)  # Normalize to [-1, 1]
rec_t = np.arange(len(rec)) / RATE  # Time vector for the recording

xf, yf = calc_psd(rec, RATE, window=1 / 5)
x_ref, y_ref = calc_psd(noise, RATE)
yf /= y_ref
# yf *= xf / xf[0]
yf /= np.max(yf)

fig, axs = plt.subplots(2, 1)
axs[0].plot(rec_t, rec)
axs[1].semilogx(xf, 10 * np.log10(yf))
axs[1].set_title("FFT of Recorded Waveform")
axs[1].set_xlabel("Frequency [Hz]")
axs[1].set_ylabel("PSD [dB]")
axs[1].grid(which="both", axis="both")
plt.show()
