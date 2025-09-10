# import pyaudio
from utils.generators import tone
from utils.audio import listinputs, listoutputs
from pprint import pp
import numpy as np
from matplotlib import pyplot as plt
import sounddevice as sd
import os

RATE = 48000
BUFSIZE = 1024

# signal = (tone(1, RATE, 440) * 2**15 * 0.9).astype(np.int16)

# # print("-------------","Available audio inputs:", sep="\n")
# # for name, key in listinputs().items():
# #     print(f"{key}: {name}")

# # print("-------------","Available audio outputs:", sep="\n")
# # for name, key in listoutputs().items():
# #     print(f"{key}: {name}")

# input_device = 15
# output_device = 14

# pa = pyaudio.PyAudio()

# # f = open("devinfo.txt", "w")
# # for i in range(pa.get_device_count()):
# #     pp(pa.get_device_info_by_index(i), f)

# outstream = pa.open(
#     format=pyaudio.paInt16,
#     channels=1,
#     output=True,
#     output_device_index=output_device,
#     rate=RATE,
# )

# instream = pa.open(
#     format=pyaudio.paInt16,
#     channels=1,
#     input=True,
#     input_device_index=input_device,
#     rate=RATE,
# )

# recbuffer = bytearray()
# # stream.write(signal[i : i + BUFSIZE].tobytes())
# for i in range(0, len(signal), BUFSIZE):
#     outstream.write(signal[i : i + BUFSIZE].tobytes())
#     recbuffer += instream.read(BUFSIZE)

# instream.stop_stream()
# outstream.stop_stream()
# instream.close()
# outstream.close()
# pa.terminate()

# record = np.frombuffer(recbuffer, dtype=np.int16).astype(np.float64) / 2**15

# plt.plot(record)
# plt.show()
os.environ["SD_ENABLE_ASIO"] = "1"
# pp(sd.query_devices())

signal = tone(1, RATE, 440)
record = np.zeros_like(signal)

start_idx = 0


def cb(indata, outdata, frames, time, status):
    global start_idx, record
    if status:
        print(status)
    if start_idx + frames > len(signal):
        frames = len(signal) - start_idx
    outdata[:frames] = signal[start_idx : start_idx + frames].reshape(-1, 1)
    record[start_idx : start_idx + frames] = indata[:frames].reshape(-1)
    start_idx += frames


stream = sd.Stream(
    samplerate=RATE,
    channels=1,
    callback=cb,
    blocksize=BUFSIZE,
    device=(15,14)

)

stream.start()
sd.sleep(3000)  # Record for 1 second
stream.stop()
stream.close()
ts=np.arange(len(record)) / RATE
plt.plot(ts,record)
plt.show()
