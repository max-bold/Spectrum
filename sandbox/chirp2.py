from scipy.signal import chirp
import sounddevice as sd
import numpy as np
from time import sleep

stream = sd.OutputStream(channels=1)


n = int(30 * stream.samplerate)
f0 = 200 / stream.samplerate
f1 = 5000 / stream.samplerate
ts = np.arange(n)
chunksize = 1024
stream.start()
for start in range(0, n, chunksize):
    end = min(start + chunksize, n)
    chunk = chirp(ts[start:end], f0, n, f1, method="logarithmic") * .5
    if stream.write(chunk.astype(np.float32)):
        print("underflow")
    sleep(0.1)
