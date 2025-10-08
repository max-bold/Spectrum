import numpy as np

n = 1000
x = np.random.uniform(-1, 1, n)

fs = 44100

f = np.fft.rfftfreq(len(x), 1 / fs)
p = np.square(np.abs(np.fft.rfft(x))) / n / fs
print(np.allclose(np.diff(f), fs / n))
print(len(p) == n / 2 + 1)
