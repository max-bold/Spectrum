from utils.generators import pink_noise
import numpy as np
from utils.audio import playaudio

RATE = 96000
BUFFER = 1024 * 4

noise = pink_noise(30, RATE)
noise_bytes = (noise * 2**15 * 0.9).astype(np.int16)
playaudio(noise_bytes, RATE)