import pyfar as pf
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import os

x = pf.signals.exponential_sweep_time(
    n_samples=2**16,
    frequency_range=[500, 5000],
    sampling_rate=44100)

ax = pf.plot.freq(x)
ax[0].set_title('Exponential sweep excitation signal')
plt.show()