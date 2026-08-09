import numpy as np

# freqs = np.geomspace(20,20e3,31)
freqs = np.pow(10.0, 0.1*np.arange(13,44))
print(freqs.round())