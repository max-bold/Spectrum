from scipy.signal import butter, lfilter

band = (20, 20e3)
samplerate = 44100
sos = butter(4, band, btype="band", fs=samplerate, output="sos")
print(type(sos))