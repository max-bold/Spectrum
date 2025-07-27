import pyaudio
import numpy as np
import time

pa = pyaudio.PyAudio()

RATE = 44100
FORMAT = pyaudio.paInt16
LENGTH = 10 * RATE  # Fixed typo: LENGHT -> LENGTH
CHUNK = 1024

# Calculate how long each chunk should take to play
CHUNK_TIME = CHUNK / RATE

stream = pa.open(format=FORMAT, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)
# print(pa.get_default_output_device_info())
# Generate the audio data
n = np.zeros(LENGTH, dtype=np.int16)
for i in range(LENGTH):
    n[i] = np.sin(2*np.pi*440*i/RATE)*2**15*0.9

def sinegen(frequency=440):
    a=0
    while a<LENGTH:
        n = np.zeros(CHUNK, dtype=np.int16)
        for i in range(a,a+CHUNK):
            n[i-a] = np.sin(2*np.pi*frequency*i/RATE)*2**15*0.9
        a+=CHUNK
        yield n.tobytes()

def sweepgen(start=440,end=880):
    a=0
    while a<LENGTH:
        n = np.zeros(CHUNK, dtype=np.int16)
        for i in range(a,a+CHUNK):  
            w = i**2/2/LENGTH/end-i**2/2/LENGTH/start+i/start
            n[i-a] = np.sin(w)*2**15*0.9
        a+=CHUNK
        yield n.tobytes()

# def sweep(start=440,end=880):
#     n=np.zeros(LENGTH,dtype=np.int16)
#     for i in range(LENGTH):
#         w = i*(2*end*LENGTH-i*(end-start))/2/end/start/LENGTH
#         n[i] = np.sin(w)*2**15*0.9
#     return n.tobytes()

for buf in sweepgen(440,880):
    stream.write(buf)