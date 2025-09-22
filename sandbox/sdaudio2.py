import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from threading import Thread, Lock


class Audio(Thread):
    def __init__(self) -> None:
        self.record = np.empty((0, 2), dtype=np.float32)
        self.record_lock = Lock()
        return super().__init__()

    def run(self):
        stream = sd.Stream(channels=2,device=(20,16))
        fs = stream.samplerate

        f = 1000
        t = 60
        n = int(t * fs)
        ts = np.linspace(0, t, n)
        cs = 1024
        rec_len = 10000

        stream.start()
        for start in range(0, len(ts), cs):
            end = min(start + cs, len(ts))
            chunk = np.sin(2 * np.pi * f * ts[start:end])*0.5
            chunk = np.column_stack((chunk, chunk)).astype(np.float32)
            stream.write(chunk)
            chunk = stream.read(cs)[0]
            with self.record_lock:
                rs = max(0, len(self.record) + len(chunk) - rec_len)
                self.record = np.append(self.record[rs:], chunk, axis=0)

    def get_record(self):
        with self.record_lock:
            return self.record.copy()


plt.ion()
line = plt.plot([])[0]
line.axes.set_xlim(0, 10000)
line.axes.set_ylim(-1.2, 1.2)
io = Audio()
io.start()
while io.is_alive():
    rec = io.get_record()
    x = np.arange(len(rec))
    line.set_data(x, rec[:, 0])
    plt.draw()
    plt.pause(0.1)
# plt.plot(record[:, 0])
plt.ioff()
plt.show()
