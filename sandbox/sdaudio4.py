import sounddevice as sd
import numpy as np
from time import sleep

def callback(indata: np.ndarray, outdata: np.ndarray, frames: int,
         time, status) -> None:
    outdata.fill(0)
    print(frames)


stream = sd.Stream(callback=callback, device=(26,20))
stream.start()
sleep(5)
stream.close()
