from utils.generators import pink_noise
import sounddevice as sd

stream = sd.OutputStream(channels=1)

wf = pink_noise(30,stream.device.)