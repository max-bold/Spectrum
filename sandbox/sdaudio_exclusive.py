import sounddevice as sd
from time import sleep
print(sd.query_devices())
ws = sd.WasapiSettings(exclusive=True)
stream = sd.Stream(device=(12,11), extra_settings=ws, blocksize=4096, channels=2)
# sd.check_input_settings(device=3, extra_settings=ws)
stream.start()
sleep(5)
stream.close()
stream.abort()

