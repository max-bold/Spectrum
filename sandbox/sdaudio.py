import sounddevice as sd
import numpy as np

# stream = sd.Stream(device=)
# print(f"Input: {sd.query_devices(stream.device[0])}\nOutput: {sd.query_devices(stream.device[1])}")
# print(f"default samplerate: {stream.samplerate}")
# print(f"default blocksize: {stream.blocksize}")

print(sd.query_devices())
print(sd.query_hostapis())

# stream.start()
# print(stream.write_available)
# for i in range(100):
    
#     noise = np.random.uniform(-1,1,1024)
#     noise = np.column_stack((noise,noise)).astype(np.float32)
#     stream.write(noise)
#     print(stream.write_available)
    # 

