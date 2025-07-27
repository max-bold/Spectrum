import pyaudio
import numpy as np
import threading
import time

class AudioPlayer:
    def __init__(self, sample_rate=48000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_data = None
        self.position = 0
        self.playing = False
        self.lock = threading.Lock()
        
    def generate_tone(self, frequency=440, duration=10):
        """Generate a sine wave tone"""
        length = int(duration * self.sample_rate)
        self.audio_data = np.zeros(length, dtype=np.int16)
        for i in range(length):
            self.audio_data[i] = np.sin(2*np.pi*frequency*i/self.sample_rate)*2**15*0.9
        self.position = 0
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio"""
        if self.audio_data is None or not self.playing:
            return (b'\x00' * frame_count * 2, pyaudio.paComplete)
            
        with self.lock:
            if self.position >= len(self.audio_data):
                return (b'\x00' * frame_count * 2, pyaudio.paComplete)
                
            end_pos = min(self.position + frame_count, len(self.audio_data))
            chunk = self.audio_data[self.position:end_pos]
            
            # Pad with zeros if needed
            if len(chunk) < frame_count:
                chunk = np.pad(chunk, (0, frame_count - len(chunk)), 'constant')
                
            self.position += frame_count
            
            return (chunk.tobytes(), pyaudio.paContinue)
    
    def play(self):
        """Play the audio using callback method"""
        if self.audio_data is None:
            print("No audio data to play. Call generate_tone() first.")
            return
            
        self.position = 0
        self.playing = True
        
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        stream.start_stream()
        
        # Wait for playback to complete
        while stream.is_active():
            time.sleep(0.1)
            
        stream.stop_stream()
        stream.close()
        pa.terminate()
        self.playing = False

# Example usage
if __name__ == "__main__":
    player = AudioPlayer()
    player.generate_tone(frequency=440, duration=5)  # 5 seconds of 440Hz tone
    player.play() 