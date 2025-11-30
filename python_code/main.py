import wave
import numpy as np

obj = wave.open("file_example_WAV_2MG.wav", "rb")
print("Number of channels", obj.getnchannels())
print("Sample width", obj.getsampwidth())
print("Frame rate", obj.getframerate())
print("Number of frames", obj.getnframes())
print("parameters", obj.getparams())

n_samples = obj.getnchannels() * obj.getnframes()


total_time = obj.getnframes() / obj.getframerate()
print(f"{total_time=}")

frames = obj.readframes(-1)

signal = np.frombuffer(frames, dtype=np.int16)
