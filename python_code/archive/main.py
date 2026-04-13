import wave
import matplotlib.pyplot as plt
import numpy as np

obj = wave.open("440Hz_44100Hz_16bit_05sec.wav", "rb")
nch = obj.getnchannels()
print("Number of channels", nch)
print("Sample width", obj.getsampwidth())
print("Frame rate", obj.getframerate())
print("Number of frames", obj.getnframes())
print("parameters", obj.getparams())

fs = obj.getframerate()
total_time = obj.getnframes() / fs
print(f"{total_time=}")

frames = obj.readframes(-1)
obj.close()

sig_i16 = np.frombuffer(frames, dtype=np.int16)
if nch > 1:
    sig_i16 = sig_i16.reshape(-1, nch)[:, 0]

x = sig_i16.astype(np.float32) / 32768.0
x = x - np.mean(x)

w = np.hanning(len(x))
xw = x * w

X = np.fft.rfft(xw)
f = np.fft.rfftfreq(len(xw), d=1.0 / fs)
mag = np.abs(X)

peak_freq = f[np.argmax(mag)]
print(f"Peak frequency ~ {peak_freq:.2f} Hz")

plt.figure()
plt.plot(f, mag)
plt.xlim(0, 2000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("|FFT|")
plt.title("Magnitude Spectrum")
plt.grid(True)
plt.show()
