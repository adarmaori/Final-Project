import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# Read the WAV file
sample_rate, data = wavfile.read('blues-electric-guitar-smooth-lo-fi-loop_85bpm_B_minor.wav')

# If stereo, convert to mono by averaging channels
if len(data.shape) > 1:
    data = data.mean(axis=1)

# Design 400Hz low-pass filter (Butterworth, 4th order)
cutoff_freq = 400  # Hz
nyquist = sample_rate / 2
normal_cutoff = cutoff_freq / nyquist
b, a = butter(4, normal_cutoff, btype='low', analog=False)

# Apply the filter
filtered_data = filtfilt(b, a, data)

# Save filtered audio to new WAV file
filtered_data_int = np.int16(filtered_data)
wavfile.write('filtered_400Hz.wav', sample_rate, filtered_data_int)
print(f"Filtered audio saved to: filtered_400Hz.wav")

# Compute FFT for original
fft_result = np.fft.fft(data)
fft_freq = np.fft.fftfreq(len(data), 1/sample_rate)

# Compute FFT for filtered
fft_result_filtered = np.fft.fft(filtered_data)
fft_freq_filtered = np.fft.fftfreq(len(filtered_data), 1/sample_rate)

# Take only positive frequencies
positive_freq_idx = fft_freq > 0
fft_freq = fft_freq[positive_freq_idx]
fft_magnitude = np.abs(fft_result[positive_freq_idx])
fft_magnitude_filtered = np.abs(fft_result_filtered[positive_freq_idx])

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot original time domain signal
time = np.arange(len(data)) / sample_rate
ax1.plot(time, data, alpha=0.7)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Original Signal (Time Domain)')
ax1.grid(True)

# Plot filtered time domain signal
ax2.plot(time, filtered_data, alpha=0.7, color='orange')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude')
ax2.set_title('Filtered Signal - 400Hz LPF (Time Domain)')
ax2.grid(True)

# Plot original frequency domain (FFT)
ax3.plot(fft_freq, fft_magnitude, alpha=0.7)
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Magnitude')
ax3.set_title('Original Signal (Frequency Domain)')
ax3.set_xlim(0, 2000)
ax3.axvline(x=400, color='r', linestyle='--', alpha=0.5, label='400Hz cutoff')
ax3.legend()
ax3.grid(True)

# Plot filtered frequency domain (FFT)
ax4.plot(fft_freq, fft_magnitude_filtered, alpha=0.7, color='orange')
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel('Magnitude')
ax4.set_title('Filtered Signal - 400Hz LPF (Frequency Domain)')
ax4.set_xlim(0, 2000)
ax4.axvline(x=400, color='r', linestyle='--', alpha=0.5, label='400Hz cutoff')
ax4.legend()
ax4.grid(True)

# Find and print peak frequency
peak_idx = np.argmax(fft_magnitude)
peak_freq = fft_freq[peak_idx]
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(data)/sample_rate:.2f} seconds")
print(f"Original peak frequency: {peak_freq:.2f} Hz")

plt.tight_layout()
plt.show()
