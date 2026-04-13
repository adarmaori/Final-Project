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

# Apply reverb effect to original signal
def apply_reverb(signal, sample_rate, decay=0.5, delay_ms=50, iterations=6):
    """Apply a simple reverb using a fixed delay increment and decay."""
    output = signal.astype(np.float64).copy()

    for i in range(1, iterations + 1):
        delay_samples = int(delay_ms * i * sample_rate / 1000)
        if delay_samples >= len(signal):
            break

        decay_factor = decay ** i
        delayed = np.zeros_like(output)
        delayed[delay_samples:] = signal[:-delay_samples] * decay_factor

        output += delayed

    # Normalize to prevent clipping
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 32767 * 0.9  # Leave some headroom

    return output

# Apply reverb with default parameters
reverb_data = apply_reverb(data, sample_rate, decay=0.5, delay_ms=120, iterations=100)

# Save reverb audio to new WAV file
reverb_data_int = np.int16(reverb_data)
wavfile.write('reverb_output.wav', sample_rate, reverb_data_int)
print(f"Reverb audio saved to: reverb_output.wav")

# --- Distortion Implementations ---

def apply_hard_clipping(signal, threshold=0.5, gain=10.0):
    """
    Apply hard clipping distortion.
    Caps the amplitude using min() and max() (equivalent to clipping).
    """
    # Normalize to float -1.0 to 1.0
    if np.issubdtype(signal.dtype, np.integer):
        signal_float = signal / 32768.0
    else:
        # Handle float data that might be in integer range (e.g. from averaging)
        if np.max(np.abs(signal)) > 1.0:
            signal_float = signal / 32768.0
        else:
            signal_float = signal.copy()

    # Apply pre-gain
    amplified = signal_float * gain

    # Hard clipping using min/max as requested
    # Clip upper bound
    clipped = np.minimum(amplified, threshold)
    # Clip lower bound
    clipped = np.maximum(clipped, -threshold)
    
    # Convert back to int16 range
    output = clipped * 32767

    return output

def apply_tanh_distortion(signal, threshold=0.5, gain=10.0):
    """
    Apply soft clipping distortion using tanh function.
    """
    # Normalize to float -1.0 to 1.0
    if np.issubdtype(signal.dtype, np.integer):
        signal_float = signal / 32768.0
    else:
        # Handle float data that might be in integer range
        if np.max(np.abs(signal)) > 1.0:
            signal_float = signal / 32768.0
        else:
            signal_float = signal.copy()

    # Apply pre-gain
    amplified = signal_float * gain

    # Apply tanh distortion
    # Soft clip to threshold
    distorted = threshold * np.tanh(amplified)

    # Convert back to int16 range
    output = distorted * 32767

    return output

# Apply Hard Clipping
hard_clipped_data = apply_hard_clipping(data, threshold=0.6, gain=60.0)
hard_clipped_int = np.int16(hard_clipped_data)
wavfile.write('hard_clipping_output.wav', sample_rate, hard_clipped_int)
print(f"Hard clipped audio saved to: hard_clipping_output.wav")

# Apply Tanh Distortion
tanh_distorted_data = apply_tanh_distortion(data, threshold=0.1, gain=1000.0)
tanh_distorted_int = np.int16(tanh_distorted_data)
wavfile.write('tanh_distortion_output.wav', sample_rate, tanh_distorted_int)
print(f"Tanh distorted audio saved to: tanh_distortion_output.wav")

# Plotting Distortion
plt.figure(figsize=(14, 8))

# Find a loud part of the signal to visualize clipping
# We'll look for the maximum amplitude in the original signal
zoom_center = np.argmax(np.abs(data))
zoom_window = 500  # Number of samples to show
start_idx = max(0, zoom_center - zoom_window // 2)
end_idx = min(len(data), zoom_center + zoom_window // 2)

time_zoom = np.arange(start_idx, end_idx) / sample_rate

# Plot Hard Clipping
plt.subplot(2, 1, 1)
plt.plot(time_zoom, data[start_idx:end_idx], label='Original', alpha=0.5)
plt.plot(time_zoom, hard_clipped_data[start_idx:end_idx], label='Hard Clipping', linestyle='--')
plt.title('Hard Clipping (Zoomed)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot Tanh Distortion
plt.subplot(2, 1, 2)
plt.plot(time_zoom, data[start_idx:end_idx], label='Original', alpha=0.5)
plt.plot(time_zoom, tanh_distorted_data[start_idx:end_idx], label='Tanh Distortion', linestyle='--')
plt.title('Tanh Distortion (Zoomed)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
