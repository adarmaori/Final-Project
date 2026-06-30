import numpy as np
import scipy.signal as signal

def aural_exciter(x, drive=2.0, mix=0.3, cutoff_freq=3000.0, fs=44100):
    """
    Aural Exciter based on DAFX Chapter 4.4.1.
    Generates new high-frequency harmonics to add "shimmer" and clarity.
    
    Args:
        x (np.ndarray): 1D float array of input audio.
        drive (float): How hard to saturate the high frequencies (creates more overtones).
        mix (float): How much of the generated sparkle to blend back with the dry signal.
        cutoff_freq (float): The frequency above which harmonics are generated.
        fs (int): System sampling rate.
    """
    # 1. Isolate the high frequencies using a 2nd-order Butterworth High-Pass Filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(2, normal_cutoff, btype='high', analog=False)
    high_passed = signal.lfilter(b, a, x)
    
    # 2. Memoryless non-linearity (Harmonic Generation)
    # We use tanh (soft clipping) to smoothly generate odd harmonics
    generated_harmonics = np.tanh(drive * high_passed)
    
    # 3. High-pass the generated harmonics again 
    # (Removes low-end intermodulation "mud" created by the distortion process)
    clean_harmonics = signal.lfilter(b, a, generated_harmonics)
    
    # 4. Mix the newly generated sparkle back with the dry, untouched input signal
    y = x + (mix * clean_harmonics)
    
    return y