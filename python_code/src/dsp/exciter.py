import numpy as np
import scipy.signal as signal

def aural_exciter(x, drive=6.0, mix=0.7, cutoff_freq=2200.0, fs=44100):
    """
    Fully Causal, High-Output Aural Exciter.
    Uses a fixed scalar gain boost instead of global track peak normalization.
    This guarantees time-invariance, allowing the TCN to optimize flawlessly.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(2, normal_cutoff, btype='high', analog=False)
    
    # 1. Isolate the upper midrange and treble 
    high_passed = signal.lfilter(b, a, x)
    
    # 2. Fixed Pre-Drive Gain (Fully Causal & Predictable for the TCN)
    # Multiplying by a constant 8.0 acts as a clean, predictable +18dB boost 
    # to push the isolated highs perfectly into the musical saturation zone.
    sidechain_boosted = high_passed * 8.0
    
    # 3. Saturation (Rich Harmonic Generation)
    # With a drive of 6.0 and a fixed boost, this produces a smooth, continuous
    # cascade of overtones that won't sound broken or overly distorted.
    generated_harmonics = np.tanh(drive * sidechain_boosted)
    
    # 4. Secondary Residual Filter Cleanup
    clean_harmonics = signal.lfilter(b, a, generated_harmonics)
    
    # 5. Fixed Output Scaling & Balance
    # Scale down the raw harmonics slightly so they don't clip your interface output
    scaled_harmonics = clean_harmonics * 0.4
    
    # 6. Summation with Dry Signal
    y = x + (mix * scaled_harmonics)
    
    return y