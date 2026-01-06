import numpy as np
from src.dsp.filters import lowpass_filter

def tanh_distortion(audio_signal, gain=1.0, output_gain=1.0):
    """
    Applies simplified tanh distortion to the audio signal.
    f(x) = tanh(k * x)
    
    Args:
        audio_signal (np.ndarray): Input audio samples.
        gain (float): Input gain factor (k).
        output_gain (float): Output gain scaling.
        
    Returns:
        np.ndarray: Distorted audio samples.
    """
    # Soft clipping using tanh
    distorted = np.tanh(gain * audio_signal)
    return distorted * output_gain

def tube_saturator(audio_signal, drive=5.0, asymmetry=0.2, tone=5000, fs=44100):
    """
    A specific "Tube-like" distortion chain.
    
    Processing Chain:
    1. Input Gain (Drive)
    2. Asymmetric Soft Clipping: f(x) = tanh(x) + a*x^2
    3. Low Pass Filter (Tone)
    
    Args:
        audio_signal (np.ndarray): Input audio.
        drive (float): Input drive amount (1.0 to 20.0+).
        asymmetry (float): Amount of even-harmonic asymmetry (0.0 to 0.5).
        tone (float): Low-pass cutoff frequency in Hz.
        fs (int): Sample rate.
        
    Returns:
        np.ndarray: Processed signal.
    """
    # 1. Drive
    x = audio_signal * drive
    
    # 2. Asymmetric Function
    # We mix a bit of squared signal (rectified) to add even harmonics
    # But we clip it to keep it stable
    # Use a modified tanh: tanh(x) + offset
    
    # Simple asymmetry: shift the operating point
    # x_shifted = x + asymmetry
    # y = np.tanh(x_shifted)
    
    # Better asymmetry for guitar:
    # y = tanh(x) + a * (x^2 if x>0 else 0) - keeps it somewhat centered but alters shape
    
    # Let's use a standard waveshaper approach:
    # y = (1+a)*x / (1 + a*abs(x))  <- Soft clipper, less expensive than tanh actually
    # But let's stick to tanh + a shift which is very "tube bias" like
    
    dc_bias = asymmetry
    y = np.tanh(x + dc_bias)
    
    # Remove the DC offset introduced by the bias so speaker doesn't pop
    # Simple DC blocker (High-pass at 20Hz) - or just subtract mean for offline
    y = y - np.mean(y)
    
    # 3. Tone (Low Pass)
    y_filtered = lowpass_filter(y, cutoff=tone, fs=fs, order=4)
    
    # Compensation Gain (rough approximation)
    output_gain = 0.8
    return y_filtered * output_gain

def hard_clipping(audio_signal, threshold=0.5):
    """
    Applies hard clipping to the audio signal.
    
    Args:
        audio_signal (np.ndarray): Input audio samples.
        threshold (float): Clipping threshold.
        
    Returns:
        np.ndarray: Clipped audio samples.
    """
    return np.clip(audio_signal, -threshold, threshold)
