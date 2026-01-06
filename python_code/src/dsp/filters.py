import numpy as np
from scipy.signal import lfilter, butter

def lowpass_filter(data, cutoff, fs, order=1):
    """
    Applies a simple Low-Pass Filter (Butterworth).
    
    Args:
        data (np.ndarray): Input signal.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.
        order (int): Filter order.
        
    Returns:
        np.ndarray: Filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def tone_stack(data, bass=0.0, mid=0.0, treble=0.0, fs=44100):
    """
    Placeholder for a more complex EQ curve. 
    For now, acts as a simple non-adjustable presence filter.
    """
    # Simple mid-boost for guitar presence
    # Center freq: 800Hz, Q: 0.7
    # This is just a dummy implementation for now
    return data
