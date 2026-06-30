import numpy as np
import scipy.signal as signal
import librosa
import os

def generate_procedural_cab_ir(fs=44100, ir_length=2048):
    """
    Procedurally designs a realistic 12-inch guitar cabinet speaker impulse response
    based on Chapter 2 (FIR Filters/Equalizers) of the DAFX textbook.
    
    Models:
    - Low-end cabinet resonance (~90 Hz)
    - Mid-range comb-filtering/cone breakup notches (1.5 kHz - 3.5 kHz)
    - Steep high-frequency brickwall roll-off above 4.5 kHz ("anti-fuzz")
    """
    nyquist = fs / 2.0
    
    # 1. Design a multi-band FIR profile using windowed frequency sampling
    num_taps = ir_length | 1  # Must be odd for type 1 FIR
    
    # Define frequency grid
    freqs = np.linspace(0, nyquist, 1024)
    gain = np.zeros_like(freqs)
    
    for i, f in enumerate(freqs):
        if f < 60:
            # Sub-bass roll-off
            gain[i] = 0.01
        elif f >= 60 and f < 110:
            # Bass resonance spike (90Hz cabinet thud)
            gain[i] = 1.2 + 0.3 * np.sin((f - 60) / 50 * np.pi)
        elif f >= 110 and f < 1500:
            # Warm guitar midrange body
            gain[i] = 1.0
        elif f >= 1500 and f < 4500:
            # Jagged cone-breakup phase notches (gives texture/color)
            gain[i] = 0.6 + 0.3 * np.sin(f * 0.01) + 0.1 * np.cos(f * 0.05)
        elif f >= 4500 and f < 5500:
            # Steep speaker cone high-end roll-off
            gain[i] = 1.0 - (f - 4500) / 1000
        else:
            # Complete high-frequency attenuation
            gain[i] = 0.0
            
    # Convert frequency response profile to time-domain impulse response coefficients
    ir = signal.firwin2(num_taps, freqs / nyquist, gain, window='hamming')
    
    # Apply a tiny exponential decay envelope to ensure smooth tail termination
    t = np.linspace(0, 1, num_taps)
    decay = np.exp(-4.0 * t)
    ir = ir * decay
    
    # Normalize IR energy
    return ir / (np.linalg.norm(ir) + 1e-8)


def cab_simulator(x, ir_path=None, fs=44100):
    """
    Guitar Cabinet Simulator effect using discrete linear convolution:
    y[n] = x[n] * h[n]
    
    Args:
        x (np.ndarray): 1D float array of input audio.
        ir_path (str, optional): Path to a custom .wav speaker impulse response file.
        fs (int): System sampling rate.
    """
    if ir_path and os.path.exists(ir_path):
        # Load custom cabinet IR wav file
        ir, ir_sr = librosa.load(ir_path, sr=fs, mono=True)
    else:
        # Fall back to high-fidelity procedural emulation curve
        ir = generate_procedural_cab_ir(fs=fs, ir_length=2048)
        
    # Perform standard linear discrete convolution
    y = signal.lfilter(ir, [1.0], x)
    
    # Protect against digital clipping
    peak = np.max(np.abs(y))
    if peak > 1.0:
        y = y / peak * 0.95
        
    return y