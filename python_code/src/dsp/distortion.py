import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
from src.dsp.filters import lowpass_filter


# ---------------------------------------------------------------------------
# Real-time (block-by-block) tube saturator
# ---------------------------------------------------------------------------

class RealtimeTubeSaturator:
    """
    Stateful, block-by-block tube saturator suitable for real-time use.

    Maintains internal filter states between successive ``process()`` calls so
    that consecutive blocks produce a seamless output stream.

    Processing chain (identical to the offline ``tube_saturator``):
        1. Input gain (drive)
        2. Asymmetric soft clipping:  y = tanh(x + dc_bias)
        3. Streaming DC blocker  (1st-order high-pass @ 20 Hz)
        4. Butterworth low-pass tone filter (stateful)
        5. Output gain compensation
    """

    def __init__(self, drive=5.0, asymmetry=0.2, tone=5000, fs=44100, output_gain=0.8):
        self.drive = drive
        self.asymmetry = asymmetry
        self.output_gain = output_gain
        self.fs = fs

        # --- Tone filter (Butterworth LP, same as offline version) ----------
        nyq = 0.5 * fs
        self._tone_b, self._tone_a = butter(4, tone / nyq, btype='low')
        # Initial filter state (scaled to handle step response properly)
        self._tone_zi = lfilter_zi(self._tone_b, self._tone_a) * 0.0

        # --- DC blocker (1st-order HP @ 20 Hz) -----------------------------
        # H(z) = (1 - z^-1) / (1 - R*z^-1),  R = 1 - (2*pi*fc / fs)
        R = 1.0 - (2.0 * np.pi * 20.0 / self.fs)
        self._dc_b = np.array([1.0, -1.0])
        self._dc_a = np.array([1.0, -R])
        self._dc_zi = lfilter_zi(self._dc_b, self._dc_a) * 0.0

    # ------------------------------------------------------------------
    def reset(self):
        """Clear all internal state (call between unrelated audio streams)."""
        self._tone_zi = lfilter_zi(self._tone_b, self._tone_a) * 0.0
        self._dc_zi = lfilter_zi(self._dc_b, self._dc_a) * 0.0

    # ------------------------------------------------------------------
    def process(self, audio_block: np.ndarray) -> np.ndarray:
        """
        Process a single block of audio samples.

        Args:
            audio_block: 1-D numpy array of float samples.

        Returns:
            Processed 1-D numpy array (same length as input).
        """
        # 1. Drive
        x = audio_block * self.drive

        # 2. Asymmetric soft clipping
        y = np.tanh(x + self.asymmetry)

        # 3. Streaming DC blocker (stateful IIR high-pass)
        y, self._dc_zi = lfilter(self._dc_b, self._dc_a, y, zi=self._dc_zi)

        # 4. Tone (stateful low-pass)
        y, self._tone_zi = lfilter(self._tone_b, self._tone_a, y, zi=self._tone_zi)

        # 5. Output gain
        return (y * self.output_gain).astype(np.float32 if audio_block.dtype == np.float32 else np.float64)

    # ------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Original offline functions (unchanged)
# ---------------------------------------------------------------------------

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
    # Use same IIR DC blocker as the real-time version for consistency
    # H(z) = (1 - z^-1) / (1 - R*z^-1),  R ≈ 1 - 2*pi*20/fs
    R = 1.0 - (2.0 * np.pi * 20.0 / fs)
    dc_b = np.array([1.0, -1.0])
    dc_a = np.array([1.0, -R])
    y = lfilter(dc_b, dc_a, y)
    
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
