"""
Auto-Wah (Envelope Filter) audio effect implementation.

Signal flow:
    x(n) ──► |·| ──► EnvelopeFollower ──► freq_control(n)
             (abs)        (attack/release)       │
                                                  ▼
    x(n) ──────────────────► BiquadPeaking(f_c(n)) ──► y(n)

Where:
    - EnvelopeFollower: asymmetric first-order LP (fast attack, slow release)
    - Frequency sweep: f_c(n) = f_min + envelope(n) * (f_max - f_min)
    - BiquadPeaking: time-varying resonant peaking filter at center freq f_c(n)
    - Q: Quality factor (resonance/bandwidth control)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Offline wah effect function
# ---------------------------------------------------------------------------

def wah_effect(
    audio_signal: np.ndarray,
    freq_min: float = 400.0,
    freq_max: float = 2500.0,
    q: float = 2.0,
    attack_ms: float = 5.0,
    release_ms: float = 100.0,
    fs: int = 44100,
) -> np.ndarray:
    """
    Offline auto-wah effect for batch audio processing.
    
    Processes the entire signal using a real-time wah object to maintain
    state consistency with streaming implementation.
    
    Args:
        audio_signal: Input audio array (1-D, mono)
        freq_min: Minimum frequency (Hz) of sweep
        freq_max: Maximum frequency (Hz) of sweep
        q: Quality factor (resonance); higher Q = narrower, more resonant
        attack_ms: Envelope attack time in milliseconds
        release_ms: Envelope release time in milliseconds
        fs: Sample rate in Hz
    
    Returns:
        Processed audio array (same shape as input)
    """
    processor = RealtimeWahWah(
        freq_min=freq_min,
        freq_max=freq_max,
        q=q,
        attack_ms=attack_ms,
        release_ms=release_ms,
        fs=fs,
    )
    
    # Process in blocks matching typical RT chunk size
    block_size = 2048
    output = np.zeros_like(audio_signal, dtype=np.float32)
    
    for i in range(0, len(audio_signal), block_size):
        block = audio_signal[i : i + block_size]
        output[i : i + block_size] = processor.process(block)
    
    return output


# ---------------------------------------------------------------------------
# Envelope Follower
# ---------------------------------------------------------------------------

class EnvelopeFollower:
    """
    Asymmetric first-order low-pass envelope follower.
    
    Detects the amplitude envelope of the input signal using:
        1. Rectification (absolute value)
        2. Asymmetric smoothing (fast attack, slow release)
    
    This follows the standard DAFX envelope follower model.
    """
    
    def __init__(self, attack_ms: float = 5.0, release_ms: float = 100.0, fs: int = 44100):
        """
        Initialize envelope follower.
        
        Args:
            attack_ms: Attack time constant in milliseconds (fast transient response)
            release_ms: Release time constant in milliseconds (smooth decay)
            fs: Sample rate in Hz
        """
        self.fs = fs
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        
        # Compute smoothing factors using first-order LP tau = 1/(2*pi*fc)
        # or tau = 1/(fs*alpha) => alpha = e^(-1/(fs*tau))
        self._alpha_attack = np.exp(-1.0 / (fs * attack_ms / 1000.0))
        self._alpha_release = np.exp(-1.0 / (fs * release_ms / 1000.0))
        
        self._state = 0.0  # Envelope state
    
    def reset(self):
        """Clear envelope state."""
        self._state = 0.0
    
    def process(self, audio_block: np.ndarray) -> np.ndarray:
        """
        Process audio block and return envelope signal.
        
        Args:
            audio_block: 1-D numpy array of float samples
        
        Returns:
            1-D numpy array of envelope values in [0, 1]
        """
        block_size = len(audio_block)
        envelope = np.zeros(block_size, dtype=np.float32)
        
        for i in range(block_size):
            # Rectify input
            x_abs = abs(audio_block[i])
            
            # Select alpha based on comparison (attack if rising, release if falling)
            if x_abs > self._state:
                alpha = self._alpha_attack  # Fast attack
            else:
                alpha = self._alpha_release  # Slow release
            
            # First-order LP: y(n) = (1 - alpha) * x_abs + alpha * y(n-1)
            self._state = (1.0 - alpha) * x_abs + alpha * self._state
            
            envelope[i] = self._state
        
        return envelope


# ---------------------------------------------------------------------------
# Biquad Peaking Filter (Time-Varying Center Frequency)
# ---------------------------------------------------------------------------

class BiquadPeaking:
    """
    Time-varying biquad peaking (parametric EQ) filter.
    
    Implements a resonant peaking filter with dynamically updated coefficients
    based on a time-varying center frequency control signal.
    
    The filter is described by:
        H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
    
    State variables maintain filter history for sample-accurate processing.
    """
    
    def __init__(self, freq_min: float = 400.0, freq_max: float = 2500.0, 
                 q: float = 2.0, fs: int = 44100):
        """
        Initialize biquad peaking filter.
        
        Args:
            freq_min: Minimum center frequency (Hz) for scaling control signal
            freq_max: Maximum center frequency (Hz) for scaling control signal
            q: Quality factor (resonance); higher Q = narrower peak
            fs: Sample rate in Hz
        """
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.q = q
        self.fs = fs
        
        # Filter state (delay line)
        self.x1 = 0.0  # x(n-1)
        self.x2 = 0.0  # x(n-2)
        self.y1 = 0.0  # y(n-1)
        self.y2 = 0.0  # y(n-2)
        
        # Current coefficients
        self._b0 = 0.0
        self._b1 = 0.0
        self._b2 = 0.0
        self._a1 = 0.0
        self._a2 = 0.0
        
        # Initialize coefficients for center frequency
        self._update_coefficients(freq_min)
    
    def reset(self):
        """Clear filter state."""
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
    
    def _update_coefficients(self, freq_hz: float):
        # Clamp frequency to safe range
        freq_hz = np.clip(freq_hz, 20.0, self.fs / 2.1)
        
        w0 = 2.0 * np.pi * freq_hz / self.fs
        sin_w0 = np.sin(w0)
        cos_w0 = np.cos(w0)
        alpha = sin_w0 / (2.0 * self.q)
        
        # Define a rich analog-style boost (15 dB)
        gain_db = 15.0 
        A = 10.0 ** (gain_db / 40.0)
        
        # Standard RBJ Peaking EQ Formulas
        a0 = 1.0 + (alpha / A)
        
        self._b0 = (1.0 + alpha * A) / a0
        self._b1 = (-2.0 * cos_w0) / a0
        self._b2 = (1.0 - alpha * A) / a0
        self._a1 = (-2.0 * cos_w0) / a0
        self._a2 = (1.0 - alpha / A) / a0
    
    def process(self, audio_block: np.ndarray, freq_control: np.ndarray) -> np.ndarray:
        """
        Process audio block with time-varying center frequency.
        
        Args:
            audio_block: 1-D numpy array of float samples
            freq_control: 1-D numpy array of normalized control values [0, 1],
                mapped to [freq_min, freq_max]
        
        Returns:
            1-D numpy array of filtered samples
        """
        block_size = len(audio_block)
        output = np.zeros(block_size, dtype=np.float32)
        
        for i in range(block_size):
            # Map control signal [0, 1] to frequency range [freq_min, freq_max]
            control_val = np.clip(freq_control[i], 0.0, 1.0)
            freq_hz = self.freq_min + control_val * (self.freq_max - self.freq_min)
            
            # Update filter coefficients for new frequency
            self._update_coefficients(freq_hz)
            
            # Apply biquad difference equation:
            # y(n) = b0*x(n) + b1*x(n-1) + b2*x(n-2) - a1*y(n-1) - a2*y(n-2)
            x_in = audio_block[i]
            y_out = (
                self._b0 * x_in + 
                self._b1 * self.x1 + 
                self._b2 * self.x2 - 
                self._a1 * self.y1 - 
                self._a2 * self.y2
            )
            
            # Update state for next sample
            self.x2 = self.x1
            self.x1 = x_in
            self.y2 = self.y1
            self.y1 = y_out
            
            output[i] = y_out
        
        return output.astype(np.float32)


# ---------------------------------------------------------------------------
# Real-time stateful wah effect
# ---------------------------------------------------------------------------

class RealtimeWahWah:
    """
    Stateful, block-by-block auto-wah (envelope filter) suitable for real-time use.
    
    Combines:
        1. Envelope follower (abs + asymmetric LP)
        2. Time-varying biquad peaking filter
    
    State is maintained between process() calls for seamless streaming.
    """
    
    def __init__(
        self,
        freq_min: float = 400.0,
        freq_max: float = 2500.0,
        q: float = 2.0,
        attack_ms: float = 5.0,
        release_ms: float = 100.0,
        fs: int = 44100,
    ):
        """
        Initialize real-time auto-wah.
        
        Args:
            freq_min: Minimum frequency (Hz) of resonant peak sweep
            freq_max: Maximum frequency (Hz) of resonant peak sweep
            q: Quality factor (resonance; higher = narrower, more pronounced)
            attack_ms: Envelope attack time in milliseconds (fast transient response)
            release_ms: Envelope release time in milliseconds (smooth decay)
            fs: Sample rate in Hz
        """
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.q = q
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.fs = fs
        
        # Components
        self._envelope_follower = EnvelopeFollower(
            attack_ms=attack_ms,
            release_ms=release_ms,
            fs=fs
        )
        self._biquad = BiquadPeaking(
            freq_min=freq_min,
            freq_max=freq_max,
            q=q,
            fs=fs
        )
    
    def reset(self):
        """Clear all internal state (call between unrelated audio streams)."""
        self._envelope_follower.reset()
        self._biquad.reset()
    
    def process(self, audio_block: np.ndarray) -> np.ndarray:
        """
        Process a single block of audio samples.
        
        Args:
            audio_block: 1-D numpy array of float samples
        
        Returns:
            Processed 1-D numpy array (same length as input)
        """
        # 1. Extract envelope from input
        envelope = self._envelope_follower.process(audio_block)
        
        # 2. Apply time-varying biquad filter driven by envelope
        output = self._biquad.process(audio_block, envelope)
        
        return output
