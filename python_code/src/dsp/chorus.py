"""
Chorus audio effect - Multi-voice simulation architecture.

Signal flow:
    x(n) ──┬────────────────────────────┬── FF*x_delayed ───┬──→ y(n)
           │                            │                   │
           │  ┌─ MOD(n) [LFO] ────┐    │                   │
           │  │                   ▼    │                   │
           │  │  z^-(M(n)+frac)   │    │                   │
           │  └───────────────────┘    │                   │
           │                            │ (feedback tap)    │
           └──── FB * x_d(n-K) ◄────────┴───────────────────┕

Where:
    - M(n): Modulated delay line = center_delay + depth * MOD(n)
    - center_delay: 15-30 ms (prevents flanging/comb filtering)
    - depth: 1-5 ms modulation width (controls pitch deviation pitch intensity)
    - FF: Feed-forward coefficient (0.7 typical for equal mix)
    - FB: Feedback coefficient (0.0 to 0.1 typical, usually left at 0 for chorus)
    - MOD(n): LFO modulation signal [-1, 1]
"""

import numpy as np
from src.dsp.lfo import SineLFO


# ---------------------------------------------------------------------------
# Offline chorus function
# ---------------------------------------------------------------------------

def chorus_effect(
    audio_signal: np.ndarray,
    rate: float = 1.0,
    depth: float = 3.0,
    center_delay: float = 20.0,
    ff: float = 0.7,
    fb: float = 0.0,
    fs: int = 44100,
) -> np.ndarray:
    """
    Offline chorus effect for batch audio processing.
    
    Args:
        audio_signal: Input audio array (1-D, mono)
        rate: LFO frequency in Hz (0.5-2.0 Hz typical)
        depth: Modulation depth in ms (1.0-5.0 ms typical)
        center_delay: Mean delay in ms (15.0-30.0 ms typical)
        ff: Feed-forward coefficient (0-1)
        fb: Feedback coefficient (0-0.2, typically 0 for clean chorus)
        fs: Sample rate in Hz
    
    Returns:
        Processed audio array (same shape as input)
    """
    processor = RealtimeChorus(
        rate=rate,
        depth=depth,
        center_delay=center_delay,
        ff=ff,
        fb=fb,
        fs=fs,
    )
    
    block_size = 2048
    output = np.zeros_like(audio_signal, dtype=np.float32)
    
    for i in range(0, len(audio_signal), block_size):
        block = audio_signal[i : i + block_size]
        output[i : i + block_size] = processor.process(block)
    
    return output


# ---------------------------------------------------------------------------
# Real-time stateful chorus
# ---------------------------------------------------------------------------

class RealtimeChorus:
    """
    Stateful, block-by-block chorus effect.
    
    Maintains internal delay buffers and LFO phases across real-time blocks.
    Modulates delay dynamically around a large center offset to generate pitch vibrato.
    """
    
    def __init__(
        self,
        rate: float = 1.0,
        depth: float = 3.0,
        center_delay: float = 20.0,
        ff: float = 0.7,
        fb: float = 0.0,
        fs: int = 44100,
    ):
        self.rate = rate
        self.depth = depth
        self.center_delay = center_delay
        self.ff = ff
        self.fb = fb
        self.fs = fs
        
        # --- LFO (sine modulation in [-1, 1]) ----
        self._lfo = SineLFO(frequency=rate, fs=fs)
        
        # --- Delay buffer allocation ----
        # Max delay is center + depth. Add a safety buffer margin for interpolation.
        max_delay_ms = center_delay + depth
        max_delay_samples = int(np.ceil(max_delay_ms * fs / 1000.0)) + 4
        
        self._buffer = np.zeros(max_delay_samples, dtype=np.float32)
        self._buffer_idx = 0
        self._buffer_size = max_delay_samples
        
        # --- Feedback tap offset ----
        self._fb_tap_delay = min(max_delay_samples - 1, max_delay_samples // 2)
        
        # --- Normalization factor ----
        fb_abs = min(abs(fb), 0.99)
        delayed_gain_bound = 1.0 / (1.0 - fb_abs)
        self._norm_factor = 1.0 / (1.0 + abs(ff) * delayed_gain_bound)
        
    def reset(self):
        """Clear delay lines and reset LFO phase."""
        self._buffer.fill(0.0)
        self._buffer_idx = 0
        self._lfo.reset()
    
    def _linear_interpolate(self, delay_samples: float) -> float:
        """Read from circular buffer with fractional linear interpolation."""
        int_delay = int(delay_samples)
        frac = delay_samples - int_delay
        
        int_delay = min(int_delay, self._buffer_size - 1)
        
        read_idx_0 = (self._buffer_idx - int_delay) % self._buffer_size
        read_idx_1 = (self._buffer_idx - int_delay - 1) % self._buffer_size
        
        sample_0 = self._buffer[read_idx_0]
        sample_1 = self._buffer[read_idx_1]
        
        return (1.0 - frac) * sample_0 + frac * sample_1
    
    def process(self, audio_block: np.ndarray) -> np.ndarray:
        block_size = len(audio_block)
        output = np.zeros(block_size, dtype=np.float32)
        
        # Retrieve LFO stream for this block
        lfo_values = self._lfo.get_samples(block_size)
        
        # Pre-calculate sample offsets
        center_samples = self.center_delay * self.fs / 1000.0
        depth_samples = self.depth * self.fs / 1000.0
        
        for i in range(block_size):
            x_in = audio_block[i]
            
            # Modulate delay: center_delay +/- depth
            delay_samples = center_samples + (depth_samples * lfo_values[i])
            
            # Guard bounds
            delay_samples = max(0.0, min(delay_samples, self._buffer_size - 2.0))
            
            # Read delayed voice
            x_delayed = self._linear_interpolate(delay_samples)
            
            # Output generation: y(n) = x(n) + FF * x_delayed(n)
            y_raw = x_in + self.ff * x_delayed
            
            # Feedback processing
            fb_read_idx = (self._buffer_idx - self._fb_tap_delay) % self._buffer_size
            x_fb = self._buffer[fb_read_idx]
            
            # Update delay line
            buffer_write = x_in + self.fb * x_fb
            self._buffer[self._buffer_idx] = buffer_write
            
            # Step circular pointer forward
            self._buffer_idx = (self._buffer_idx + 1) % self._buffer_size
            
            # Write normalized output
            output[i] = y_raw * self._norm_factor
            
        return output.astype(np.float32)