"""
Flanger audio effect - Datorro standard architecture implementation.

Signal flow (per Figure 2.34):
    x(n) ──┬────────────────────────────┬── FF*x_delayed ───┬──→ y(n)
           │                            │                   │
           │  ┌─ MOD(n) [LFO] ────┐    │                   │
           │  │                   ▼    │                   │
           │  │  z^-(M(n)+frac)   │    │                   │
           │  └───────────────────┘    │                   │
           │                            │ (feedback tap)    │
           └──── FB * x_d(n-K) ◄────────┴───────────────────┕

Where:
    - M(n): Center delay (usually 1-4 ms for audible sweeping)
  - frac: Fractional delay for smooth modulation
  - FF: Feed-forward coefficient (0.7 typical)
  - FB: Feedback coefficient (0.7 typical, stable if < 1)
  - K: Feedback tap location (fixed offset)
    - MOD(n): LFO modulation signal [-1, 1]
"""

import numpy as np
from src.dsp.lfo import SineLFO


# ---------------------------------------------------------------------------
# Offline flanger function
# ---------------------------------------------------------------------------

def flanger_effect(
    audio_signal: np.ndarray,
    rate: float = 0.5,
    depth: float = 2.0,
    center_delay: float = 2.5,
    ff: float = 0.7,
    fb: float = 0.2,
    fs: int = 44100,
) -> np.ndarray:
    """
    Offline flanger effect for batch audio processing.
    
    Processes the entire signal using a real-time flanger object to maintain
    state consistency with streaming implementation.
    
    Args:
        audio_signal: Input audio array (1-D, mono)
        rate: LFO frequency in Hz (0.1-1.0 typical)
        depth: Modulation depth in ms (typically 1-5 ms)
        center_delay: Center delay in ms around which LFO sweeps
        ff: Feed-forward coefficient (0-1, typically 0.7)
        fb: Feedback coefficient (-0.9 to 0.9, abs must be < 1)
        fs: Sample rate in Hz
    
    Returns:
        Processed audio array (same shape as input)
    """
    processor = RealtimeFlanger(
        rate=rate,
        depth=depth,
        center_delay=center_delay,
        ff=ff,
        fb=fb,
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
# Real-time stateful flanger
# ---------------------------------------------------------------------------

class RealtimeFlanger:
    """
    Stateful, block-by-block flanger suitable for real-time use.
    
    Maintains internal state (delay buffer, LFO phase) between successive
    process() calls, enabling seamless streaming.
    
    Datorro standard architecture:
        1. LFO generates modulation signal MOD(n) ∈ [-1, 1]
        2. MOD(n) modulates delay around center:
            delay(n) = center_delay + depth * MOD(n)
      3. Delayed signal x_d reads from modulated position
      4. Output: y(n) = x(n) + FF * x_d(n)
      5. Feedback tap at K samples back fed to buffer
      6. Normalization prevents clipping
    
    Parameters (Industry Standard - Table 2.9):
      - rate: 0.1-1 Hz (flanger LFO frequency)
    - depth: 1-5 ms (flanger modulation range)
    - center_delay: 1-4 ms
      - ff: ~0.7 (feed-forward coefficient)
    - fb: ~0.1 to 0.3 for cleaner tone
    """
    
    def __init__(
        self,
        rate: float = 0.5,
        depth: float = 2.0,
        center_delay: float = 2.5,
        ff: float = 0.7,
        fb: float = 0.2,
        fs: int = 44100,
    ):
        """
        Initialize real-time flanger.
        
        Args:
            rate: LFO frequency in Hz
            depth: Modulation depth in ms
            center_delay: Center delay in ms
            ff: Feed-forward gain (0-1)
            fb: Feedback coefficient (-0.9 to 0.9)
            fs: Sample rate in Hz
        """
        self.rate = rate
        self.depth = depth  # in ms
        self.center_delay = center_delay  # in ms
        self.ff = ff
        self.fb = fb
        self.fs = fs
        
        # --- LFO (sine modulation in [-1, 1]) ----
        self._lfo = SineLFO(frequency=rate, fs=fs)
        
        # --- Delay buffer (circular) ----
        # Max delay is center + depth; add margin for interpolation.
        max_delay_ms = max(0.0, center_delay + abs(depth))
        max_delay_samples = int(np.ceil(max_delay_ms * fs / 1000.0)) + 2
        max_delay_samples = max(max_delay_samples, 8)
        self._buffer = np.zeros(max_delay_samples, dtype=np.float32)
        self._buffer_idx = 0
        self._buffer_size = max_delay_samples
        
        # --- Feedback tap offset ----
        # Typical tap location: K samples back from write position
        # For flanger, K ~= buffer_size/2 to K-1
        self._fb_tap_delay = min(max_delay_samples - 1, max_delay_samples // 2)
        
        # --- Normalization factor ----
        # To avoid clipping with gain structure: 1 / (1 + FF)
        self._norm_factor = 1.0 / (1.0 + abs(ff))
        
    # ------------------------------------------------------------------
    def reset(self):
        """Clear all internal state (call between unrelated audio streams)."""
        self._buffer.fill(0.0)
        self._buffer_idx = 0
        self._lfo.reset()
    
    # ------------------------------------------------------------------
    def _linear_interpolate(self, delay_samples: float) -> float:
        """
        Read from delay buffer with linear interpolation for fractional delays.
        
        Args:
            delay_samples: Delay in samples (fractional allowed)
        
        Returns:
            Interpolated sample value
        """
        int_delay = int(delay_samples)
        frac = delay_samples - int_delay
        
        # Clamp to valid buffer range
        int_delay = min(int_delay, self._buffer_size - 1)
        
        # Read positions
        read_idx_0 = (self._buffer_idx - int_delay) % self._buffer_size
        read_idx_1 = (self._buffer_idx - int_delay - 1) % self._buffer_size
        
        # Linear interpolation: (1-frac)*y[0] + frac*y[1]
        sample_0 = self._buffer[read_idx_0]
        sample_1 = self._buffer[read_idx_1]
        
        return (1.0 - frac) * sample_0 + frac * sample_1
    
    # ------------------------------------------------------------------
    def process(self, audio_block: np.ndarray) -> np.ndarray:
        """
        Process a single block of audio samples.
        
        Args:
            audio_block: 1-D numpy array of float samples
        
        Returns:
            Processed 1-D numpy array (same length as input)
        """
        block_size = len(audio_block)
        output = np.zeros(block_size, dtype=np.float32)
        
        # Get LFO modulation for this block in [-1, 1]
        lfo_values = self._lfo.get_samples(block_size)
        
        # Convert delay controls from ms to samples
        center_samples = self.center_delay * self.fs / 1000.0
        depth_samples = self.depth * self.fs / 1000.0
        
        for i in range(block_size):
            # 1. Get input sample
            x_in = audio_block[i]
            
            # 2. Modulate delay around center with bipolar LFO.
            delay_samples = center_samples + depth_samples * lfo_values[i]
            delay_samples = max(0.0, min(delay_samples, self._buffer_size - 2.0))
            
            # 3. Read delayed sample with interpolation
            x_delayed = self._linear_interpolate(delay_samples)
            
            # 4. Compute feed-forward output
            # y(n) = x(n) + FF * x_delayed(n)
            y_raw = x_in + self.ff * x_delayed
            
            # 5. Read feedback tap (K samples back)
            fb_read_idx = (self._buffer_idx - self._fb_tap_delay) % self._buffer_size
            x_fb = self._buffer[fb_read_idx]
            
            # 6. Write to buffer: input + feedback
            # buffer[write_idx] = x(n) + FB * x_fb(n-K)
            buffer_write = x_in + self.fb * x_fb
            self._buffer[self._buffer_idx] = buffer_write
            
            # 7. Advance write pointer
            self._buffer_idx = (self._buffer_idx + 1) % self._buffer_size
            
            # 8. Normalize and store output
            output[i] = y_raw * self._norm_factor
        
        return output.astype(np.float32)
    
    # ------------------------------------------------------------------
