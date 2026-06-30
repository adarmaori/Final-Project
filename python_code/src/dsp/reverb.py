"""
Schroeder Reverberator architecture for electric guitar.

Signal flow:
               ┌─► [ Comb Filter 1 ] ─┐
               ├─► [ Comb Filter 2 ] ─┼─► [ Allpass 1 ] ──► [ Allpass 2 ] ──┐
    x(n) ──┬───┼─► [ Comb Filter 3 ] ─┤                                   │
           │   └─► [ Comb Filter 4 ] ─┘                                   ▼
           │                                                        (Wet Gain)
           └─────────────────────────────────────────────────────► ( + ) ──► y(n)

Where:
    - Parallel Comb Filters create the dense, frequency-dependent decay tail.
    - Series Allpass Filters increase the reflection density over time.
    - Delay times are mathematically chosen as prime ratios to prevent metallic ringing.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helper DSP Classes for Sub-Modules
# ---------------------------------------------------------------------------

class FeedbackCombFilter:
    """Feedback Comb Filter: y(n) = x(n - D) + g * y(n - D)"""
    def __init__(self, delay_ms: float, gain: float, fs: int):
        self.delay_samples = int(round(delay_ms * fs / 1000.0))
        self.g = gain
        self.buffer = np.zeros(self.delay_samples, dtype=np.float32)
        self.idx = 0

    def process_sample(self, x_in: float) -> float:
        # Read delayed output
        out = self.buffer[self.idx]
        # Feedback equation
        self.buffer[self.idx] = x_in + self.g * out
        # Increment circular buffer
        self.idx = (self.idx + 1) % self.delay_samples
        return out


class AllPassFilter:
    """All-Pass Filter: y(n) = -g * x(n) + x(n - D) + g * y(n - D)"""
    def __init__(self, delay_ms: float, gain: float, fs: int):
        self.delay_samples = int(round(delay_ms * fs / 1000.0))
        self.g = gain
        self.buffer = np.zeros(self.delay_samples, dtype=np.float32)
        self.idx = 0

    def process_sample(self, x_in: float) -> float:
        # Read delayed buffer value
        v_delayed = self.buffer[self.idx]
        # Main feed-forward/feedback node calculation
        v_new = x_in + self.g * v_delayed
        # Allpass output equation
        y_out = -self.g * v_new + v_delayed
        # Store in buffer
        self.buffer[self.idx] = v_new
        # Increment circular buffer
        self.idx = (self.idx + 1) % self.delay_samples
        return y_out


# ---------------------------------------------------------------------------
# Offline Reverb Function
# ---------------------------------------------------------------------------

def reverb_effect(
    audio_signal: np.ndarray,
    room_size: float = 0.75,
    wet_level: float = 0.35,
    fs: int = 44100,
) -> np.ndarray:
    """
    Offline Schroeder reverb effect for batch processing.
    """
    processor = RealtimeReverb(room_size=room_size, wet_level=wet_level, fs=fs)
    
    block_size = 2048
    output = np.zeros_like(audio_signal, dtype=np.float32)
    
    for i in range(0, len(audio_signal), block_size):
        block = audio_signal[i : i + block_size]
        output[i : i + block_size] = processor.process(block)
    
    return output


# ---------------------------------------------------------------------------
# Real-Time Stateful Reverb
# ---------------------------------------------------------------------------

class RealtimeReverb:
    """
    Stateful, block-by-block Schroeder Reverberator.
    
    Maintains internal states of 4 parallel comb filters and 2 series allpass filters.
    Optimized for simulating vintage room/spring style behaviors for electric guitar.
    """
    def __init__(
        self,
        room_size: float = 0.75,
        wet_level: float = 0.35,
        fs: int = 44100,
    ):
        """
        Args:
            room_size: Internal feedback gain factor (0.0 to 0.95 controls decay time)
            wet_level: Mix amount of wet signal (0.0 to 1.0)
            fs: Sample rate in Hz
        """
        self.room_size = np.clip(room_size, 0.0, 0.95)
        self.wet_level = np.clip(wet_level, 0.0, 1.0)
        self.fs = fs
        
        # Classic Schroeder delay matrices (in milliseconds)
        # Prime/Coprime values reduce distinct echoing patterns
        comb_delays = [29.7, 37.1, 41.1, 43.7]
        allpass_delays = [5.0, 1.7]
        
        # Fixed scaling for comb decay layout
        # Giving each comb slightly different gains spaces out individual resonances
        comb_gains = [self.room_size, self.room_size - 0.03, self.room_size - 0.05, self.room_size - 0.07]
        
        # Initialize sub-modules
        self._combs = [
            FeedbackCombFilter(delay_ms=d, gain=g, fs=fs)
            for d, g in zip(comb_delays, comb_gains)
        ]
        
        # Fixed stable gain for standard diffusion pass
        self._allpasses = [
            AllPassFilter(delay_ms=d, gain=0.707, fs=fs)
            for d in allpass_delays
        ]
        
    def reset(self):
        """Clear all internal buffers."""
        for comb in self._combs:
            comb.buffer.fill(0.0)
            comb.idx = 0
        for ap in self._allpasses:
            ap.buffer.fill(0.0)
            ap.idx = 0

    def process(self, audio_block: np.ndarray) -> np.ndarray:
        block_size = len(audio_block)
        output = np.zeros(block_size, dtype=np.float32)
        
        for i in range(block_size):
            x_in = audio_block[i]
            
            # 1. Process through the 4 parallel comb filters
            comb_out_mix = 0.0
            for comb in self._combs:
                comb_out_mix += comb.process_sample(x_in)
            
            # Normalizing parallel gain accumulator path
            comb_out_mix *= 0.25
            
            # 2. Process sequentially through series all-pass diffusion network
            wet_sample = comb_out_mix
            for ap in self._allpasses:
                wet_sample = ap.process_sample(wet_sample)
                
            # 3. Apply dry/wet linear blend formula
            # y(n) = (1 - wet) * x(n) + wet * wet_sample(n)
            output[i] = ((1.0 - self.wet_level) * x_in) + (self.wet_level * wet_sample)
            
        return output.astype(np.float32)