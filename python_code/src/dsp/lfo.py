"""
Low-Frequency Oscillator (LFO) module for modulation effects.

Provides stateful, phase-based LFO generators suitable for real-time
block-by-block processing. Phase accumulates across blocks to maintain
continuity and efficiency.
"""

import numpy as np


class SineLFO:
    """
    Sine-wave Low-Frequency Oscillator.
    
    Outputs values in range [-1, 1] with smooth sinusoidal modulation.
    Phase is maintained across process() calls for streaming compatibility.
    
    Typical use:
    - Flanger/Chorus: 0.1-1 Hz
    - Vibrato: 1-10 Hz
    
    Attributes:
        frequency (float): LFO frequency in Hz
        phase (float): Current phase in [0, 1), persists across calls
    """
    
    def __init__(self, frequency: float = 0.5, fs: int = 44100):
        """
        Initialize sine LFO.
        
        Args:
            frequency: LFO frequency in Hz (e.g., 0.5 for slow flanger)
            fs: Sample rate in Hz
        """
        self.frequency = frequency
        self.fs = fs
        self.phase = 0.0
        self._phase_increment = frequency / fs
        
    def reset(self):
        """Reset phase to 0 (call between unrelated audio streams)."""
        self.phase = 0.0
    
    def get_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate n_samples of sine modulation.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            1-D numpy array of shape (n_samples,) with values in [-1, 1]
        """
        output = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(n_samples):
            # Compute sine from phase
            output[i] = np.sin(2.0 * np.pi * self.phase)
            # Advance phase
            self.phase += self._phase_increment
            # Wrap phase to [0, 1)
            if self.phase >= 1.0:
                self.phase -= 1.0
        
        return output
    
    def set_frequency(self, frequency: float):
        """Update LFO frequency (thread-safe for block processing)."""
        self.frequency = frequency
        self._phase_increment = frequency / self.fs


class TriangleLFO:
    """
    Triangle-wave Low-Frequency Oscillator.
    
    Outputs values in range [-1, 1] with linear ramps (more aggressive
    sweep than sine, useful for pronounced modulation effects).
    
    Phase is maintained across process() calls for streaming compatibility.
    
    Typical use:
    - Vintage flanger/chorus effects
    """
    
    def __init__(self, frequency: float = 0.5, fs: int = 44100):
        """
        Initialize triangle LFO.
        
        Args:
            frequency: LFO frequency in Hz
            fs: Sample rate in Hz
        """
        self.frequency = frequency
        self.fs = fs
        self.phase = 0.0
        self._phase_increment = frequency / fs
        
    def reset(self):
        """Reset phase to 0 (call between unrelated audio streams)."""
        self.phase = 0.0
    
    def get_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate n_samples of triangle modulation.
        
        Triangle waveform:
          - phase [0, 0.5): value = -1 + 4*phase (ramp from -1 to 1)
          - phase [0.5, 1): value = 3 - 4*phase (ramp from 1 to -1)
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            1-D numpy array of shape (n_samples,) with values in [-1, 1]
        """
        output = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(n_samples):
            # Triangle wave generation
            if self.phase < 0.5:
                # Ascending ramp: -1 to 1
                output[i] = -1.0 + 4.0 * self.phase
            else:
                # Descending ramp: 1 to -1
                output[i] = 3.0 - 4.0 * self.phase
            
            # Advance phase
            self.phase += self._phase_increment
            # Wrap phase to [0, 1)
            if self.phase >= 1.0:
                self.phase -= 1.0
        
        return output
    
    def set_frequency(self, frequency: float):
        """Update LFO frequency (thread-safe for block processing)."""
        self.frequency = frequency
        self._phase_increment = frequency / self.fs


class NormalizedLFO:
    """
    Wraps an LFO instance to output values in [0, 1] (normalized).
    
    Useful for delay modulation where we need non-negative values:
    normalized_value = (raw_lfo_value + 1) / 2
    
    This makes it easy to compute modulated delays:
    delay_samples = center + normalized_lfo * depth
    """
    
    def __init__(self, lfo_instance):
        """
        Initialize normalized LFO wrapper.
        
        Args:
            lfo_instance: An LFO object with get_samples() method
        """
        self.lfo = lfo_instance
    
    def reset(self):
        """Reset underlying LFO."""
        self.lfo.reset()
    
    def get_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate normalized LFO samples in [0, 1].
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            1-D numpy array with values in [0, 1]
        """
        # Get raw LFO output ([-1, 1])
        raw = self.lfo.get_samples(n_samples)
        # Map to [0, 1]
        return (raw + 1.0) / 2.0
    
    def set_frequency(self, frequency: float):
        """Update underlying LFO frequency."""
        self.lfo.set_frequency(frequency)
