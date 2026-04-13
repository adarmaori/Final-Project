"""
Digital Signal Processing (DSP) module for audio effects.

Provides deterministic, real-time audio effect implementations:
  - Distortion/Saturation: tube_saturator
  - Flanger: flanger_effect
  - Filtering: lowpass, tone stack
  - LFO: sine/triangle modulation generators
"""

# Import DSP effects
from src.dsp.distortion import RealtimeTubeSaturator, tube_saturator, tanh_distortion
from src.dsp.filters import lowpass_filter, tone_stack
from src.dsp.flanger import RealtimeFlanger, flanger_effect
from src.dsp.lfo import SineLFO, TriangleLFO, NormalizedLFO

__all__ = [
    # Distortion
    "RealtimeTubeSaturator",
    "tube_saturator",
    "tanh_distortion",
    # Flanger
    "RealtimeFlanger",
    "flanger_effect",
    # LFO
    "SineLFO",
    "TriangleLFO",
    "NormalizedLFO",
    # Filters
    "lowpass_filter",
    "tone_stack",
]
