"""
Test suite for flanger DSP effect.

Validates:
  1. LFO generation (sine, triangle, normalized)
  2. Flanger processing (real-time and offline)
  3. Integration with engine wrapper
  4. Parameter variations
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from src.dsp.lfo import SineLFO, TriangleLFO, NormalizedLFO
from src.dsp.flanger import RealtimeFlanger, flanger_effect
from src.engine.wrapper import RealtimeDSPWrapper


def test_lfo_sine():
    """Test sine LFO generation."""
    print("\n[TEST] Sine LFO")
    lfo = SineLFO(frequency=0.5, fs=44100)
    
    # Generate 1 second of LFO
    samples = lfo.get_samples(44100)
    
    # Should complete exactly 0.5 cycles (0.5 Hz for 1 sec)
    assert len(samples) == 44100, "Sample count mismatch"
    assert samples.min() >= -1.0 and samples.max() <= 1.0, "LFO out of range [-1, 1]"
    
    # Phase should advance by 0.5 (half cycle)
    assert 0.499 < lfo.phase < 0.501, f"Phase should be ~0.5; got {lfo.phase}"
    
    print(f"  ✓ Generated {len(samples)} samples")
    print(f"  ✓ Range: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"  ✓ Phase advanced correctly to {lfo.phase:.4f}")


def test_lfo_triangle():
    """Test triangle LFO generation."""
    print("\n[TEST] Triangle LFO")
    lfo = TriangleLFO(frequency=1.0, fs=44100)
    
    samples = lfo.get_samples(44100)
    
    assert samples.min() >= -1.0 and samples.max() <= 1.0, "LFO out of range"
    # Triangle should reach ±1 values
    assert samples.max() > 0.95, "Triangle should reach near +1"
    assert samples.min() < -0.95, "Triangle should reach near -1"
    
    print(f"  ✓ Generated {len(samples)} samples")
    print(f"  ✓ Range: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"  ✓ Triangle reaches extremes as expected")


def test_normalized_lfo():
    """Test normalized LFO wrapper."""
    print("\n[TEST] Normalized LFO (sine)")
    lfo = NormalizedLFO(SineLFO(frequency=0.5, fs=44100))
    
    samples = lfo.get_samples(44100)
    
    assert samples.min() >= 0.0 and samples.max() <= 1.0, "Normalized LFO out of [0, 1]"
    # For 0.5 cycles, the mean will be skewed, so just check it's in reasonable range
    assert 0.5 < np.mean(samples) < 0.95, f"Normalized sine mean unexpected: {np.mean(samples)}"
    
    print(f"  ✓ Normalized range: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"  ✓ Mean: {np.mean(samples):.3f} (valid range [0, 1])")


def test_flanger_offline():
    """Test offline flanger processing."""
    print("\n[TEST] Offline Flanger Processing")
    
    # Create test signal: 440 Hz sine, 1 second
    fs = 44100
    duration = 1.0
    t = np.arange(int(fs * duration)) / fs
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Process with flanger
    output = flanger_effect(
        signal,
        rate=0.5,
        depth=1.0,
        ff=0.7,
        fb=0.7,
        fs=fs,
    )
    
    assert output.shape == signal.shape, "Output shape mismatch"
    assert output.dtype == np.float32, "Output should be float32"
    assert np.all(np.isfinite(output)), "Output contains NaN/Inf"
    assert np.abs(output).max() <= 1.2, f"Output should be normalized, got max {np.abs(output).max()}"
    
    print(f"  ✓ Processed {len(output)} samples (1 second)")
    print(f"  ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"  ✓ No NaN/Inf values")


def test_flanger_realtime():
    """Test real-time flanger with block processing."""
    print("\n[TEST] Real-time Flanger (Block Processing)")
    
    fs = 44100
    flanger = RealtimeFlanger(rate=0.5, depth=1.0, ff=0.7, fb=0.7, fs=fs)
    
    # Create test signal
    t = np.arange(44100) / fs
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Process in blocks (typical RT chunk size)
    block_size = 2048
    output_blocks = []
    
    for i in range(0, len(signal), block_size):
        block = signal[i : i + block_size]
        output_block = flanger.process(block)
        output_blocks.append(output_block)
    
    output = np.concatenate(output_blocks)
    
    assert len(output) == len(signal), "Output length mismatch"
    assert np.all(np.isfinite(output)), "Output contains NaN/Inf"
    
    print(f"  ✓ Processed {len(signal)} samples in {len(output_blocks)} blocks")
    print(f"  ✓ Block size: {block_size} samples")
    print(f"  ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")


def test_flanger_state_persistence():
    """Test that flanger maintains state between blocks."""
    print("\n[TEST] Flanger State Persistence")
    
    fs = 44100
    flanger = RealtimeFlanger(rate=0.5, depth=1.0, ff=0.7, fb=0.7, fs=fs)
    
    # Create continuous signal
    t = np.arange(44100) / fs
    signal = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    
    # Process all at once
    output_continuous = flanger.process(signal)
    
    # Process in blocks
    flanger.reset()
    output_blocks = []
    for i in range(0, len(signal), 2048):
        block = signal[i : i + 2048]
        output_blocks.append(flanger.process(block))
    output_blocked = np.concatenate(output_blocks)
    
    # Results should be identical (state persistence is transparent)
    assert np.allclose(output_continuous, output_blocked, atol=1e-5), \
        "Block processing should match continuous"
    
    print(f"  ✓ State persistence verified")
    print(f"  ✓ Block-based output matches continuous processing")


def test_flanger_engine_wrapper():
    """Test integration with engine wrapper."""
    print("\n[TEST] Engine Wrapper Integration")
    
    # Create flanger
    flanger = RealtimeFlanger(rate=0.5, depth=1.0, fs=44100)
    
    # Wrap in RealtimeDSPWrapper
    wrapper = RealtimeDSPWrapper(flanger)
    
    assert wrapper.name == "RealtimeDSP", "Wrapper name incorrect"
    assert hasattr(wrapper, 'process'), "Wrapper missing process method"
    assert hasattr(wrapper, 'reset'), "Wrapper missing reset method"
    
    # Test processing
    test_signal = np.random.randn(4410).astype(np.float32)
    output = wrapper.process(test_signal)
    
    assert output.shape == test_signal.shape, "Output shape mismatch"
    assert np.all(np.isfinite(output)), "Output contains NaN/Inf"
    
    print(f"  ✓ Flanger wraps correctly in RealtimeDSPWrapper")
    print(f"  ✓ Wrapper.process() works")
    print(f"  ✓ Output valid: range [{output.min():.3f}, {output.max():.3f}]")


def test_flanger_parameter_sweep():
    """Test flanger with different parameter settings."""
    print("\n[TEST] Parameter Sweep")
    
    fs = 44100
    signal = np.sin(2 * np.pi * 440 * np.arange(4410) / fs).astype(np.float32)
    
    params_list = [
        {"rate": 0.1, "depth": 0.5, "ff": 0.5, "fb": 0.5},
        {"rate": 0.5, "depth": 1.0, "ff": 0.7, "fb": 0.7},
        {"rate": 1.0, "depth": 2.0, "ff": 0.9, "fb": 0.9},
    ]
    
    for i, params in enumerate(params_list, 1):
        flanger = RealtimeFlanger(**params, fs=fs)
        output = flanger.process(signal)
        
        assert np.all(np.isfinite(output)), f"Param set {i} produced NaN/Inf"
        print(f"  ✓ Param set {i}: rate={params['rate']}Hz, depth={params['depth']}ms "
              f"→ output range [{output.min():.3f}, {output.max():.3f}]")


def test_flanger_reset():
    """Test flanger reset functionality."""
    print("\n[TEST] Flanger Reset")
    
    fs = 44100
    flanger = RealtimeFlanger(rate=0.5, depth=1.0, fs=fs)
    
    # Process some audio
    signal1 = np.random.randn(1000).astype(np.float32)
    output1 = flanger.process(signal1)
    
    # Reset state
    flanger.reset()
    
    # Process same audio again
    signal2 = np.random.randn(1000).astype(np.float32)
    signal2[:] = signal1[:]  # Copy signal for comparison
    output2 = flanger.process(signal2)
    
    # Outputs should match now (same input, fresh state)
    assert np.allclose(output1, output2, atol=1e-5), "Reset did not clear state"
    
    print(f"  ✓ Reset clears internal state")
    print(f"  ✓ After reset, same input produces same output")


if __name__ == "__main__":
    print("=" * 60)
    print("FLANGER DSP TEST SUITE")
    print("=" * 60)
    
    try:
        test_lfo_sine()
        test_lfo_triangle()
        test_normalized_lfo()
        test_flanger_offline()
        test_flanger_realtime()
        test_flanger_state_persistence()
        test_flanger_engine_wrapper()
        test_flanger_parameter_sweep()
        test_flanger_reset()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
