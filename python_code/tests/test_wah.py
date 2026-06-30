"""
Test suite for wah-wah (auto-wah) DSP effect.

Validates:
  1. Envelope follower (abs + asymmetric LP)
  2. Biquad peaking filter (time-varying coefficients)
  3. Real-time and offline wah processing
  4. Integration with engine wrapper
  5. Parameter variations
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from src.dsp.wah_wah import EnvelopeFollower, BiquadPeaking, RealtimeWahWah, wah_effect
from src.engine.wrapper import RealtimeDSPWrapper


def test_envelope_follower():
    """Test envelope follower with attack/release."""
    print("\n[TEST] Envelope Follower")
    
    fs = 44100
    env = EnvelopeFollower(attack_ms=5.0, release_ms=100.0, fs=fs)
    
    # Create test signal: quiet, then spike, then decay
    t = np.arange(fs) / fs
    signal = np.concatenate([
        np.ones(11025) * 0.1,      # Quiet (0.25 sec)
        np.ones(11025) * 0.8,      # Spike (0.25 sec)
        np.ones(22050) * 0.1,      # Decay back to quiet (0.5 sec)
    ]).astype(np.float32)
    
    envelope = env.process(signal)
    
    assert envelope.shape == signal.shape, "Envelope shape mismatch"
    assert envelope.min() >= 0.0 and envelope.max() <= 1.0, "Envelope out of range [0, 1]"
    assert np.all(np.isfinite(envelope)), "Envelope contains NaN/Inf"
    
    # Envelope should track: rise from 0.1 to ~0.8, then slowly decay
    initial_level = envelope[0]
    peak_level = np.max(envelope[11025:22050])
    final_level = envelope[-1]
    
    assert peak_level > 0.5, f"Peak should be high; got {peak_level}"
    assert initial_level < 0.2, f"Initial should be low; got {initial_level}"
    
    print(f"  ✓ Envelope tracked signal dynamics")
    print(f"  ✓ Initial: {initial_level:.3f}, Peak: {peak_level:.3f}, Final: {final_level:.3f}")
    print(f"  ✓ Range: [{envelope.min():.3f}, {envelope.max():.3f}]")


def test_biquad_peaking():
    """Test biquad peaking filter with static frequency."""
    print("\n[TEST] Biquad Peaking Filter (Static)")
    
    fs = 44100
    freq_min = 400.0
    freq_max = 2500.0
    q = 2.0
    
    biquad = BiquadPeaking(freq_min=freq_min, freq_max=freq_max, q=q, fs=fs)
    
    # Create test signal: white noise
    signal = np.random.randn(4410).astype(np.float32)
    
    # Control signal: fixed at 0.5 (middle of range = 1450 Hz)
    control = np.ones(4410) * 0.5
    
    output = biquad.process(signal, control)
    
    assert output.shape == signal.shape, "Output shape mismatch"
    assert output.dtype == np.float32, "Output should be float32"
    assert np.all(np.isfinite(output)), "Output contains NaN/Inf"
    
    print(f"  ✓ Processed {len(output)} samples")
    print(f"  ✓ Filter range: [{freq_min:.0f}, {freq_max:.0f}] Hz at center {(freq_min + freq_max)/2:.0f} Hz")
    print(f"  ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")


def test_biquad_time_varying():
    """Test biquad with time-varying control signal."""
    print("\n[TEST] Biquad Peaking Filter (Time-Varying)")
    
    fs = 44100
    biquad = BiquadPeaking(freq_min=400.0, freq_max=2500.0, q=2.0, fs=fs)
    
    # Create test signal
    signal = np.sin(2 * np.pi * 440 * np.arange(4410) / fs).astype(np.float32)
    
    # Control signal: slowly sweep from 0 to 1
    control = np.linspace(0.0, 1.0, 4410).astype(np.float32)
    
    output = biquad.process(signal, control)
    
    assert output.shape == signal.shape, "Output shape mismatch"
    assert np.all(np.isfinite(output)), "Output contains NaN/Inf"
    
    print(f"  ✓ Time-varying control processed correctly")
    print(f"  ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")


def test_wah_offline():
    """Test offline wah processing."""
    print("\n[TEST] Offline Wah Processing")
    
    # Create test signal: 440 Hz sine, 1 second
    fs = 44100
    duration = 1.0
    t = np.arange(int(fs * duration)) / fs
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Process with wah
    output = wah_effect(
        signal,
        freq_min=400.0,
        freq_max=2500.0,
        q=2.0,
        attack_ms=5.0,
        release_ms=100.0,
        fs=fs,
    )
    
    assert output.shape == signal.shape, "Output shape mismatch"
    assert output.dtype == np.float32, "Output should be float32"
    assert np.all(np.isfinite(output)), "Output contains NaN/Inf"
    
    print(f"  ✓ Processed {len(output)} samples (1 second)")
    print(f"  ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"  ✓ No NaN/Inf values")


def test_wah_realtime():
    """Test real-time wah with block processing."""
    print("\n[TEST] Real-time Wah (Block Processing)")
    
    fs = 44100
    wah = RealtimeWahWah(
        freq_min=400.0,
        freq_max=2500.0,
        q=2.0,
        attack_ms=5.0,
        release_ms=100.0,
        fs=fs,
    )
    
    # Create test signal
    t = np.arange(44100) / fs
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Process in blocks (typical RT chunk size)
    block_size = 2048
    output_blocks = []
    
    for i in range(0, len(signal), block_size):
        block = signal[i : i + block_size]
        output_block = wah.process(block)
        output_blocks.append(output_block)
    
    output = np.concatenate(output_blocks)
    
    assert len(output) == len(signal), "Output length mismatch"
    assert np.all(np.isfinite(output)), "Output contains NaN/Inf"
    
    print(f"  ✓ Processed {len(signal)} samples in {len(output_blocks)} blocks")
    print(f"  ✓ Block size: {block_size} samples")
    print(f"  ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")


def test_wah_state_persistence():
    """Test that wah maintains state between blocks."""
    print("\n[TEST] Wah State Persistence")
    
    fs = 44100
    wah = RealtimeWahWah(fs=fs)
    
    # Create continuous signal with amplitude variations
    t = np.arange(44100) / fs
    signal = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    signal *= np.abs(np.sin(2 * np.pi * 0.5 * t))  # Amplitude modulation
    
    # Process all at once
    output_continuous = wah.process(signal)
    
    # Process in blocks
    wah.reset()
    output_blocks = []
    for i in range(0, len(signal), 2048):
        block = signal[i : i + 2048]
        output_blocks.append(wah.process(block))
    output_blocked = np.concatenate(output_blocks)
    
    # Results should be identical (state persistence is transparent)
    assert np.allclose(output_continuous, output_blocked, atol=1e-5), \
        "Block processing should match continuous"
    
    print(f"  ✓ State persistence verified")
    print(f"  ✓ Block-based output matches continuous processing")


def test_wah_engine_wrapper():
    """Test integration with engine wrapper."""
    print("\n[TEST] Engine Wrapper Integration")
    
    # Create wah
    wah = RealtimeWahWah(fs=44100)
    
    # Wrap in RealtimeDSPWrapper
    wrapper = RealtimeDSPWrapper(wah)
    
    assert wrapper.name == "RealtimeDSP", "Wrapper name incorrect"
    assert hasattr(wrapper, 'process'), "Wrapper missing process method"
    assert hasattr(wrapper, 'reset'), "Wrapper missing reset method"
    
    # Test processing
    test_signal = np.random.randn(4410).astype(np.float32)
    output = wrapper.process(test_signal)
    
    assert output.shape == test_signal.shape, "Output shape mismatch"
    assert np.all(np.isfinite(output)), "Output contains NaN/Inf"
    
    print(f"  ✓ Wah wraps correctly in RealtimeDSPWrapper")
    print(f"  ✓ Wrapper.process() works")
    print(f"  ✓ Output valid: range [{output.min():.3f}, {output.max():.3f}]")


def test_wah_parameter_sweep():
    """Test wah with different parameter settings."""
    print("\n[TEST] Parameter Sweep")
    
    fs = 44100
    signal = np.sin(2 * np.pi * 440 * np.arange(4410) / fs).astype(np.float32)
    
    params_list = [
        {"freq_min": 200.0, "freq_max": 1000.0, "q": 1.0, "attack_ms": 2.0, "release_ms": 50.0},
        {"freq_min": 400.0, "freq_max": 2500.0, "q": 2.0, "attack_ms": 5.0, "release_ms": 100.0},
        {"freq_min": 500.0, "freq_max": 3000.0, "q": 3.0, "attack_ms": 10.0, "release_ms": 200.0},
    ]
    
    for i, params in enumerate(params_list, 1):
        wah = RealtimeWahWah(**params, fs=fs)
        output = wah.process(signal)
        
        assert np.all(np.isfinite(output)), f"Param set {i} produced NaN/Inf"
        print(f"  ✓ Param set {i}: freq=[{params['freq_min']:.0f}, {params['freq_max']:.0f}], "
              f"Q={params['q']:.1f} → output range [{output.min():.3f}, {output.max():.3f}]")


def test_wah_reset():
    """Test wah reset functionality."""
    print("\n[TEST] Wah Reset")
    
    fs = 44100
    wah = RealtimeWahWah(fs=fs)
    
    # Process some audio
    signal1 = np.random.randn(1000).astype(np.float32)
    output1 = wah.process(signal1)
    
    # Reset state
    wah.reset()
    
    # Process same audio again
    signal2 = np.random.randn(1000).astype(np.float32)
    signal2[:] = signal1[:]  # Copy signal for comparison
    output2 = wah.process(signal2)
    
    # Outputs should match now (same input, fresh state)
    assert np.allclose(output1, output2, atol=1e-5), "Reset did not clear state"
    
    print(f"  ✓ Reset clears internal state")
    print(f"  ✓ After reset, same input produces same output")


if __name__ == "__main__":
    print("=" * 60)
    print("WAH-WAH DSP TEST SUITE")
    print("=" * 60)
    
    try:
        test_envelope_follower()
        test_biquad_peaking()
        test_biquad_time_varying()
        test_wah_offline()
        test_wah_realtime()
        test_wah_state_persistence()
        test_wah_engine_wrapper()
        test_wah_parameter_sweep()
        test_wah_reset()
        
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
