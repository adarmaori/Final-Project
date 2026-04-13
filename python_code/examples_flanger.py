"""
Quick-Start: Using the Flanger DSP Effect

This file demonstrates how to use the new flanger effect in your audio pipeline.
"""

import numpy as np
from src.dsp.flanger import RealtimeFlanger, flanger_effect
from src.engine.wrapper import RealtimeDSPWrapper

# ============================================================================
# EXAMPLE 1: Offline Processing (Batch, Non-Real-Time)
# ============================================================================

def example_offline():
    """Process an entire audio file with flanger effect."""
    
    # Load or generate audio
    fs = 44100
    duration = 1.0  # 1 second
    t = np.arange(int(fs * duration)) / fs
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Apply flanger offline
    output = flanger_effect(
        audio,
        rate=0.5,           # LFO frequency (Hz)
        depth=1.0,          # Modulation depth (ms)
        ff=0.7,             # Feed-forward coefficient
        fb=0.7,             # Feedback coefficient
        fs=fs,              # Sample rate
    )
    
    print(f"Input  range: [{audio.min():.3f}, {audio.max():.3f}]")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    return output


# ============================================================================
# EXAMPLE 2: Real-Time Block Processing
# ============================================================================

def example_realtime():
    """Process audio in blocks (real-time capable)."""
    
    # Create a stateful flanger processor
    flanger = RealtimeFlanger(
        rate=0.5,           # 0.5 Hz sweep
        depth=1.0,          # 1 ms modulation
        ff=0.7,
        fb=0.7,
        fs=44100,
    )
    
    # Generate test signal
    fs = 44100
    audio = np.sin(2 * np.pi * 440 * np.arange(44100) / fs).astype(np.float32)
    
    # Process in typical RT block size
    block_size = 2048
    output = []
    
    for i in range(0, len(audio), block_size):
        block = audio[i : i + block_size]
        processed_block = flanger.process(block)
        output.append(processed_block)
    
    # Concatenate all blocks
    output = np.concatenate(output)
    
    print(f"Processed {len(audio)} samples in {len(output)} blocks")
    print(f"Block size: {block_size} samples ({block_size/fs*1000:.1f} ms)")
    
    return output


# ============================================================================
# EXAMPLE 3: Integration with Engine Wrapper
# ============================================================================

def example_with_wrapper():
    """Use flanger with the unified engine wrapper."""
    
    # Create flanger
    flanger = RealtimeFlanger(rate=0.5, depth=1.0, fs=44100)
    
    # Wrap it (same wrapper works for any stateful DSP processor)
    wrapper = RealtimeDSPWrapper(flanger)
    
    # Generate audio
    audio = np.random.randn(4410).astype(np.float32)
    
    # Process
    output = wrapper.process(audio)
    
    print(f"Wrapper name: {wrapper.name}")
    print(f"Output shape: {output.shape}")
    
    return output


# ============================================================================
# EXAMPLE 4: Parameter Sweep (Different Settings)
# ============================================================================

def example_parameter_variations():
    """Try different flanger parameter combinations."""
    
    fs = 44100
    audio = np.sin(2 * np.pi * 440 * np.arange(4410) / fs).astype(np.float32)
    
    settings = [
        {"name": "Subtle", "rate": 0.2, "depth": 0.5, "ff": 0.5, "fb": 0.5},
        {"name": "Classic", "rate": 0.5, "depth": 1.0, "ff": 0.7, "fb": 0.7},
        {"name": "Deep", "rate": 1.0, "depth": 2.0, "ff": 0.7, "fb": 0.7},
        {"name": "Fast", "rate": 2.0, "depth": 1.5, "ff": 0.8, "fb": 0.8},
    ]
    
    for setting in settings:
        name = setting.pop("name")
        flanger = RealtimeFlanger(**setting, fs=fs)
        output = flanger.process(audio)
        
        print(f"{name:10} | rate={setting['rate']}Hz, depth={setting['depth']}ms → "
              f"out=[{output.min():.3f}, {output.max():.3f}]")


# ============================================================================
# EXAMPLE 5: State Reset (Between Independent Streams)
# ============================================================================

def example_state_reset():
    """Demonstrates resetting state between different audio streams."""
    
    flanger = RealtimeFlanger(rate=0.5, depth=1.0, fs=44100)
    
    # Process first audio stream
    stream1 = np.random.randn(1000).astype(np.float32)
    output1 = flanger.process(stream1)
    
    print(f"Stream 1 processed ({len(stream1)} samples)")
    print(f"  Output range: [{output1.min():.3f}, {output1.max():.3f}]")
    
    # Reset state before processing unrelated stream
    flanger.reset()
    
    # Process second stream
    stream2 = np.random.randn(1000).astype(np.float32)
    output2 = flanger.process(stream2)
    
    print(f"Stream 2 processed ({len(stream2)} samples) - state was reset")
    print(f"  Output range: [{output2.min():.3f}, {output2.max():.3f}]")


# ============================================================================
# EXAMPLE 6: Comparison with Offline Function
# ============================================================================

def example_offline_vs_realtime():
    """Show that offline and real-time produce identical results."""
    
    fs = 44100
    audio = np.sin(2 * np.pi * 440 * np.arange(4410) / fs).astype(np.float32)
    
    # Offline processing
    output_offline = flanger_effect(audio, rate=0.5, depth=1.0, ff=0.7, fb=0.7, fs=fs)
    
    # Real-time processing (same audio)
    flanger = RealtimeFlanger(rate=0.5, depth=1.0, ff=0.7, fb=0.7, fs=fs)
    output_realtime = flanger.process(audio)
    
    # Compare
    difference = np.abs(output_offline - output_realtime).max()
    
    print(f"Offline output   range: [{output_offline.min():.3f}, {output_offline.max():.3f}]")
    print(f"Real-time output range: [{output_realtime.min():.3f}, {output_realtime.max():.3f}]")
    print(f"Max difference: {difference:.2e} (should be << 1e-5)")
    
    if difference < 1e-5:
        print("✓ Outputs are identical!")
    else:
        print("⚠ Outputs differ (check block boundaries)")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FLANGER DSP - QUICK START EXAMPLES")
    print("=" * 70)
    
    print("\n[EXAMPLE 1] Offline Processing")
    print("-" * 70)
    example_offline()
    
    print("\n[EXAMPLE 2] Real-Time Block Processing")
    print("-" * 70)
    example_realtime()
    
    print("\n[EXAMPLE 3] Engine Wrapper Integration")
    print("-" * 70)
    example_with_wrapper()
    
    print("\n[EXAMPLE 4] Parameter Variations")
    print("-" * 70)
    example_parameter_variations()
    
    print("\n[EXAMPLE 5] State Reset Between Streams")
    print("-" * 70)
    example_state_reset()
    
    print("\n[EXAMPLE 6] Offline vs. Real-Time Comparison")
    print("-" * 70)
    example_offline_vs_realtime()
    
    print("\n" + "=" * 70)
    print("✓ All examples completed successfully!")
    print("=" * 70)
