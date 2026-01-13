import time
import numpy as np
import librosa
import soundfile as sf
import torch
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dsp.distortion import tube_saturator
from src.engine.wrapper import NNWrapper

def run_phase1(input_file, output_dir=None):
    if output_dir is None:
        # Default to data/processed relative to project root
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

    print(f"--- Phase 1 Benchmark: {input_file} ---")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Audio
    try:
        y, sr = librosa.load(input_file, sr=None) # Load at native SR
        print(f"Loaded audio: {len(y)} samples, {sr} Hz ({len(y)/sr:.2f}s)")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 2. Run Deterministic DSP (Improved Tube Saturator)
    print("\nRunning DSP (Tube Saturator)...")
    start_time = time.perf_counter()
    # Updated parameters from training generation (Heavy Distortion)
    y_dsp = tube_saturator(y, drive=70.0, asymmetry=0.4, tone=5000, fs=sr)
    dsp_time = time.perf_counter() - start_time
    print(f"DSP Processing Time: {dsp_time:.6f}s")
    
    # Save DSP output
    dsp_out_path = os.path.join(output_dir, "phase1_dsp_output.wav")
    sf.write(dsp_out_path, y_dsp, sr)
    print(f"Saved DSP output to {dsp_out_path}")

    # 3. Run NN Inference (Trained Model)
    print("\nRunning NN Inference (Trained TCN)...")
    
    # Locate model checkpoint
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'checkpoints', 'tcn_final.pt')
    wrapper = NNWrapper(model_path=model_path)
    
    # Get model stats
    param_count = sum(p.numel() for p in wrapper.model.parameters())
    print(f"Model Parameters: {param_count:,}")

    start_time = time.perf_counter()
    y_nn = wrapper.process(y)
    nn_time = time.perf_counter() - start_time
    
    print(f"NN Processing Time: {nn_time:.6f}s")
    
    # Save NN output
    nn_out_path = os.path.join(output_dir, "phase1_nn_output.wav")
    
    # Normalize NN output if needed (safety check)
    if np.max(np.abs(y_nn)) > 1.0:
        y_nn = y_nn / np.max(np.abs(y_nn)) * 0.99
        
    sf.write(nn_out_path, y_nn, sr)
    print(f"Saved NN output to {nn_out_path}")

    # 4. Analysis & Reporting
    print("\n--- Generating Report ---")
    
    # Metrics
    duration_s = len(y) / sr
    
    # RTF (Real Time Factor) = Processing Time / Audio Duration
    # RTF < 1.0 means faster than real-time
    dsp_rtf = dsp_time / duration_s
    nn_rtf = nn_time / duration_s
    
    # Speedup
    ratio_nn_dsp = nn_time / dsp_time
    
    # Accuracy (MSE) - how close is NN to DSP?
    # Ensure lengths match (NN might change length slightly if padding is off, but wrapper handles it usually)
    min_len = min(len(y_dsp), len(y_nn))
    mse = np.mean((y_dsp[:min_len] - y_nn[:min_len])**2)
    max_error = np.max(np.abs(y_dsp[:min_len] - y_nn[:min_len]))
    
    report_lines = [
        "Phase 1 Benchmark Report",
        "========================",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input File: {os.path.basename(input_file)}",
        f"Duration: {duration_s:.2f}s ({len(y)} samples @ {sr}Hz)",
        "",
        "Model Info",
        "----------",
        f"Architecture: Causal TCN",
        f"Parameter Count: {param_count:,}",
        f"Model Path: {os.path.basename(model_path)}",
        "",
        "Performance Metrics",
        "-------------------",
        f"DSP Time: {dsp_time:.6f}s",
        f"NN Time:  {nn_time:.6f}s",
        f"Ratio:    NN is {ratio_nn_dsp:.2f}x slower than DSP",
        "",
        "Real-Time Factor (RTF)",
        "----------------------",
        "(RTF < 1.0 means faster than real-time)",
        f"DSP RTF: {dsp_rtf:.6f}",
        f"NN RTF:  {nn_rtf:.6f}",
        f"NN Max Throughput: {len(y)/nn_time:,.0f} samples/sec",
        "",
        "Accuracy (NN validity)",
        "----------------------",
        f"MSE (Mean Squared Error): {mse:.2e}",
        f"Max Absolute Error:       {max_error:.4f}",
        "",
        "Notes",
        "-----",
        "- DSP is the 'Ground Truth' target.",
        "- Low MSE indicates the NN has successfully learned the distortion function.",
        "- If NN RTF > 1.0, the model is too heavy for real-time use on this CPU."
    ]
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save timestamped report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_filename = f"phase1_report_{timestamp}.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nSaved detailed report to {report_path}")

    # Save latest copy (overwrite)
    latest_path = os.path.join(output_dir, "phase1_report_latest.txt")
    with open(latest_path, "w") as f:
        f.write(report_text)
    print(f"Saved latest report copy to {latest_path}")

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.title("Original Signal")
    plt.plot(y[:1000]) # Plot first 1000 samples for visibility
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.title("DSP Output (Tanh)")
    plt.plot(y_dsp[:1000])
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.title("NN Output (Trained)")
    plt.plot(y_nn[:1000])
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "phase1_waveform_comparison.png")
    plt.savefig(plot_path)
    print(f"Saved comparison plot to {plot_path}")

if __name__ == "__main__":
    # Use a file from the workspace if available
    # Look in project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_file_name = "../raw_sound_files/funk-soul-guitar-clean-4_90bpm_G.wav"
    test_file = os.path.join(project_root, test_file_name)
    
    if not os.path.exists(test_file):
        # Generate synthetic if file doesn't exist
        print(f"Test file not found at {test_file}, generating synthetic sin wave signal...")
        sr = 44100
        t = np.linspace(0, 5, 5*sr)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(test_file, y, sr)
        print(f"Generated synthetic file at {test_file}")
        
    run_phase1(test_file)
