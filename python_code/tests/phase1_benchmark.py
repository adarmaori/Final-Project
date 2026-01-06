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

from src.dsp.distortion import tanh_distortion, tube_saturator
from src.nn.architecture import SimpleTCN

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
    # Drive=10 gives nice crunch, Asymmetry=0.3 adds warmth, Tone=4000 removes fizz
    y_dsp = tube_saturator(y, drive=20.0, asymmetry=0.3, tone=4000, fs=sr)
    dsp_time = time.perf_counter() - start_time
    print(f"DSP Processing Time: {dsp_time:.6f}s")
    
    # Save DSP output
    dsp_out_path = os.path.join(output_dir, "phase1_dsp_output.wav")
    sf.write(dsp_out_path, y_dsp, sr)
    print(f"Saved DSP output to {dsp_out_path}")

    # 3. Run NN Inference (Dummy Model)
    print("\nRunning NN Inference (Simple TCN)...")
    # Prepare input for PyTorch (Batch, Channels, Length)
    # TCN expects (Batch, Channels, Length)
    y_tensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0) 
    
    model = SimpleTCN()
    model.eval() # Set to evaluation mode
    
    start_time = time.perf_counter()
    with torch.no_grad():
        y_nn_tensor = model(y_tensor)
    nn_time = time.perf_counter() - start_time
    
    y_nn = y_nn_tensor.squeeze().numpy()
    print(f"NN Processing Time: {nn_time:.6f}s")
    
    # Save NN output
    nn_out_path = os.path.join(output_dir, "phase1_nn_output.wav")
    # Normalize NN output if needed (random weights can produce huge values)
    if np.max(np.abs(y_nn)) > 1.0:
        y_nn = y_nn / np.max(np.abs(y_nn)) * 0.9
        
    sf.write(nn_out_path, y_nn, sr)
    print(f"Saved NN output to {nn_out_path}")

    # 4. Compare
    print("\n--- Comparison ---")
    print(f"DSP Time: {dsp_time:.6f}s")
    print(f"NN Time:  {nn_time:.6f}s")
    print(f"Ratio (NN/DSP): {nn_time/dsp_time:.2f}x slower")
    
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
    plt.title("NN Output (Untrained)")
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
    test_file_name = "funk-soul-guitar-clean-4_90bpm_G.wav"
    test_file = os.path.join(project_root, test_file_name)
    
    if not os.path.exists(test_file):
        # Generate synthetic if file doesn't exist
        print(f"Test file not found at {test_file}, generating synthetic signal...")
        sr = 44100
        t = np.linspace(0, 5, 5*sr)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(test_file, y, sr)
        print(f"Generated synthetic file at {test_file}")
        
    run_phase1(test_file)
