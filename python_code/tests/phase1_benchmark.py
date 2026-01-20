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
from src.engine.wrapper import NNWrapper, DSPWrapper

def run_benchmark_suite(input_file, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

    print(f"--- Phase 1 Benchmark Suite: {os.path.basename(input_file)} ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Configuration ---
    N_RUNS = 10  # Number of runs for statistics
    BLOCK_SIZE = 512 # For real-time simulation
    TEST_DURATIONS = [1.0, 5.0, 10.0] # Seconds to test scalability
    
    # Define Models to Benchmark (Add new models here)
    project_models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'checkpoints')
    
    models_config = [
        {
            "name": "Causal TCN (Final)",
            "path": os.path.join(project_models_dir, 'tcn_final.pt'),
            "active": True,
            "color": "blue"
        },
        {
            "name": "TCN (Small) [Placeholder]", 
            "path": os.path.join(project_models_dir, 'tcn_small.pt'),
            "active": False, # Set to True when model exists
            "color": "green"
        },
        {
            "name": "Use Optimized ONNX [Placeholder]",
            "path": os.path.join(project_models_dir, 'model_opt.onnx'),
            "active": False, 
            "color": "red"
        }
    ]

    # --- 1. Load Data ---
    try:
        y_full, sr = librosa.load(input_file, sr=None)
        print(f"Loaded audio: {len(y_full)} samples, {sr} Hz ({len(y_full)/sr:.2f}s)")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    report_lines = []
    report_lines.append(f"Phase 1 Detailed Benchmark Report")
    report_lines.append(f"==============================")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Input: {os.path.basename(input_file)} ({len(y_full)/sr:.2f}s @ {sr}Hz)")
    report_lines.append(f"Test Configuration: {N_RUNS} runs/test, Block Size: {BLOCK_SIZE}")
    report_lines.append("")

    # --- 2. Initialize Models ---
    wrappers = {}
    
    # 2a. DSP Baseline
    dsp_wrapper = DSPWrapper(tube_saturator, drive=70.0, asymmetry=0.4, tone=5000, fs=sr)
    wrappers["DSP Match"] = dsp_wrapper
    
    # 2b. NN Models
    for cfg in models_config:
        if not cfg['active']:
            report_lines.append(f"Skipping {cfg['name']} (Inactive/Placeholder)")
            continue
            
        if os.path.exists(cfg['path']):
            # Instantiate Wrapper
            w = NNWrapper(model_path=cfg['path'])
            wrappers[cfg['name']] = w
            
            # File size stats
            size_mb = os.path.getsize(cfg['path']) / (1024 * 1024)
            param_count = sum(p.numel() for p in w.model.parameters())
            report_lines.append(f"Loaded {cfg['name']}: {size_mb:.2f} MB, {param_count:,} params")
        else:
            report_lines.append(f"Warning: {cfg['name']} file not found at {cfg['path']}")

    report_lines.append("")
    
    # --- 3. Experiment A: Full File Statitics (Accuracy & Speed) ---
    print("\nStarting Experiment A: Full File Stats...")
    results_A = {}
    
    report_lines.append("Experiment A: Batch Processing Checks")
    report_lines.append("-------------------------------------")
    report_lines.append(f"{'Model':<20} | {'Time(ms)':<10} | {'RTF':<8} | {'MSE':<10} | {'Speed vs DSP':<12}")
    
    # Values for plotting
    model_names_plot = []
    rtfs_plot = []
    
    # Get DSP ground truth first for MSE
    y_dsp_ref = wrappers["DSP Match"].process(y_full)
    
    for name, wrapper in wrappers.items():
        # Timing Loop
        times = []
        for i in range(N_RUNS):
            start = time.perf_counter()
            y_out = wrapper.process(y_full)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        rtf = avg_time / (len(y_full)/sr)
        
        # Accuracy Loop (Once)
        y_out = wrapper.process(y_full)
        if name == "DSP Match":
            mse = 0.0
        else:
            # Normalize length
            L = min(len(y_out), len(y_dsp_ref))
            mse = np.mean((y_out[:L] - y_dsp_ref[:L])**2)
            
            # Save Output
            out_name = f"output_{name.replace(' ', '_')}.wav"
            sf.write(os.path.join(output_dir, out_name), y_out, sr)
            
        ratio_str = "1.0x"
        if name != "DSP Match":
            dsp_time = results_A.get("DSP Match", {}).get("avg_time", avg_time) # minor bug handling if DSP not first, but it is inserted first
            ratio_str = f"{avg_time/dsp_time:.2f}x slower"
            
        results_A[name] = {"avg_time": avg_time, "rtf": rtf, "mse": mse}
        model_names_plot.append(name)
        rtfs_plot.append(rtf)
        
        report_lines.append(f"{name:<20} | {avg_time*1000:8.2f}ms | {rtf:6.4f}   | {mse:.2e}   | {ratio_str}")

    report_lines.append("")

    # --- 4. Experiment B: Scalability (File Size) ---
    print("Starting Experiment B: Scalability...")
    report_lines.append("Experiment B: Scalability (RTF over different durations)")
    report_lines.append("------------------------------------------------------")
    
    scalability_results = {name: [] for name in wrappers.keys()}
    
    header = f"{'Duration':<10} |" + " | ".join([f"{name:<15}" for name in wrappers.keys()])
    report_lines.append(header)
    
    for dur in TEST_DURATIONS:
        # Prepare slice
        n_samples = int(dur * sr)
        if n_samples > len(y_full): break
        y_slice = y_full[:n_samples]
        
        row_str = f"{dur:<10.1f} |"
        
        for name, wrapper in wrappers.items():
            start = time.perf_counter()
            _ = wrapper.process(y_slice)
            t = time.perf_counter() - start
            rtf = t / dur
            scalability_results[name].append(rtf)
            row_str += f" {rtf:<15.4f} |"
        
        report_lines.append(row_str)
            
    report_lines.append("")

    # --- 5. Experiment C: Simulated Real-Time (Block Jitter) ---
    print("Starting Experiment C: Simulated Real-Time...")
    report_lines.append(f"Experiment C: Simulated Real-Time (Block Size: {BLOCK_SIZE})")
    report_lines.append("-------------------------------------------------------")
    report_lines.append(f"{'Model':<20} | {'Avg Latency':<12} | {'Max Latency':<12} | {'Processing Load (%)':<20}")

    block_data_plot = []
    
    num_blocks = len(y_full) // BLOCK_SIZE
    # Limit to e.g. 500 blocks to save time if file is huge
    num_blocks = min(num_blocks, 500)
    
    block_duraion_ms = (BLOCK_SIZE / sr) * 1000
    report_lines.append(f"Budget per block: {block_duraion_ms:.2f} ms")

    for name, wrapper in wrappers.items():
        if "DSP" in name: 
            # We skip DSP for complex jitter plot generally, or keep it. Let's keep it.
            pass
            
        latencies_ms = []
        for i in range(num_blocks):
            # Grab chunk
            chunk = y_full[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE]
            
            t0 = time.perf_counter()
            _ = wrapper.process(chunk)
            dt_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(dt_ms)
            
        avg_lat = np.mean(latencies_ms)
        max_lat = np.max(latencies_ms)
        p99_lat = np.percentile(latencies_ms, 99)
        
        # Load % = Avg Latency / Budget
        load_pct = (avg_lat / block_duraion_ms) * 100
        
        report_lines.append(f"{name:<20} | {avg_lat:8.4f} ms | {max_lat:8.4f} ms | {load_pct:6.2f}% (P99: {p99_lat:.2f}ms)")
        block_data_plot.append(latencies_ms)

    # --- 6. Plotting ---
    print("Generating Plots...")
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: RTF Comparison (Experiment A)
    ax1 = plt.subplot(2, 2, 1)
    ax1.bar(model_names_plot, rtfs_plot, color=['gray' if 'DSP' in x else 'blue' for x in model_names_plot])
    ax1.set_title("Real-Time Factor (Lower is Better)")
    ax1.set_ylabel("RTF (Proc Time / Audio Time)")
    ax1.axhline(y=1.0, color='r', linestyle='--', label="Real-Time Limit")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Waveforms (DSP vs NN)
    ax2 = plt.subplot(2, 2, 2)
    slc = Slice_for_vis = slice(1000, 2000) # Zoom in
    ax2.plot(y_full[slc], label="Input", alpha=0.5, color='gray')
    ax2.plot(y_dsp_ref[slc], label="DSP Target", color='black', linewidth=1)
    
    for name, wrapper in wrappers.items():
        if "NN" in name or "TCN" in name:
            # Quick process to get fresh plot data if needed, or use saved logic? 
            # We'll just run inference on this slice specifically to be cleaner
            out_slice = wrapper.process(y_full[slc])
            ax2.plot(out_slice, label=f"Output: {name}", linestyle='--')
            
    ax2.set_title("Waveform Detail (Sample 1000-2000)")
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Latency Distribution (Experiment C)
    ax3 = plt.subplot(2, 1, 2)
    ax3.boxplot(block_data_plot, labels=wrappers.keys())
    ax3.set_title(f"Simulated Real-Time Latency Jitter (Block Size {BLOCK_SIZE})")
    ax3.set_ylabel("Latency (ms)")
    ax3.axhline(y=block_duraion_ms, color='r', linestyle='--', label=f"Budget ({block_duraion_ms:.2f}ms)")
    ax3.legend()
    ax3.grid(True, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "benchmark_suite_plots.png")
    plt.savefig(plot_path)
    print(f"Saved plots to {plot_path}")

    # --- 7. Save Report ---
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    # Save timestamped
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"report_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
        
    # Save latest
    with open(os.path.join(output_dir, "report_latest.txt"), "w") as f:
        f.write(report_text)
        
    print(f"Reports saved to {output_dir}")

if __name__ == "__main__":
    # Use a file from the workspace if available
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_file_name = "../raw_sound_files/funk-soul-guitar-clean-4_90bpm_G.wav"
    test_file = os.path.join(project_root, test_file_name)
    
    # Fallback generation
    if not os.path.exists(test_file):
        print(f"Test file not found at {test_file}, generating synthetic signal...")
        sr = 44100
        t = np.linspace(0, 10, 10*sr) # Make it 10 seconds for scalability test
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        sf.write(test_file, y, sr)
        print(f"Generated synthetic file at {test_file}")
        
    run_benchmark_suite(test_file)
