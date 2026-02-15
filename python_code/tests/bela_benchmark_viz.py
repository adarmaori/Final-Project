"""
Bela Benchmark Visualisation
=============================
Reads the CSV timing logs and distorted WAV exported from the Bela board and
produces plots comparable to phase1_benchmark.py (Experiment C style).

Expected files in  ../cpp_code/:
    benchmark1.csv        – NN model timing (block_idx, model_ns, block_ns)
    benchmark_baseline.csv – DSP baseline timing
    distorted.wav         – Bela NN output audio

Usage:
    python tests/bela_benchmark_viz.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

# Add project root to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from src.dsp.distortion import RealtimeTubeSaturator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
CPP_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'cpp_code'))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Bela hardware constants
BELA_SR = 44100
BELA_BLOCK_SIZE = 16  # Bela default audio frames per block

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_bela_csv(path: str) -> pd.DataFrame:
    """Load a Bela benchmark CSV and add derived columns."""
    df = pd.read_csv(path)
    df['model_ms'] = df['model_ns'] / 1e6
    df['block_ms'] = df['block_ns'] / 1e6
    return df


def summary_stats(df: pd.DataFrame, col: str = 'block_ms') -> dict:
    """Return timing summary for a given column (in ms)."""
    vals = df[col].values
    budget_ms = (BELA_BLOCK_SIZE / BELA_SR) * 1e3  # ~0.363 ms
    return {
        'mean': np.mean(vals),
        'std': np.std(vals),
        'median': np.median(vals),
        'max': np.max(vals),
        'p99': np.percentile(vals, 99),
        'load_pct': np.mean(vals) / budget_ms * 100,
        'budget_ms': budget_ms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load CSVs ---
    nn_csv_path = os.path.join(CPP_DIR, 'benchmark1.csv')
    dsp_csv_path = os.path.join(CPP_DIR, 'benchmark_baseline.csv')
    distorted_wav_path = os.path.join(CPP_DIR, 'distorted.wav')

    for p in [nn_csv_path, dsp_csv_path]:
        if not os.path.exists(p):
            print(f"Error: {p} not found"); return

    df_nn = load_bela_csv(nn_csv_path)
    df_dsp = load_bela_csv(dsp_csv_path)

    budget_ms = (BELA_BLOCK_SIZE / BELA_SR) * 1e3

    # --- Summary report ---
    print("=" * 60)
    print("Bela Real-Time Benchmark Report")
    print("=" * 60)
    print(f"Sample Rate: {BELA_SR} Hz  |  Block Size: {BELA_BLOCK_SIZE}")
    print(f"Budget per block: {budget_ms:.3f} ms")
    print(f"Total blocks: {len(df_nn)}")
    print(f"Audio duration: {len(df_nn) * BELA_BLOCK_SIZE / BELA_SR:.2f} s")
    print()

    report_lines = []
    report_lines.append(f"{'Metric':<14s} | {'All-Pass (y=x)':<16s} | {'NN Model':<16s}")
    report_lines.append("-" * 52)

    stats_ap = summary_stats(df_dsp, 'block_ms')
    stats_nn = summary_stats(df_nn, 'block_ms')

    for label, key in [('Mean (ms)', 'mean'), ('Std (ms)', 'std'),
                        ('Median (ms)', 'median'), ('Max (ms)', 'max'),
                        ('P99 (ms)', 'p99'), ('Proc. Load (%)', 'load_pct')]:
        report_lines.append(
            f"{label:<14s} | {stats_ap[key]:<16.4f} | {stats_nn[key]:<16.4f}"
        )

    model_stats_ap = summary_stats(df_dsp, 'model_ms')
    model_stats_nn = summary_stats(df_nn, 'model_ms')
    report_lines.append("")
    report_lines.append("Model-only timing (excluding I/O overhead):")
    report_lines.append(f"{'Mean (ms)':<14s} | {model_stats_ap['mean']:<16.4f} | {model_stats_nn['mean']:<16.4f}")
    report_lines.append(f"{'Max (ms)':<14s} | {model_stats_ap['max']:<16.4f} | {model_stats_nn['max']:<16.4f}")

    report_text = "\n".join(report_lines)
    print(report_text)
    print()

    # --- Plotting ---
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Bela Real-Time Benchmark", fontsize=14, fontweight='bold')

    # ---- Plot 1: Block latency over time ----
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(df_dsp['block_idx'], df_dsp['block_ms'], alpha=0.4, linewidth=0.5, label='All-Pass (y=x)', color='gray')
    ax1.plot(df_nn['block_idx'], df_nn['block_ms'], alpha=0.4, linewidth=0.5, label='NN Model', color='blue')
    ax1.axhline(y=budget_ms, color='r', linestyle='--', linewidth=1, label=f'Budget ({budget_ms:.3f} ms)')
    ax1.set_xlabel('Block Index')
    ax1.set_ylabel('Block Latency (ms)')
    ax1.set_title('Block Latency Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- Plot 2: Model-only latency over time ----
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(df_dsp['block_idx'], df_dsp['model_ms'], alpha=0.4, linewidth=0.5, label='All-Pass (y=x)', color='gray')
    ax2.plot(df_nn['block_idx'], df_nn['model_ms'], alpha=0.4, linewidth=0.5, label='NN Model', color='blue')
    ax2.set_xlabel('Block Index')
    ax2.set_ylabel('Model Latency (ms)')
    ax2.set_title('Model-Only Latency Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ---- Plot 3: Latency distribution (box plot) ----
    ax3 = plt.subplot(3, 2, 3)
    bp = ax3.boxplot(
        [df_dsp['block_ms'].values, df_nn['block_ms'].values,
         df_dsp['model_ms'].values, df_nn['model_ms'].values],
        labels=['All-Pass\n(block)', 'NN\n(block)', 'All-Pass\n(model)', 'NN\n(model)'],
        patch_artist=True,
    )
    colors = ['lightgray', 'lightblue', 'lightgray', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax3.axhline(y=budget_ms, color='r', linestyle='--', linewidth=1, label=f'Budget ({budget_ms:.3f} ms)')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('Latency Distribution')
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)

    # ---- Plot 4: Histogram ----
    ax4 = plt.subplot(3, 2, 4)
    ax4.hist(df_dsp['block_ms'], bins=80, alpha=0.5, label='All-Pass (y=x)', color='gray', density=True)
    ax4.hist(df_nn['block_ms'], bins=80, alpha=0.5, label='NN Model', color='blue', density=True)
    ax4.axvline(x=budget_ms, color='r', linestyle='--', linewidth=1, label=f'Budget ({budget_ms:.3f} ms)')
    ax4.set_xlabel('Block Latency (ms)')
    ax4.set_ylabel('Density')
    ax4.set_title('Block Latency Histogram')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ---- Plot 5: CPU Load bar chart ----
    ax5 = plt.subplot(3, 2, 5)
    labels_bar = ['All-Pass (y=x)', 'NN Model']
    loads = [stats_ap['load_pct'], stats_nn['load_pct']]
    bars = ax5.bar(labels_bar, loads, color=['gray', 'blue'], alpha=0.7)
    ax5.axhline(y=100, color='r', linestyle='--', linewidth=1, label='100% (xrun)')
    ax5.set_ylabel('Processing Load (%)')
    ax5.set_title('Average Processing Load')
    for bar, val in zip(bars, loads):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax5.legend()
    ax5.grid(True, axis='y', alpha=0.3)

    # ---- Plot 6: Waveform comparison (Bela NN vs DSP RT vs Clean) ----
    ax6 = plt.subplot(3, 2, 6)
    input_wav_path = os.path.join(PROJECT_ROOT, '..', 'raw_sound_files',
                                  'funk-soul-guitar-clean-4_90bpm_G.wav')
    slc = slice(1000, 2000)

    has_waveforms = False
    if os.path.exists(input_wav_path):
        y_in, sr_in = librosa.load(input_wav_path, sr=BELA_SR)
        ax6.plot(y_in[slc], label='Clean Input', alpha=0.4, color='gray')
        has_waveforms = True

        # Generate DSP RT reference from the same input
        rt_sat = RealtimeTubeSaturator(drive=70.0, asymmetry=0.4, tone=5000, fs=sr_in)
        y_dsp_rt = rt_sat.process(y_in)
        ax6.plot(y_dsp_rt[slc], label='DSP RT', color='orange', linewidth=1)

    if os.path.exists(distorted_wav_path):
        y_dist, _ = librosa.load(distorted_wav_path, sr=BELA_SR)
        ax6.plot(y_dist[slc], label='Bela NN Output', color='blue', linewidth=1)
        has_waveforms = True

    if has_waveforms:
        ax6.set_title('Waveform Detail (Samples 1000–2000)')
        ax6.set_xlabel('Sample')
        ax6.set_ylabel('Amplitude')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No audio files found', ha='center', va='center',
                 transform=ax6.transAxes, fontsize=12, color='gray')
        ax6.set_title('Waveform Detail (not available)')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'bela_benchmark_plots.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plots to {plot_path}")

    # --- Save text report ---
    report_path = os.path.join(OUTPUT_DIR, 'bela_benchmark_report.txt')
    with open(report_path, 'w') as f:
        f.write("Bela Real-Time Benchmark Report\n")
        f.write(f"Block Size: {BELA_BLOCK_SIZE}  |  Sample Rate: {BELA_SR} Hz\n")
        f.write(f"Budget: {budget_ms:.3f} ms/block\n\n")
        f.write(report_text)
    print(f"Saved report to {report_path}")


if __name__ == '__main__':
    main()
