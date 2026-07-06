import argparse
import os
import sys
import time
from typing import cast

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import soundfile as sf

# Add project root to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dsp.distortion import RealtimeTubeSaturator, tube_saturator
from src.dsp.exciter import aural_exciter
from src.dsp.flanger import RealtimeFlanger, flanger_effect
from src.dsp.wah_wah import RealtimeWahWah, wah_effect
from src.engine.wrapper import DSPWrapper, NNWrapper, RealtimeDSPWrapper


def _format_ascii_table(headers, rows, alignments=None):
    if alignments is None:
        alignments = ["left"] * len(headers)

    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def format_cell(value, width, alignment):
        text = str(value)
        if alignment == "right":
            return text.rjust(width)
        return text.ljust(width)

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [border]
    lines.append("| " + " | ".join(format_cell(header, width, "left") for header, width in zip(headers, widths)) + " |")
    lines.append(border)

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                format_cell(cell, width, alignment)
                for cell, width, alignment in zip(row, widths, alignments)
            )
            + " |"
        )

    lines.append(border)
    return lines


def _compute_log_spectrogram(audio, sr, n_fft=1024, hop_length=128):
    effective_nperseg = min(n_fft, len(audio))
    if effective_nperseg < 2:
        effective_nperseg = len(audio)

    effective_noverlap = min(effective_nperseg - 1, max(0, effective_nperseg - hop_length))
    frequencies, times, spectrogram = signal.spectrogram(
        audio,
        fs=sr,
        window="hann",
        nperseg=effective_nperseg,
        noverlap=effective_noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    spectrogram_db = 20.0 * np.log10(np.maximum(spectrogram, 1e-12))
    return frequencies, times, spectrogram_db


def _save_spectrogram_grid(output_path, tracks, sr, preview_duration_sec=1.0):
    if not tracks:
        return

    segment_length = min(len(tracks[0][1]), int(preview_duration_sec * sr))
    segment_length = max(segment_length, 1)

    prepared_tracks = []
    min_db = None
    max_db = None

    for title, audio in tracks:
        clipped_audio = audio[:segment_length]
        frequencies, times, spectrogram_db = _compute_log_spectrogram(clipped_audio, sr)
        prepared_tracks.append((title, frequencies, times, spectrogram_db))
        local_min = float(np.min(spectrogram_db))
        local_max = float(np.max(spectrogram_db))
        min_db = local_min if min_db is None else min(min_db, local_min)
        max_db = local_max if max_db is None else max(max_db, local_max)

    fig_width = 3
    fig_height = max(10.5, 2.75 * len(prepared_tracks))
    fig, axes = plt.subplots(len(prepared_tracks), 1, figsize=(fig_width, fig_height), sharex=True, constrained_layout=True)
    if len(prepared_tracks) == 1:
        axes = [axes]

    mesh = None
    for axis, (title, frequencies, times, spectrogram_db) in zip(axes, prepared_tracks):
        mesh = axis.pcolormesh(times, frequencies, spectrogram_db, shading="auto", cmap="magma", vmin=min_db, vmax=max_db)
        axis.set_title(title)
        axis.set_ylabel("Hz")
        axis.set_ylim(0, min(sr / 2, 8000))
        axis.grid(False)

    axes[-1].set_xlabel("Seconds")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, orientation="horizontal", pad=0.05, fraction=0.06, shrink=0.9, label="dB")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_models_config(project_models_dir, effect_mode, quant_bits=0, compare_all_quant=False):
    """Select active models based on requested effect mode."""
    if effect_mode == "wah":
        crnn_model_name = 'crnn_wah_final.pt'
    else:
        crnn_model_name = 'crnn_final.pt'

    crnn_model_path = os.path.join(project_models_dir, 'crnn', crnn_model_name)
    if not os.path.exists(crnn_model_path):
        crnn_model_path = os.path.join(project_models_dir, crnn_model_name)

    def resolve_tcn_path(*candidate_names):
        for candidate_name in candidate_names:
            candidate_path = os.path.join(project_models_dir, candidate_name)
            if os.path.exists(candidate_path):
                return candidate_path
        return os.path.join(project_models_dir, candidate_names[0])

    tcn_effect_name = effect_mode if effect_mode in ("distortion", "exciter") else None

    tcn_variants = []
    color_map = {16: 'blue', 8: 'cyan', 4: 'green'}

    if compare_all_quant:
        float_candidates = [
            f"{tcn_effect_name}_tcn_final.pt" if tcn_effect_name else "tcn_final.pt",
            "tcn_final.pt",
        ]
        float_tcn = resolve_tcn_path(*float_candidates)
        tcn_variants.append({
            "name": "Causal TCN (float)",
            "path": float_tcn,
            "model_type": "tcn",
            "quant_bits": 0,
            "active": effect_mode in ("distortion", "exciter") and os.path.exists(float_tcn),
            "color": 'navy',
        })

        for bits in [16, 8, 4]:
            tcn_candidates = []
            if tcn_effect_name:
                tcn_candidates.append(f"{tcn_effect_name}_tcn_q{bits}_final.pt")
            tcn_candidates.append(f"tcn_q{bits}_final.pt")
            tcn_path = resolve_tcn_path(*tcn_candidates)
            tcn_variants.append({
                "name": f"Causal TCN ({bits}-bit)",
                "path": tcn_path,
                "model_type": "tcn",
                "quant_bits": bits,
                "active": effect_mode in ("distortion", "exciter") and os.path.exists(tcn_path),
                "color": color_map.get(bits, 'gray'),
            })
    else:
        if quant_bits > 0:
            tcn_candidates = []
            if tcn_effect_name:
                tcn_candidates.append(f"{tcn_effect_name}_tcn_q{quant_bits}_final.pt")
            tcn_candidates.append(f"tcn_q{quant_bits}_final.pt")
            tcn_model_name = resolve_tcn_path(*tcn_candidates)
            color = color_map.get(quant_bits, 'blue')
        else:
            tcn_candidates = []
            if tcn_effect_name:
                tcn_candidates.append(f"{tcn_effect_name}_tcn_final.pt")
            tcn_candidates.append('tcn_final.pt')
            tcn_model_name = resolve_tcn_path(*tcn_candidates)
            color = 'navy'

        tcn_variants.append({
            "name": f"Causal TCN ({quant_bits}-bit)" if quant_bits > 0 else "Causal TCN (float)",
            "path": tcn_model_name,
            "model_type": "tcn",
            "quant_bits": quant_bits,
            "active": effect_mode in ("distortion", "exciter") and os.path.exists(tcn_model_name),
            "color": color,
        })

        if effect_mode in ("distortion", "exciter") and quant_bits == 0:
            tcn_4bit_path = resolve_tcn_path(
                f"{tcn_effect_name}_tcn_q4_final.pt" if tcn_effect_name else "tcn_q4_final.pt",
                'tcn_q4_final.pt',
            )
            if os.path.exists(tcn_4bit_path):
                tcn_variants.append({
                    "name": "Causal TCN (4-bit)",
                    "path": tcn_4bit_path,
                    "model_type": "tcn",
                    "quant_bits": 4,
                    "active": True,
                    "color": color_map.get(4, 'green'),
                })

    return tcn_variants + [
        {
            "name": "CRNN (Final)",
            "path": crnn_model_path,
            "model_type": "crnn",
            "active": effect_mode in ("flange", "wah"),
            "color": "cyan",
        },
        {
            "name": "LSTM (Final)",
            "path": os.path.join(project_models_dir, 'lstm_final.pt'),
            "model_type": "lstm",
            "active": effect_mode in ("flange", "wah"),
            "color": "purple",
        },
        {
            "name": "STFT U-Net (Final)",
            "path": os.path.join(project_models_dir, 'unet_final.pt'),
            "model_type": "unet",
            "active": effect_mode == "reverb",
            "color": "magenta",
        },
        {
            "name": "TCN (Small) [Placeholder]",
            "path": os.path.join(project_models_dir, 'tcn_small.pt'),
            "model_type": "tcn",
            "active": False,
            "color": "green",
        },
        {
            "name": "Use Optimized ONNX [Placeholder]",
            "path": os.path.join(project_models_dir, 'model_opt.onnx'),
            "active": False,
            "color": "red",
        },
    ]


def run_benchmark_suite(input_file, output_dir=None, effect_mode="flange", quant_bits=0, compare_all_quant=False):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

    print(f"--- Phase 1 Benchmark Suite: {os.path.basename(input_file)} [{effect_mode}] ---")
    os.makedirs(output_dir, exist_ok=True)

    n_runs = 10
    block_size = 512
    test_durations = [1.0, 5.0, 10.0, 30.0]

    project_models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'checkpoints')
    models_config = _build_models_config(
        project_models_dir,
        effect_mode,
        quant_bits=quant_bits,
        compare_all_quant=compare_all_quant,
    )

    try:
        y_full, sr = librosa.load(input_file, sr=None)
        print(f"Loaded audio: {len(y_full)} samples, {sr} Hz ({len(y_full) / sr:.2f}s)")
    except Exception as exc:
        print(f"Error loading file: {exc}")
        return

    effect_tag = effect_mode.replace(" ", "_")

    report_lines = [
        f"Phase 1 Detailed Benchmark Report [{effect_mode}]",
        "=================================",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input: {os.path.basename(input_file)} ({len(y_full) / sr:.2f}s @ {sr}Hz)",
        f"Effect Mode: {effect_mode}",
        "",
    ]

    wrappers = {}

    if effect_mode == "distortion":
        dsp_wrapper = DSPWrapper(tube_saturator, drive=70.0, asymmetry=0.4, tone=5000, fs=int(sr))
        wrappers["DSP baseline"] = dsp_wrapper
        wrappers["DSP RT"] = RealtimeDSPWrapper(RealtimeTubeSaturator(drive=70.0, asymmetry=0.4, tone=5000, fs=int(sr)))
    elif effect_mode == "flange":
        dsp_wrapper = DSPWrapper(flanger_effect, rate=0.5, center_delay=2.5, ff=0.7, fb=0.2, fs=int(sr))
        wrappers["DSP baseline"] = dsp_wrapper
        wrappers["DSP RT"] = RealtimeDSPWrapper(RealtimeFlanger(rate=0.5, center_delay=2.5, ff=0.7, fb=0.2, fs=int(sr)))
    elif effect_mode == "wah":
        dsp_wrapper = DSPWrapper(wah_effect, freq_min=400.0, freq_max=2500.0, q=2.0, attack_ms=5.0, release_ms=100.0, fs=int(sr))
        wrappers["DSP baseline"] = dsp_wrapper
        wrappers["DSP RT"] = RealtimeDSPWrapper(RealtimeWahWah(freq_min=400.0, freq_max=2500.0, q=2.0, attack_ms=5.0, release_ms=100.0, fs=int(sr)))
    elif effect_mode == "reverb":
        def simple_reverb_dsp(x, fs=44100):
            out = x.copy()
            delay_samples = int(0.04 * fs)
            feedback = 0.5
            for index in range(delay_samples, len(x)):
                out[index] += feedback * out[index - delay_samples]
            return out

    elif effect_mode == "exciter":
        dsp_wrapper = DSPWrapper(aural_exciter, drive=6, mix=0.8, cutoff_freq=2200.0, fs=int(sr))
        wrappers["DSP baseline"] = dsp_wrapper

        class RealtimeExciter:
            def __init__(self, drive=6, mix=0.8, cutoff_freq=2200.0, fs=44100):
                self.drive = drive
                self.mix = mix
                nyquist = 0.5 * fs
                normal_cutoff = cutoff_freq / nyquist
                butter_result = cast(tuple[np.ndarray, np.ndarray], signal.butter(2, normal_cutoff, btype='high', analog=False))
                self.b, self.a = butter_result
                self.zi1 = signal.lfilter_zi(self.b, self.a)
                self.zi2 = signal.lfilter_zi(self.b, self.a)

            def process(self, block):
                high_passed, self.zi1 = signal.lfilter(self.b, self.a, block, zi=self.zi1)
                generated_harmonics = np.tanh(self.drive * high_passed)
                clean_harmonics, self.zi2 = signal.lfilter(self.b, self.a, generated_harmonics, zi=self.zi2)
                return block + (self.mix * clean_harmonics)

        wrappers["DSP RT"] = RealtimeDSPWrapper(RealtimeExciter(fs=int(sr)))

    for cfg in models_config:
        if not cfg['active']:
            continue
        if os.path.exists(cfg['path']):
            model = NNWrapper(
                model_path=cfg['path'],
                model_type=cfg.get('model_type', 'tcn'),
                quant_bits=cfg.get('quant_bits', 0),
            )
            if cfg.get('quant_bits', 0) > 0 and hasattr(model, 'calibrate'):
                model.calibrate(y_full)
            wrappers[cfg['name']] = model
        else:
            print(f"Warning: {cfg['name']} file not found at {cfg['path']}")

    report_lines.append("Experiment A: Batch Processing Checks")
    report_lines.append("-------------------------------------")

    results_a = {}
    outputs_a = {}
    model_names_plot = []
    rtfs_plot = []
    rows_a = []

    y_dsp_ref = wrappers["DSP baseline"].process(y_full)
    ref_energy = np.mean(y_dsp_ref ** 2)

    for name, wrapper in wrappers.items():
        times = []
        for _ in range(n_runs):
            if hasattr(wrapper, 'reset'):
                wrapper.reset()
            start = time.perf_counter()
            _ = wrapper.process(y_full)
            times.append(time.perf_counter() - start)

        avg_time = float(np.mean(times))
        rtf = avg_time / (len(y_full) / sr)

        if hasattr(wrapper, 'reset'):
            wrapper.reset()
        y_out = wrapper.process(y_full)
        outputs_a[name] = y_out

        if name == "DSP baseline":
            mse = 0.0
            esr = 0.0
            ratio_str = "1.0x"
        else:
            length = min(len(y_out), len(y_dsp_ref))
            y_slice = y_out[:length]
            ref_slice = y_dsp_ref[:length]
            mse = float(np.mean((y_slice - ref_slice) ** 2))
            esr = mse / (ref_energy + 1e-10)
            dsp_time = results_a.get("DSP baseline", {}).get("avg_time", avg_time)
            ratio_str = f"{avg_time / dsp_time:.2f}x slower"

        input_stem = os.path.splitext(os.path.basename(input_file))[0]
        output_name = f"output_{input_stem}_{name.replace(' ', '_')}.wav"
        sf.write(os.path.join(output_dir, output_name), y_out, sr)

        results_a[name] = {
            "avg_time": avg_time,
            "rtf": rtf,
            "mse": mse,
            "esr": esr,
        }
        model_names_plot.append(name)
        rtfs_plot.append(rtf)
        rows_a.append([
            name,
            f"{avg_time * 1000:.2f}",
            f"{rtf:.4f}",
            f"{mse:.2e}",
            f"{esr * 100:.2f}%",
            ratio_str,
        ])

    report_lines.extend(_format_ascii_table(
        ["Model", "Time(ms)", "RTF", "MSE", "ESR", "Speed vs DSP"],
        rows_a,
        ["left", "right", "right", "right", "right", "left"],
    ))
    report_lines.append("")

    report_lines.append("Experiment B: Scalability (RTF over different durations)")
    report_lines.append("------------------------------------------------------")

    scalability_results = {name: [] for name in wrappers.keys()}
    rows_b = []
    for duration in test_durations:
        n_samples = int(duration * sr)
        if n_samples > len(y_full):
            break
        y_slice = y_full[:n_samples]
        row_values = [f"{duration:.1f}"]
        for name, wrapper in wrappers.items():
            if hasattr(wrapper, 'reset'):
                wrapper.reset()
            start = time.perf_counter()
            _ = wrapper.process(y_slice)
            elapsed = time.perf_counter() - start
            rtf = elapsed / duration
            scalability_results[name].append(rtf)
            row_values.append(f"{rtf:.4f}")
        rows_b.append(row_values)

    report_lines.extend(_format_ascii_table(
        ["Duration"] + list(wrappers.keys()),
        rows_b,
        ["right"] + ["right"] * len(wrappers),
    ))
    report_lines.append("")

    report_lines.append(f"Experiment C: Simulated Real-Time (Block Size: {block_size})")
    report_lines.append("-------------------------------------------------------")
    block_data_plot = []
    rows_c = []

    num_blocks = min(len(y_full) // block_size, 500)
    block_duration_ms = (block_size / sr) * 1000
    report_lines.append(f"Budget per block: {block_duration_ms:.2f} ms")

    for name, wrapper in wrappers.items():
        if hasattr(wrapper, 'reset'):
            wrapper.reset()

        latencies_ms = []
        for index in range(num_blocks):
            chunk = y_full[index * block_size:(index + 1) * block_size]
            t0 = time.perf_counter()
            _ = wrapper.process(chunk)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        avg_lat = float(np.mean(latencies_ms))
        max_lat = float(np.max(latencies_ms))
        p99_lat = float(np.percentile(latencies_ms, 99))
        load_pct = (avg_lat / block_duration_ms) * 100

        rows_c.append([
            name,
            f"{avg_lat:.4f} ms",
            f"{max_lat:.4f} ms",
            f"{load_pct:.2f}% (P99: {p99_lat:.2f} ms)",
        ])
        block_data_plot.append(latencies_ms)

    report_lines.extend(_format_ascii_table(
        ["Model", "Avg Latency", "Max Latency", "Load"],
        rows_c,
        ["left", "right", "right", "left"],
    ))

    print("Generating Plots...")
    fig, (ax_waveform, ax_latency) = plt.subplots(2, 1, figsize=(6.8, 10.8), constrained_layout=True)

    waveform_start = 1000
    waveform_end = min(len(y_full), waveform_start + 1000)
    waveform_slice = slice(waveform_start, waveform_end)

    ax_waveform.plot(y_full[waveform_slice], label="Input", alpha=0.5, color='gray')
    ax_waveform.plot(y_dsp_ref[waveform_slice], label="DSP", color='black', linewidth=1)
    for name, output_audio in outputs_a.items():
        if name not in ("DSP baseline", "DSP RT"):
            ax_waveform.plot(output_audio[waveform_slice], label=name, linestyle='--')
    ax_waveform.set_title("Waveform Detail")
    ax_waveform.legend(ncol=2)
    ax_waveform.grid(True, alpha=0.3)

    ax_latency.boxplot(block_data_plot, tick_labels=list(wrappers.keys()))
    ax_latency.set_title(f"Simulated Real-Time Latency Jitter (Block Size {block_size})")
    ax_latency.set_ylabel("Latency (ms)")
    ax_latency.axhline(y=block_duration_ms, color='r', linestyle='--', label=f"Budget ({block_duration_ms:.2f}ms)")
    ax_latency.legend()
    ax_latency.grid(True, axis='y')

    plot_path = os.path.join(output_dir, f"benchmark_suite_{effect_tag}_plots.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plots to {plot_path}")

    spectrogram_tracks = [("Input", y_full), ("DSP", y_dsp_ref)]
    for name, output_audio in outputs_a.items():
        if name not in ("DSP baseline", "DSP RT"):
            spectrogram_tracks.append((name, output_audio))

    spectrogram_path = os.path.join(output_dir, f"benchmark_suite_{effect_tag}_spectrograms.png")
    _save_spectrogram_grid(spectrogram_path, spectrogram_tracks, sr, preview_duration_sec=1.0)
    print(f"Saved spectrograms to {spectrogram_path}")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"report_{effect_tag}_{timestamp}.txt")
    with open(report_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(report_text)

    with open(os.path.join(output_dir, "report_latest.txt"), "w", encoding="utf-8") as file_handle:
        file_handle.write(report_text)

    with open(os.path.join(output_dir, f"report_latest_{effect_tag}.txt"), "w", encoding="utf-8") as file_handle:
        file_handle.write(report_text)

    print(f"Reports saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run phase1 benchmark for flanger/distortion/wah/reverb models")
    parser.add_argument("--effect", choices=["flange", "distortion", "wah", "reverb", "exciter"], default="flange", help="Select effect/model mapping")
    parser.add_argument("--input_file", type=str, default=None, help="Filename in ../raw_sound_files/ to benchmark. If omitted, uses built-in defaults.")
    parser.add_argument("--quant_bits", type=int, default=0, help="Set to 16/8/4 to benchmark a specific quantized TCN")
    parser.add_argument("--compare_all", action="store_true", help="Load all available quantized variants (16, 8, 4-bit) for comparison in a single run")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if args.input_file:
        test_file = os.path.join(project_root, "..", "raw_sound_files", args.input_file)
        if not os.path.exists(test_file):
            print(f"Input file not found: {test_file}")
            sys.exit(1)
        run_benchmark_suite(test_file, effect_mode=args.effect, quant_bits=args.quant_bits, compare_all_quant=args.compare_all)
        sys.exit(0)

    test_file_name = "../raw_sound_files/funk-soul-guitar-clean-4_90bpm_G.wav"
    test_file = os.path.join(project_root, test_file_name)

    if not os.path.exists(test_file):
        print(f"Test file not found at {test_file}, generating synthetic signal...")
        sr = 44100
        t = np.linspace(0, 10, 10 * sr)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        sf.write(test_file, y, sr)
        print(f"Generated synthetic file at {test_file}")

    run_benchmark_suite(test_file, effect_mode=args.effect, quant_bits=args.quant_bits, compare_all_quant=args.compare_all)

    test_file_2_name = "../raw_sound_files/romantic-electric-guitar-riff-mixed_143bpm_F#_minor.wav"
    test_file_2 = os.path.join(project_root, test_file_2_name)
    if os.path.exists(test_file_2):
        run_benchmark_suite(test_file_2, effect_mode=args.effect, quant_bits=args.quant_bits, compare_all_quant=args.compare_all)
    else:
        print(f"Skipping second benchmark — file not found: {test_file_2}")
        run_benchmark_suite(test_file, effect_mode=args.effect, quant_bits=args.quant_bits, compare_all_quant=args.compare_all)
