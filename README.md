# Neural Network vs. DSP Audio Effects Comparison

This project explores the capability of Neural Networks to emulate deterministic DSP audio effects and benchmarks them against traditional algorithms. The active track is now a **Flanger** effect (with a tube saturator baseline still available in the codebase).

## Project Status

We have implemented a complete end-to-end pipeline for **Phase 1 (Non-Real-Time Benchmarking)**:

*   **DSP Baseline (Active)**: Flanger implementation with sinusoidal LFO modulation.
	* Delay law: `delay(t) = center_delay * (1 + sin(2*pi*rate*t))`
	* Sweep range: `0 .. 2*center_delay` ms.
*   **Neural Networks**:
	* **LSTM** is the active model for **flanger** emulation.
	* **TCN** is retained for the **distortion** track.
*   **Dataset Generation**: `generate_targets.py` supports flanger target creation.
*   **Training Loop**: Configurable model type (`lstm`/`tcn`) with checkpointing.
*   **Inference Engine**: Unified wrappers for DSP and NN paths.
*   **Benchmark/Testbench**: `tests/phase1_benchmark.py` measures speed (RTF), block latency, and signal error against DSP flanger reference.

## Quick Start

### 1. Installation

This project uses `uv` for dependency management.

```bash
cd python_code
uv sync
```

All Python commands in this project are expected to run through `uv` (`uv sync`, `uv run ...`).

### CUDA / GPU Usage (Optional but Recommended for Training)

If a CUDA-capable GPU is available, training should run on GPU automatically when PyTorch in the active `uv` environment is CUDA-enabled.

Quick checks:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If `torch.cuda.is_available()` is `False`, your active environment likely has a CPU-only PyTorch build.

### 2. Workflow

**Step A: Prepare Data**
1.  Place your raw/clean audio files (wav) in `python_code/data/datasets/inputs/`.
2.  Generate flanger target files:

```bash
uv run generate_targets.py \
	--input_dir data/datasets/inputs \
	--target_dir data/datasets/targets_flange \
	--effect flanger \
	--flanger_rate 0.2 \
	--flanger_center_delay 3.0 \
	--flanger_ff 0.70 \
	--flanger_fb 0.12
```

(This saves processed files to `python_code/data/datasets/targets_flange/` with matching filenames.)

**Step B: Train the Model**
Train the LSTM to mimic the DSP flanger effect:
```bash
uv run src/nn/train.py \
	--model_type lstm \
	--data_root data/datasets \
	--input_subdir inputs \
	--target_subdir targets_flange \
	--epochs 100 \
	--batch_size 16
```
The final model will be saved to `python_code/models/checkpoints/lstm_final.pt`.
When CUDA is available in the active environment, this command trains on GPU; otherwise it falls back to CPU.

**Step C: Run Inference**
Apply the trained model to a new audio file:

```bash
uv run inference.py --input_file "path/to/my_riff.wav"
```

Output will be saved to `python_code/data/processed/`.

**Step D: Benchmark**
Compare DSP flanger vs trained NN model for a selected effect mode:

```bash
uv run tests/phase1_benchmark.py --effect flange --input_file powerchords-mute.wav
```

Effect-mode model mapping in `tests/phase1_benchmark.py`:
- `--effect flange` -> activates `LSTM (Final)` and keeps TCN inactive.
- `--effect distortion` -> activates `Causal TCN (Final)` and keeps LSTM inactive.

If `--input_file` is omitted, the benchmark uses its built-in default files.

*TODO: add file size comparisons (several runs)*
*TODO: add statistics (several runs)*
*TODO: add different models (different size, optimized) to compare against eachother.*
*TODO: add real-time inference implementation*

## Comparisons

| Feature | Deterministic DSP (Flanger) | Neural Network (LSTM/TCN) |
| :--- | :--- | :--- |
| **Method** | Modulated fractional delay + feedback/feed-forward | Learned sequence mapping |
| **Complexity** | Low-level DSP operations | Higher model-dependent compute |
| **Sound** | Deterministic reference | Learned approximation of reference |
| **Speed** | Fast reference baseline | Slower but benchmarked with RTF/latency |

## File Structure

*   `src/dsp/`: Reference DSP implementations.
*   `src/nn/`: PyTorch model architecture and training logic.
*   `src/engine/`: Wrappers for unified inference.
*   `data/`: Storage for datasets and processed audio.
*   `tests/`: Benchmarking scripts.
*   `python_code/archive/`: Legacy one-off scripts moved out of active workflow (`main.py`, `fft_analyzer.py`).

---

### Previous Research Link
https://docs.google.com/document/d/1PU49m20RlBC7QgCGgH99PVEWIrra0MUrChNkFV747vk/edit?tab=t.0
