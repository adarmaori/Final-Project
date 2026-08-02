# Neural Network vs. DSP Audio Effects

This project develops a pipeline and platform for testing, benchmarking, and optimizing neural-network audio processing across three levels of hardware abstraction: PC, microcontroller, and FPGA. The PC workflow is used for DSP target generation, model training, offline inference, and detailed performance analysis; the microcontroller and FPGA implementations investigate how trained models can be adapted for low-latency and resource-constrained deployment.

The work focuses on two audio effects—distortion and exciter—and on Temporal Convolutional Network (TCN) models that learn to reproduce their DSP behavior. The repository includes floating-point and quantized TCN experiments, benchmark tooling, deployment artifacts, Bela/microcontroller renderers, and Verilog implementations and testbenches for FPGA evaluation.

## Repository layout

```text
.
├── README.md
├── python_code/
│   ├── pyproject.toml       # Python dependencies and project metadata
│   ├── uv.lock              # Locked environment
│   ├── generate_targets.py  # Create DSP target WAVs
│   ├── inference.py         # Apply a trained model to a WAV file
│   ├── export_*.py          # Deployment/export utilities
│   ├── src/
│   │   ├── dsp/             # Deterministic effects
│   │   ├── nn/              # Architectures, dataset, training, quantization
│   │   └── engine/          # DSP/NN wrappers
│   ├── tests/               # Benchmarks and effect tests
│   ├── data/                # Input, target, processed audio and plots
│   └── models/              # Checkpoints and sweep results
├── Bela/                    # Bela renderers, models, and benchmark results
├── FPGA/                    # Verilog engines, testbenches, and reports
├── raw_sound_files/         # Source WAV material used by examples/benchmarks
└── instructions/            # Platform and effect-specific notes
```

Generated audio, plots, reports, and model files can be large. Check whether they should be stored in Git or in external artifact storage before adding new runs.

## Python setup

Python 3.12 or 3.13 is required by `python_code/pyproject.toml`. From the repository root:

```bash
cd python_code
uv sync
```

Run Python commands from `python_code`, because the scripts use paths relative to that directory:

```bash
uv run python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

The checked-in environment on the development machine currently runs PyTorch without CUDA. Training falls back to CPU (or MPS when available in the runtime). A CUDA-enabled PyTorch installation is recommended for training.

## Generating DSP targets

Put mono or stereo WAV files in `python_code/data/datasets/inputs/`. The generator reads WAV files directly in that directory; it does not recursively scan subdirectories. It preserves the input sample rate and writes mono targets.

Exciter:

```bash
uv run python generate_targets.py \
  --effect exciter \
  --exciter_drive 6.0 \
  --exciter_mix 0.8 \
  --exciter_cutoff 2200
```

Tube distortion:

```bash
uv run python generate_targets.py \
  --effect tube \
  --tube_drive 70.0 \
  --tube_asymmetry 0.4 \
  --tube_tone 5000
```

By default, targets are written to `data/datasets/targets_exciter` for `exciter` and `data/datasets/targets_distortion` for `tube` (distortion).

## Training a model

Training uses matching filenames from an input directory and a target directory.

Train a floating-point distortion TCN:

```bash
uv run python src/nn/train.py \
  --effect distortion \
  --model_type tcn \
  --data_root data/datasets \
  --input_subdir inputs \
  --target_subdir targets_distortion \
  --chunk_size 16384 \
  --epochs 100 \
  --batch_size 16
```

Train a floating-point exciter TCN:

```bash
uv run python src/nn/train.py \
  --effect exciter \
  --model_type tcn \
  --data_root data/datasets \
  --input_subdir inputs \
  --target_subdir targets_exciter \
  --chunk_size 16384 \
  --epochs 100 \
  --batch_size 16
```

For Brevitas quantization-aware training, set `--quant_bits` to the desired width, such as `8` or `4`. The training script supports `tcn`, `lstm`, `crnn`, and `unet`; see all options with:

```bash
uv run python src/nn/train.py --help
```

The loss selected by default is multi-resolution STFT plus waveform L1 and first-difference L1 terms. The dataset also supports a context window; use `--context_size` when the effect needs more history than the training chunk alone.

## Run inference

`inference.py` expects the input filename to exist in `../raw_sound_files/` relative to `python_code` and writes output to `data/processed/`.

```bash
uv run python inference.py \
  --input_file funk-soul-guitar-clean-4_90bpm_G.wav \
  --model_path models/checkpoints/distortion_tcn_final.pt \
  --model_type tcn
```

Use `--output_file` to choose the output filename. The wrapper can infer several TCN configurations from checkpoint keys, but the checkpoint and model type still need to match the intended architecture. Whole-file TCN inference is convenient for offline use; memory and latency should be measured before treating it as real-time capable.

## Run benchmarks

From `python_code`:

```bash
uv run python tests/phase1_benchmark.py \
  --effect distortion \
  --input_file funk-soul-guitar-clean-4_90bpm_G.wav \
  --quant_bits 8
```

Supported benchmark modes are `distortion` and `exciter`. The benchmark writes reports and visualizations to `data/processed/`. Useful options include:

```bash
uv run python tests/phase1_benchmark.py --help
uv run python tests/phase1_benchmark.py --effect distortion --compare_all
uv run python tests/phase1_benchmark.py --model_path models/checkpoints/distortion_tcn_final.pt
```

The default model selection expects effect-specific files such as `distortion_tcn_final.pt`, `distortion_tcn_q8_final.pt`, `exciter_tcn_final.pt`, and `exciter_tcn_q8_final.pt`. If a selected checkpoint is missing, the benchmark reports that model as unavailable; use `--model_path` to select an existing checkpoint explicitly.

## FPGA and Bela

The FPGA work is independent of the Python setup. Basic Verilog simulation and synthesis targets are defined in `FPGA/Makefile`:

```bash
cd FPGA
make test-network
make throughput-network
make report
```

The default synthesis target is an iCE40 design and requires tools such as `iverilog`, `yosys`, and `nextpnr-ice40`. See [`instructions/FPGA.md`](instructions/FPGA.md) for the current hardware notes.

The Bela implementation is under `Bela/code/`; platform setup and deployment notes are in [`instructions/Bela.md`](instructions/Bela.md). The repository also includes C++ benchmark artifacts under `cpp_code/`.

## Research notes

Effect-specific technical notes are in the [`instructions/`](instructions/) directory. The original research document is available [here](https://docs.google.com/document/d/1PU49m20RlBC7QgCGgH99PVEWIrra0MUrChNkFV747vk/edit?tab=t.0).
