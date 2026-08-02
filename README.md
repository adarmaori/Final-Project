# Neural Network vs. DSP Audio Effects

This project compares deterministic digital signal processing (DSP) effects with neural-network models that learn to reproduce them. The main workflow is offline WAV-file processing: generate DSP targets, train a PyTorch model, run inference, and compare speed and signal error.

The repository also contains deployment experiments for Bela and FPGA targets.

## Current status

| Area | Current state |
| --- | --- |
| DSP effects | Flanger, tube distortion, wah-wah, reverb, cabinet simulation, and aural exciter implementations are present in `python_code/src/dsp/`. |
| Neural models | TCN, quantized TCN, CRNN, LSTM, and STFT U-Net architectures are implemented in `python_code/src/nn/`. |
| Training | Configurable training loop with train/validation split, automatic checkpointing, L1 + derivative + multi-resolution STFT loss, and optional Brevitas quantization. |
| Benchmarking | `python_code/tests/phase1_benchmark.py` measures DSP/NN processing time, real-time block behavior, MSE/ESR, waveforms, and spectrograms. |
| Included checkpoints | Distortion and exciter checkpoints are present. A flanger CRNN checkpoint is not present in this checkout, so flanger NN benchmarking requires training or supplying a checkpoint. |
| Real-time audio | Stateful DSP wrappers and Bela C++ renderers exist. A general-purpose live Python audio callback is not yet implemented. |
| FPGA | Verilog TCN/convolution engines, testbenches, synthesis reports, and an iCE40-oriented `FPGA/Makefile` are included. |

The project is therefore best described as an offline research/benchmarking pipeline with hardware-deployment prototypes, not yet as a finished real-time plugin or standalone pedal.

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
├── cpp_code/                # C++ audio/benchmark artifacts
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

## Generate DSP targets

Put mono or stereo WAV files in `python_code/data/datasets/inputs/`. The generator reads WAV files directly in that directory; it does not recursively scan subdirectories. It preserves the input sample rate and writes mono targets.

Flanger:

```bash
uv run python generate_targets.py \
  --effect flanger \
  --flanger_rate 0.2 \
  --flanger_depth 2.0 \
  --flanger_center_delay 3.0 \
  --flanger_ff 0.70 \
  --flanger_fb 0.12
```

Tube distortion:

```bash
uv run python generate_targets.py \
  --effect tube \
  --tube_drive 70.0 \
  --tube_asymmetry 0.4 \
  --tube_tone 5000
```

By default, targets are written to `data/datasets/targets_flange` for `flanger` and `data/datasets/targets_distortion` for `tube`. Other supported effects are `wah`, `reverb`, `cab`, and `exciter`; use `--target_dir` or `--help` to customize them.

For cabinet simulation, provide a WAV impulse response with `--cab_ir_path`.

## Train a model

Training uses matching filenames from an input directory and a target directory. The command below trains a CRNN for flanger emulation:

```bash
uv run python src/nn/train.py \
  --effect flange \
  --model_type crnn \
  --data_root data/datasets \
  --input_subdir inputs \
  --target_subdir targets_flange \
  --chunk_size 88200 \
  --epochs 100 \
  --batch_size 16
```

The checkpoint is written to `models/checkpoints/crnn_final.pt`. If no effect is supplied, the effect is inferred from the target directory.

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

Supported benchmark modes are `flange`, `distortion`, `wah`, `reverb`, and `exciter`. The benchmark writes reports and visualizations to `data/processed/`. Useful options include:

```bash
uv run python tests/phase1_benchmark.py --help
uv run python tests/phase1_benchmark.py --effect distortion --compare_all
uv run python tests/phase1_benchmark.py --model_path models/checkpoints/distortion_tcn_final.pt
```

The default model selection expects effect-specific files such as `distortion_tcn_final.pt`, `distortion_tcn_q8_final.pt`, and `crnn_final.pt`. If a selected checkpoint is missing, the benchmark reports that model as unavailable; use `--model_path` to select an existing checkpoint explicitly.

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

## Known limitations

- No top-level automated test command or CI workflow is currently defined.
- The README and code do not yet provide a single reproducible benchmark table with hardware, sample rate, block size, and exact checkpoint provenance.
- Some model files referenced by benchmark-selection code are not included in the current checkpoint directory.
- The project contains generated artifacts and experimental sweep results alongside source code, which makes it harder to distinguish canonical outputs from exploratory runs.
- Real-time audio I/O, plugin packaging, and end-to-end FPGA/Bela deployment are still separate prototypes rather than one integrated path.

## Suggested next additions

The most valuable next documentation and project additions would be:

1. Add a small smoke-test command or CI job that checks imports, DSP effects, target generation, model loading, and the Verilog testbench.
2. Add a checked-in benchmark report with the machine, OS, Python/PyTorch versions, sample rate, block size, model file, parameter count, latency, RTF, MSE, and ESR.
3. Add a model manifest describing each checkpoint’s effect, architecture, training data, target parameters, quantization width, and intended input shape.
4. Separate source, reproducible examples, and generated artifacts; document which large WAVs/checkpoints are required versus optional.
5. Document audio conventions explicitly: mono conversion, sample-rate handling, normalization/clipping policy, state reset behavior, and whether each effect is causal.
6. Add a real-time milestone with a fixed block-size API and a measured latency budget before pursuing plugin or standalone deployment.

## Research notes

Effect-specific technical notes are in the [`instructions/`](instructions/) directory. The original research document is available [here](https://docs.google.com/document/d/1PU49m20RlBC7QgCGgH99PVEWIrra0MUrChNkFV747vk/edit?tab=t.0).
