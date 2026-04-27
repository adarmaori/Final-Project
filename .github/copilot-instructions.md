# Project Guidelines

## Scope
- This repository has multiple tracks (`python_code`, `cpp_code`, `verilog_code`), but active implementation and benchmarking are currently centered in `python_code`.
- Prefer making Python pipeline changes unless the user explicitly asks for C++/FPGA updates.

## Build And Test
- Run commands from `python_code` unless stated otherwise.
- Install dependencies:
  - `cd python_code`
  - `uv sync`
- Generate training targets from clean inputs:
  - `uv run generate_targets.py`
- Train the neural model:
  - `uv run src/nn/train.py --epochs 100 --batch_size 16`
- Run inference on an input file (looked up under `../raw_sound_files/`):
  - `uv run inference.py --input_file "my_file.wav"`
- Run benchmark suite:
  - `uv run tests/phase1_benchmark.py --effect flange --input_file "my_file.wav"`

## Architecture
- `python_code/src/dsp/`: deterministic audio effects baseline (`tube_saturator`, real-time stateful saturator).
- `python_code/src/nn/`: model architecture, dataset loader, and training loop.
- `python_code/src/engine/wrapper.py`: unified wrappers for DSP and NN inference.
- `python_code/tests/phase1_benchmark.py`: benchmark experiments (batch speed/quality, scalability, block-latency simulation).
- `python_code/data/`: datasets (`inputs`, `targets`) and processed outputs/reports.
- `python_code/models/checkpoints/`: saved model checkpoints.

## Conventions
- Dataset pairing convention: each WAV in `data/datasets/inputs/` must have the same filename in `data/datasets/targets/`.
- Effect/model convention:
  - `LSTM` is for flanger (`targets_flange`)
  - `TCN` is for distortion (`targets_distortion`)
- Typical defaults are 44.1 kHz sample rate and chunked sequence training.
- NN path uses channel-first tensors for TCN (`batch, channels, length`).
- Real-time DSP relies on persistent filter state; call `reset()` between unrelated streams.
- Scripts assume repository-relative paths; keeping `python_code` as working directory avoids path issues.

## Known Pitfalls
- `soundfile` is imported by scripts but is not declared in `python_code/pyproject.toml`; install it if imports fail.
- `NNWrapper` warns and proceeds with random weights when a checkpoint path is missing.
- `AudioEffectDataset` currently hardcodes zero overlap behavior in chunk stride.
- Benchmark model list contains placeholder entries that are inactive by default.
- Benchmark supports `--effect flange|distortion` and optional `--input_file`.
- CUDA training depends on the active `uv` environment resolving a CUDA-enabled PyTorch build; verify with:
  - `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

## Documentation Links
- Project overview and workflow: `README.md`
- Platform plans and hardware direction:
  - `instructions/PC.md`
  - `instructions/Bela.md`
  - `instructions/FPGA.md`
- Roadmap artifact: `work_plan.pdf`