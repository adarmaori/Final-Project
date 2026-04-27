---
description: "Use when editing Python audio pipeline files (training, inference, DSP, dataset, benchmarks). Enforces command location, path conventions, and audio-model safeguards."
name: "Python Audio Pipeline Guidelines"
applyTo: "python_code/**/*.py"
---
# Python Audio Pipeline Guidelines

- Run project commands from `python_code/` unless a task explicitly targets another track.
- Preserve dataset pairing: every file in `data/datasets/inputs/` must map to the same filename in `data/datasets/targets/`.
- Use effect-specific targets and model ownership:
  - flanger: `targets_flange` with `lstm` models
  - distortion: `targets_distortion` with `tcn` models
- Keep tensor and signal conventions stable unless migration is requested:
  - NN tensors are channel-first (`batch, channels, length`) for TCN paths.
  - Default sample rate is typically 44.1 kHz.
- Do not assume GPU execution: verify CUDA availability in the active environment before claiming GPU-backed results.
- Treat real-time DSP as stateful when applicable: preserve filter state across blocks and call `reset()` between unrelated streams.
- Preserve repository-relative path behavior in scripts (`generate_targets.py`, `inference.py`, benchmarks). If changing path logic, update all affected scripts together.
- If code introduces new Python imports, update `python_code/pyproject.toml` dependencies in the same change.

## Validation Checklist

- For dataset-generation changes: run `uv run generate_targets.py`.
- For training/data-path changes: run at least a short training smoke test (small epochs/batch if needed).
- For performance/training claims on GPU: run
  - `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
- For inference/wrapper changes: run `uv run inference.py --input_file "<file>.wav"`.
- For benchmark-related changes: run `uv run tests/phase1_benchmark.py` when feasible.
- Prefer effect-explicit benchmark commands when feasible:
  - `uv run tests/phase1_benchmark.py --effect flange --input_file <file>.wav`
  - `uv run tests/phase1_benchmark.py --effect distortion --input_file <file>.wav`

## Reference Docs

- Project workflow and status: `README.md`
- DSP/NN comparison plan: `instructions/PC.md`
