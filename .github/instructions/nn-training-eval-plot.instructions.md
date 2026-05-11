---
description: "Use when editing NN training, evaluation CSV generation, and plotting/report scripts under src/nn and tests."
name: "NN Training, Evaluation, and Plotting Guidelines"
applyTo: "python_code/{src/nn/train.py,src/nn/trainer.py,src/nn/evaluator.py,src/nn/plot_report.py,tests/phase1_benchmark.py,tests/bela_benchmark_viz.py}"
---
# NN Training, Evaluation, and Plotting Guidelines

- Run commands from `python_code/`.
- Keep this workflow stable unless migration is requested:
  1. train experiments (`src/nn/train.py`)
  2. evaluate latency/NMSE to `data/processed/nn_latency_report.csv`
  3. generate plots from that CSV (`src/nn/plot_report.py`)
- Preserve metric naming and units used by existing tooling:
  - `avg_batch_ms`, `min_batch_ms`, `max_batch_ms`
  - `avg_sample_us`, `samples_per_second`
  - `nmse_percent`, `num_parameters`
- Keep CSV compatibility for downstream readers:
  - metadata lines must stay `# key: value`
  - tabular rows must remain parseable with `pandas.read_csv(..., comment="#")`
- If changing model search spaces or experiment naming in `src/nn/train.py`, keep deterministic experiment names and ensure evaluator CSV columns remain consistent for plotting scripts.
- Preserve architecture-specific batch shaping (`tcn` channel-first, `lstm` sequence-last) unless all call sites are migrated together.
- Keep output artifacts in `python_code/data/processed/` unless explicitly requested otherwise.

## Validation Checklist

- Dependency setup: `uv sync`
- Training/eval path smoke test: `uv run src/nn/train.py`
- Plot generation from evaluation CSV: `uv run src/nn/plot_report.py`
- Benchmark plotting/report checks when touched:
  - `uv run tests/phase1_benchmark.py`
  - `uv run tests/bela_benchmark_viz.py`

## Reference Docs

- Workflow overview: [README](../../README.md)
- General Python pipeline guardrails: [python-code.instructions](./python-code.instructions.md)
- Platform benchmarking context: [instructions/PC.md](../../instructions/PC.md)