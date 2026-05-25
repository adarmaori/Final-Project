---
description: "Run or update the Phase 1 benchmark workflow, then summarize timing, RTF, latency, and quality metrics with actionable conclusions."
name: "Run Benchmark Suite"
argument-hint: "Input audio, model checkpoints to compare, and any experiment constraints"
agent: "agent"
---
You are benchmarking this workspace's DSP vs NN audio pipeline.

Use the user's arguments as constraints for inputs/models/runtime budget. Then:

1. Confirm benchmark setup from `python_code/tests/phase1_benchmark.py`.
2. Run the benchmark workflow from `python_code/`.
3. Prefer using effect mode flags (`--effect flange|distortion`) over manual model toggles; for distortion, use the quantized TCN checkpoint selected by `--quant_bits` (for example `--quant_bits 16`). Only edit model entries if explicitly requested.
4. Summarize results with:
   - Processing time and Real-Time Factor (RTF)
   - Block-latency behavior and budget headroom
   - MSE/ESR quality deltas vs DSP reference
5. Provide a concise recommendation for next optimization steps.

Output format:

- Configuration used
- Key numeric results
- Risks/limitations observed
- Recommended next steps
