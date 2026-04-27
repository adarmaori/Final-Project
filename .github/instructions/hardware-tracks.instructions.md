---
description: "Use when working on cpp_code or verilog_code, or when planning Bela/FPGA implementation details. Covers low-latency constraints and cross-track consistency."
name: "Hardware Track Guidelines"
applyTo:
  - "cpp_code/**"
  - "verilog_code/**"
---
# Hardware Track Guidelines

- Keep hardware-track changes isolated to `cpp_code/` or `verilog_code/` unless the request explicitly includes Python pipeline updates.
- Optimize for predictable low-latency behavior first, then quality improvements.
- For Bela/C++ real-time callbacks:
  - Avoid allocations or blocking operations in the audio callback.
  - Prefer fixed-size/stateful processing patterns.
- For FPGA work, keep a measurable progression:
  - Baseline deterministic/DSP implementation.
  - Naive NN implementation.
  - Optimized NN (quantization/pipelining or equivalent).
- Document latency/resource tradeoffs with concrete metrics when introducing architecture changes.

## Cross-Track Consistency

- Keep effect naming and parameter semantics aligned with Python baseline where practical (e.g., drive/asymmetry/tone concepts).
- Keep effect/model mapping aligned with Python baseline:
  - `TCN` for distortion
  - `LSTM` for flanger
- If output behavior diverges by design, record the reason in code comments or adjacent docs.

## Reference Docs

- Bela implementation notes: `instructions/Bela.md`
- FPGA implementation notes: `instructions/FPGA.md`
- Overall roadmap context: `work_plan.pdf`
