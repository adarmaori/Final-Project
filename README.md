# Neural Network vs. DSP Audio Effects Comparison

This project explores the capability of Neural Networks to emulate analog-style audio effects (specifically **Distortion/Saturation**) and benchmarks them against traditional deterministic Digital Signal Processing (DSP) algorithms.

## Project Status

We have implemented a complete end-to-end pipeline for **Phase 1 (Non-Real-Time Benchmarking)**:

*   **DSP Baseline**: An "Improved Tube Saturator" algorithm using asymmetric tanh waveshaping and filtering.
*   **Neural Network**: A Causal Temporal Convolutional Network (TCN) implemented in PyTorch.
*   **Dataset Generation**: A script to process raw audio through the DSP effect to create training target pairs.
*   **Training Loop**: A complete PyTorch training pipeline with validation and checkpointing.
*   **Inference Engine**: A wrapper to run both DSP and NN models on audio files.
*   **Benchmark**: A test script to measure processing speed (Real-Time Factor) and signal quality.

## Quick Start

### 1. Installation

This project uses `uv` for dependency management.

```bash
cd python_code
uv sync
```

### 2. Workflow

**Step A: Prepare Data**
1.  Place your raw/clean audio files (wav) in `python_code/data/datasets/inputs/`.
2.  Run the generator to create target "distorted" files:

```bash
uv run generate_targets.py
```

(This saves processed files to `python_code/data/datasets/targets/`)

**Step B: Train the Model**
Train the TCN to mimic the DSP effect:
```bash
uv run src/nn/train.py --epochs 100 --batch_size 16
```
The best model will be saved to `python_code/models/checkpoints/tcn_final.pt`.

**Step C: Run Inference**
Apply the trained model to a new audio file:

```bash
uv run inference.py --input_file "path/to/my_riff.wav"
```

Output will be saved to `python_code/data/processed/`.

**Step D: Benchmark**
Compare the speed and output of the DSP vs. the Trained NN:

```bash
uv run tests/phase1_benchmark.py
```

*TODO: add file size comparisons (several runs)*
*TODO: add statistics (several runs)*
*TODO: add different models (different size, optimized) to compare against eachother.*
*TODO: add real-time inference implementation*

## Comparisons

| Feature | Deterministic DSP (Tube Saturator) | Neural Network (TCN) |
| :--- | :--- | :--- |
| **Method** | Math (`tanh`, filters) | Causal Dilated Convolutions |
| **Complexity** | Extremely Low ($O(N)$) | Higher ($O(N \cdot Layers)$) |
| **Sound** | Precise, Defined | "Learned" imitation, can be warmer/noisier |
| **Speed** | ~0.001s per clip (Fast) | ~0.025s per clip (Slower but feasible) |

## File Structure

*   `src/dsp/`: Reference DSP implementations.
*   `src/nn/`: PyTorch model architecture and training logic.
*   `src/engine/`: Wrappers for unified inference.
*   `data/`: Storage for datasets and processed audio.
*   `tests/`: Benchmarking scripts.

---

### Previous Research Link
https://docs.google.com/document/d/1PU49m20RlBC7QgCGgH99PVEWIrra0MUrChNkFV747vk/edit?tab=t.0
