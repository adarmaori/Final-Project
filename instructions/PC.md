# PC plan 

This project plan outlines a comparative study between a Neural Network (NN) and deterministic Digital Signal Processing (DSP) algorithms for audio effects. The current active focus is **Flanger** (modulated delay), with distortion/tube-saturation retained as a secondary baseline in the codebase.

The initial LSTM flanger baseline reached benchmark runs but failed to converge to the desired ESR target. The active flanger architecture has been upgraded to a causal residual CRNN.

## **Phase 1: The "Simple Start" (Non-Real-Time)**

Before tackling real-time streams, build a foundation that processes `.wav` files. This isolates the algorithmic performance from audio I/O latency.

* **Objective:** Read a file, apply effects, write to disk, and measure "Processing Speed" vs "Signal Quality."
* **Workflow:**
1. Load audio segment (e.g., 5 seconds).
2. Run Deterministic function.
3. Run NN Inference.
4. Compare output waveforms and processing time.

---

### **Phase 2: Project Architecture & Platform (Implemented)**

#### **1. The Neural Network Platform**

*   **Framework:** **PyTorch**.
*   **Model Architectures:**
    *   **CRNN (active)**: causal residual convolution + GRU recurrence model for learning time-varying flanger response.
    *   **TCN (supported)**: causal convolution model for direct comparison.
    *   *Input/Output:* both models consume chunked mono audio and produce chunked mono targets.
*   **Training Loop:** Implemented with split validation, L1 + MR-STFT loss, warm-up masking, and automatic checkpointing.
*   **CUDA Usage:**
    *   Training is expected to use GPU when CUDA-enabled PyTorch is installed in the active `uv` environment.
    *   Quick verification:
        *   `nvidia-smi`
        *   `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
    *   If CUDA is unavailable in this check, training will run on CPU.

#### **2. Deterministic Implementations**

*   **Flanger (active reference):**
    $$delay(t) = c \cdot (1 + \sin(2\pi f t))$$
    where $c$ is `center_delay` (ms) and $f$ is `rate` (Hz).
    *   Sweep range is fixed to $[0, 2c]$ ms.
    *   Includes feed-forward and feedback paths with a fractional-delay interpolated read head.

*   **Tube Saturator (Improved):**
    A multi-stage chain for warm, analog-style distortion:
    $$f(x) = \text{LPF}(\tanh(\text{drive} \cdot x + \text{asymmetry}) - \text{dc\_offset})$$
    *   *Stages:*
        1.  **Drive**: High input gain (e.g., 70.0) to push signal into non-linearity.
        2.  **Asymmetry**: Adds a DC bias (e.g., 0.3-0.4) to create even-order harmonics (warmth).
        3.  **Soft Clip**: Uses `tanh` to round off peaks.
        4.  **Tone Stack**: A 4kHz Low-Pass Filter (Butterworth) to remove harsh high-frequency aliasing/fizz.

#### **3. Real-Time Audio Engine (Planned for Phase 3)**

*   **Library:** **PyAudio** (wrapper for PortAudio).
*   **Mechanism:** Callback Mode.

---

### **Phase 3: The Testbench (Measuring Performance)**

We have implemented `tests/phase1_benchmark.py` which performs the following:

#### **Key Metrics Measured:**

1.  **Processing Speed:** Time to process a fixed length file.
2.  **Ratio:** Comparison of NN inference time vs. DSP execution time.
3.  **Quality Error:** MSE/ESR versus deterministic DSP reference.
4.  **Real-Time Simulation:** block-by-block latency and jitter under fixed block size.
5.  **Visual Quality:** Plots waveforms of Original vs. DSP vs. NN output for visual inspection.

Current benchmark defaults compare:
* DSP Match (offline flanger)
* DSP RT (stateful real-time flanger)
* One NN checkpoint selected by effect mode:
    * `--effect flange` => `crnn_final.pt`
    * `--effect distortion` => `tcn_final.pt`

Recommended benchmark invocation:
```bash
cd python_code
uv run tests/phase1_benchmark.py --effect flange --input_file powerchords-mute.wav
```

---

### **Project File Structure (Current)**

```text
python_code/
│
├── README.md                   # Main project documentation
├── pyproject.toml              # Dependencies (uv managed)
├── main.py                     # (Currently unused placeholder)
│
├── data/                       # Audio Data Storage
│   ├── datasets/               # Training Data
│   │   ├── inputs/             # Clean wav files
│   │   ├── targets/            # Legacy/default target path
│   │   └── targets_flange/     # Active flanger target path
│   ├── processed/              # Inference/Benchmark outputs
│   └── raw/                    # Miscellaneous raw files
│
├── models/                     # Saved Models
│   └── checkpoints/            # PyTorch .pt training checkpoints
│
├── src/                        # Source Code
│   ├── dsp/                    # Deterministic Algorithms
│   │   ├── distortion.py       # Tube Saturator & simple tanh
│   │   └── filters.py          # LPF/HPF filter helpers
│   │
│   ├── nn/                     # Neural Network Logic
│   │   ├── architecture.py     # CRNN + TCN model definitions
│   │   ├── dataset.py          # AudioEffectDataset (configurable input/target subdirs)
│   │   └── train.py            # Training loop (supports --model_type tcn|lstm|crnn)
│   │
│   └── engine/                 # Unified Interfaces
│       └── wrapper.py          # NNWrapper / DSPWrapper classes
│
├── tests/                      # Testing & Benchmarking
│   └── phase1_benchmark.py     # Comparison Script
│
├── generate_targets.py         # Script to create training data from inputs
└── inference.py                # Script to run trained model on files
```

