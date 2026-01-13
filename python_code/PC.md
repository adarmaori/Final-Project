# PC plan 

This project plan outlines a comparative study between a Neural Network (NN) and deterministic Digital Signal Processing (DSP) algorithms for audio effects. The focus is on **Distortion**, a perfect candidate because it is computationally cheap in DSP (simple math) but complex to model accurately with NNs (non-linearities).

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
*   **Model Architecture:** **Causal Temporal Convolutional Network (TCN)**.
    *   *Features:* Dilated convolutions (1, 2, 4...) allow learning long-term dependencies.
    *   *Causality:* Uses custom `Chomp1d` layers and padding to ensure the model *only* looks at past samples, making it suitable for real-time processing simulation.
    *   *Input/Output:* Accepts raw audio chunks (1D), outputs processed audio chunks (1D).
*   **Training Loop:** Implemented with split validation, MSE Loss, and automatic checkpointing.

#### **2. Deterministic Implementations**

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
3.  **Visual Quality:** Plots waveforms of Original vs. DSP vs. NN output for visual inspection.

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
│   │   └── targets/            # DSP-processed wav files (Generated)
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
│   │   ├── architecture.py     # TCN, Causal Convs, Chomp1d
│   │   ├── dataset.py          # AudioEffectDataset (Slicing/Loading)
│   │   └── train.py            # Training Loop Script
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

