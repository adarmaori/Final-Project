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

### **Phase 2: Project Architecture & Platform**

#### **1. The Neural Network Platform**

* **Framework:** **PyTorch** is recommended for its dynamic nature and easy switching to **TorchScript/ONNX** for optimization.
* **Model Architecture:** Start with a lightweight **1D Convolutional Network (TCN)** or a tiny **LSTM/GRU**. These are standard for modeling time-series audio effects.
* *Input:* Raw audio samples (buffer size , e.g., 512 samples).
* *Output:* Processed audio samples (same size).


* **Optimization:** Pure Python inference is slow. You must plan to export the trained model to **ONNX Runtime** for the actual benchmark. It drastically reduces Python overhead.

#### **2. Deterministic Implementations**

Implement these in pure Python (using NumPy) and optionally optimized with **Numba** to create a fair "fast Python" comparison.


* **Tube Saturator (Improved):**
  A multi-stage chain for warm, analog-style distortion:
  $$f(x) = \text{LPF}(\tanh(k \cdot x + \text{bias}))$$
    *   *Stages:* Input Gain $\rightarrow$ Asymmetry (DC Bias) $\rightarrow$ Soft Clip $\rightarrow$ Low-Pass Filter (4kHz) to remove digital fizz.

* **Bitcrushing:**
Reduces signal resolution (e.g., quantize float signal to 8-bit integers and back).

#### **3. Real-Time Audio Engine**

* **Library:** **PyAudio** (wrapper for PortAudio). It is the standard for low-level audio I/O in Python.
* **Mechanism:** Use a **Callback Mode** (non-blocking).
* The audio card requests data (e.g., 256 samples).
* Your code must fill this buffer and return it before the next request arrives.
* *Fail condition:* If you take too long, you hear "glitches" (buffer underruns).



---

### **Phase 3: The Testbench (Measuring Performance)**

The testbench needs to measure metrics without interfering with the audio stream itself.

#### **Key Metrics to Measure:**

1. **Real-Time Factor (RTF):**


* *Goal:*  (Ideally  for safety).


2. **Throughput:** Samples processed per second.
3. **Latency Jitter:** The variance in processing time. NNs often have "spikes" in latency (e.g., garbage collection or cache misses) even if average RTF is low.

#### **Proposed Testbench Class Structure (Python):**

```python
import time
import numpy as np

class AudioBenchmark:
    def __init__(self, processor_func, buffer_size, sample_rate):
        self.processor = processor_func
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.latencies = []

    def run_benchmark(self, input_buffer):
        start_time = time.perf_counter()
        
        # Run the effect
        output = self.processor(input_buffer)
        
        end_time = time.perf_counter()
        
        process_time = end_time - start_time
        buffer_duration = self.buffer_size / self.sample_rate
        
        # Log metrics
        self.latencies.append(process_time)
        return output

    def report(self):
        avg_lat = np.mean(self.latencies) * 1000 # ms
        p99_lat = np.percentile(self.latencies, 99) * 1000 # 99th percentile
        rtf = np.mean(self.latencies) / (self.buffer_size / self.sample_rate)
        
        print(f"Avg Latency: {avg_lat:.2f}ms")
        print(f"99% Latency: {p99_lat:.2f}ms (Glitches likely here)")
        print(f"Real-Time Factor: {rtf:.4f}")

```

---

### **Comparison Strategy (The "Vs" Part)**

To make the comparison meaningful, you need to compare apples to apples.

| Feature | Deterministic (NumPy/Numba) | Neural Network (PyTorch/ONNX) |
| --- | --- | --- |
| **Complexity** | O(N) - Linear per sample | O(N * Layers * Kernel) |
| **Memory** | Negligible | High (Weights + Activation Buffers) |
| **Buffer Sensitivity** | Low (can process 1 sample) | High (Often needs batching for speed) |
| **Quality** | Mathematically precise, "Cold" | Can learn "Warm" analog quirks |

### **Additional Considerations**

1. **Block-Based Processing (Statefulness):**
* **DSP:** Filters (like IIR) need to remember the *previous* sample state.
* **NN:** If using Recurrent Networks (LSTM/GRU), you must pass the "hidden state" from one audio chunk to the next. If you don't, the audio will have clicks at the buffer boundaries.
* *Plan:* Ensure your `process(buffer)` function accepts and returns a `state` variable.


2. **Overhead vs. Compute:**
* In Python, the overhead of *calling* the NN (moving data from CPU RAM to PyTorch Tensor) might be higher than the calculation itself for small buffers (e.g., 64 samples).
* *Test:* Measure how performance degrades as you *decrease* buffer size.


3. **Visualizing the Difference:**
* Include a real-time spectral analyzer (using `matplotlib` animation or `vispy`) to visually show the harmonic distortion added by the NN vs. the DSP.


Here is a recommended file structure for the project.

This structure segregates the "offline" Machine Learning workflow (training/data prep) from the "online" real-time application (inference/streaming), while keeping shared utilities accessible.

### **Project File Structure**

```text
audio_neural_vs_dsp/
│
├── README.md                   # Project overview and setup instructions
├── requirements.txt            # Dependencies (torch, numpy, pyaudio, onnxruntime, etc.)
├── main_benchmark.py           # Entry point for the real-time comparison testbench
│
├── data/                       # Storage for audio files
│   ├── raw/                    # Original dry audio samples (wav)
│   ├── processed/              # Audio with effects applied (for offline analysis)
│   └── datasets/               # Prepared datasets for NN training (input/target pairs)
│
├── models/                     # Saved models and checkpoints
│   ├── checkpoints/            # PyTorch .pt training checkpoints
│   └── exported/               # Optimized .onnx models for fast inference
│
├── src/                        # Source code
│   ├── __init__.py
│   │
│   ├── dsp/                    # Deterministic implementations (The Baseline)
│   │   ├── __init__.py
│   │   ├── distortion.py       # Hard/Soft clipping algorithms (NumPy/Numba)
│   │   └── filters.py          # Helper filters (e.g., simple low-pass if needed)
│   │
│   ├── nn/                     # Neural Network logic (The Challenger)
│   │   ├── __init__.py
│   │   ├── architecture.py     # PyTorch Model classes (TCN, LSTM, etc.)
│   │   ├── train.py            # Script to train the model
│   │   └── dataset.py          # PyTorch Dataset/Dataloader logic
│   │
│   ├── engine/                 # Real-time processing core
│   │   ├── __init__.py
│   │   ├── audio_io.py         # PyAudio wrapper (handles stream callbacks)
│   │   └── wrapper.py          # Unified wrapper class for NN and DSP models
│   │
│   └── analysis/               # Measurement tools
│       ├── __init__.py
│       ├── metrics.py          # Calculations for RTF, Throughput, Jitter
│       └── plotting.py         # Matplotlib scripts for waveforms/spectrograms
│
└── tests/                      # Unit tests
    ├── test_dsp.py             # Verify DSP math correctness
    └── test_nn_latency.py      # Basic isolated speed checks for models

```

### **Key Modules Explained**

* **`src/dsp/`**: Contains the "Reference" implementations. These are pure mathematical functions (e.g., `tanh(x)`). Using **Numba** decorators here allows you to compare "Optimized Python" vs "Neural Network."
* **`src/nn/`**: The "Lab." This is where you define the model layers and run the training loop. This code is mostly run offline.
* **`models/exported/`**: The "Deployment" folder. While you train in PyTorch, for the benchmark you should export the model here (e.g., to **ONNX**) to ensure you are benchmarking the model's speed, not the overhead of the Python training framework.
* **`src/engine/wrapper.py`**: The "Interface." This script defines a standard class structure (e.g., a `.process_buffer()` method) so the main benchmark loop doesn't need to know if it is talking to a Neural Net or a DSP function.
