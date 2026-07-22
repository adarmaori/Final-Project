import itertools
import os
import sys
import glob
import time
import shutil
import subprocess
import torch
import librosa
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nn.architecture import SimpleTCN
from src.dsp.exciter import aural_exciter
from src.dsp.distortion import tube_saturator

# ---------------------------------------------------------
# 1. SWEEP CONFIGURATION
# ---------------------------------------------------------
TEST_FILE = "../raw_sound_files/funk-soul-guitar-clean-4_90bpm_G.wav"
CHECKPOINT_DIR = "models/checkpoints"
SWEEP_OUT_DIR = "models/sweep_results"

EPOCHS = 50
LR = 0.003
BATCH_SIZE = 32
BENCHMARK_ONLY = False

CHANNELS = [4, 8, 12]
LAYERS = [2, 4]
KERNELS = [5, 7]

# Define the targets and their exact DSP matches for the slim benchmark
EFFECTS = {
    "exciter": {
        "target_dir": "targets_exciter",
        "dsp_func": lambda x: aural_exciter(x, drive=6.0, mix=0.8, cutoff_freq=2200.0),
    },
    "distortion": {
        "target_dir": "targets_distortion",
        "dsp_func": lambda x: tube_saturator(x, drive=100.0, asymmetry=0.4, tone=3000),
    },
}

# The Architectural Grid: (Hidden Channels, Num Layers, Kernel Size)
ARCH_GRID = list(itertools.product(CHANNELS, LAYERS, KERNELS))

print(f"Total architectures to test per effect: {len(ARCH_GRID)}")


# ---------------------------------------------------------
# 2. SLIM BENCHMARKER
# ---------------------------------------------------------
def calculate_esr(pred, target):
    """Calculates the Error-to-Signal Ratio."""
    return torch.sum((target - pred) ** 2) / torch.sum(target**2)


def slim_evaluate(model_path, effect_name, channels, layers, kernel):
    """Loads the model, processes a test file, and returns the ESR vs pure DSP."""
    # 1. Load Audio
    x, sr = librosa.load(TEST_FILE, sr=44100, mono=True)

    # 2. Generate Ground Truth using native DSP
    y_target = EFFECTS[effect_name]["dsp_func"](x)

    # 3. Load Model
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps"
        # if torch.backends.mps.is_available()
        # else "cpu"
    )

    model = SimpleTCN(
        hidden_channels=channels, num_layers=layers, kernel_size=kernel
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Run Inference
    x_tensor = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        y_pred = model(x_tensor).squeeze().cpu().numpy()

    # 5. Trim padding to match lengths
    min_len = min(len(y_target), len(y_pred))
    y_target = torch.tensor(y_target[:min_len])
    y_pred = torch.tensor(y_pred[:min_len])

    # 6. Calculate ESR
    esr = calculate_esr(y_pred, y_target).item()
    mse = torch.mean((y_pred - y_target) ** 2).item()
    return esr * 100.0, mse  # Return ESR as percentage and MSE as raw power


# ---------------------------------------------------------
# 3. SWEEP ORCHESTRATOR
# ---------------------------------------------------------
os.makedirs(SWEEP_OUT_DIR, exist_ok=True)
report_path = os.path.join(SWEEP_OUT_DIR, "sweep_report_small.csv")

with open(report_path, "w") as f:
    f.write("Effect,Channels,Layers,Kernel,Params,ESR_Percent,MSE,Model_File\n")

for effect, config in EFFECTS.items():
    print(f"\n{'=' * 50}\nStarting Sweep for: {effect.upper()}\n{'=' * 50}")

    for channels, layers, kernel in ARCH_GRID:
        model_name = f"{effect}_tcn_c{channels}_l{layers}_k{kernel}"
        final_model_path = os.path.join(SWEEP_OUT_DIR, f"{model_name}.pt")

        if 1:
            # if not os.path.exists(final_model_path):
            if BENCHMARK_ONLY:
                print(
                    f"\n--- Missing {model_name}.pt; skipping (benchmark-only mode) ---"
                )
                continue

            print(f"\n--- Training {model_name} ---")

            # 0. CLEAR OLD CHECKPOINTS
            # Prevent the script from accidentally loading a model from a previous run
            for f in glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt")):
                os.remove(f)

            # 1. Execute Training (subprocess)
            # Set save_interval directly to EPOCHS so it saves exactly once at the end
            cmd = [
                sys.executable,
                "src/nn/train.py",
                "--effect",
                effect,
                "--target_subdir",
                config["target_dir"],
                "--model_type",
                "tcn",
                "--tcn_hidden_channels",
                str(channels),
                "--lr",
                str(LR),
                "--tcn_num_layers",
                str(layers),
                "--tcn_kernel_size",
                str(kernel),
                "--epochs",
                str(EPOCHS),
                "--batch_size",
                str(BATCH_SIZE),
                "--save_interval",
                str(EPOCHS),  # <-- CHANGED: Forces a final save
            ]

            start_time = time.time()
            subprocess.run(cmd, check=True)

            # 2. Find the final checkpoint and rename it
            # Now we use getmtime (Modified Time), which works flawlessly on Windows
            list_of_files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt"))

            if not list_of_files:
                print(f"ERROR: train.py did not output any .pt files for {model_name}!")
                continue

            latest_file = max(list_of_files, key=os.path.getmtime)
            shutil.copy(latest_file, final_model_path)

            # Optional: Delete the original to keep the checkpoints folder perfectly clean
            os.remove(latest_file)
        else:
            print(f"\n--- Using existing checkpoint for {model_name} ---")

        # 3. Run Slim Benchmark
        print(f"Benchmarking {model_name}...")
        esr_score, mse_score = slim_evaluate(
            final_model_path, effect, channels, layers, kernel
        )

        # Calculate params roughly for the log
        param_count = sum(
            p.numel()
            for p in SimpleTCN(
                hidden_channels=channels, num_layers=layers, kernel_size=kernel
            ).parameters()
        )

        print(
            f"Result: {esr_score:.2f}% ESR | MSE: {mse_score:.6f} | {param_count} params"
        )

        # 4. Log to CSV
        with open(report_path, "a") as f:
            f.write(
                f"{effect},{channels},{layers},{kernel},{param_count},{esr_score:.2f},{mse_score:.6f},{model_name}.pt\n"
            )

print(f"\nSweep complete! Report saved to {report_path}")
