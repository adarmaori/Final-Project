import os
import sys
import torch
import librosa
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.nn.architecture import SimpleTCN
from src.dsp.distortion import tube_saturator

# Config
CHANNELS = 4
LAYERS = 4
KERNEL = 5
TEST_FILE = "raw_sound_files/funk-soul-guitar-clean-4_90bpm_G.wav"

def calculate_esr(pred, target):
    return torch.sum((target - pred) ** 2) / torch.sum(target**2)

def evaluate_and_plot(models):
    print("\n--- Evaluating Models ---")
    x, sr = librosa.load(TEST_FILE, sr=44100, mono=True)
    y_target = tube_saturator(x, drive=100.0, asymmetry=0.4, tone=5000.0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    results = {}
    preds = {}
    
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"Skipping {name}, not found at {path}")
            continue
            
        model = SimpleTCN(hidden_channels=CHANNELS, num_layers=LAYERS, kernel_size=KERNEL).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        
        x_tensor = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            y_pred = model(x_tensor).squeeze().cpu().numpy()
            
        min_len = min(len(y_target), len(y_pred))
        targ = torch.tensor(y_target[:min_len])
        pred = torch.tensor(y_pred[:min_len])
        
        esr = calculate_esr(pred, targ).item() * 100.0
        results[name] = esr
        preds[name] = y_pred[:min_len]
        print(f"{name} ESR: {esr:.2f}%")
        
    if not results:
        print("No models evaluated. Exiting.")
        return

    # Plotting 1000 point waveform
    print("\n--- Plotting Waveforms ---")
    fig, ax = plt.subplots(figsize=(10, 6))
    start_idx = 1000
    end_idx = start_idx + 1000
    
    ax.plot(y_target[start_idx:end_idx], label="DSP Target", color='black', linewidth=1.5)
    
    colors = ['r', 'b', 'g']
    for idx, (name, y_pred) in enumerate(preds.items()):
        ax.plot(y_pred[start_idx:end_idx], label=f"{name} (ESR: {results[name]:.2f}%)", 
                linestyle='--', color=colors[idx % len(colors)], alpha=0.8)
        
    ax.set_title("1000 Point Waveform Comparison (Incomplete Loss Functions)")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    out_plot = "models/sweep_results/loss_experiment_waveforms.png"
    os.makedirs("models/sweep_results", exist_ok=True)
    plt.savefig(out_plot, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {out_plot}")

if __name__ == "__main__":
    models = {
        "MRSTFT Only": "models/checkpoints/distortion_tcn_c4_l4_k5_mrstft.pt",
        "L1 Only": "models/checkpoints/distortion_tcn_c4_l4_k5_l1.pt"
    }
    evaluate_and_plot(models)
