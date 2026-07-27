import os
import torch
import librosa
import numpy as np

from src.nn.architecture import SimpleTCN, BrevitasQuantizedSimpleTCN
from src.nn.quantization import export_simple_tcn_quantization_artifact, QuantizationConfig

def main():
    models_to_quantize = [
        {
            "name": "exciter",
            "path": "models/sweep_results/exciter_tcn_c8_l4_k5.pt",
            "channels": 8,
            "layers": 4,
            "kernel": 5
        },
        {
            "name": "distortion",
            "path": "models/sweep_results/distortion_tcn_c8_l4_k7.pt",
            "channels": 8,
            "layers": 4,
            "kernel": 7
        }
    ]

    bit_widths = [4, 8, 16]

    # Load a small snippet of audio for calibration (PTQ max/min range collection)
    audio_path = "data/datasets/inputs/funk-soul-guitar-clean-4_90bpm_G.wav"
    audio, _ = librosa.load(audio_path, sr=44100, mono=True)
    # Use just the first 1 second for calibration
    calibration_data = [audio[:44100]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for m in models_to_quantize:
        print(f"\nProcessing {m['name']} ({m['path']})...")
        
        # Load the Float model
        float_model = SimpleTCN(
            hidden_channels=m["channels"], 
            num_layers=m["layers"], 
            kernel_size=m["kernel"]
        ).to(device)
        
        state_dict = torch.load(m["path"], map_location=device)
        float_model.load_state_dict(state_dict)
        float_model.eval()

        for bits in bit_widths:
            print(f"  -> Quantizing to {bits}-bit...")
            
            # 1. Export JSON Artifact for C++ Deployment
            config = QuantizationConfig(
                weight_bits=bits, 
                activation_bits=bits,
                accumulator_bits=32
            )
            
            json_out_path = f"models/sweep_results/{m['name']}_tcn_c{m['channels']}_l{m['layers']}_k{m['kernel']}_q{bits}.json"
            
            export_simple_tcn_quantization_artifact(
                model=float_model,
                calibration_data=calibration_data,
                output_path=json_out_path,
                config=config,
                device=str(device)
            )
            
            # 2. Export a Brevitas .pt file for PyTorch benchmarking
            # Instantiate Brevitas equivalent
            brevitas_model = BrevitasQuantizedSimpleTCN(
                weight_bits=bits, 
                act_bits=bits,
                hidden_channels=m["channels"], 
                num_layers=m["layers"], 
                kernel_size=m["kernel"]
            ).to(device)
            
            # Copy standard float weights into the Brevitas model
            # SimpleTCN has layers: Conv(0), Chomp(1), Tanh(2), Conv(3), ...
            # Brevitas has: QuantConv(0), Chomp(1), Tanh(2), QuantIdentity(3), QuantConv(4), ...
            b_dict = brevitas_model.state_dict()
            for key, val in state_dict.items():
                if 'tcn.' in key:
                    parts = key.split('.')
                    idx = int(parts[1])
                    if idx % 3 == 0:  # It's a Conv1d layer
                        new_idx = (idx // 3) * 4
                        new_key = f"tcn.{new_idx}.{'.'.join(parts[2:])}"
                        if new_key in b_dict:
                            b_dict[new_key] = val
                else:
                    if key in b_dict:
                        b_dict[key] = val
                        
            brevitas_model.load_state_dict(b_dict, strict=False)
            
            # Run one forward pass to initialize any dynamic quantizers
            brevitas_model.eval()
            with torch.no_grad():
                test_tensor = torch.from_numpy(calibration_data[0]).float().unsqueeze(0).unsqueeze(0).to(device)
                brevitas_model(test_tensor)
                
            pt_out_path = f"models/sweep_results/{m['name']}_tcn_c{m['channels']}_l{m['layers']}_k{m['kernel']}_q{bits}.pt"
            
            # Force add a dummy 'quant_flag' so wrapper.py knows it's a quantized model
            final_dict = brevitas_model.state_dict()
            final_dict['quant_flag'] = torch.tensor(1)
            torch.save(final_dict, pt_out_path)
            
            print(f"     Saved JSON: {json_out_path}")
            print(f"     Saved .pt : {pt_out_path}")

if __name__ == "__main__":
    main()
