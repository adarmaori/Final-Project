import os
import argparse
import librosa
import soundfile as sf
import numpy as np
from src.engine.wrapper import NNWrapper

def run_inference(args):
    # 1. Setup paths
    # Concat default input directory
    input_path = os.path.join("..", "raw_sound_files", args.input_file)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    # Determine Output Path
    processed_dir = os.path.join("data", "processed")
    
    if args.output_file is None:
        input_basename = os.path.splitext(os.path.basename(input_path))[0]
        model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
        output_filename = f"{input_basename}_{model_basename}.wav"
        output_path = os.path.join(processed_dir, output_filename)
    else:
        # Concat default output directory
        output_path = os.path.join(processed_dir, args.output_file)

    # 2. Initialize Model
    print("Initializing Model...")
    wrapper = NNWrapper(model_path=args.model_path)
    
    # 3. Load Audio
    print(f"Loading {input_path}...")
    y, sr = librosa.load(input_path, sr=None, mono=True)
    
    # 4. Process
    # Since TCNs are convolutional, we can perform inference on the whole file 
    # if it fits in VRAM/RAM.
    print("Processing...")
    y_out = wrapper.process(y)
    
    # 5. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, y_out, sr)
    print(f"Saved processed audio to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference with Trained Model")
    parser.add_argument("--input_file", type=str, default="input.wav", help="Filename of input wav (in ../raw_sound_files/)")
    parser.add_argument("--model_path", type=str, default="models/checkpoints/tcn_final.pt", help="Path to trained model .pt file")
    parser.add_argument("--output_file", type=str, default=None, help="Filename to save output wav (in data/processed/). Auto-generated if None.")

    args = parser.parse_args()
    run_inference(args)
