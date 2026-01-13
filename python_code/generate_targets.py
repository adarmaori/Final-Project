import os
import glob
import soundfile as sf
import librosa
import numpy as np
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.dsp.distortion import tube_saturator

def generate_targets(input_dir, target_dir):
    """
    Reads all .wav files from input_dir, processes them with the DSP effect,
    and saves them to target_dir.
    """
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Find all wav files
    input_files = glob.glob(os.path.join(input_dir, "*.wav"))
    
    if not input_files:
        print(f"No .wav files found in {input_dir}")
        return

    print(f"Found {len(input_files)} files. Processing...")
    
    for in_path in input_files:
        filename = os.path.basename(in_path)
        out_path = os.path.join(target_dir, filename)
        
        try:
            # Load Audio
            # Load at native SR
            y, sr = librosa.load(in_path, sr=None, mono=True)
            
            # Apply DSP Effect (Tube Saturator)
            # Parameters: drive=100.0, asymmetry=0.4, tone=3000
            y_processed = tube_saturator(y, drive=100.0, asymmetry=0.4, tone=3000, fs=sr)
            
            # Normalize output to prevent clipping before write (optional but good practice for datasets)
            # Find peak
            # peak = np.max(np.abs(y_processed))
            # if peak > 1.0:
            #      y_processed = y_processed / peak * 0.95
            
            # Save
            sf.write(out_path, y_processed, sr)
            print(f"Processed: {filename}")
            
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    # Define paths relative to the script location
    base_dir = os.path.dirname(__file__)
    
    # User requested paths: data/datasets/inputs -> data/datasets/targets
    # Note: Previous context had data/datasets/train/input, but user asked specifically for data/datasets/inputs
    input_dir = os.path.join(base_dir, "data", "datasets", "inputs")
    target_dir = os.path.join(base_dir, "data", "datasets", "targets")
    
    print(f"Input Directory: {input_dir}")
    print(f"Target Directory: {target_dir}")
    
    generate_targets(input_dir, target_dir)
