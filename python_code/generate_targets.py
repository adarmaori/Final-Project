import argparse
import glob
import os
import sys

import librosa
import numpy as np
import soundfile as sf

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.dsp.distortion import tube_saturator
from src.dsp.flanger import flanger_effect


def _collect_wav_files(input_dir):
    return sorted(glob.glob(os.path.join(input_dir, "*.wav")))


def _pick_shortest_and_longest(file_paths):
    if len(file_paths) < 2:
        return file_paths

    durations = []
    for path in file_paths:
        info = sf.info(path)
        durations.append((info.frames / info.samplerate, path))

    durations.sort(key=lambda x: x[0])
    shortest = durations[0][1]
    longest = durations[-1][1]
    if shortest == longest:
        return [shortest]
    return [shortest, longest]


def _process_with_effect(y, sr, effect, args):
    if effect == "tube":
        return tube_saturator(
            y,
            drive=args.tube_drive,
            asymmetry=args.tube_asymmetry,
            tone=args.tube_tone,
            fs=sr,
        )

    if effect == "flanger":
        return flanger_effect(
            y,
            rate=args.flanger_rate,
            depth=args.flanger_depth,
            center_delay=args.flanger_center_delay,
            ff=args.flanger_ff,
            fb=args.flanger_fb,
            fs=sr,
        )

    raise ValueError(f"Unsupported effect: {effect}")


def generate_targets(input_dir, target_dir, args):
    """
    Reads all .wav files from input_dir, processes them with the DSP effect,
    and saves them to target_dir.
    """
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Find all wav files
    input_files = _collect_wav_files(input_dir)
    
    if not input_files:
        print(f"No .wav files found in {input_dir}")
        return

    if args.preview_short_long:
        input_files = _pick_shortest_and_longest(input_files)
        print("Preview mode enabled: processing shortest and longest WAV only.")

    print(f"Found {len(input_files)} files to process with effect='{args.effect}'.")
    
    for in_path in input_files:
        filename = os.path.basename(in_path)
        out_path = os.path.join(target_dir, filename)
        
        try:
            # Load Audio
            # Load at native SR
            y, sr = librosa.load(in_path, sr=None, mono=True)
            
            # Apply selected DSP effect
            y_processed = _process_with_effect(y, sr, args.effect, args)
            
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

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate DSP targets from input WAV files.")

    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input WAV directory. Defaults to data/datasets/inputs.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=None,
        help="Output WAV directory. Defaults to data/datasets/targets.",
    )
    parser.add_argument(
        "--effect",
        type=str,
        choices=["tube", "flanger"],
        default="tube",
        help="DSP effect to apply.",
    )
    parser.add_argument(
        "--preview_short_long",
        action="store_true",
        help="Process only shortest and longest WAV from input_dir.",
    )

    # Tube saturator parameters
    parser.add_argument("--tube_drive", type=float, default=100.0)
    parser.add_argument("--tube_asymmetry", type=float, default=0.4)
    parser.add_argument("--tube_tone", type=float, default=3000.0)

    # Flanger parameters
    parser.add_argument("--flanger_rate", type=float, default=0.5, help="LFO rate in Hz.")
    parser.add_argument("--flanger_depth", type=float, default=2.0, help="Sweep depth in ms.")
    parser.add_argument("--flanger_center_delay", type=float, default=2.5, help="Center delay in ms.")
    parser.add_argument("--flanger_ff", type=float, default=0.7, help="Feed-forward coefficient.")
    parser.add_argument("--flanger_fb", type=float, default=0.2, help="Feedback coefficient (-0.9 to 0.9).")

    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    input_dir = args.input_dir or os.path.join(base_dir, "data", "datasets", "inputs")
    target_dir = args.target_dir or os.path.join(base_dir, "data", "datasets", "targets")

    print(f"Input Directory: {input_dir}")
    print(f"Target Directory: {target_dir}")
    print(f"Effect: {args.effect}")

    generate_targets(input_dir, target_dir, args)
