import argparse
import os
import sys
from pathlib import Path

import librosa
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.nn.architecture import SimpleTCN
from src.nn.quantization import QuantizationConfig, export_simple_tcn_quantization_artifact, summarize_artifact


def _load_calibration_audio(calibration_dir: Path, max_files: int | None = None):
    wav_files = sorted(calibration_dir.glob("*.wav"))
    if max_files is not None:
        wav_files = wav_files[:max_files]

    for wav_path in wav_files:
        audio, _ = librosa.load(wav_path, sr=None, mono=True)
        yield audio


def main(args):
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    state_dict = torch.load(args.model_path, map_location="cpu")
    model = SimpleTCN()
    model.load_state_dict(state_dict)

    calibration_dir = Path(args.calibration_dir)
    if not calibration_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {calibration_dir}")

    config = QuantizationConfig(
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        input_bits=args.input_bits,
        output_bits=args.output_bits,
        accumulator_bits=args.accumulator_bits,
    )

    artifact = export_simple_tcn_quantization_artifact(
        model=model,
        calibration_data=_load_calibration_audio(calibration_dir, args.max_files),
        output_path=args.output_path,
        config=config,
    )

    print(summarize_artifact(artifact))
    print(f"Saved quantized export to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a quantization artifact for the distortion TCN")
    parser.add_argument("--model_path", type=str, default="models/checkpoints/tcn_final.pt", help="Path to the float checkpoint")
    parser.add_argument("--calibration_dir", type=str, default="data/datasets/inputs", help="Directory of WAV files for calibration")
    parser.add_argument("--output_path", type=str, default="models/exported/tcn_quantized.json", help="Output JSON artifact")
    parser.add_argument("--weight_bits", type=int, default=8, help="Weight bit width")
    parser.add_argument("--activation_bits", type=int, default=8, help="Activation bit width")
    parser.add_argument("--input_bits", type=int, default=None, help="Input bit width (defaults to activation bits)")
    parser.add_argument("--output_bits", type=int, default=None, help="Output bit width (defaults to activation bits)")
    parser.add_argument("--accumulator_bits", type=int, default=32, help="Accumulator bit width")
    parser.add_argument("--max_files", type=int, default=8, help="Limit calibration WAV files")

    main(parser.parse_args())
