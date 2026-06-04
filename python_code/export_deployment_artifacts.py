import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch

from src.nn.architecture import BrevitasQuantizedSimpleTCN


@dataclass(frozen=True)
class TensorQuantization:
    bits: int
    scale: float
    zero_point: int
    qmin: int
    qmax: int
    signed: bool


@dataclass(frozen=True)
class QuantizedParameter:
    name: str
    bits: int
    scale: list[float] | float
    zero_point: int | list[int]
    values: list


@dataclass(frozen=True)
class QuantizedLayer:
    name: str
    layer_type: str
    input_quant: TensorQuantization | None
    output_quant: TensorQuantization | None
    weight: QuantizedParameter | None = None
    bias: QuantizedParameter | None = None


@dataclass(frozen=True)
class QuantizedArtifact:
    model_name: str
    model_type: str
    quant_bits: int
    activation_bits: int
    accumulator_bits: int
    checkpoint: str
    calibration: dict[str, dict[str, float]]
    layers: list[QuantizedLayer]
    metadata: dict


def _qbounds(num_bits: int, signed: bool = True) -> tuple[int, int]:
    if num_bits < 2:
        raise ValueError("Quantization bit width must be at least 2 bits.")
    if signed:
        qmin = -(1 << (num_bits - 1))
        qmax = (1 << (num_bits - 1)) - 1
    else:
        qmin = 0
        qmax = (1 << num_bits) - 1
    return qmin, qmax


def _prepare_audio_tensor(sample: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = sample if isinstance(sample, torch.Tensor) else torch.from_numpy(np.asarray(sample))
    tensor = tensor.detach().clone().float()
    if tensor.dim() == 1:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() == 3:
        return tensor
    raise ValueError(f"Unsupported calibration sample shape: {tuple(tensor.shape)}")


def _load_calibration_audio(calibration_dir: Path, max_files: int | None = None) -> Iterable[np.ndarray]:
    wav_files = sorted(calibration_dir.glob("*.wav"))
    if max_files is not None:
        wav_files = wav_files[:max_files]

    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in calibration directory: {calibration_dir}")

    for wav_path in wav_files:
        audio, _ = librosa.load(wav_path, sr=None, mono=True)
        yield audio


def _collect_ranges(model: BrevitasQuantizedSimpleTCN, calibration_data: Iterable[np.ndarray]) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    hooks = []

    def register(name: str, module: torch.nn.Module) -> None:
        def _hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            tensor = tensor.detach().float()
            stats = ranges.setdefault(name, {"min": float("inf"), "max": float("-inf")})
            stats["min"] = min(stats["min"], float(tensor.min().item()))
            stats["max"] = max(stats["max"], float(tensor.max().item()))

        hooks.append(module.register_forward_hook(_hook))

    register("conv1", model.conv1)
    register("relu", model.relu)
    register("conv2", model.conv2)
    register("output", model)

    model.eval()
    saw_sample = False
    with torch.no_grad():
        for sample in calibration_data:
            saw_sample = True
            tensor = _prepare_audio_tensor(sample)
            input_stats = ranges.setdefault("input", {"min": float("inf"), "max": float("-inf")})
            input_stats["min"] = min(input_stats["min"], float(tensor.min().item()))
            input_stats["max"] = max(input_stats["max"], float(tensor.max().item()))
            model(tensor)

    for h in hooks:
        h.remove()

    if not saw_sample:
        raise ValueError("Calibration data is empty.")

    return ranges


def _build_tensor_quantization(bits: int, stats: dict[str, float], signed: bool) -> TensorQuantization:
    qmin, qmax = _qbounds(bits, signed=signed)
    max_abs = max(abs(stats["min"]), abs(stats["max"]))
    scale = max(max_abs / float(qmax), 1e-12)
    return TensorQuantization(bits=bits, scale=float(scale), zero_point=0, qmin=qmin, qmax=qmax, signed=signed)


def _quantize_per_channel_weight(weight: torch.Tensor, num_bits: int, channel_axis: int = 0) -> tuple[torch.Tensor, list[float], list[int]]:
    qmin, qmax = _qbounds(num_bits, signed=True)
    reduce_dims = tuple(dim for dim in range(weight.dim()) if dim != channel_axis)
    max_abs = torch.amax(torch.abs(weight), dim=reduce_dims)
    scales = torch.clamp(max_abs / float(qmax), min=1e-12)

    view_shape = [1] * weight.dim()
    view_shape[channel_axis] = weight.shape[channel_axis]
    quantized = torch.clamp(torch.round(weight / scales.view(view_shape)), qmin, qmax).to(torch.int64)
    return quantized, scales.tolist(), [0] * weight.shape[channel_axis]


def _quantize_bias(bias: torch.Tensor, input_scale: float, weight_scales: list[float], num_bits: int) -> tuple[torch.Tensor, list[float], list[int]]:
    qmin, qmax = _qbounds(num_bits, signed=True)
    bias_scales = [max(float(input_scale) * float(w_scale), 1e-12) for w_scale in weight_scales]
    quantized = []
    for value, scale in zip(bias.tolist(), bias_scales):
        quantized.append(int(np.clip(np.round(value / scale), qmin, qmax)))
    return torch.tensor(quantized, dtype=torch.int64), bias_scales, [0] * len(bias_scales)


def _infer_activation_bits_from_model(model: BrevitasQuantizedSimpleTCN, quant_bits: int) -> int:
    for attr in ("input_quant", "output_quant", "relu"):
        module = getattr(model, attr, None)
        if module is None:
            continue
        bit_width = getattr(module, "bit_width", None)
        if bit_width is None:
            continue
        if callable(bit_width):
            try:
                value = bit_width()
            except TypeError:
                value = None
        else:
            value = bit_width
        if value is not None:
            return int(float(value))
    return 8 if quant_bits == 4 else quant_bits


def _accumulator_min_bits(weight_bits: int, activation_bits: int, kernel_size: int, in_channels: int) -> int:
    mac_terms = kernel_size * in_channels
    growth = int(np.ceil(np.log2(max(mac_terms, 1))))
    return weight_bits + activation_bits + growth + 1


def _checkpoint_quant_bits(path: Path) -> int | None:
    stem = path.stem
    parts = stem.split("_")
    for part in parts:
        if part.startswith("q") and part[1:].isdigit():
            return int(part[1:])
    return None


def _build_quantized_artifact(
    model: BrevitasQuantizedSimpleTCN,
    checkpoint_path: Path,
    calibration_data: Iterable[np.ndarray],
    quant_bits: int,
    accumulator_bits: int | None,
) -> QuantizedArtifact:
    ranges = _collect_ranges(model, calibration_data)
    activation_bits = _infer_activation_bits_from_model(model, quant_bits)

    input_quant = _build_tensor_quantization(activation_bits, ranges["input"], signed=True)
    conv1_out_quant = _build_tensor_quantization(activation_bits, ranges["conv1"], signed=True)
    relu_out_quant = _build_tensor_quantization(activation_bits, ranges["relu"], signed=False)
    conv2_out_quant = _build_tensor_quantization(activation_bits, ranges["conv2"], signed=True)
    output_quant = _build_tensor_quantization(activation_bits, ranges["output"], signed=True)

    conv1_weight_q, conv1_weight_scales, conv1_weight_zp = _quantize_per_channel_weight(model.conv1.weight.detach().cpu(), quant_bits)
    conv2_weight_q, conv2_weight_scales, conv2_weight_zp = _quantize_per_channel_weight(model.conv2.weight.detach().cpu(), quant_bits)

    acc_min_conv1 = _accumulator_min_bits(quant_bits, activation_bits, model.conv1.kernel_size[0], model.conv1.in_channels)
    acc_min_conv2 = _accumulator_min_bits(quant_bits, activation_bits, model.conv2.kernel_size[0], model.conv2.in_channels)
    min_required_acc = max(acc_min_conv1, acc_min_conv2)
    resolved_acc_bits = max(accumulator_bits or min_required_acc, min_required_acc)

    conv1_bias_q, conv1_bias_scales, conv1_bias_zp = _quantize_bias(
        model.conv1.bias.detach().cpu(), input_quant.scale, conv1_weight_scales, resolved_acc_bits
    ) if model.conv1.bias is not None else (None, [], [])

    conv2_bias_q, conv2_bias_scales, conv2_bias_zp = _quantize_bias(
        model.conv2.bias.detach().cpu(), relu_out_quant.scale, conv2_weight_scales, resolved_acc_bits
    ) if model.conv2.bias is not None else (None, [], [])

    layers = [
        QuantizedLayer(
            name="conv1",
            layer_type="conv1d",
            input_quant=input_quant,
            output_quant=conv1_out_quant,
            weight=QuantizedParameter(
                name="weight",
                bits=quant_bits,
                scale=conv1_weight_scales,
                zero_point=conv1_weight_zp,
                values=conv1_weight_q.tolist(),
            ),
            bias=(
                QuantizedParameter(
                    name="bias",
                    bits=resolved_acc_bits,
                    scale=conv1_bias_scales,
                    zero_point=conv1_bias_zp,
                    values=conv1_bias_q.tolist(),
                ) if conv1_bias_q is not None else None
            ),
        ),
        QuantizedLayer(
            name="relu",
            layer_type="relu",
            input_quant=conv1_out_quant,
            output_quant=relu_out_quant,
        ),
        QuantizedLayer(
            name="conv2",
            layer_type="conv1d",
            input_quant=relu_out_quant,
            output_quant=conv2_out_quant,
            weight=QuantizedParameter(
                name="weight",
                bits=quant_bits,
                scale=conv2_weight_scales,
                zero_point=conv2_weight_zp,
                values=conv2_weight_q.tolist(),
            ),
            bias=(
                QuantizedParameter(
                    name="bias",
                    bits=resolved_acc_bits,
                    scale=conv2_bias_scales,
                    zero_point=conv2_bias_zp,
                    values=conv2_bias_q.tolist(),
                ) if conv2_bias_q is not None else None
            ),
        ),
        QuantizedLayer(
            name="output",
            layer_type="identity",
            input_quant=conv2_out_quant,
            output_quant=output_quant,
        ),
    ]

    metadata = {
        "hidden_channels": int(getattr(model, "hidden_channels", model.conv1.out_channels)),
        "kernel_size": int(model.conv1.kernel_size[0]),
        "dilation": int(model.conv1.dilation[0]),
        "residual": bool(getattr(model, "residual", False)),
        "output_gain": float(getattr(model, "output_gain", torch.tensor(1.0)).detach().cpu().item()),
        "accumulator_min_bits": {
            "conv1": int(acc_min_conv1),
            "conv2": int(acc_min_conv2),
        },
    }

    return QuantizedArtifact(
        model_name=model.__class__.__name__,
        model_type="brevitas_quantized_simple_tcn",
        quant_bits=int(quant_bits),
        activation_bits=int(activation_bits),
        accumulator_bits=int(resolved_acc_bits),
        checkpoint=str(checkpoint_path),
        calibration=ranges,
        layers=layers,
        metadata=metadata,
    )


def _artifact_to_payload(artifact: QuantizedArtifact, target: str) -> dict:
    return {
        "schema_version": 1,
        "target": target,
        "model_name": artifact.model_name,
        "model_type": artifact.model_type,
        "quant_bits": artifact.quant_bits,
        "activation_bits": artifact.activation_bits,
        "accumulator_bits": artifact.accumulator_bits,
        "checkpoint": artifact.checkpoint,
        "metadata": artifact.metadata,
        "calibration": artifact.calibration,
        "layers": [
            {
                "name": layer.name,
                "layer_type": layer.layer_type,
                "input_quant": asdict(layer.input_quant) if layer.input_quant else None,
                "output_quant": asdict(layer.output_quant) if layer.output_quant else None,
                "weight": asdict(layer.weight) if layer.weight else None,
                "bias": asdict(layer.bias) if layer.bias else None,
            }
            for layer in artifact.layers
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _export_bela_json(artifact: QuantizedArtifact, output_dir: Path, checkpoint_stem: str) -> Path:
    out_path = output_dir / "bela" / f"{checkpoint_stem}_bela.json"
    payload = _artifact_to_payload(artifact, target="bela")
    payload["runtime"] = {
        "format": "json",
        "sample_rate": 44100,
        "process_mode": "sample",
    }
    _write_json(out_path, payload)
    return out_path


def _export_fpga_package(
    model: BrevitasQuantizedSimpleTCN,
    artifact: QuantizedArtifact,
    output_dir: Path,
    checkpoint_stem: str,
    io_type: str,
    strategy: str,
    board_part: str | None,
) -> tuple[Path, Path, Path]:
    fpga_dir = output_dir / "fpga" / checkpoint_stem
    fpga_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = fpga_dir / f"{checkpoint_stem}.onnx"
    model.eval()
    dummy = torch.zeros(1, 1, 64, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["audio_in"],
        output_names=["audio_out"],
        dynamic_axes={"audio_in": {2: "samples"}, "audio_out": {2: "samples"}},
        opset_version=18,
        dynamo=False,
    )

    layer_precision = {
        "conv1": {
            "weight": f"ap_fixed<{artifact.quant_bits},1>",
            "bias": f"ap_fixed<{artifact.accumulator_bits},4>",
            "result": f"ap_fixed<{artifact.activation_bits + 2},2>",
            "accum": f"ap_fixed<{artifact.accumulator_bits},6>",
        },
        "conv2": {
            "weight": f"ap_fixed<{artifact.quant_bits},1>",
            "bias": f"ap_fixed<{artifact.accumulator_bits},4>",
            "result": f"ap_fixed<{artifact.activation_bits + 2},2>",
            "accum": f"ap_fixed<{artifact.accumulator_bits},6>",
        },
    }

    hls_config = {
        "Model": {
            "Precision": f"ap_fixed<{artifact.activation_bits + 2},2>",
            "ReuseFactor": 1,
            "Strategy": strategy,
            "BramFactor": 1000000000,
        },
        "LayerName": layer_precision,
    }

    hls_package = {
        "schema_version": 1,
        "backend": "Vitis",
        "io_type": io_type,
        "strategy": strategy,
        "part": board_part,
        "onnx_model": str(onnx_path).replace("\\", "/"),
        "hls_config": hls_config,
        "quantization": {
            "weight_bits": artifact.quant_bits,
            "activation_bits": artifact.activation_bits,
            "accumulator_bits": artifact.accumulator_bits,
            "accumulator_min_bits": artifact.metadata.get("accumulator_min_bits", {}),
        },
    }

    package_path = fpga_dir / "hls4ml_package.json"
    _write_json(package_path, hls_package)

    script_path = fpga_dir / "run_hls4ml.py"
    script_path.write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "",
                "import hls4ml",
                "",
                "pkg_path = Path(__file__).resolve().parent / 'hls4ml_package.json'",
                "pkg = json.loads(pkg_path.read_text(encoding='utf-8'))",
                "cfg = hls4ml.utils.config_from_onnx_model(pkg['onnx_model'], granularity='name')",
                "cfg.update(pkg['hls_config'])",
                "prj = hls4ml.converters.convert_from_onnx_model(",
                "    pkg['onnx_model'],",
                "    hls_config=cfg,",
                "    output_dir=str((Path(__file__).resolve().parent / 'hls4ml_prj')),",
                "    backend=pkg['backend'],",
                "    io_type=pkg['io_type'],",
                ")",
                "prj.compile()",
                "print('hls4ml project generated at', Path(__file__).resolve().parent / 'hls4ml_prj')",
            ]
        ),
        encoding="utf-8",
    )

    return onnx_path, package_path, script_path


def _find_quantized_checkpoints(checkpoints_dir: Path) -> list[Path]:
    matches = []
    for path in sorted(checkpoints_dir.glob("tcn_q*_final.pt")):
        bits = _checkpoint_quant_bits(path)
        if bits is not None:
            matches.append(path)
    return matches


def _load_quantized_model(checkpoint_path: Path, quant_bits: int, hidden_channels: int) -> BrevitasQuantizedSimpleTCN:
    model = BrevitasQuantizedSimpleTCN(quant_bits=quant_bits, hidden_channels=hidden_channels)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"[warn] Missing keys in {checkpoint_path.name}: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"[warn] Unexpected keys in {checkpoint_path.name}: {load_result.unexpected_keys}")
    return model


def _export_for_checkpoint(
    checkpoint_path: Path,
    calibration_dir: Path,
    output_dir: Path,
    target: str,
    max_files: int,
    hidden_channels: int,
    accumulator_bits: int | None,
    io_type: str,
    strategy: str,
    board_part: str | None,
) -> dict:
    quant_bits = _checkpoint_quant_bits(checkpoint_path)
    if quant_bits is None:
        raise ValueError(f"Could not infer quant bits from checkpoint name: {checkpoint_path.name}")

    model = _load_quantized_model(checkpoint_path, quant_bits=quant_bits, hidden_channels=hidden_channels)
    calibration_data = list(_load_calibration_audio(calibration_dir, max_files=max_files))

    artifact = _build_quantized_artifact(
        model=model,
        checkpoint_path=checkpoint_path,
        calibration_data=calibration_data,
        quant_bits=quant_bits,
        accumulator_bits=accumulator_bits,
    )

    checkpoint_stem = checkpoint_path.stem
    result = {
        "checkpoint": str(checkpoint_path),
        "quant_bits": quant_bits,
        "activation_bits": artifact.activation_bits,
        "accumulator_bits": artifact.accumulator_bits,
        "outputs": {},
    }

    if target in {"bela", "all"}:
        bela_path = _export_bela_json(artifact, output_dir=output_dir, checkpoint_stem=checkpoint_stem)
        result["outputs"]["bela_json"] = str(bela_path)

    if target in {"fpga", "all"}:
        onnx_path, package_path, script_path = _export_fpga_package(
            model=model,
            artifact=artifact,
            output_dir=output_dir,
            checkpoint_stem=checkpoint_stem,
            io_type=io_type,
            strategy=strategy,
            board_part=board_part,
        )
        result["outputs"]["fpga_onnx"] = str(onnx_path)
        result["outputs"]["fpga_hls4ml_package"] = str(package_path)
        result["outputs"]["fpga_hls4ml_runner"] = str(script_path)

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export quantized TCN deployment artifacts for Bela JSON and FPGA hls4ml workflows."
    )
    parser.add_argument("--target", choices=["bela", "fpga", "all"], default="all")
    parser.add_argument("--bela_format", choices=["json"], default="json")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a quantized checkpoint (e.g., tcn_q8_final.pt)")
    parser.add_argument("--export_all_quantized", action="store_true", help="Export every tcn_q*_final.pt checkpoint")
    parser.add_argument("--checkpoints_dir", type=str, default="models/checkpoints")
    parser.add_argument("--calibration_dir", type=str, default="data/datasets/inputs")
    parser.add_argument("--max_files", type=int, default=8)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--accumulator_bits", type=int, default=None, help="Optional override; will be clamped to required minimum")
    parser.add_argument("--output_dir", type=str, default="models/exported/deployment")
    parser.add_argument("--io_type", choices=["io_stream", "io_parallel"], default="io_stream")
    parser.add_argument("--strategy", choices=["Latency", "Resource"], default="Latency")
    parser.add_argument("--board_part", type=str, default=None, help="Optional FPGA part, e.g., xc7z020clg400-1")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.bela_format != "json":
        raise ValueError("Only JSON export is currently supported for Bela.")

    checkpoints_dir = Path(args.checkpoints_dir)
    calibration_dir = Path(args.calibration_dir)
    output_dir = Path(args.output_dir)

    if not calibration_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {calibration_dir}")

    checkpoint_paths: list[Path]
    if args.export_all_quantized:
        checkpoint_paths = _find_quantized_checkpoints(checkpoints_dir)
        if not checkpoint_paths:
            raise FileNotFoundError(f"No quantized checkpoints found in {checkpoints_dir}")
    elif args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        checkpoint_paths = [checkpoint]
    else:
        default_checkpoint = checkpoints_dir / "tcn_q8_final.pt"
        if not default_checkpoint.exists():
            raise FileNotFoundError(
                "No checkpoint provided. Use --checkpoint or --export_all_quantized. "
                f"Default checkpoint not found: {default_checkpoint}"
            )
        checkpoint_paths = [default_checkpoint]

    results = []
    for checkpoint_path in checkpoint_paths:
        print(f"[info] Exporting {checkpoint_path.name} for target={args.target}")
        result = _export_for_checkpoint(
            checkpoint_path=checkpoint_path,
            calibration_dir=calibration_dir,
            output_dir=output_dir,
            target=args.target,
            max_files=args.max_files,
            hidden_channels=args.hidden_channels,
            accumulator_bits=args.accumulator_bits,
            io_type=args.io_type,
            strategy=args.strategy,
            board_part=args.board_part,
        )
        results.append(result)

    summary = {
        "target": args.target,
        "count": len(results),
        "results": results,
    }
    summary_path = output_dir / "export_summary.json"
    _write_json(summary_path, summary)

    print(json.dumps(summary, indent=2))
    print(f"[done] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
