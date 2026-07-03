"""Deployment exporters for the TCN audio effects models.

Provides two paths:
- Bela-friendly plain JSON/YAML export with model metadata and weights.
- FPGA export through a float proxy model into hls4ml.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .architecture import BrevitasQuantizedSimpleTCN, SimpleTCN

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

try:
    import hls4ml
except ImportError:  # pragma: no cover - optional dependency
    hls4ml = None


def _infer_hidden_channels_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    conv1_weight = state_dict.get("conv1.weight")
    if conv1_weight is None:
        return 16
    return int(conv1_weight.shape[0])


def load_tcn_checkpoint(model_path: str | Path, quant_bits: int = 0) -> torch.nn.Module:
    """Load a float or Brevitas TCN checkpoint."""
    state_dict = torch.load(model_path, map_location="cpu")
    hidden_channels = _infer_hidden_channels_from_state_dict(state_dict)
    if quant_bits > 0:
        model = BrevitasQuantizedSimpleTCN(quant_bits=quant_bits, hidden_channels=hidden_channels)
    else:
        model = SimpleTCN(hidden_channels=hidden_channels)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _tensor_to_payload(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "values": tensor.tolist(),
    }


def _state_dict_to_payload(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {name: _tensor_to_payload(tensor) for name, tensor in state_dict.items()}


class _TCNExportProxy(torch.nn.Module):
    """Trace-safe TCN wrapper for hls4ml export.

    This mirrors the trained TCN structure but assumes a 3D channel-first input,
    which avoids the `x.dim()` control-flow branch that Torch FX cannot trace.
    """

    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(input_channels, hidden_channels, kernel_size=kernel_size, dilation=dilation, padding=0)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_channels, output_channels, kernel_size=kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


def _model_architecture_payload(model: torch.nn.Module) -> dict[str, Any]:
    conv1 = getattr(model, "conv1", None)
    conv2 = getattr(model, "conv2", None)
    payload: dict[str, Any] = {
        "class": model.__class__.__name__,
        "input_channels": int(getattr(conv1, "in_channels", 1)),
        "hidden_channels": int(getattr(conv1, "out_channels", 16)),
        "output_channels": int(getattr(conv2, "out_channels", 1)),
        "kernel_size": int(getattr(conv1, "kernel_size", (3,))[0] if conv1 is not None else 3),
        "dilation": int(getattr(conv1, "dilation", (1,))[0] if conv1 is not None else 1),
        "residual": bool(getattr(model, "residual", False)),
    }
    if hasattr(model, "quant_bits"):
        payload["quant_bits"] = int(getattr(model, "quant_bits"))
    if hasattr(model, "output_gain"):
        payload["output_gain"] = float(model.output_gain.detach().cpu().item())
    return payload


def export_bela_artifact(
    model: torch.nn.Module,
    output_path: str | Path,
    file_format: str = "json",
) -> Path:
    """Export a plain serializable artifact for Bela."""
    file_format = file_format.lower()
    if file_format not in {"json", "yaml", "yml"}:
        raise ValueError("file_format must be 'json' or 'yaml'")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": "final-project.audio-tcn.export.v1",
        "target": "bela",
        "model": _model_architecture_payload(model),
        "state_dict": _state_dict_to_payload(model.state_dict()),
    }

    if file_format == "json":
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        if yaml is None:
            raise ImportError("PyYAML is required for YAML export")
        output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    return output_path


def _build_float_proxy_tcn(model: torch.nn.Module) -> torch.nn.Module:
    """Build a trace-safe float TCN with the same trained conv weights for FPGA export."""
    conv1 = getattr(model, "conv1", None)
    conv2 = getattr(model, "conv2", None)
    input_channels = int(getattr(conv1, "in_channels", 1))
    hidden_channels = int(getattr(conv1, "out_channels", 16))
    output_channels = int(getattr(conv2, "out_channels", 1))
    kernel_size = int(getattr(conv1, "kernel_size", (3,))[0] if conv1 is not None else 3)
    dilation = int(getattr(conv1, "dilation", (1,))[0] if conv1 is not None else 1)

    float_model = _TCNExportProxy(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        output_channels=output_channels,
        kernel_size=kernel_size,
        dilation=dilation,
    )

    source_state = model.state_dict()
    float_state = float_model.state_dict()
    matched_state = {name: tensor for name, tensor in source_state.items() if name in float_state and float_state[name].shape == tensor.shape}
    float_model.load_state_dict(matched_state, strict=False)
    float_model.eval()
    return float_model


def export_hls4ml_project(
    model: torch.nn.Module,
    output_dir: str | Path,
    onnx_path: str | Path | None = None,
    backend: str = "Vitis",
    io_type: str = "io_stream",
    default_precision: str = "fixed<16,6>",
    sample_length: int = 16384,
) -> dict[str, Path]:
    """Export a float proxy TCN and convert it into an hls4ml project."""
    if hls4ml is None:
        raise ImportError("hls4ml is required for FPGA export")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(onnx_path) if onnx_path is not None else output_dir / "tcn_proxy.onnx"

    float_proxy_model = _build_float_proxy_tcn(model).cpu().eval()
    example_input = torch.zeros(1, 1, sample_length, dtype=torch.float32)

    torch.onnx.export(
        float_proxy_model,
        example_input,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "time"}, "output": {0: "batch", 2: "time"}},
        dynamo=False,
    )

    config = hls4ml.utils.config_from_pytorch_model(
        float_proxy_model,
        input_shape=tuple(example_input.shape),
        granularity="name",
        backend=backend,
        default_precision=default_precision,
    )

    hls_model = hls4ml.converters.convert_from_pytorch_model(
        float_proxy_model,
        output_dir=str(output_dir),
        io_type=io_type,
        backend=backend,
        hls_config=config,
    )
    hls_model.write()

    return {
        "onnx_path": onnx_path,
        "project_dir": output_dir,
    }


def export_quantized_model_bundle(
    checkpoint_path: str | Path,
    output_root: str | Path,
    quant_bits: int,
    bela_format: str = "json",
    export_bela: bool = True,
    export_fpga: bool = True,
    backend: str = "Vitis",
    io_type: str = "io_stream",
    default_precision: str = "fixed<16,6>",
    sample_length: int = 16384,
) -> dict[str, Path]:
    """Export one quantized checkpoint into a dedicated Bela and/or hls4ml bundle."""
    checkpoint_path = Path(checkpoint_path)
    output_root = Path(output_root)
    model_name = checkpoint_path.stem
    bundle_dir = output_root / model_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    model = load_tcn_checkpoint(checkpoint_path, quant_bits=quant_bits)
    outputs: dict[str, Path] = {}

    if export_bela:
        bela_filename = f"tcn_bela.{ 'yml' if bela_format == 'yml' else bela_format }"
        bela_path = bundle_dir / bela_filename
        outputs["bela_path"] = export_bela_artifact(model, bela_path, file_format=bela_format)

    if export_fpga:
        fpga_dir = bundle_dir / "hls4ml"
        fpga_outputs = export_hls4ml_project(
            model=model,
            output_dir=fpga_dir,
            onnx_path=bundle_dir / f"{model_name}.onnx",
            backend=backend,
            io_type=io_type,
            default_precision=default_precision,
            sample_length=sample_length,
        )
        outputs.update(fpga_outputs)

    return outputs


def export_all_quantized_tcn_bundles(
    checkpoint_dir: str | Path,
    output_root: str | Path,
    quant_bits_values: Iterable[int] = (16, 8, 4),
    bela_format: str = "json",
    export_bela: bool = True,
    export_fpga: bool = True,
    backend: str = "Vitis",
    io_type: str = "io_stream",
    default_precision: str = "fixed<16,6>",
    sample_length: int = 16384,
) -> dict[int, dict[str, Path]]:
    """Export all requested quantized TCN checkpoints into separate bundles."""
    checkpoint_dir = Path(checkpoint_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    exports: dict[int, dict[str, Path]] = {}
    for quant_bits in quant_bits_values:
        checkpoint_name = f"tcn_q{quant_bits}_final.pt"
        checkpoint_path = checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        exports[quant_bits] = export_quantized_model_bundle(
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            quant_bits=quant_bits,
            bela_format=bela_format,
            export_bela=export_bela,
            export_fpga=export_fpga,
            backend=backend,
            io_type=io_type,
            default_precision=default_precision,
            sample_length=sample_length,
        )

    return exports
