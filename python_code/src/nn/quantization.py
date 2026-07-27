"""Quantization helpers for the TCN audio effects models.

This module focuses on deployment-friendly fixed-point export for the
SimpleTCN / Brevitas TCN architectures used by distortion and exciter.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from .architecture import BrevitasQuantizedSimpleTCN, SimpleTCN


@dataclass(frozen=True)
class QuantizationConfig:
    weight_bits: int
    activation_bits: int
    input_bits: int | None = None
    output_bits: int | None = None
    accumulator_bits: int = 32
    per_channel_weights: bool = True
    symmetric_weights: bool = True
    signed_activations: bool = True

    def resolved_input_bits(self) -> int:
        return self.input_bits if self.input_bits is not None else self.activation_bits

    def resolved_output_bits(self) -> int:
        return self.output_bits if self.output_bits is not None else self.activation_bits


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
    config: QuantizationConfig
    calibration: dict[str, dict[str, float]]
    layers: list[QuantizedLayer]


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


def _to_tensor(sample: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(sample, torch.Tensor):
        return sample.detach().clone().float()
    return torch.from_numpy(np.asarray(sample)).float()


def _prepare_audio_tensor(sample: torch.Tensor | np.ndarray) -> torch.Tensor:
    tensor = _to_tensor(sample)
    if tensor.dim() == 1:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() == 3:
        return tensor
    raise ValueError(f"Unsupported calibration sample shape: {tuple(tensor.shape)}")


def _quantize_symmetric(tensor: torch.Tensor, num_bits: int, signed: bool = True) -> tuple[torch.Tensor, TensorQuantization]:
    qmin, qmax = _qbounds(num_bits, signed=signed)
    max_abs = torch.max(torch.abs(tensor)).item()
    scale = max(max_abs / float(qmax), 1e-12)
    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int64)
    return quantized, TensorQuantization(bits=num_bits, scale=scale, zero_point=0, qmin=qmin, qmax=qmax, signed=signed)


def _quantize_per_channel_weight(weight: torch.Tensor, num_bits: int, channel_axis: int = 0) -> tuple[torch.Tensor, list[float], list[int], tuple[int, int]]:
    qmin, qmax = _qbounds(num_bits, signed=True)
    reduce_dims = tuple(dim for dim in range(weight.dim()) if dim != channel_axis)
    max_abs = torch.amax(torch.abs(weight), dim=reduce_dims)
    scales = torch.clamp(max_abs / float(qmax), min=1e-12)

    view_shape = [1] * weight.dim()
    view_shape[channel_axis] = weight.shape[channel_axis]
    quantized = torch.clamp(torch.round(weight / scales.view(view_shape)), qmin, qmax).to(torch.int64)
    return quantized, scales.tolist(), [0] * weight.shape[channel_axis], (qmin, qmax)


def _quantize_bias(bias: torch.Tensor, input_scale: float, weight_scales: Sequence[float], num_bits: int) -> tuple[torch.Tensor, list[float], list[int]]:
    qmin, qmax = _qbounds(num_bits, signed=True)
    bias_scales = [max(float(input_scale) * float(w_scale), 1e-12) for w_scale in weight_scales]
    bias_int = []
    for value, scale in zip(bias.tolist(), bias_scales):
        bias_int.append(int(np.clip(np.round(value / scale), qmin, qmax)))
    return torch.tensor(bias_int, dtype=torch.int64), bias_scales, [0] * len(bias_scales)


def _collect_simple_tcn_ranges(model: nn.Module, calibration_data: Iterable[torch.Tensor | np.ndarray], device: str = "cpu") -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    handles = []

    def register(name: str, module: nn.Module) -> None:
        def _hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            tensor = tensor.detach().float()
            stats = ranges.setdefault(name, {"min": float("inf"), "max": float("-inf")})
            stats["min"] = min(stats["min"], float(tensor.min().item()))
            stats["max"] = max(stats["max"], float(tensor.max().item()))

        handles.append(module.register_forward_hook(_hook))

    if hasattr(model, 'tcn'):
        for i, layer in enumerate(model.tcn):
            if isinstance(layer, nn.Conv1d):
                register(f"tcn_{i}_conv", layer)
            elif isinstance(layer, nn.Tanh):
                register(f"tcn_{i}_tanh", layer)
        register("final_conv", model.final_conv)
    elif hasattr(model, 'conv1'):
        register("conv1", model.conv1)
        if hasattr(model, 'relu'):
            register("relu", model.relu)
        register("conv2", model.conv2)
    
    register("output", model)

    model.eval()
    saw_sample = False
    with torch.no_grad():
        for sample in calibration_data:
            saw_sample = True
            tensor = _prepare_audio_tensor(sample).to(device)
            input_stats = ranges.setdefault("input", {"min": float("inf"), "max": float("-inf")})
            input_stats["min"] = min(input_stats["min"], float(tensor.min().item()))
            input_stats["max"] = max(input_stats["max"], float(tensor.max().item()))
            model(tensor)

    if not saw_sample:
        raise ValueError("Calibration data is empty; provide at least one audio file or tensor.")

    for handle in handles:
        handle.remove()

    return ranges


def _build_tensor_quantization(bits: int, stats: dict[str, float], signed: bool) -> TensorQuantization:
    qmin, qmax = _qbounds(bits, signed=signed)
    max_abs = max(abs(stats["min"]), abs(stats["max"]))
    scale = max(max_abs / float(qmax), 1e-12)
    return TensorQuantization(bits=bits, scale=scale, zero_point=0, qmin=qmin, qmax=qmax, signed=signed)


def export_simple_tcn_quantization_artifact(
    model: nn.Module,
    calibration_data: Iterable[torch.Tensor | np.ndarray],
    output_path: str | Path | None = None,
    config: QuantizationConfig | None = None,
    device: str = "cpu",
) -> QuantizedArtifact:
    """Create a fixed-point export package for SimpleTCN.

    The export is intended for Bela / FPGA workflows where weights and
    activations are represented in fixed-point, while accumulators stay wider.
    """

    if not isinstance(model, (SimpleTCN, BrevitasQuantizedSimpleTCN)):
        raise TypeError("export_simple_tcn_quantization_artifact only supports SimpleTCN or BrevitasQuantizedSimpleTCN.")

    cfg = config or QuantizationConfig(weight_bits=8, activation_bits=8)
    model = model.to(device)
    ranges = _collect_simple_tcn_ranges(model, calibration_data, device=device)

    input_quant = _build_tensor_quantization(cfg.resolved_input_bits(), ranges["input"], signed=True)
    layers = []
    
    if hasattr(model, 'tcn'):
        prev_quant = input_quant
        for i, layer in enumerate(model.tcn):
            if isinstance(layer, nn.Conv1d):
                out_quant = _build_tensor_quantization(cfg.activation_bits, ranges[f"tcn_{i}_conv"], signed=True)
                weight_q, weight_scales, weight_zp, _ = _quantize_per_channel_weight(layer.weight.detach().cpu(), cfg.weight_bits)
                bias_q, bias_scales, bias_zp = _quantize_bias(
                    layer.bias.detach().cpu(), prev_quant.scale, weight_scales, cfg.accumulator_bits
                ) if layer.bias is not None else (None, [], [])
                
                layers.append(QuantizedLayer(
                    name=f"tcn_{i}_conv",
                    layer_type="conv1d",
                    input_quant=prev_quant,
                    output_quant=out_quant,
                    weight=QuantizedParameter(name="weight", bits=cfg.weight_bits, scale=weight_scales, zero_point=weight_zp, values=weight_q.tolist()),
                    bias=QuantizedParameter(name="bias", bits=cfg.accumulator_bits, scale=bias_scales, zero_point=bias_zp, values=bias_q.tolist()) if bias_q is not None else None
                ))
                prev_quant = out_quant
            elif isinstance(layer, nn.Tanh):
                out_quant = _build_tensor_quantization(cfg.activation_bits, ranges[f"tcn_{i}_tanh"], signed=True)
                layers.append(QuantizedLayer(
                    name=f"tcn_{i}_tanh",
                    layer_type="tanh",
                    input_quant=prev_quant,
                    output_quant=out_quant
                ))
                prev_quant = out_quant
        
        # Final conv
        out_quant = _build_tensor_quantization(cfg.resolved_output_bits(), ranges["output"], signed=True)
        fc = model.final_conv
        weight_q, weight_scales, weight_zp, _ = _quantize_per_channel_weight(fc.weight.detach().cpu(), cfg.weight_bits)
        bias_q, bias_scales, bias_zp = _quantize_bias(
            fc.bias.detach().cpu(), prev_quant.scale, weight_scales, cfg.accumulator_bits
        ) if fc.bias is not None else (None, [], [])
        
        layers.append(QuantizedLayer(
            name="final_conv",
            layer_type="conv1d",
            input_quant=prev_quant,
            output_quant=out_quant,
            weight=QuantizedParameter(name="weight", bits=cfg.weight_bits, scale=weight_scales, zero_point=weight_zp, values=weight_q.tolist()),
            bias=QuantizedParameter(name="bias", bits=cfg.accumulator_bits, scale=bias_scales, zero_point=bias_zp, values=bias_q.tolist()) if bias_q is not None else None
        ))
    else:
        conv1_out_quant = _build_tensor_quantization(cfg.activation_bits, ranges["conv1"], signed=True)
        relu_out_quant = _build_tensor_quantization(cfg.activation_bits, ranges.get("relu", ranges["conv1"]), signed=False)
        conv2_out_quant = _build_tensor_quantization(cfg.activation_bits, ranges["conv2"], signed=True)
        output_quant = _build_tensor_quantization(cfg.resolved_output_bits(), ranges["output"], signed=True)

        conv1_weight_q, conv1_weight_scales, conv1_weight_zp, _ = _quantize_per_channel_weight(model.conv1.weight.detach().cpu(), cfg.weight_bits)
        conv1_bias_q, conv1_bias_scales, conv1_bias_zp = _quantize_bias(
            model.conv1.bias.detach().cpu(), input_quant.scale, conv1_weight_scales, cfg.accumulator_bits
        ) if model.conv1.bias is not None else (None, [], [])

        conv2_weight_q, conv2_weight_scales, conv2_weight_zp, _ = _quantize_per_channel_weight(model.conv2.weight.detach().cpu(), cfg.weight_bits)
        conv2_bias_q, conv2_bias_scales, conv2_bias_zp = _quantize_bias(
            model.conv2.bias.detach().cpu(), conv1_out_quant.scale, conv2_weight_scales, cfg.accumulator_bits
        ) if model.conv2.bias is not None else (None, [], [])

        layers = [
            QuantizedLayer(
                name="conv1",
                layer_type="conv1d",
                input_quant=input_quant,
                output_quant=conv1_out_quant,
                weight=QuantizedParameter(
                    name="weight", bits=cfg.weight_bits, scale=conv1_weight_scales, zero_point=conv1_weight_zp, values=conv1_weight_q.tolist(),
                ),
                bias=(QuantizedParameter(
                        name="bias", bits=cfg.accumulator_bits, scale=conv1_bias_scales, zero_point=conv1_bias_zp, values=conv1_bias_q.tolist(),
                    ) if conv1_bias_q is not None else None),
            ),
            QuantizedLayer(
                name="relu", layer_type="relu", input_quant=conv1_out_quant, output_quant=relu_out_quant,
            ),
            QuantizedLayer(
                name="conv2",
                layer_type="conv1d",
                input_quant=relu_out_quant,
                output_quant=conv2_out_quant,
                weight=QuantizedParameter(
                    name="weight", bits=cfg.weight_bits, scale=conv2_weight_scales, zero_point=conv2_weight_zp, values=conv2_weight_q.tolist(),
                ),
                bias=(QuantizedParameter(
                        name="bias", bits=cfg.accumulator_bits, scale=conv2_bias_scales, zero_point=conv2_bias_zp, values=conv2_bias_q.tolist(),
                    ) if conv2_bias_q is not None else None),
            ),
            QuantizedLayer(
                name="output", layer_type="identity", input_quant=conv2_out_quant, output_quant=output_quant,
            ),
        ]

    artifact = QuantizedArtifact(
        model_name=model.__class__.__name__,
        model_type="simple_tcn",
        config=cfg,
        calibration=ranges,
        layers=layers,
    )

    if output_path is not None:
        export_path = Path(output_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": artifact.model_name,
            "model_type": artifact.model_type,
            "config": asdict(artifact.config),
            "calibration": artifact.calibration,
            "layers": [
                {
                    "name": layer.name,
                    "layer_type": layer.layer_type,
                    "input_quant": asdict(layer.input_quant) if layer.input_quant is not None else None,
                    "output_quant": asdict(layer.output_quant) if layer.output_quant is not None else None,
                    "weight": asdict(layer.weight) if layer.weight is not None else None,
                    "bias": asdict(layer.bias) if layer.bias is not None else None,
                }
                for layer in artifact.layers
            ],
        }
        export_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return artifact


def summarize_artifact(artifact: QuantizedArtifact) -> str:
    lines = [
        f"Model: {artifact.model_name}",
        f"Type: {artifact.model_type}",
        f"Weights: {artifact.config.weight_bits}-bit",
        f"Activations: {artifact.config.activation_bits}-bit",
        f"Accumulator: {artifact.config.accumulator_bits}-bit",
    ]
    for layer in artifact.layers:
        if layer.weight is None:
            continue
        if isinstance(layer.weight.scale, list):
            lines.append(
                f"{layer.name}: weight scale range [{min(layer.weight.scale):.3e}, {max(layer.weight.scale):.3e}]"
            )
        else:
            lines.append(f"{layer.name}: weight scale {layer.weight.scale:.3e}")
    return "\n".join(lines)
