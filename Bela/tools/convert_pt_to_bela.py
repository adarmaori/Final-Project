#!/usr/bin/env python3
"""Convert the small Torch checkpoint state dictionaries used by Bela.

This deliberately avoids importing torch: on the development machine the
checkpoint was produced on MPS and importing the local Torch installation is
not reliable.  The .pt files are standard zip-based Torch checkpoints, and
the tensor storage payloads are plain little-endian float32 values.
"""

from __future__ import annotations

import argparse
import json
import pickle
import struct
import zipfile
from pathlib import Path


class Storage:
    def __init__(self, key: str, size: int):
        self.key = key
        self.size = size


class Tensor:
    def __init__(self, storage: Storage, offset: int, shape: tuple[int, ...], stride: tuple[int, ...]):
        self.storage = storage
        self.offset = offset
        self.shape = shape
        self.stride = stride


def rebuild_tensor(storage, offset, size, stride, _requires_grad, _hooks):
    return Tensor(storage, offset, tuple(size), tuple(stride))


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return rebuild_tensor
        if module == "torch" and name == "FloatStorage":
            return object
        return super().find_class(module, name)

    def persistent_load(self, pid):
        _kind, _storage_type, key, _device, size = pid
        return Storage(str(key), int(size))


def read_checkpoint(path: Path) -> dict[str, list | float]:
    with zipfile.ZipFile(path) as archive:
        root = archive.namelist()[0].split("/")[0]
        state = Unpickler(__import__("io").BytesIO(archive.read(root + "/data.pkl"))).load()
        out = {}
        for name, tensor in state.items():
            if not isinstance(tensor, Tensor):
                continue
            raw = archive.read(root + "/data/" + tensor.storage.key)
            values = struct.unpack("<" + "f" * tensor.storage.size, raw)

            def at(index):
                flat = tensor.offset + sum(i * s for i, s in zip(index, tensor.stride))
                return float(values[flat])

            def nested(shape, prefix=()):
                if not shape:
                    return at(prefix)
                return [nested(shape[1:], prefix + (i,)) for i in range(shape[0])]

            out[name] = nested(tensor.shape)
    return out


def convert(path: Path, output: Path) -> None:
    state = read_checkpoint(path)
    conv_names = sorted((n for n in state if n.startswith("tcn.") and n.endswith(".weight")),
                        key=lambda n: int(n.split(".")[1]))
    layers = []
    for name in conv_names:
        index = int(name.split(".")[1])
        layers.append({
            "name": name,
            "weight": state[name],
            "bias": state[f"tcn.{index}.bias"],
            "dilation": 2 ** (len(layers)),
            "activation": "tanh",
        })
    final_weight = state["final_conv.weight"]
    output.write_text(json.dumps({
        "schema": "final-project.audio-tcn.float-bela.v1",
        "target": "bela",
        "checkpoint": str(path),
        "model": {
            "input_channels": 1,
            "hidden_channels": len(layers[0]["weight"]),
            "output_channels": len(final_weight),
            "kernel_size": len(layers[0]["weight"][0][0]),
            "num_layers": len(layers),
            "dilations": [layer["dilation"] for layer in layers],
            "activation": "tanh",
            "causal": True,
        },
        "layers": layers,
        "final_conv": {"weight": final_weight, "bias": state["final_conv.bias"]},
    }, indent=2) + "\n", encoding="utf-8")


def c_float(value: float) -> str:
    return f"{value:.9g}f"


def cpp_array(values, indent="\t") -> str:
    if not isinstance(values, list):
        return c_float(values)
    if not values:
        return "{}"
    child = [cpp_array(value, indent + "\t") for value in values]
    if isinstance(values[0], list):
        return "{\n" + ",\n".join(indent + item for item in child) + "\n" + indent[:-1] + "}"
    return "{ " + ", ".join(child) + " }"


def render_cpp(json_path: Path, output: Path, effect: str) -> None:
    artifact = json.loads(json_path.read_text(encoding="utf-8"))
    model = artifact["model"]
    layers = artifact["layers"]
    hidden = model["hidden_channels"]
    kernel = model["kernel_size"]
    num_layers = model["num_layers"]
    max_history = max(model["dilations"]) * (kernel - 1)
    weights = [layer["weight"] for layer in layers]
    biases = [layer["bias"] for layer in layers]
    final_weight = artifact["final_conv"]["weight"]
    final_bias = artifact["final_conv"]["bias"][0]
    final_values = [row[0][0] for row in final_weight]
    cpp = f'''#include <Bela.h>
#include <cmath>
#include <cstdint>
#include <ctime>

// Generated from {artifact["checkpoint"]}.
// {effect} TCN: {num_layers} causal Tanh layers, {hidden} hidden channels,
// kernel {kernel}, dilations {{{", ".join(map(str, model["dilations"]))}}}.
namespace {{
constexpr int kLayers = {num_layers};
constexpr int kHidden = {hidden};
constexpr int kKernel = {kernel};
constexpr int kMaxHistory = {max_history};
constexpr int kMaxChannels = 2;
constexpr int kDilations[kLayers] = {{{", ".join(map(str, model["dilations"]))}}};

static const float kWeights[kLayers][kHidden][kHidden][kKernel] = {cpp_array(weights)};
static const float kBiases[kLayers][kHidden] = {cpp_array(biases)};
static const float kFinalWeight[kHidden] = {cpp_array(final_values)};
static const float kFinalBias = {c_float(final_bias)};

struct VoiceState {{
    float history[kLayers][kMaxHistory][kHidden] = {{}};
    void reset() {{
        for(int layer = 0; layer < kLayers; ++layer)
            for(int delay = 0; delay < kMaxHistory; ++delay)
                for(int channel = 0; channel < kHidden; ++channel)
                    history[layer][delay][channel] = 0.0f;
    }}
}};

VoiceState gState[kMaxChannels];
uint64_t gLatencyBlocks = 0;
uint64_t gLatencyTotalNs = 0;
uint64_t gLatencyMaxNs = 0;

static uint64_t nowNs()
{{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}}

static float processSample(float input, VoiceState& state)
{{
    float layerInput[kHidden] = {{ input }};
    float layerOutput[kHidden] = {{}};
    for(int layer = 0; layer < kLayers; ++layer) {{
        const int inputChannels = layer == 0 ? 1 : kHidden;
        for(int out = 0; out < kHidden; ++out) {{
            float value = kBiases[layer][out];
            for(int in = 0; in < inputChannels; ++in) {{
                value += kWeights[layer][out][in][0] * layerInput[in];
                for(int tap = 1; tap < kKernel; ++tap)
                    value += kWeights[layer][out][in][tap] * state.history[layer][tap * kDilations[layer] - 1][in];
            }}
            layerOutput[out] = std::tanh(value);
        }}
        const int historyLength = (kKernel - 1) * kDilations[layer];
        for(int delay = historyLength - 1; delay > 0; --delay)
            for(int channel = 0; channel < inputChannels; ++channel)
                state.history[layer][delay][channel] = state.history[layer][delay - 1][channel];
        for(int channel = 0; channel < inputChannels; ++channel)
            state.history[layer][0][channel] = layerInput[channel];
        for(int channel = 0; channel < kHidden; ++channel)
            layerInput[channel] = layerOutput[channel];
    }}
    float output = kFinalBias;
    for(int channel = 0; channel < kHidden; ++channel)
        output += kFinalWeight[channel] * layerInput[channel];
    return output;
}}
}}

bool setup(BelaContext*, void*)
{{
    for(int channel = 0; channel < kMaxChannels; ++channel)
        gState[channel].reset();
    return true;
}}

void render(BelaContext* context, void*)
{{
    const uint64_t startNs = nowNs();
    const unsigned int channels = context->audioInChannels < kMaxChannels ? context->audioInChannels : kMaxChannels;
    for(unsigned int frame = 0; frame < context->audioFrames; ++frame) {{
        float outputs[kMaxChannels] = {{}};
        if(channels == 0)
            outputs[0] = processSample(0.0f, gState[0]);
        for(unsigned int channel = 0; channel < channels; ++channel)
            outputs[channel] = processSample(audioRead(context, frame, channel), gState[channel]);
        for(unsigned int channel = 0; channel < context->audioOutChannels; ++channel) {{
            const unsigned int inputChannel = channel < channels ? channel : 0;
            audioWrite(context, frame, channel, outputs[inputChannel]);
        }}
    }}
    const uint64_t elapsedNs = nowNs() - startNs;
    ++gLatencyBlocks;
    gLatencyTotalNs += elapsedNs;
    if(elapsedNs > gLatencyMaxNs)
        gLatencyMaxNs = elapsedNs;
}}

void cleanup(BelaContext*, void*)
{{
    const double averageNs = gLatencyBlocks > 0
        ? static_cast<double>(gLatencyTotalNs) / static_cast<double>(gLatencyBlocks)
        : 0.0;
    rt_printf("{effect} average latency: %.0f ns (%.3f ms)\\n", averageNs, averageNs / 1e6);
    rt_printf("{effect} maximum latency: %llu ns (%.3f ms)\\n",
              static_cast<unsigned long long>(gLatencyMaxNs),
              static_cast<double>(gLatencyMaxNs) / 1e6);
}}
'''
    output.write_text(cpp, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpp", type=Path)
    parser.add_argument("--effect", default="effect")
    args = parser.parse_args()
    convert(args.checkpoint, args.output)
    if args.cpp:
        render_cpp(args.output, args.cpp, args.effect)
