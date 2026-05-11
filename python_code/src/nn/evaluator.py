from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.nn.dataset import AudioEffectDataset
from src.nn.trainer import (
    ExperimentConfig,
    build_model,
    forward_model,
    prepare_batch,
    resolve_device,
)


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 1
    num_workers: int = 0
    warmup_batches: int = 3
    timed_batches: int | None = 20
    device: str | None = None
    csv_path: str = "data/processed/nn_latency_report.csv"
    description: str = ""


def evaluate_experiments(
    experiment_summaries: list[dict[str, Any]],
    experiments: dict[str, ExperimentConfig],
    config: EvaluationConfig,
) -> list[dict[str, Any]]:
    results = []
    csv_path = Path(config.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    for summary in experiment_summaries:
        experiment_name = summary["name"]
        experiment = experiments[experiment_name]
        checkpoint_path = summary["best_checkpoint"]
        results.append(evaluate_experiment(experiment, checkpoint_path, config))

    metadata = build_csv_metadata(experiment_summaries, config)
    write_results_csv(csv_path, results, metadata)
    print(f"\nWrote evaluation CSV to {csv_path}")
    return results


def evaluate_experiment(
    experiment: ExperimentConfig,
    checkpoint_path: str,
    config: EvaluationConfig,
) -> dict[str, Any]:
    device = resolve_device(config.device or experiment.training.device)
    dataset = AudioEffectDataset(experiment.dataset)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_model(experiment.model).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()

    warmup_batches = max(0, config.warmup_batches)
    timed_batches = config.timed_batches if config.timed_batches is not None else len(loader)

    batch_times_ms: list[float] = []
    total_samples = 0
    total_batches = 0
    total_squared_error = 0.0
    total_target_energy = 0.0

    with torch.inference_mode():
        for batch_index, (raw_inputs, raw_targets) in enumerate(loader):
            inputs, targets = prepare_batch(raw_inputs, raw_targets, device, experiment.model.name)

            if batch_index < warmup_batches:
                _ = forward_model(model, inputs, experiment.model.name)
                continue

            if total_batches >= timed_batches:
                break

            start = time.perf_counter()
            outputs = forward_model(model, inputs, experiment.model.name)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            batch_times_ms.append(elapsed_ms)
            total_batches += 1
            total_samples += int(raw_inputs.shape[0] * raw_inputs.shape[-1])
            total_squared_error += float(torch.sum((outputs - targets) ** 2).item())
            total_target_energy += float(torch.sum(targets ** 2).item())

    if not batch_times_ms:
        raise ValueError(
            f"No timed batches were evaluated for experiment {experiment.name}. "
            "Increase dataset size or reduce warmup_batches."
        )

    avg_batch_ms = sum(batch_times_ms) / len(batch_times_ms)
    min_batch_ms = min(batch_times_ms)
    max_batch_ms = max(batch_times_ms)
    p50_batch_ms = float(np.percentile(batch_times_ms, 50))
    p95_batch_ms = float(np.percentile(batch_times_ms, 95))
    p99_batch_ms = float(np.percentile(batch_times_ms, 99))
    samples_per_second = total_samples / (sum(batch_times_ms) / 1000.0)
    avg_sample_us = (sum(batch_times_ms) * 1000.0) / total_samples
    nmse_percent = 100.0 * total_squared_error / max(total_target_energy, 1e-12)

    result = {
        "experiment": experiment.name,
        "model": experiment.model.name,
        "checkpoint": checkpoint_path,
        "device": str(device),
        "sample_rate_hz": experiment.dataset.sample_rate,
        "chunk_size": experiment.dataset.chunk_size,
        "eval_batch_size": config.batch_size,
        "timed_batches": len(batch_times_ms),
        "avg_batch_ms": avg_batch_ms,
        "min_batch_ms": min_batch_ms,
        "p50_batch_ms": p50_batch_ms,
        "p95_batch_ms": p95_batch_ms,
        "p99_batch_ms": p99_batch_ms,
        "max_batch_ms": max_batch_ms,
        "avg_sample_us": avg_sample_us,
        "samples_per_second": samples_per_second,
        "nmse_percent": nmse_percent,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    result.update(flatten_model_kwargs(experiment.model.kwargs))
    return result


def write_results_csv(path: Path, results: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    if not results:
        return

    base_fieldnames = [
        "experiment",
        "model",
        "checkpoint",
        "device",
        "sample_rate_hz",
        "chunk_size",
        "eval_batch_size",
        "timed_batches",
        "avg_batch_ms",
        "min_batch_ms",
        "p50_batch_ms",
        "p95_batch_ms",
        "p99_batch_ms",
        "max_batch_ms",
        "avg_sample_us",
        "samples_per_second",
        "nmse_percent",
        "num_parameters",
    ]
    extra_fieldnames = sorted(
        {
            key
            for result in results
            for key in result.keys()
            if key not in base_fieldnames
        }
    )
    fieldnames = base_fieldnames + extra_fieldnames

    with path.open("w", newline="", encoding="utf-8") as handle:
        for key, value in metadata.items():
            handle.write(f"# {key}: {value}\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def flatten_model_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {f"model_{key}": value for key, value in kwargs.items()}


def build_csv_metadata(
    experiment_summaries: list[dict[str, Any]],
    config: EvaluationConfig,
) -> dict[str, Any]:
    experiment_names = [summary["name"] for summary in experiment_summaries]
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": config.description or "Latency evaluation report",
        "experiments": ",".join(experiment_names),
        "eval_batch_size": config.batch_size,
        "warmup_batches": config.warmup_batches,
        "timed_batches": config.timed_batches if config.timed_batches is not None else "all",
    }
