from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.nn.architecture import SimpleLSTM, SimpleTCN
from src.nn.dataset import AudioEffectDataset, DatasetConfig


@dataclass(frozen=True)
class ModelConfig:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    validation_split: float = 0.2
    checkpoint_root: str = "models/checkpoints"
    save_every: int = 10
    seed: int = 7
    num_workers: int = 0
    device: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig


MODEL_REGISTRY = {
    "tcn": SimpleTCN,
    "lstm": SimpleLSTM,
}


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    set_seed(config.training.seed)
    device = resolve_device(config.training.device)
    dataset = AudioEffectDataset(config.dataset)
    train_dataset, val_dataset = split_dataset(dataset, config.training)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    model = build_model(config.model).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    run_dir = Path(config.training.checkpoint_root) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", asdict(config))

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    print(
        f"Starting {config.name}: model={config.model.name}, "
        f"train_chunks={len(train_dataset)}, val_chunks={len(val_dataset)}, device={device}"
    )

    for epoch in range(1, config.training.epochs + 1):
        start_time = time.time()
        train_loss = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            model_name=config.model.name,
            train=True,
        )
        val_loss = run_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            model_name=config.model.name,
            train=False,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "duration_sec": time.time() - start_time,
        }
        history.append(epoch_metrics)

        print(
            f"[{config.name}] epoch {epoch}/{config.training.epochs} "
            f"train={train_loss:.6f} val={val_loss:.6f} "
            f"time={epoch_metrics['duration_sec']:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(run_dir / "best.pt", model, config, epoch, history)

        if epoch % config.training.save_every == 0:
            save_checkpoint(run_dir / f"epoch_{epoch}.pt", model, config, epoch, history)

    save_checkpoint(run_dir / "final.pt", model, config, config.training.epochs, history)
    write_json(run_dir / "history.json", history)

    return {
        "name": config.name,
        "best_val_loss": best_val_loss,
        "run_dir": str(run_dir),
        "epochs": config.training.epochs,
        "best_checkpoint": str(run_dir / "best.pt"),
        "final_checkpoint": str(run_dir / "final.pt"),
    }


def build_model(config: ModelConfig) -> nn.Module:
    model_cls = MODEL_REGISTRY.get(config.name.lower())
    if model_cls is None:
        raise ValueError(f"Unknown model type: {config.name}")
    return model_cls(**config.kwargs)


def split_dataset(dataset: AudioEffectDataset, config: TrainingConfig):
    if not 0.0 < config.validation_split < 1.0:
        raise ValueError("validation_split must be in the range (0.0, 1.0)")

    val_size = max(1, int(len(dataset) * config.validation_split))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset split produced an empty training set")

    generator = torch.Generator().manual_seed(config.seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    model_name: str,
    train: bool,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    for raw_inputs, raw_targets in data_loader:
        inputs, targets = prepare_batch(raw_inputs, raw_targets, device, model_name)

        if train and optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = forward_model(model, inputs, model_name)
            loss = criterion(outputs, targets)
            if train and optimizer is not None:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(data_loader))


def prepare_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    model_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = inputs.to(device)
    targets = targets.to(device)

    if model_name == "tcn":
        return inputs.unsqueeze(1), targets.unsqueeze(1)
    if model_name == "lstm":
        return inputs.unsqueeze(-1), targets.unsqueeze(-1)

    raise ValueError(f"Unsupported model type for batch preparation: {model_name}")


def forward_model(model: nn.Module, inputs: torch.Tensor, model_name: str) -> torch.Tensor:
    if model_name == "lstm":
        outputs, _ = model(inputs)
        return outputs
    return model(inputs)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: ExperimentConfig,
    epoch: int,
    history: list[dict[str, Any]],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "experiment": asdict(config),
            "epoch": epoch,
            "history": history,
        },
        path,
    )


def resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
