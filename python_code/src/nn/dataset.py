from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetConfig:
    data_root: str = "data/datasets"
    sample_rate: int = 44_100
    chunk_size: int = 16_384
    overlap: float = 0.0
    normalize: bool = False


@dataclass(frozen=True)
class ChunkRecord:
    input_path: str
    target_path: str
    start: int
    stop: int


class AudioEffectDataset(Dataset):
    """
    Paired audio dataset for supervised effect modeling.

    Directory structure:
        data_root/
            inputs/
                clip_a.wav
            targets/
                clip_a.wav
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data_root = Path(config.data_root)
        self.input_dir = self.data_root / "inputs"
        self.target_dir = self.data_root / "targets"

        if not self.input_dir.exists() or not self.target_dir.exists():
            raise FileNotFoundError(
                f"Expected paired dataset folders at {self.input_dir} and {self.target_dir}"
            )

        if not 0.0 <= config.overlap < 1.0:
            raise ValueError("overlap must be in the range [0.0, 1.0)")
        if config.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.chunk_size = config.chunk_size
        self.stride = max(1, int(round(config.chunk_size * (1.0 - config.overlap))))
        self.chunks: list[ChunkRecord] = []
        self.file_pairs: list[tuple[Path, Path]] = []
        self._audio_cache: dict[str, np.ndarray] = {}

        self._scan_dataset()

        if not self.chunks:
            raise ValueError(
                f"No usable chunks found under {self.data_root}. "
                "Check that inputs/targets contain matching wav files longer than chunk_size."
            )

    def _scan_dataset(self) -> None:
        for input_path in sorted(self.input_dir.glob("*.wav")):
            target_path = self.target_dir / input_path.name
            if not target_path.exists():
                print(f"Warning: missing target file for {input_path.name}")
                continue

            input_audio = self._load_audio(input_path)
            target_audio = self._load_audio(target_path)
            pair_length = min(len(input_audio), len(target_audio))

            if pair_length < self.chunk_size:
                print(
                    f"Warning: skipping {input_path.name} because "
                    f"{pair_length} < chunk_size {self.chunk_size}"
                )
                continue

            self.file_pairs.append((input_path, target_path))

            last_start = pair_length - self.chunk_size
            for start in range(0, last_start + 1, self.stride):
                stop = start + self.chunk_size
                self.chunks.append(
                    ChunkRecord(
                        input_path=str(input_path),
                        target_path=str(target_path),
                        start=start,
                        stop=stop,
                    )
                )

    def _load_audio(self, path: Path) -> np.ndarray:
        cache_key = str(path)
        cached = self._audio_cache.get(cache_key)
        if cached is not None:
            return cached

        audio, _ = librosa.load(path, sr=self.config.sample_rate, mono=True)
        audio = audio.astype(np.float32, copy=False)

        if self.config.normalize:
            peak = float(np.max(np.abs(audio)))
            if peak > 0.0:
                audio = audio / peak

        self._audio_cache[cache_key] = audio
        return audio

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.chunks[idx]
        x = self._audio_cache[record.input_path][record.start:record.stop]
        y = self._audio_cache[record.target_path][record.start:record.stop]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

