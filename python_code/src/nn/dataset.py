import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import os
import glob

class AudioEffectDataset(Dataset):
    """
    Memory-efficient dataset for paired audio files (Input -> Target).
    Implements Receptive Field Padding (Context Windows) for time-smeared effects.
    """
    def __init__(
        self,
        data_root,
        sample_rate=44100,
        chunk_size=65536,
        context_size=44100,  # Number of samples to look back into the past
        overlap=0.0,
        input_subdir="inputs",
        target_subdir="targets",
    ):
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.input_dir = os.path.join(data_root, input_subdir)
        self.target_dir = os.path.join(data_root, target_subdir)
        
        # Keep full tracks in memory (avoids massive chunk duplication)
        self.audio_data = [] # List of tuples: (x_full, y_full)
        self.indices = []    # List of tuples: (file_idx, start_idx, end_idx)
        
        input_files = sorted(glob.glob(os.path.join(self.input_dir, "*.wav")))
        if not input_files:
            raise FileNotFoundError(f"No WAV files found in {self.input_dir}")
            
        for in_path in input_files:
            filename = os.path.basename(in_path)
            tgt_path = os.path.join(self.target_dir, filename)
            
            if os.path.exists(tgt_path):
                self._process_file(in_path, tgt_path)
            else:
                print(f"Warning: Missing target for {filename}")

        print(f"Dataset loaded: {len(self.indices)} chunks from {len(self.audio_data)} files.")

    def _process_file(self, in_path, tgt_path):
        # Load audio
        x, _ = librosa.load(in_path, sr=self.sample_rate, mono=True)
        y, _ = librosa.load(tgt_path, sr=self.sample_rate, mono=True)
        
        # Ensure lengths match
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        
        file_idx = len(self.audio_data)
        self.audio_data.append((x, y))
        
        stride = int(self.chunk_size * (1 - 0.0)) # 0 overlap for now
        num_chunks = (min_len - self.chunk_size) // stride + 1
        
        for i in range(num_chunks):
            start = i * stride
            end = start + self.chunk_size
            self.indices.append((file_idx, start, end))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, start, end = self.indices[idx]
        x_full, y_full = self.audio_data[file_idx]
        
        # Target is exactly the chunk size
        y_chunk = y_full[start:end]
        
        # Input gets prepended with historical context
        context_start = start - self.context_size
        if context_start < 0:
            # If we are at the very beginning of the song, pad with silence
            pad_len = -context_start
            x_chunk = np.pad(x_full[0:end], (pad_len, 0), mode='constant')
        else:
            # Otherwise, grab the real audio context from the past
            x_chunk = x_full[context_start:end]
            
        return torch.from_numpy(x_chunk).unsqueeze(0), torch.from_numpy(y_chunk).unsqueeze(0)