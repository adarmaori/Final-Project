import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import os
import glob

class AudioEffectDataset(Dataset):
    """
    Dataset for paired audio files (Input -> Target).
    Assumes a folder structure:
        data_root/
            inputs/
                file1.wav
                ...
            targets/
                file1.wav  (must match input filename)
                ...
    """
    def __init__(self, data_root, sample_rate=44100, chunk_size=2048, overlap=0.0):
        """
        Args:
            data_root (str): Path to dataset folder containing 'input' and 'target' subfolders.
            sample_rate (int): Target sample rate.
            chunk_size (int): Number of samples per training example (sequence length).
            overlap (float): Overlap between chunks (0.0 to 1.0).
        """
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.input_dir = os.path.join(data_root, "inputs")
        self.target_dir = os.path.join(data_root, "targets")
        
        # Verify directories exist
        if not os.path.exists(self.input_dir) or not os.path.exists(self.target_dir):
             raise FileNotFoundError(f"Data directory must contain 'inputs' and 'targets' folders at {data_root}")

        self.file_list = []
        input_files = sorted(glob.glob(os.path.join(self.input_dir, "*.wav")))
        
        # Pre-calculate chunks to allow random access
        self.chunks = []

        print(f"Scanning dataset at {data_root}...")
        for in_path in input_files:
            filename = os.path.basename(in_path)
            tgt_path = os.path.join(self.target_dir, filename)
            
            if os.path.exists(tgt_path):
                # We store file paths and load on demand, or pre-load if small enough.
                # For simplicity and speed with small datasets, let's pre-load and slice.
                # For massive datasets, you'd want to lazy-load.
                self._process_file(in_path, tgt_path)
            else:
                print(f"Warning: Missing target for {filename}")

        print(f"Dataset loaded: {len(self.chunks)} chunks.")

    def _process_file(self, in_path, tgt_path):
        # Load audio
        # Using librosa for convenience, returns float32 
        x, _ = librosa.load(in_path, sr=self.sample_rate, mono=True)
        y, _ = librosa.load(tgt_path, sr=self.sample_rate, mono=True)
        
        # Ensure lengths match
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        
        # Create chunks
        stride = int(self.chunk_size * (1 - 0.0)) # 0 overlap for now implementation correctness
        
        # If overlap is needed later: stride = int(self.chunk_size * (1 - self.overlap))
        
        num_chunks = (min_len - self.chunk_size) // stride + 1
        
        for i in range(num_chunks):
            start = i * stride
            end = start + self.chunk_size
            self.chunks.append((x[start:end], y[start:end]))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        x, y = self.chunks[idx]
        
        # Convert to Tensor and add Channel dimension
        # Shape: (1, Length)
        x_tensor = torch.from_numpy(x).float().unsqueeze(0)
        y_tensor = torch.from_numpy(y).float().unsqueeze(0)
        
        return x_tensor, y_tensor
