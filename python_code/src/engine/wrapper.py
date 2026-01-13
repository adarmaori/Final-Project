import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.nn.architecture import SimpleTCN

class DSPWrapper:
    def __init__(self, processor_func, **kwargs):
        """
        Wraps a deterministic DSP function.
        """
        self.processor = processor_func
        self.kwargs = kwargs
        self.name = "DSP"

    def process(self, audio_buffer):
        return self.processor(audio_buffer, **self.kwargs)

class NNWrapper:
    def __init__(self, model_path=None, model_class=SimpleTCN, device='cpu'):
        """
        Wraps a PyTorch Neural Network.
        """
        self.device = torch.device(device)
        self.model = model_class()
        
        if model_path:
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}...")
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                print(f"Warning: Model path {model_path} not found. Using random weights.")
        
        self.model.to(self.device)
        self.model.eval()
        self.name = "NeuralNetwork"

    def process(self, audio_buffer):
        """
        Process a buffer of audio. 
        Note: This naive implementation assumes the buffer is the whole context.
        For real-time streaming, a ring buffer is needed for TCNs.
        """
        # Prepare input: (1, 1, Length)
        x_tensor = torch.from_numpy(audio_buffer).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            y_tensor = self.model(x_tensor)
        
        # Output: (Length,)
        return y_tensor.squeeze().cpu().numpy()
