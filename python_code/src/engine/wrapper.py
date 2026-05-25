import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.nn.architecture import SimpleTCN, BrevitasQuantizedSimpleTCN, AudioLSTM, FlangerCRNN

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


class RealtimeDSPWrapper:
    """Wraps a *stateful* DSP processor that exposes a ``.process(block)`` method.

    Unlike ``DSPWrapper`` (which calls a pure function), this wrapper holds a
    processor object whose internal state persists between ``process()`` calls,
    making it suitable for block-by-block real-time simulation.
    """

    def __init__(self, processor_instance):
        """
        Args:
            processor_instance: An object with a ``process(audio_block)`` method
                                (e.g. ``RealtimeTubeSaturator``).
        """
        self.processor = processor_instance
        self.name = "RealtimeDSP"

    def process(self, audio_buffer):
        return self.processor.process(audio_buffer)

    def reset(self):
        """Reset internal state between unrelated audio streams."""
        if hasattr(self.processor, 'reset'):
            self.processor.reset()

class NNWrapper:
    def __init__(self, model_path=None, model_class=None, model_type='tcn', device='cpu', quant_bits=0):
        """
        Wraps a PyTorch Neural Network.
        """
        self.device = torch.device(device)
        self.model_type = model_type
        self.quant_bits = quant_bits
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {os.path.basename(model_path)}...")
            state_dict = torch.load(model_path, map_location=self.device)
            
            if model_type == 'lstm':
                hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
                num_layers = 1
                while f'lstm.weight_ih_l{num_layers}' in state_dict:
                    num_layers += 1
                self.model = AudioLSTM(hidden_size=hidden_size, num_layers=num_layers)
            elif model_type == 'crnn':
                # Simplified loading for FlangerCRNN
                self.model = FlangerCRNN()
            elif model_type == 'tcn' and quant_bits > 0:
                self.model = BrevitasQuantizedSimpleTCN(quant_bits=quant_bits)
            else:
                self.model = SimpleTCN()

            strict_loading = True
            load_result = self.model.load_state_dict(state_dict, strict=strict_loading)
            if load_result.missing_keys or load_result.unexpected_keys:
                print(f"Quantized load summary: missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}")
        else:
            if model_class is None:
                if model_type == 'lstm':
                    model_class = AudioLSTM
                elif model_type == 'crnn':
                    model_class = FlangerCRNN
                elif model_type == 'tcn' and quant_bits > 0:
                    model_class = lambda: BrevitasQuantizedSimpleTCN(quant_bits=quant_bits)
                else:
                    model_class = SimpleTCN
            self.model = model_class()
            if model_path:
                print(f"Warning: Model path {model_path} not found. Using random weights.")
        
        self.model.to(self.device)
        self.model.eval()
        self.name = "NeuralNetwork"

    def calibrate(self, audio_buffer):
        """Kept for compatibility; Brevitas models do not need a separate calibration pass here."""
        return

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
            if isinstance(y_tensor, tuple):
                y_tensor = y_tensor[0]
        
        # Output: (Length,)
        return y_tensor.squeeze().cpu().numpy()
