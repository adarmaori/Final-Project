import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import STFTUNet along with the existing architectures
from src.nn.architecture import SimpleTCN, BrevitasQuantizedSimpleTCN, AudioLSTM, FlangerCRNN, STFTUNet


def _infer_tcn_hidden_channels(state_dict):
    """
    Infers the number of hidden channels from the saved PyTorch state dictionary.
    Supports both legacy TCN architectures (conv1) and new deep sequential TCNs (tcn.0).
    """
    # Check for old architecture naming
    conv1_weight = state_dict.get('conv1.weight')
    if conv1_weight is not None:
        return int(conv1_weight.shape[0])
    
    # Check for new deep architecture naming
    tcn_0_weight = state_dict.get('tcn.0.weight')
    if tcn_0_weight is not None:
        return int(tcn_0_weight.shape[0])
        
    # Default fallback
    return 16

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


class NNWrapper:
    def __init__(self, model_path=None, model_type='tcn', quant_bits=0):
        """
        Wraps a neural network model to handle PyTorch tensor execution.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # Instantiate correct model architecture based on string identifier
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            hidden_channels = _infer_tcn_hidden_channels(state_dict) if model_type == 'tcn' else None

            # --- Smart Checkpoint Detection ---
            if model_type == 'tcn' and quant_bits > 0:
                # If there are no Brevitas 'quant' keys in the saved weights, it's a float model
                is_quant_checkpoint = any("quant" in k for k in state_dict.keys())
                if not is_quant_checkpoint:
                    print(f"Warning: {model_path} is a floating-point checkpoint. Auto-switching to Float TCN.")
                    quant_bits = 0  # Force fallback to SimpleTCN
            # ----------------------------------

            if model_type == 'lstm':
                model_class = AudioLSTM
            if model_type == 'lstm':
                model_class = AudioLSTM
            elif model_type == 'crnn':
                model_class = FlangerCRNN
            elif model_type == 'unet':
                model_class = STFTUNet
            elif model_type == 'tcn' and quant_bits > 0:
                model_class = lambda: BrevitasQuantizedSimpleTCN(weight_bits=quant_bits, act_bits=quant_bits, hidden_channels=hidden_channels or 16)
            else:
                model_class = lambda: SimpleTCN(hidden_channels=hidden_channels or 16)
                
            self.model = model_class()
            
            # Load trained weights matching the architecture
            self.model.load_state_dict(state_dict)
            print(f"Loaded {model_type} model from {model_path} onto {self.device}")
        else:
            if model_type == 'lstm':
                model_class = AudioLSTM
            elif model_type == 'crnn':
                model_class = FlangerCRNN
            elif model_type == 'unet':
                model_class = STFTUNet
            elif model_type == 'tcn' and quant_bits > 0:
                model_class = lambda: BrevitasQuantizedSimpleTCN(quant_bits=quant_bits, hidden_channels=32)
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
        """
        # Save original length to trim off any real-time padding later
        orig_len = len(audio_buffer)
        
        # For U-Net real-time block simulation: if the block is smaller than 2048 samples,
        # pad it with zeros so torch.stft doesn't crash on reflection padding sizes.
        if self.model_type == 'unet' and orig_len < 2048:
            # Pad up to 2048 samples
            pad_amount = 2048 - orig_len
            audio_buffer = np.pad(audio_buffer, (0, pad_amount), mode='constant')

        # Prepare input tensor shape.
        x_tensor = torch.from_numpy(audio_buffer).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Strip additional dimensions for models like U-Net that expect (B, T)
        if self.model_type == 'unet':
            x_tensor = x_tensor.squeeze(1) 

        with torch.no_grad():
            y_tensor = self.model(x_tensor)
            if isinstance(y_tensor, tuple):
                y_tensor = y_tensor[0]
                
        # Send tensor back to CPU and convert to a flat 1D numpy array
        out_array = y_tensor.cpu().numpy().squeeze()
        
        # Trim off the trailing zero-padded values if we padded earlier
        if self.model_type == 'unet' and orig_len < 2048:
            out_array = out_array[:orig_len]
            
        return out_array