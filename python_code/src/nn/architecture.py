import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

try:
    import brevitas.nn as qnn
except ImportError as exc:  # pragma: no cover - dependency issue should fail loudly
    qnn = None
    _BREVITAS_IMPORT_ERROR = exc
else:
    _BREVITAS_IMPORT_ERROR = None

class Chomp1d(nn.Module):
    """
    Removes the last elements of the sequence to ensure causality.
    Used after padding.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class SimpleTCN(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, kernel_size=3, dilation=1):
        super(SimpleTCN, self).__init__()
        
        # Causal Padding = (kernel_size - 1) * dilation
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.chomp1 = Chomp1d(padding)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(16, output_channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.chomp2 = Chomp1d(padding)

    def forward(self, x):
        # x shape: (batch_size, channels, length)
        # Ensure input is 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.chomp1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.chomp2(x)
        return x


class BrevitasQuantizedSimpleTCN(nn.Module):
    """Brevitas-based quantized TCN for 16/8/4-bit training and export.
    
    Args:
        input_channels: Input channel count (default 1).
        output_channels: Output channel count (default 1).
        kernel_size: Conv kernel size (default 3).
        dilation: Conv dilation (default 1).
        quant_bits: Weight quantization bit-width (default 16). 
                    When quant_bits=4, activations use 8-bit (mixed precision) for stability.
        hidden_channels: Number of hidden channels in the first/second convs (default 32).
    """

    def __init__(self, input_channels=1, output_channels=1, kernel_size=3, dilation=1, quant_bits=16, hidden_channels=32):
        super().__init__()
        if qnn is None:
            raise ImportError("brevitas is required for BrevitasQuantizedSimpleTCN") from _BREVITAS_IMPORT_ERROR

        if quant_bits < 2:
            raise ValueError("quant_bits must be >= 2")

        padding = (kernel_size - 1) * dilation
        self.quant_bits = quant_bits
        self.hidden_channels = hidden_channels
        
        # Mixed precision for 4-bit: keep activations at 8-bit for stability
        act_bit_width = 8 if quant_bits == 4 else quant_bits

        self.input_quant = qnn.QuantIdentity(bit_width=act_bit_width)
        self.residual = input_channels == output_channels
        self.conv1 = qnn.QuantConv1d(
            input_channels,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=True,
            weight_bit_width=quant_bits,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu = qnn.QuantReLU(bit_width=act_bit_width)
        self.conv2 = qnn.QuantConv1d(
            hidden_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=True,
            weight_bit_width=quant_bits,
        )
        self.chomp2 = Chomp1d(padding)
        self.output_quant = qnn.QuantIdentity(bit_width=act_bit_width)
        self.output_gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        residual = x if self.residual else None

        x = self.input_quant(x)
        x = self.conv1(x)
        x = self.chomp1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.chomp2(x)
        x = self.output_quant(x)

        if residual is not None:
            x = x + self.output_gain * residual

        return x

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, input_size) or (batch, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden


class AudioLSTM(nn.Module):
    """Sequence-to-sequence LSTM for audio effect learning.

    Accepts channel-first audio tensors and returns channel-first output to match
    the existing dataset/training pipeline.
    """

    def __init__(
        self,
        input_channels=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        output_channels=1,
        residual=True,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        padding = 4
        self.residual = residual and input_channels == output_channels
        self.input_proj = nn.Conv1d(input_channels, hidden_size, kernel_size=5, padding=padding)
        self.chomp = Chomp1d(padding)
        self.pre_norm = nn.GroupNorm(1, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.post_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, output_channels)

    def forward(self, x):
        # Supported input shapes:
        # (B, C, T), (B, T, C), or (B, T)
        residual_input = None
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, T, 1)
        elif x.dim() == 3:
            # Convert channel-first (B, C, T) -> (B, T, C)
            if x.size(1) <= 4 and x.size(2) > x.size(1):
                x = x.transpose(1, 2)

        residual_input = x.transpose(1, 2)
        x = self.input_proj(residual_input)
        x = self.chomp(x)
        x = self.pre_norm(x)
        x = x.transpose(1, 2)

        y, _ = self.lstm(x)
        y = self.post_norm(y)
        y = self.dropout(y)
        y = self.proj(y)           # (B, T, output_channels)
        y = y.transpose(1, 2)      # (B, output_channels, T)

        if self.residual:
            y = y + residual_input
        return y

class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_channels, out_channels * 2, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp = Chomp1d(padding)
        self.proj = weight_norm(nn.Conv1d(in_channels, out_channels, 1)) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.proj is None else self.proj(x)
        out = self.conv(x)
        out = self.chomp(out)
        
        # WaveNet-style Gated Activation (GLU)
        gate, filter_act = torch.chunk(out, 2, dim=1)
        out = torch.tanh(filter_act) * torch.sigmoid(gate)
        
        return out + residual, out

class FlangerCRNN(nn.Module):
    """
    Dilated CRNN architecture specifically designed for effects with large delay lines (like flangers).
    Uses a TCN front-end with GLUs and skip connections to maintain high-frequency fidelity,
    while using an LSTM back-end to track the slow LFO sweep.
    """
    def __init__(
        self, 
        input_channels=1, 
        tcn_channels=32, 
        lstm_hidden=64, 
        output_channels=1,
        dilations=[1, 2, 4, 8, 16, 32, 64, 128]
    ):
        super().__init__()
        self.input_conv = nn.Conv1d(input_channels, tcn_channels, 1)
        
        self.tcn_blocks = nn.ModuleList()
        for d in dilations:
            self.tcn_blocks.append(
                TCNResidualBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=d)
            )
            
        self.skip_proj = nn.Conv1d(tcn_channels * len(dilations), tcn_channels, 1)
        
        # GRU acts as the LFO phase tracker (lighter, faster than LSTM)
        self.lstm = nn.GRU(input_size=tcn_channels, hidden_size=lstm_hidden, batch_first=True)
        
        # Output projection combines the pristine skip connections with the LSTM's control signal
        self.out_proj = nn.Linear(tcn_channels + lstm_hidden, output_channels)
        # Initialize final projection to zero so network starts as identity (residual pass-through)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        # Supported input shapes: (B, C, T) or (B, T, C) or (B, T)
        residual_input = None
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)
        elif x.dim() == 3:
            if x.size(1) > 4 and x.size(2) <= 4: # (B, T, C)
                x = x.transpose(1, 2)
        
        residual_input = x
        
        # Front-end projection
        tcn_out = self.input_conv(x)
        
        skip_connections = []
        for block in self.tcn_blocks:
            tcn_out, skip = block(tcn_out)
            skip_connections.append(skip)
            
        # Combine skip connections
        skips_cat = torch.cat(skip_connections, dim=1) # (B, tcn_C * num_blocks, T)
        skips_proj = self.skip_proj(skips_cat)       # (B, tcn_C, T)
        
        # LSTM takes the combined TCN features
        lstm_in = skips_proj.transpose(1, 2)         # (B, T, tcn_C)
        lstm_out, _ = self.lstm(lstm_in)             # (B, T, lstm_hidden)
        
        # Concat the clean TCN audio pipeline with the LSTM control signal
        out = torch.cat([lstm_in, lstm_out], dim=-1) # (B, T, tcn_C + lstm_hidden)
        out = self.out_proj(out)                     # (B, T, output_channels)
        out = out.transpose(1, 2)                    # (B, output_channels, T)
        
        return out + residual_input # Adding raw residual like AudioLSTM
