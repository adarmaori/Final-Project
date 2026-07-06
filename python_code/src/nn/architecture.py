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
    """
    A true deep Temporal Convolutional Network.
    Uses exponentially increasing dilations to capture complex time-series data.
    """
    def __init__(self, input_channels=1, output_channels=1, hidden_channels=32, kernel_size=15, num_layers=8):
        super(SimpleTCN, self).__init__()
        
        layers = []
        in_c = input_channels
        
        for i in range(num_layers):
            # Exponentially increasing dilation: 1, 2, 4, 8, 16...
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            
            layers.append(nn.Conv1d(in_c, hidden_channels, kernel_size=kernel_size, dilation=dilation, padding=padding))
            layers.append(Chomp1d(padding))
            layers.append(nn.Tanh()) # Tanh is great for learning distortion curves
            
            in_c = hidden_channels
            
        self.tcn = nn.Sequential(*layers)
        
        # Final linear mapping back to 1 audio channel
        self.final_conv = nn.Conv1d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.tcn(x)
        return self.final_conv(x)


class BrevitasQuantizedSimpleTCN(nn.Module):
    """
    A deep Quantized Temporal Convolutional Network.
    Uses Brevitas to simulate 8-bit or 4-bit integer quantization during training.
    """
    def __init__(self, weight_bits=8, act_bits=8, input_channels=1, output_channels=1, 
                 hidden_channels=32, kernel_size=15, num_layers=8):
        super(BrevitasQuantizedSimpleTCN, self).__init__()
        
        # Quantize the incoming floating-point audio signal
        self.quant_input = qnn.QuantIdentity(bit_width=act_bits, return_quant_tensor=True)
        
        layers = []
        in_c = input_channels
        
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            
            # Quantized Convolution
            layers.append(qnn.QuantConv1d(
                in_c, hidden_channels, 
                kernel_size=kernel_size, 
                dilation=dilation, 
                padding=padding,
                weight_bit_width=weight_bits,
                bias=True,
                return_quant_tensor=True
            ))
            
            layers.append(Chomp1d(padding))
            
            # Non-linearity followed by Activation Quantization
            layers.append(nn.Tanh())
            layers.append(qnn.QuantIdentity(bit_width=act_bits, return_quant_tensor=True))
            
            in_c = hidden_channels
            
        self.tcn = nn.Sequential(*layers)
        
        # Final projection back to 1 audio channel (outputs standard float tensor)
        self.final_conv = qnn.QuantConv1d(
            hidden_channels, output_channels, 
            kernel_size=1,
            weight_bit_width=weight_bits,
            bias=True,
            return_quant_tensor=False
        )

    def forward(self, x):
        x = self.quant_input(x)
        x = self.tcn(x)
        return self.final_conv(x)
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

class STFTUNet(nn.Module):
    """
    STFT-Domain 2D U-Net designed for learning time-invariant effects with long tails (Reverb).
    Converts 1D audio to a 2-channel (Real/Imaginary) Spectrogram, processes it via 2D Convolutions,
    and returns to the 1D time domain natively.
    """
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Register the window function as a buffer so it moves to GPU automatically
        self.register_buffer("window", torch.hann_window(win_length))

        # Encoders (Downsampling)
        # Input shape: (B, 2, F_bins, Time_frames)
        self.enc1 = self._conv_block(2, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)
        
        # --- NEW: DILATED TEMPORAL BOTTLENECK ---
        # Expands the receptive field in the Time axis (dim 1 of the 2D spatial plane)
        # padding=(1, d) ensures the tensor size remains perfectly unchanged.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 2), dilation=(1, 2)),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 4), dilation=(1, 4)),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 8), dilation=(1, 8)),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 16), dilation=(1, 16)),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        
        # Decoders (Upsampling + Skip Connections)
        self.dec3 = self._upconv_block(256, 128)
        self.dec2 = self._upconv_block(256, 64)   # Input = 128 (dec3) + 128 (enc3 skip)
        self.dec1 = self._upconv_block(128, 32)   # Input = 64 (dec2) + 64 (enc2 skip)
        
        self.final_deconv = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_conv = nn.Conv2d(32, 2, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )

    def _upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        
    def _match_shape(self, x, target):
        """Helper to crop padding differences to match shapes for U-Net skip connections."""
        diffY = x.size(2) - target.size(2)
        diffX = x.size(3) - target.size(3)
        
        if diffY > 0 or diffX > 0:
            x = x[:, :, :x.size(2)-max(0, diffY), :x.size(3)-max(0, diffX)]
            
        if diffY < 0 or diffX < 0:
            padY = -min(0, diffY)
            padX = -min(0, diffX)
            x = F.pad(x, (0, padX, 0, padY))
            
        return x

    def forward(self, x):
        B = x.size(0)
        T_orig = x.size(-1)
        
        # Flatten input to 1D for STFT
        x_1d = x.view(B, -1)
        
        # Compute Complex STFT
        stft_out = torch.stft(
            x_1d, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=self.window, 
            return_complex=True, 
            pad_mode='reflect', 
            center=True
        )
        
        # Shape: (B, F_bins, Time_frames) -> With n_fft=1024, F_bins is 513
        # Remove the Nyquist bin so F becomes 512 (allowing clean maxpooling/striding)
        nyquist_bin = stft_out[:, -1:, :]
        stft_out = stft_out[:, :-1, :]
        
        # Stack Real and Imaginary parts as channels -> Shape: (B, 2, 512, Time_frames)
        spec = torch.stack([stft_out.real, stft_out.imag], dim=1)
        
        # Pad Time dimension to ensure it divides cleanly by 16 (for 4 strided downsamples)
        pad_t = (16 - (spec.size(-1) % 16)) % 16
        if pad_t > 0:
            spec = F.pad(spec, (0, pad_t, 0, 0))
            
        # --- Encoder ---
        e1 = self.enc1(spec)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # --- Bottleneck ---
        b = self.bottleneck(e4)
        
        # --- Decoder ---
        d3 = self.dec3(b)
        d3 = self._match_shape(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = self._match_shape(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = self._match_shape(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        
        out = self.final_deconv(d1)
        out = self._match_shape(out, spec)
        out = self.final_conv(out)
        
        # Remove time padding if applied
        if pad_t > 0:
            out = out[..., :-pad_t]
            
        # --- Reconstruct to Complex STFT ---
        out_complex = torch.complex(out[:, 0], out[:, 1])
        
        # Add back a zeroed Nyquist bin to restore F=513
        zero_nyquist = torch.zeros(B, 1, out_complex.size(-1), device=out_complex.device, dtype=out_complex.dtype)
        out_complex = torch.cat([out_complex, zero_nyquist], dim=1)
        
        # Inverse STFT to generate audio waveform
        y = torch.istft(
            out_complex, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.win_length, 
            window=self.window, 
            length=T_orig, 
            center=True
        )
        
        # Restore standard shape: (B, 1, T)
        return y.view(B, 1, -1)