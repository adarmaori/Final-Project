import torch
import torch.nn as nn

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
