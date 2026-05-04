import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            Chomp1d(padding),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class SimpleTCN(nn.Module):
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        hidden_channels=16,
        kernel_size=3,
        dilation=1,
        num_blocks=1,
    ):
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        blocks = []
        in_channels = input_channels
        for block_index in range(num_blocks):
            block_dilation = dilation ** (block_index + 1)
            blocks.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=block_dilation,
                )
            )
            in_channels = hidden_channels

        self.feature_extractor = nn.Sequential(*blocks)

        output_padding = (kernel_size - 1) * dilation
        self.output_conv = nn.Conv1d(
            hidden_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=output_padding,
        )
        self.output_chomp = Chomp1d(output_padding)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.feature_extractor(x)
        x = self.output_conv(x)
        x = self.output_chomp(x)
        return x


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1, num_layers=1):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
