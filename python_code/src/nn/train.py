import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
import os
import argparse
import time

try:
    import auraloss
except ImportError:
    auraloss = None

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.nn.architecture import (
    SimpleTCN,
    BrevitasQuantizedSimpleTCN,
    AudioLSTM,
    FlangerCRNN,
    STFTUNet,
)
from src.nn.dataset import AudioEffectDataset


def _infer_effect_name(target_subdir):
    normalized = (target_subdir or "").lower()
    mapping = {
        "targets_flange": "flange",
        "targets_flanger": "flange",
        "targets_distortion": "distortion",
        "targets_exciter": "exciter",
        "targets_wah": "wah",
        "targets_reverb": "reverb",
    }
    for key, value in mapping.items():
        if key in normalized:
            return value
    return "reverb"


def _checkpoint_stem(args):
    effect_name = args.effect or _infer_effect_name(args.target_subdir)
    if args.model_type == "tcn" and args.quant_bits > 0:
        return f"tcn_{effect_name}_q{args.quant_bits}"
    return f"{args.model_type}_{effect_name}"


def _build_mrstft_loss():
    """Build a reusable multi-resolution STFT loss instance."""
    if auraloss is not None:
        return auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048],
            hop_sizes=[128, 256, 512],
            win_lengths=[512, 1024, 2048],
            w_sc=1.0,
            w_log_mag=1.0,
        )

    class _FallbackMRSTFTLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.fft_sizes = [512, 1024, 2048]
            self.hop_sizes = [128, 256, 512]
            self.win_lengths = [512, 1024, 2048]

        def forward(self, pred, target):
            total_loss = 0.0
            for fft_size, hop_size, win_length in zip(
                self.fft_sizes, self.hop_sizes, self.win_lengths
            ):
                window = torch.hann_window(win_length, device=pred.device)
                pred_spec = torch.stft(
                    pred.squeeze(1),
                    n_fft=fft_size,
                    hop_length=hop_size,
                    win_length=win_length,
                    window=window,
                    center=True,
                    return_complex=True,
                )
                target_spec = torch.stft(
                    target.squeeze(1),
                    n_fft=fft_size,
                    hop_length=hop_size,
                    win_length=win_length,
                    window=window,
                    center=True,
                    return_complex=True,
                )

                pred_mag = torch.abs(pred_spec)
                target_mag = torch.abs(target_spec)

                # Spectral Convergence
                spectral_convergence = torch.norm(target_mag - pred_mag, p="fro") / (
                    torch.norm(target_mag, p="fro") + 1e-8
                )

                # FIX: True Log-Magnitude scaling for low-energy reverb tails
                eps = 1e-7
                log_mag = torch.mean(
                    torch.abs(torch.log(target_mag + eps) - torch.log(pred_mag + eps))
                )

                total_loss = total_loss + spectral_convergence + log_mag

            return total_loss / len(self.fft_sizes)

    return _FallbackMRSTFTLoss()


def _audio_training_loss(pred, target, mrstft_loss, alpha=10.0, diff_alpha=50.0, loss_type="all"):
    """
    Tri-Band Audio Loss.
    1. STFT: Learns the general frequency balance.
    2. L1 (Macro): Locks the phase of the low and mid frequencies.
    3. L1 Derivative (Micro): Forces the network to reconstruct high-frequency
       transients and fast-moving harmonic slopes.
    """
    pred = pred.squeeze(1)
    target = target.squeeze(1)

    # 1. Spectral Loss
    stft_loss = mrstft_loss(pred.unsqueeze(1), target.unsqueeze(1))

    # 2. Standard Time-Domain Loss (Lowered alpha so it doesn't overpower)
    l1_loss = torch.nn.functional.l1_loss(pred, target)

    # 3. Derivative / Pre-Emphasis Loss
    # torch.diff calculates the difference between consecutive samples (the slope)
    pred_diff = torch.diff(pred, dim=-1)
    target_diff = torch.diff(target, dim=-1)
    diff_loss = torch.nn.functional.l1_loss(pred_diff, target_diff)

    # Combine all three
    if loss_type == "mrstft_only":
        return stft_loss
    elif loss_type == "l1_only":
        return l1_loss
    elif loss_type == "l1_diff_only":
        return diff_loss
    else:
        return stft_loss + (alpha * l1_loss) + (diff_alpha * diff_loss)


def _build_model(args):
    if args.model_type == "unet":
        return STFTUNet()
    elif args.model_type == "lstm":
        return AudioLSTM(
            input_channels=1,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.lstm_num_layers,
            dropout=args.lstm_dropout,
            output_channels=1,
        )
    elif args.model_type == "crnn":
        return FlangerCRNN()
    elif args.model_type == "tcn":
        if args.quant_bits > 0:
            return BrevitasQuantizedSimpleTCN(
                weight_bits=args.quant_bits,
                act_bits=args.quant_bits,
                hidden_channels=args.tcn_hidden_channels,  # <-- Pass capacity args here
                num_layers=args.tcn_num_layers,
                kernel_size=args.tcn_kernel_size,
            )
        else:
            return SimpleTCN(
                hidden_channels=args.tcn_hidden_channels,
                num_layers=args.tcn_num_layers,
                kernel_size=args.tcn_kernel_size,
            )
    return SimpleTCN(hidden_channels=args.tcn_hidden_channels)


def _build_checkpoint_tag(args):
    if args.model_type == "tcn":
        base_name = f"{args.effect}_tcn"
        if args.quant_bits > 0:
            base_name = f"{base_name}_q{args.quant_bits}"
        return base_name
    return args.model_type


def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.model_type == "tcn" and args.quant_bits > 0:
        # Brevitas calls tensor.rename() (named tensors), which MPS does not implement.
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
    print(f"Using device: {device}")

    print("Loading dataset...")
    try:
        full_dataset = AudioEffectDataset(
            data_root=args.data_root,
            sample_rate=args.sample_rate,
            chunk_size=args.chunk_size,
            context_size=args.context_size,  # Pass context size to dataset
            input_subdir=args.input_subdir,
            target_subdir=args.target_subdir,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if len(full_dataset) == 0:
        print("Error: Dataset has zero chunks.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = _build_model(args).to(device)
    mrstft_loss = _build_mrstft_loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    print(
        f"Model '{args.model_type}' initialized. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}"
    )
    print(f"Chunk size: {args.chunk_size} | Context Lookback: {args.context_size}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if args.context_size > 0:
                outputs = outputs[..., args.context_size :]

            loss = _audio_training_loss(outputs, targets, mrstft_loss, loss_type=args.loss_type)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # CROP CONTEXT
                if args.context_size > 0:
                    outputs = outputs[..., args.context_size :]

                loss = _audio_training_loss(outputs, targets, mrstft_loss, loss_type=args.loss_type)
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        scheduler.step(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} ({time.time() - start_time:.2f}s)"
        )

        if (epoch + 1) % args.save_interval == 0:
            model_tag = _build_checkpoint_tag(args)
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"{model_tag}_epoch_{epoch + 1}.pt"
            )
            torch.save(model.state_dict(), ckpt_path)

    model_tag = _build_checkpoint_tag(args)
    final_path = os.path.join(args.checkpoint_dir, f"{model_tag}_final.pt")
    torch.save(model.state_dict(), final_path)
    print("Training Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/datasets")
    parser.add_argument("--input_subdir", type=str, default="inputs")
    parser.add_argument("--target_subdir", type=str, default="targets_reverb")
    parser.add_argument(
        "--chunk_size", type=int, default=65536, help="Audio Chunk Size"
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=44100,
        help="Samples to look backward in time for context",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument(
        "--effect",
        type=str,
        choices=["flange", "distortion", "wah", "reverb", "exciter"],
        default=None,
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["tcn", "lstm", "crnn", "unet"],
        default="unet",
    )
    parser.add_argument("--loss_type", type=str, choices=["all", "mrstft_only", "l1_only", "l1_diff_only"], default="all")
    parser.add_argument("--quant_bits", type=int, default=0)
    parser.add_argument("--lstm_hidden_size", type=int, default=64)
    parser.add_argument("--lstm_num_layers", type=int, default=2)
    parser.add_argument("--lstm_dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    # TCN Capacity Arguments
    parser.add_argument(
        "--tcn_hidden_channels",
        type=int,
        default=32,
        help="Number of filters per TCN layer",
    )
    parser.add_argument(
        "--tcn_num_layers",
        type=int,
        default=8,
        help="Number of dilated convolutional layers",
    )
    parser.add_argument(
        "--tcn_kernel_size",
        type=int,
        default=15,
        help="Size of the convolutional kernel",
    )

    args = parser.parse_args()
    print(args)
    if args.effect is None:
        args.effect = _infer_effect_name(args.target_subdir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args)

