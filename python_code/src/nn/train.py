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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.nn.architecture import SimpleTCN, AudioLSTM, FlangerCRNN
from src.nn.dataset import AudioEffectDataset


def _masked_loss_region(tensor, warmup_fraction):
    """Return the slice used for loss computation after the warm-up region."""
    start_idx = int(tensor.shape[-1] * warmup_fraction)
    return tensor[..., start_idx:]


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
            for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
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
                spectral_convergence = torch.norm(target_mag - pred_mag, p="fro") / (torch.norm(target_mag, p="fro") + 1e-8)
                log_mag = torch.mean(torch.abs(torch.log1p(target_mag) - torch.log1p(pred_mag)))
                total_loss = total_loss + spectral_convergence + log_mag

            return total_loss / len(self.fft_sizes)

    return _FallbackMRSTFTLoss()


def _flanger_training_loss(pred, target, mrstft_loss, warmup_fraction):
    """Combined waveform L1 + MR-STFT loss with a masked warm-up region."""
    pred = pred.squeeze(1)
    target = target.squeeze(1)

    pred_tail = _masked_loss_region(pred, warmup_fraction)
    target_tail = _masked_loss_region(target, warmup_fraction)

    if pred_tail.shape[-1] == 0:
        pred_tail = pred
        target_tail = target

    l1_loss = torch.mean(torch.abs(pred_tail - target_tail))
    stft_loss = mrstft_loss(pred_tail.unsqueeze(1), target_tail.unsqueeze(1))

    return l1_loss + stft_loss


def _build_model(args):
    if args.model_type == "lstm":
        return AudioLSTM(
            input_channels=1,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.lstm_num_layers,
            dropout=args.lstm_dropout,
            output_channels=1,
        )
    elif args.model_type == "crnn":
        return FlangerCRNN()
    return SimpleTCN()

def train(args):
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # 2. Data
    print("Loading dataset...")
    try:
        full_dataset = AudioEffectDataset(
            data_root=args.data_root, 
            sample_rate=args.sample_rate, 
            chunk_size=args.chunk_size,
            input_subdir=args.input_subdir,
            target_subdir=args.target_subdir,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please create the dataset folder structure ({args.data_root}/{args.input_subdir} and {args.data_root}/{args.target_subdir}).")
        return

    if len(full_dataset) == 0:
        print("Error: Dataset has zero chunks. Check chunk_size and source files.")
        return

    # Split Train/Val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Model
    model = _build_model(args).to(device)
    mrstft_loss = _build_mrstft_loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    
    print(
        f"Model '{args.model_type}' initialized. "
        f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}"
    )
    
    # 4. Loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = _flanger_training_loss(
                outputs,
                targets,
                mrstft_loss=mrstft_loss,
                warmup_fraction=args.warmup_fraction,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = _flanger_training_loss(
                    outputs,
                    targets,
                    mrstft_loss=mrstft_loss,
                    warmup_fraction=args.warmup_fraction,
                )
                val_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} "
              f"({time.time() - start_time:.2f}s)")
        
        # Save Checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model_type}_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final Save
    final_path = os.path.join(args.checkpoint_dir, f"{args.model_type}_final.pt")
    torch.save(model.state_dict(), final_path)
    print("Training Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCN for Audio Effects")
    parser.add_argument("--data_root", type=str, default="data/datasets", help="Path to dataset root")
    parser.add_argument("--input_subdir", type=str, default="inputs", help="Input folder under data_root")
    parser.add_argument("--target_subdir", type=str, default="targets", help="Target folder under data_root")
    parser.add_argument("--chunk_size", type=int, default=16384, help="Audio Chunk Size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Sample Rate")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints", help="Where to save models")
    parser.add_argument("--save_interval", type=int, default=10, help="Save model every N epochs")
    parser.add_argument("--model_type", type=str, choices=["tcn", "lstm", "crnn"], default="crnn", help="Model architecture")
    parser.add_argument("--lstm_hidden_size", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--lstm_num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--lstm_dropout", type=float, default=0.1, help="LSTM dropout (if num_layers > 1)")
    parser.add_argument("--warmup_fraction", type=float, default=0.10, help="Fraction of each chunk ignored at the start of the loss")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm")

    args = parser.parse_args()
    
    # Ensure checkpoint dir exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Relative path fix for VSCode execution from root
    if not os.path.isabs(args.data_root):
        # Assuming run from project root
        pass 

    train(args)
