import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
import os
import argparse
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.nn.architecture import SimpleTCN
from src.nn.dataset import AudioEffectDataset

def train(args):
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Data
    print("Loading dataset...")
    try:
        full_dataset = AudioEffectDataset(
            data_root=args.data_root, 
            sample_rate=args.sample_rate, 
            chunk_size=args.chunk_size
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create the dataset folder structure (data/datasets/train/input and data/datasets/train/target).")
        return

    # Split Train/Val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Model
    model = SimpleTCN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Model initialized. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # 4. Loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} "
              f"({time.time() - start_time:.2f}s)")
        
        # Save Checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"tcn_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final Save
    final_path = os.path.join(args.checkpoint_dir, "tcn_final.pt")
    torch.save(model.state_dict(), final_path)
    print("Training Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCN for Audio Effects")
    parser.add_argument("--data_root", type=str, default="data/datasets", help="Path to inputs/targets dataset")
    parser.add_argument("--chunk_size", type=int, default=16384, help="Audio Chunk Size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Sample Rate")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints", help="Where to save models")
    parser.add_argument("--save_interval", type=int, default=10, help="Save model every N epochs")

    args = parser.parse_args()
    
    # Ensure checkpoint dir exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Relative path fix for VSCode execution from root
    if not os.path.isabs(args.data_root):
        # Assuming run from project root
        pass 

    train(args)
