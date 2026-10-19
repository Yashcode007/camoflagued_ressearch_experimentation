import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.unet import UNet, AttentionUNet
from src.data.cod10k import COD10KDataset
from src.training.trainer import train_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Camouflage Pre-training on COD10K")
    parser.add_argument("--model_type", type=str, default="unet", choices=["unet", "attention_unet"],
                        help="Architecture to pre-train: 'unet' (CU1) or 'attention_unet' (CUA1)")
    parser.add_argument("--data_root", type=str, default="./data/COD10K", help="Path to COD10K dataset root")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum pre-training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--image_size", type=int, default=256, help="Input image resolution")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel width")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save model weights")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory for logs and histories")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes (0 recommended on Windows)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    tag = "CU1" if args.model_type == "unet" else "CUA1"
    print(f"=== Stage 1 Pre-training: {tag} ({args.model_type}) ===")
    print(f"Data root: {args.data_root} | Device: {args.device} | Epochs: {args.epochs}")

    # Build COD10K datasets
    train_ds = COD10KDataset(data_root=args.data_root, split="Train", image_size=args.image_size, augment=True)
    val_ds = COD10KDataset(data_root=args.data_root, split="Test", image_size=args.image_size, augment=False)

    print(f"Dataset loaded: {len(train_ds)} train samples, {len(val_ds)} validation samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Instantiate model
    if args.model_type == "unet":
        model = UNet(in_channels=3, out_channels=1, base_channels=args.base_channels)
    else:
        model = AttentionUNet(in_channels=3, out_channels=1, base_channels=args.base_channels)

    ckpt_path = os.path.join(args.checkpoint_dir, f"{tag}_best.pth")
    history_csv = os.path.join(args.output_dir, f"history_{tag}.csv")

    summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        ckpt_path=ckpt_path,
        history_csv=history_csv,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        tag=tag,
        use_amp=True
    )
    print("Stage 1 completed successfully:", summary)

if __name__ == "__main__":
    main()
