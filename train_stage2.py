import os
import csv
import argparse
import random
import json
import torch
from torch.utils.data import DataLoader

from src.models.registry import build_model, MODEL_SPECS
from src.data.medical import build_kvasir_split
from src.training.trainer import train_model

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def is_run_complete(ckpt_file, hist_file, max_epochs, patience):
    """
    Checks if a training run has successfully and completely finished.
    A run is complete if both checkpoint and history exist, and EITHER:
      1. total_epochs >= max_epochs, OR
      2. early stopping was triggered (patience consecutive epochs without improvement).
    """
    if not (os.path.exists(ckpt_file) and os.path.exists(hist_file)):
        return False, None
    try:
        with open(hist_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        if not reader:
            return False, None
        total_epochs = len(reader)
        dices = [float(r["val_dice"]) for r in reader]
        best_dice = -1.0
        best_epoch = -1
        for idx, d in enumerate(dices):
            if d > best_dice:
                best_dice = d
                best_epoch = idx + 1
        
        epochs_without_improvement = total_epochs - best_epoch
        is_finished = (total_epochs >= max_epochs) or (epochs_without_improvement >= patience)
        if is_finished:
            best_row = reader[best_epoch - 1]
            return True, {
                "best_val_dice": best_dice,
                "best_val_iou": float(best_row["val_iou"]),
                "best_epoch": best_epoch,
                "epochs_run": total_epochs,
            }
        return False, {
            "epochs_run": total_epochs,
            "best_epoch": best_epoch,
            "best_val_dice": best_dice,
            "epochs_without_improvement": epochs_without_improvement,
        }
    except Exception:
        return False, None

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Medical Segmentation with Multi-Seed Training")
    parser.add_argument("--models", nargs="+", default=["M1", "G1", "G1R", "M1W", "G1_ImageNet", "UNetPlusPlus", "G2", "G4", "G1_early", "G1_deep"],
                        help="List of model keys to train")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026],
                        help="List of random seeds for multi-seed training")
    parser.add_argument("--data_root", type=str, default="./data/Kvasir-SEG", help="Path to Kvasir-SEG root")
    parser.add_argument("--cu1_path", type=str, default="./checkpoints/CU1_best.pth", help="CU1 teacher checkpoint path")
    parser.add_argument("--cua1_path", type=str, default="./checkpoints/CUA1_best.pth", help="CUA1 teacher checkpoint path")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs per run")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--image_size", type=int, default=256, help="Image resolution")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/stage2", help="Checkpoint output folder")
    parser.add_argument("--output_dir", type=str, default="./outputs/stage2", help="History logs output folder")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 recommended on Windows)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", default=True, help="Automatically skip completed runs and re-run incomplete ones")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    summary_path = os.path.join(args.output_dir, "stage2_training_summary.json")
    all_summaries = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                all_summaries = json.load(f)
        except Exception:
            all_summaries = []

    teacher_paths = {
        "CU1": args.cu1_path,
        "CUA1": args.cua1_path
    }

    print(f"=== Stage 2 Training Matrix ===")
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {args.device} | Epochs: {args.epochs} | Resume mode: {args.resume}")

    for seed in args.seeds:
        set_seed(seed)
        train_loader = None
        val_loader = None

        for model_key in args.models:
            run_tag = f"{model_key}_seed{seed}"
            ckpt_file = os.path.join(args.checkpoint_dir, f"{run_tag}_best.pth")
            hist_file = os.path.join(args.output_dir, f"history_{run_tag}.csv")

            # Check if this model run already completed
            if args.resume:
                completed, info = is_run_complete(ckpt_file, hist_file, args.epochs, args.patience)
                if completed:
                    print("\n" + "=" * 70)
                    print(f"--> [SKIP] {run_tag} already completed! (Best Val Dice: {info['best_val_dice']:.4f} at Epoch {info['best_epoch']}/{info['epochs_run']} epochs)")
                    print("=" * 70)
                    # Ensure summary entry exists
                    if not any(s.get("tag") == run_tag for s in all_summaries):
                        info["tag"] = run_tag
                        info["model_key"] = model_key
                        info["seed"] = seed
                        all_summaries.append(info)
                        with open(summary_path, "w", encoding="utf-8") as f:
                            json.dump(all_summaries, f, indent=2)
                    continue
                elif info is not None:
                    # Incomplete run detected (like interrupted training)
                    print("\n" + "=" * 70)
                    print(f"--> [INCOMPLETE RUN DETECTED] {run_tag} stopped at Epoch {info['epochs_run']} without reaching early-stopping criterion (Patience {args.patience}).")
                    print(f"    Cleaning up partial logs/checkpoints and re-training from scratch...")
                    print("=" * 70)
                    if os.path.exists(ckpt_file):
                        try: os.remove(ckpt_file)
                        except Exception: pass
                    if os.path.exists(hist_file):
                        try: os.remove(hist_file)
                        except Exception: pass
                    all_summaries = [s for s in all_summaries if s.get("tag") != run_tag]

            # Lazy data loading only when at least one model in this seed needs training
            if train_loader is None:
                train_ds, val_ds, _ = build_kvasir_split(args.data_root, seed=seed, image_size=args.image_size, augment=True)
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            print("\n" + "=" * 70)
            print(f"--> Training {run_tag} | Family: {MODEL_SPECS[model_key]['family']}")
            print("=" * 70)

            # Build model
            model = build_model(model_key, teacher_paths=teacher_paths, device=args.device)

            # Adaptive loader for UNetPlusPlus to avoid Windows PCIe VRAM swapping & OOM
            if model_key == "UNetPlusPlus":
                model_bs = max(1, args.batch_size // 4)
                accum_steps = 4
                model_train_loader = DataLoader(train_ds, batch_size=model_bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
                model_val_loader = DataLoader(val_ds, batch_size=model_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            else:
                accum_steps = 1
                model_train_loader = train_loader
                model_val_loader = val_loader

            summary = train_model(
                model=model,
                train_loader=model_train_loader,
                val_loader=model_val_loader,
                epochs=args.epochs,
                ckpt_path=ckpt_file,
                history_csv=hist_file,
                lr=args.lr,
                patience=args.patience,
                device=args.device,
                tag=run_tag,
                use_amp=True,
                accumulation_steps=accum_steps
            )
            summary["model_key"] = model_key
            summary["seed"] = seed

            del model, model_train_loader, model_val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            # Update summary and save incrementally
            all_summaries = [s for s in all_summaries if s.get("tag") != run_tag]
            all_summaries.append(summary)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(all_summaries, f, indent=2)

    print(f"\nAll Stage 2 runs complete! Full summary recorded to {summary_path}")

if __name__ == "__main__":
    main()

