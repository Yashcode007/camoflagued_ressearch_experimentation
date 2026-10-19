import os
import time
import csv
import copy
import torch
from tqdm.auto import tqdm
from .losses import BCEDiceLoss
from .metrics import compute_binary_metrics

def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    ckpt_path,
    history_csv,
    lr=1e-4,
    patience=10,
    device="cuda",
    tag="model",
    use_amp=True,
    allow_overwrite=True,
    accumulation_steps=1
):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_csv), exist_ok=True)

    model.to(device)
    # Only train parameters that require grad (freezes teacher in guided models)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = BCEDiceLoss()

    amp_enabled = use_amp and (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    best_val_dice = -1.0
    best_epoch = -1
    best_val_iou = -1.0
    patience_counter = 0

    history = []
    start_time = time.time()

    print(f"\n[{tag}] Training started for {epochs} epochs (Patience: {patience})")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        n_train_batches = len(train_loader)
        optimizer.zero_grad(set_to_none=True)

        for step_idx, (imgs, masks) in enumerate(tqdm(train_loader, desc=f"[{tag}] Epoch {epoch}/{epochs} [Train]", leave=False)):
            imgs = imgs.to(device)
            masks = masks.to(device)

            with torch.amp.autocast('cuda', enabled=amp_enabled):
                logits = model(imgs)
                loss = criterion(logits, masks)
                loss_scaled = loss / accumulation_steps

            scaler.scale(loss_scaled).backward()
            if (step_idx + 1) % accumulation_steps == 0 or (step_idx + 1) == n_train_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()

        train_loss /= max(1, n_train_batches)
        scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_dices, val_ious = [], []
        n_val_batches = len(val_loader)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    logits = model(imgs)
                    loss = criterion(logits, masks)

                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                gts = (masks > 0.5).cpu().numpy()

                for b in range(len(preds)):
                    m = compute_binary_metrics(preds[b, 0], gts[b, 0])
                    val_dices.append(m["dice"])
                    val_ious.append(m["iou"])

        val_loss /= max(1, n_val_batches)
        mean_val_dice = float(sum(val_dices) / max(1, len(val_dices)))
        mean_val_iou = float(sum(val_ious) / max(1, len(val_ious)))

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": mean_val_dice,
            "val_iou": mean_val_iou,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(f"[{tag}] Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {mean_val_dice:.4f} | Val IoU: {mean_val_iou:.4f}")

        # Flush training history CSV after every epoch
        with open(history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr"])
            writer.writeheader()
            writer.writerows(history)

        # Check for improvement
        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            best_val_iou = mean_val_iou
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_val_dice,
                "val_iou": best_val_iou,
                "tag": tag,
            }, ckpt_path)
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{tag}] Early stopping triggered after {patience} epochs without improvement.")
                break

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"[{tag}] Complete in {elapsed_min:.1f}m. Best Val Dice: {best_val_dice:.4f} (Epoch {best_epoch})")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "tag": tag,
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "duration_min": round(elapsed_min, 2),
    }
