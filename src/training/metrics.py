import numpy as np

def compute_binary_metrics(pred_binary, target_binary, eps=1e-7):
    """
    Computes per-image segmentation metrics:
    Dice, IoU, Precision, Recall, Specificity, Accuracy.
    Both inputs should be boolean or {0, 1} numpy arrays.
    """
    p = pred_binary.astype(bool).ravel()
    t = target_binary.astype(bool).ravel()

    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, np.logical_not(t)).sum()
    fn = np.logical_and(np.logical_not(p), t).sum()
    tn = np.logical_and(np.logical_not(p), np.logical_not(t)).sum()

    # If both mask and target are empty: perfect match
    if t.sum() == 0 and p.sum() == 0:
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
        specificity = 1.0
    else:
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
    }
