import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.registry import build_model
from src.data.medical import build_kvasir_split

def compute_layer_activation_variance(feature_map):
    """
    Computes spatial variance of activation maps across channels: Var(F)
    """
    # feature_map shape: [B, C, H, W]
    var_per_channel = torch.var(feature_map, dim=(-2, -1)) # [B, C]
    return torch.mean(var_per_channel).item()

def compute_cosine_similarity(feat_teacher, feat_student):
    """
    Computes cosine similarity between teacher and student representations.
    Projected to matching spatial shape and channels via 1x1 conv or pooling if needed.
    """
    # Global average pool both to [B, C] vectors for invariant representation similarity
    t_vec = F.adaptive_avg_pool2d(feat_teacher, (1, 1)).flatten(1)
    s_vec = F.adaptive_avg_pool2d(feat_student, (1, 1)).flatten(1)
    
    # If channel counts differ, interpolate / project or compute across common dimensions
    min_c = min(t_vec.shape[1], s_vec.shape[1])
    sim = F.cosine_similarity(t_vec[:, :min_c], s_vec[:, :min_c], dim=1)
    return sim.mean().item()

def run_feature_analysis(g1_ckpt, g1r_ckpt, cu1_path, kvasir_dir="./data/Kvasir-SEG", output_dir="./outputs/stage3", num_samples=20, device="cuda"):
    """
    Fulfills Reviewer Item 10: Quantitative Feature Alignment Analysis across 20 test images.
    """
    os.makedirs(output_dir, exist_ok=True)
    _, _, test_ds = build_kvasir_split(kvasir_dir, seed=42, augment=False)
    
    teacher_paths = {"CU1": cu1_path}
    model_g1 = build_model("G1", teacher_paths=teacher_paths, device=device)
    ck1 = torch.load(g1_ckpt, map_location=device)
    model_g1.load_state_dict(ck1.get("model_state_dict", ck1), strict=True)
    model_g1.eval()

    model_g1r = build_model("G1R", teacher_paths=teacher_paths, device=device)
    ckr = torch.load(g1r_ckpt, map_location=device)
    model_g1r.load_state_dict(ckr.get("model_state_dict", ckr), strict=True)
    model_g1r.eval()

    levels = ["enc1", "enc2", "enc3", "enc4", "bottleneck"]
    records = []

    print(f"Analyzing {num_samples} Kvasir-SEG test samples for feature variance and cosine similarity...")

    with torch.no_grad():
        for i in range(min(num_samples, len(test_ds))):
            img, _ = test_ds[i]
            x = img.unsqueeze(0).to(device)

            # Extract teacher features
            _, t_cu1_list = model_g1.teacher(x, return_features=True)
            _, t_rand_list = model_g1r.teacher(x, return_features=True)
            s1_list = model_g1._encode(x)

            for idx, lvl in enumerate(levels):
                t_cu1 = t_cu1_list[idx]
                t_rand = t_rand_list[idx]
                s1 = s1_list[idx]
                
                var_cu1 = compute_layer_activation_variance(t_cu1)
                var_rand = compute_layer_activation_variance(t_rand)
                sim_val = compute_cosine_similarity(t_cu1, s1)

                records.append({
                    "sample_idx": i,
                    "level": lvl,
                    "var_camou_teacher": var_cu1,
                    "var_random_teacher": var_rand,
                    "cosine_sim_teacher_student": sim_val
                })

    df = pd.DataFrame(records)
    summary = df.groupby("level").agg({
        "var_camou_teacher": ["mean", "std"],
        "var_random_teacher": ["mean", "std"],
        "cosine_sim_teacher_student": ["mean", "std"]
    }).reset_index()

    csv_path = os.path.join(output_dir, "feature_analysis_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Feature analysis results written to {csv_path}")
    return summary
