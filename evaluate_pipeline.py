import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.models.registry import build_model, MODEL_SPECS
from src.data.medical import build_kvasir_split, load_external_dataset
from src.training.metrics import compute_binary_metrics
from src.evaluation.stats import compare_models_paired, holm_bonferroni

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Full Multi-Tier Evaluation and Statistical Testing")
    parser.add_argument("--kvasir_dir", type=str, default="./data/Kvasir-SEG")
    parser.add_argument("--isic_dir", type=str, default="./data/ISIC2016")
    parser.add_argument("--clinicdb_dir", type=str, default="./data/CVC-ClinicDB")
    parser.add_argument("--colondb_dir", type=str, default="./data/CVC-ColonDB")
    parser.add_argument("--etis_dir", type=str, default="./data/ETIS-LaribPolypDB")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/stage2")
    parser.add_argument("--cu1_path", type=str, default="./checkpoints/CU1_best.pth")
    parser.add_argument("--cua1_path", type=str, default="./checkpoints/CUA1_best.pth")
    parser.add_argument("--output_dir", type=str, default="./outputs/stage3")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

@torch.no_grad()
def evaluate_dataset(model, dataloader, device="cuda"):
    model.eval()
    results = []
    for imgs, masks in dataloader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        gts = (masks > 0.5).cpu().numpy()

        for b in range(len(preds)):
            m = compute_binary_metrics(preds[b, 0], gts[b, 0])
            results.append(m)
    return results

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Stage 3 Evaluation & Statistical Analysis ===")
    # Discover available evaluation datasets
    eval_sets = {}
    if os.path.exists(args.kvasir_dir):
        _, _, kvasir_test = build_kvasir_split(args.kvasir_dir, seed=42, augment=False)
        if len(kvasir_test) > 0:
            eval_sets["Kvasir-SEG (Test)"] = DataLoader(kvasir_test, batch_size=8, shuffle=False)
    if os.path.exists(args.isic_dir):
        isic_ds = load_external_dataset(args.isic_dir)
        if len(isic_ds) > 0:
            eval_sets["ISIC 2016 (Zero-Shot)"] = DataLoader(isic_ds, batch_size=8, shuffle=False)
    if os.path.exists(args.clinicdb_dir):
        clinic_ds = load_external_dataset(args.clinicdb_dir)
        if len(clinic_ds) > 0:
            eval_sets["CVC-ClinicDB (Domain Gen)"] = DataLoader(clinic_ds, batch_size=8, shuffle=False)
    if os.path.exists(args.colondb_dir):
        colon_ds = load_external_dataset(args.colondb_dir)
        if len(colon_ds) > 0:
            eval_sets["CVC-ColonDB (Domain Gen)"] = DataLoader(colon_ds, batch_size=8, shuffle=False)
    if os.path.exists(args.etis_dir):
        etis_ds = load_external_dataset(args.etis_dir)
        if len(etis_ds) > 0:
            eval_sets["ETIS-LaribPolypDB (Domain Gen)"] = DataLoader(etis_ds, batch_size=8, shuffle=False)

    print(f"Discovered {len(eval_sets)} non-empty evaluation datasets: {list(eval_sets.keys())}")

    # Discover checkpoints
    ckpt_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith("_best.pth")]
    print(f"Discovered {len(ckpt_files)} trained model checkpoints in {args.checkpoint_dir}")

    all_per_image_results = []
    summary_rows = []

    teacher_paths = {
        "CU1": args.cu1_path if hasattr(args, "cu1_path") else "./checkpoints/CU1_best.pth",
        "CUA1": args.cua1_path if hasattr(args, "cua1_path") else "./checkpoints/CUA1_best.pth",
    }

    for cf in ckpt_files:
        run_name = cf.replace("_best.pth", "")
        model_key = run_name.split("_seed")[0]
        seed = run_name.split("_seed")[1] if "_seed" in run_name else "42"
        ckpt_path = os.path.join(args.checkpoint_dir, cf)

        print(f"\nEvaluating {run_name}...")
        try:
            model = build_model(model_key, teacher_paths=teacher_paths, device=args.device)
            ck = torch.load(ckpt_path, map_location=args.device)
            state = ck.get("model_state_dict", ck)
            model.load_state_dict(state, strict=True)
            model.eval()
        except Exception as e:
            print(f"Skipping {run_name}: failed to load ({e})")
            continue

        for ds_name, loader in eval_sets.items():
            metrics_list = evaluate_dataset(model, loader, device=args.device)
            if not metrics_list:
                continue
            df_m = pd.DataFrame(metrics_list)
            df_m["model_run"] = run_name
            df_m["model_key"] = model_key
            df_m["seed"] = seed
            df_m["dataset"] = ds_name
            all_per_image_results.append(df_m)

            summary_rows.append({
                "dataset": ds_name,
                "model_key": model_key,
                "seed": seed,
                "dice_mean": df_m["dice"].mean(),
                "dice_std": df_m["dice"].std(),
                "iou_mean": df_m["iou"].mean(),
                "iou_std": df_m["iou"].std(),
                "precision": df_m["precision"].mean(),
                "recall": df_m["recall"].mean(),
                "specificity": df_m["specificity"].mean(),
            })

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(args.output_dir, "benchmark_summary.csv")
        df_summary.to_csv(summary_csv, index=False)
        print(f"\nBenchmark Summary written to {summary_csv}")
        print(df_summary.to_string(index=False))

    if all_per_image_results:
        full_df = pd.concat(all_per_image_results, ignore_index=True)
        full_csv = os.path.join(args.output_dir, "per_image_all.csv")
        full_df.to_csv(full_csv, index=False)
        print(f"Per-image metric logs written to {full_csv}")

        # Compute paired statistical comparisons (Reviewer Item 2)
        print("\n=== Direct Paired Statistical Comparisons (Wilcoxon Signed-Rank Test) ===")
        comparisons = [
            ("G1", "G1R"),
            ("G1", "M1W"),
            ("G1", "M1"),
            ("G1", "G1_ImageNet"),
            ("G1", "UNetPlusPlus")
        ]

        stat_rows = []
        datasets_present = full_df["dataset"].unique()
        for ds in datasets_present:
            ds_df = full_df[full_df["dataset"] == ds]
            pvals_to_correct = []
            comp_records = []

            for mA, mB in comparisons:
                dfA = ds_df[ds_df["model_key"] == mA]
                dfB = ds_df[ds_df["model_key"] == mB]
                if dfA.empty or dfB.empty:
                    continue

                scoresA = dfA["dice"].values
                scoresB = dfB["dice"].values
                min_len = min(len(scoresA), len(scoresB))
                res = compare_models_paired(scoresA[:min_len], scoresB[:min_len], metric_name="dice")
                res["dataset"] = ds
                res["comparison"] = f"{mA} vs {mB}"
                pvals_to_correct.append(res["p_value"])
                comp_records.append(res)

            if pvals_to_correct:
                p_adj = holm_bonferroni(pvals_to_correct)
                for rec, adj_p in zip(comp_records, p_adj):
                    rec["p_value_holm"] = float(adj_p)
                    stat_rows.append(rec)

        if stat_rows:
            df_stats = pd.DataFrame(stat_rows)
            stat_csv = os.path.join(args.output_dir, "wilcoxon_comparisons.csv")
            df_stats.to_csv(stat_csv, index=False)
            print(f"Wilcoxon statistical results written to {stat_csv}")
            print(df_stats[["dataset", "comparison", "delta_mean", "won_pct", "p_value", "p_value_holm", "ci_95_low", "ci_95_high"]].to_string(index=False))

if __name__ == "__main__":
    main()
