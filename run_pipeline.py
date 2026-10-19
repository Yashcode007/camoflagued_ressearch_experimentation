import os
import sys
import time
import subprocess
import psutil

_venv_python = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
PYTHON_EXE = _venv_python if os.path.exists(_venv_python) else sys.executable

def is_cu1_running():
    my_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            if proc.info['pid'] == my_pid:
                continue
            cmd = proc.info.get('cmdline')
            if cmd and any(str(arg).endswith('train_stage1.py') for arg in cmd) and any('unet' in str(arg) for arg in cmd) and not any('attention' in str(arg) for arg in cmd):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def wait_for_cu1():
    pid = is_cu1_running()
    if pid:
        print(f"--> Detected CU1 training running (PID {pid}). Waiting for completion...")
        while is_cu1_running():
            time.sleep(30)
            if os.path.exists("./outputs/history_CU1.csv"):
                with open("./outputs/history_CU1.csv", "r") as f:
                    lines = f.readlines()
                    last_epoch = len(lines) - 1
                print(f"[Wait Monitor] CU1 running... Completed epochs: {last_epoch}", end="\r")
        print("\n[Wait Monitor] CU1 training has finished!")
    else:
        print("--> CU1 is not currently running.")

def run_cmd(cmd, desc):
    print("\n" + "=" * 80)
    print(f"=== [STARTING] {desc} ===")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    start_time = time.time()
    res = subprocess.run(cmd)
    elapsed = (time.time() - start_time) / 60.0
    if res.returncode != 0:
        print(f"ERROR: {desc} exited with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"=== [COMPLETED] {desc} in {elapsed:.1f} minutes ===")

def main():
    print("================================================================================")
    print("  CAMOUFLAGE-AWARE MEDICAL IMAGE SEGMENTATION: END-TO-END MASTER PIPELINE")
    print("  In Full Alignment with Improvements.docx (Items 1 - 11)")
    print("================================================================================")

    # 1. Wait for CU1 (Stage 1 Teacher: Standard U-Net on COD10K)
    wait_for_cu1()
    if not os.path.exists("./checkpoints/CU1_best.pth"):
        print("CU1_best.pth not found. Running CU1 pre-training...")
        run_cmd([PYTHON_EXE, "-u", "train_stage1.py", "--model_type", "unet", "--epochs", "50", "--batch_size", "8", "--num_workers", "0", "--device", "cuda"],
                "Stage 1 Pre-training: CU1 (U-Net on COD10K)")

    # 2. Train CUA1 (Stage 1 Teacher: Attention U-Net on COD10K)
    if not os.path.exists("./checkpoints/CUA1_best.pth"):
        run_cmd([PYTHON_EXE, "-u", "train_stage1.py", "--model_type", "attention_unet", "--epochs", "50", "--batch_size", "8", "--num_workers", "0", "--device", "cuda"],
                "Stage 1 Pre-training: CUA1 (Attention U-Net on COD10K)")
    else:
        print("--> CUA1_best.pth already exists. Skipping CUA1 training.")

    # 3. Stage 2 Multi-Seed Training on Kvasir-SEG (Seeds 42, 123, 2026)
    # Models: M1, G1, G1R, M1W, G1_ImageNet, UNetPlusPlus, G2, G4, G1_early, G1_deep
    stage2_models = ["M1", "G1", "G1R", "M1W", "G1_ImageNet", "UNetPlusPlus", "G2", "G4", "G1_early", "G1_deep"]
    run_cmd([
        PYTHON_EXE, "-u", "train_stage2.py",
        "--models", *stage2_models,
        "--seeds", "42", "123", "2026",
        "--epochs", "50",
        "--patience", "10",
        "--batch_size", "8",
        "--num_workers", "0",
        "--device", "cuda"
    ], "Stage 2 Multi-Seed Training (Baselines, Controls, Guided Models & Ablations)")

    # 4. Stage 3 Evaluation & Statistical Wilcoxon Testing
    run_cmd([
        PYTHON_EXE, "-u", "evaluate_pipeline.py",
        "--kvasir_dir", "./data/Kvasir-SEG",
        "--isic_dir", "./data/ISIC2016",
        "--clinicdb_dir", "./data/CVC-ClinicDB",
        "--colondb_dir", "./data/CVC-ColonDB",
        "--etis_dir", "./data/ETIS-LaribPolypDB",
        "--checkpoint_dir", "./checkpoints/stage2",
        "--cu1_path", "./checkpoints/CU1_best.pth",
        "--cua1_path", "./checkpoints/CUA1_best.pth",
        "--output_dir", "./outputs/stage3",
        "--device", "cuda"
    ], "Stage 3 Benchmark Evaluation & Paired Wilcoxon Statistical Testing")

    # 5. Feature Representation Analysis (Reviewer Item 10)
    print("\n--> Running Feature Representation Analysis (Item 10)...")
    from src.evaluation.features import run_feature_analysis
    g1_ckpt = "./checkpoints/stage2/G1_seed42_best.pth"
    g1r_ckpt = "./checkpoints/stage2/G1R_seed42_best.pth"
    if os.path.exists(g1_ckpt) and os.path.exists(g1r_ckpt):
        run_feature_analysis(
            g1_ckpt=g1_ckpt,
            g1r_ckpt=g1r_ckpt,
            cu1_path="./checkpoints/CU1_best.pth",
            kvasir_dir="./data/Kvasir-SEG",
            output_dir="./outputs/stage3",
            num_samples=20,
            device="cuda"
        )
    else:
        print(f"Skipping feature analysis: {g1_ckpt} or {g1r_ckpt} not found.")

    print("\n================================================================================")
    print("  ALL EXPERIMENTAL STAGES COMPLETED SUCCESSFULLY!")
    print("  Outputs saved under ./outputs/ and checkpoints under ./checkpoints/")
    print("================================================================================")

if __name__ == "__main__":
    main()
