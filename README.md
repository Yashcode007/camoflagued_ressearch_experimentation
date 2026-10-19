# Camouflage-Aware Medical Image Segmentation: Research & Experimentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete experimental framework and codebase for **Camouflage-Aware Medical Image Segmentation**, addressing the challenge of segmenting low-contrast, visually ambiguous, and camouflaged anatomical targets (e.g., gastrointestinal polyps and skin lesions).

The methodology introduces **intermediate feature guidance** derived from models pre-trained on Camouflaged Object Detection (**COD10K**), evaluated against matched-capacity baselines, rigorous control teachers, and competitive architectures across multiple random seeds with paired non-parametric statistical significance testing.

---

## 🔬 Key Methodological Highlights

1. **Stage 1 — Camouflage Representation Pre-training**:
   - `CU1`: Standard U-Net pre-trained on COD10K.
   - `CUA1`: Attention U-Net pre-trained on COD10K.
2. **Stage 2 — Multi-Seed Medical Segmentation on Kvasir-SEG**:
   - Trained across **3 independent random seeds** (`42`, `123`, `2026`) with early stopping (patience = 10).
   - **Guided Models**: `G1` (single-block guidance), `G2` (two blocks), `G4` (four blocks).
   - **Ablation Models**: `G1_early` (early encoder guidance), `G1_deep` (deep encoder guidance).
   - **Rigorous Controls**:
     - `G1R`: Same guidance architecture paired with an untrained (randomly initialized) teacher to test whether guidance benefits arise from learned features or structural capacity.
     - `G1_ImageNet`: Guidance paired with an ImageNet-pretrained teacher to test domain specificity.
   - **Baselines**:
     - `M1`: Standard medical U-Net baseline.
     - `M1W`: Wider medical baseline parameter-matched to account for guidance module capacity.
     - `UNetPlusPlus`: SOTA nested dense skip connection benchmark.
3. **Stage 3 — Cross-Domain Benchmark & Statistical Testing**:
   - Out-of-distribution evaluation across Kvasir-SEG (in-domain), ISIC 2016 (zero-shot skin lesion), CVC-ClinicDB, CVC-ColonDB, and ETIS-LaribPolypDB.
   - **Paired Wilcoxon signed-rank tests** with **Holm-Bonferroni correction** for multiple comparisons.
4. **Feature Representation Analysis (CKA & Activation Variance)**:
   - Quantifies teacher-student feature alignment and verifies preservation of non-trivial feature variance across encoder hierarchies.

---

## 📁 Repository Structure

```text
├── .env.example               # Environment variables template
├── environment.yml            # Conda environment definition
├── requirements.txt           # Pip dependencies
├── download_datasets.py       # Automated dataset downloader (COD10K, Kvasir-SEG, ISIC, etc.)
├── train_stage1.py            # Stage 1 pre-training script on COD10K
├── train_stage2.py            # Stage 2 multi-seed training matrix (10 models × 3 seeds)
├── evaluate_pipeline.py       # Stage 3 cross-dataset evaluation & Wilcoxon statistical testing
├── run_pipeline.py            # Master end-to-end orchestration pipeline
├── generate_notebooks.py      # Programmatic generator for reproducible Jupyter notebooks
├── notebooks/                 # Standalone, step-by-step experiment notebooks
│   ├── Stage1_COD10K_Training.ipynb
│   └── Stage2_Stage3_Full_Pipeline.ipynb
├── src/                       # Modular source code
│   ├── data/                  # Dataset loaders, medical preprocessing, augmentations
│   │   ├── medical.py
│   │   └── transforms.py
│   ├── models/                # Architecture definitions & model registry
│   │   ├── unet.py
│   │   ├── attention_unet.py
│   │   ├── unet_plus_plus.py
│   │   ├── guided_models.py
│   │   └── registry.py
│   ├── training/              # Training loops, loss functions (Dice + BCE), metrics
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   └── evaluation/            # Statistical testing (Wilcoxon, Holm-Bonferroni) & CKA
│       ├── stats.py
│       └── features.py
└── outputs/                   # Experimental records, logs, and benchmark summaries
    ├── stage2/                # Per-epoch training curves (CSV) & training summary (JSON)
    └── stage3/                # Benchmark summaries, per-image metrics, Wilcoxon tests
```

---

## 🚀 Quick Start

### 1. Installation

Clone this repository and set up a virtual environment:

```bash
git clone https://github.com/Yashcode007/camoflagued_ressearch_experimentation.git
cd camoflagued_ressearch_experimentation

# Using Conda:
conda env create -f environment.yml
conda activate camo-med

# Or using Pip:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and fill in your Kaggle API credentials (for automated dataset downloads):

```bash
cp .env.example .env
```

### 3. Running the Pipeline

To run the complete automated pipeline (Stage 1 -> Stage 2 -> Stage 3 -> Feature Analysis):

```bash
python run_pipeline.py
```

Alternatively, run individual stages:

```bash
# Stage 1: Pre-train CU1 (COD10K)
python train_stage1.py --model_type unet --epochs 50 --batch_size 8 --device cuda

# Stage 2: Train multi-seed experimental matrix
python train_stage2.py --models M1 G1 G1R M1W G1_ImageNet UNetPlusPlus G2 G4 G1_early G1_deep \
                       --seeds 42 123 2026 --epochs 50 --patience 10 --device cuda

# Stage 3: Benchmark evaluation and paired statistical tests
python evaluate_pipeline.py --checkpoint_dir ./checkpoints/stage2 --output_dir ./outputs/stage3
```

---

## 📊 Summary of Benchmark Results

Detailed per-epoch training logs are stored in [`outputs/stage2/`](outputs/stage2/) and cross-dataset evaluations are archived in [`outputs/stage3/`](outputs/stage3/).

* **Validation Convergence**: All 30 models converged reliably across seeds 42, 123, and 2026, achieving 0.83–0.85 mean Dice on Kvasir-SEG validation splits.
* **Out-of-Distribution Generalization**: Guided models (`G1`) demonstrate statistically significant gains over baseline architectures on zero-shot out-of-distribution targets (e.g., $p < 10^{-33}$ on ISIC 2016 vs U-Net++).
* **Representation Analysis**: Camouflage-guided representations retain non-trivial activation variance across all encoder layers ($0.116 - 0.243$), avoiding the representation collapse observed in random-guidance controls.

---

## 📜 License

This project is licensed under the MIT License.
