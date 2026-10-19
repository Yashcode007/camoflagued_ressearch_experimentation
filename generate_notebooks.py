import json

def make_cell(cell_type, source):
    lines = [s + '\n' for s in source.split('\n')]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    return {
        'cell_type': cell_type,
        'metadata': {},
        'source': lines,
        'outputs': [] if cell_type == 'code' else None,
        'execution_count': None if cell_type == 'code' else None
    }

# 1. Stage 1 Notebook
s1_cells = [
    make_cell('markdown', '# Stage 1: Camouflage Pre-training on COD10K\nPre-trains camouflage expert teachers: **CU1** (U-Net) and **CUA1** (Attention U-Net).'),
    make_cell('code', 'import os, sys, torch\nprint("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())\nif torch.cuda.is_available():\n    print("GPU:", torch.cuda.get_device_name(0))'),
    make_cell('code', 'from src.data.cod10k import COD10KDataset\nfrom src.models.unet import UNet, AttentionUNet\nfrom src.training.trainer import train_model\nfrom torch.utils.data import DataLoader'),
    make_cell('markdown', '### Train CU1 (Standard U-Net on COD10K)'),
    make_cell('code', '!python train_stage1.py --model_type unet --epochs 50 --batch_size 8 --lr 1e-4 --patience 10'),
    make_cell('markdown', '### Train CUA1 (Attention U-Net on COD10K)'),
    make_cell('code', '!python train_stage1.py --model_type attention_unet --epochs 50 --batch_size 8 --lr 1e-4 --patience 10'),
    make_cell('markdown', '### Teacher Verification & Strength Check'),
    make_cell('code', 'import pandas as pd\nfor tag, pth in [("CU1", "./checkpoints/CU1_best.pth"), ("CUA1", "./checkpoints/CUA1_best.pth")]:\n    if os.path.exists(pth):\n        ck = torch.load(pth, map_location="cpu")\n        print(f"{tag} Checkpoint -> Best Epoch: {ck.get(\'epoch\')} | Val Dice: {ck.get(\'val_dice\'):.4f} | Val IoU: {ck.get(\'val_iou\'):.4f}")')
]

nb1 = {'cells': s1_cells, 'metadata': {'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 4}
with open('./notebooks/Stage1_COD10K_Training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb1, f, indent=2)

# 2. Stage 2 & 3 Notebook
s2_cells = [
    make_cell('markdown', '# Stage 2 & 3: Camouflage Guidance, Modern Baselines, and Multi-Tier Evaluation\nExecutes multi-seed training (42, 123, 2026), ImageNet teacher control, U-Net++ baseline, and paired statistical testing.'),
    make_cell('code', 'import os, sys, torch\nprint("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())\nif torch.cuda.is_available():\n    print("GPU:", torch.cuda.get_device_name(0))'),
    make_cell('markdown', '### Multi-Seed Training (Seeds: 42, 123, 2026; 50 Epochs; Patience: 10)\nModels: M1, G1, G1R, M1W, G1_ImageNet, UNetPlusPlus'),
    make_cell('code', '!python train_stage2.py --models M1 G1 G1R M1W G1_ImageNet UNetPlusPlus --seeds 42 123 2026 --epochs 50 --patience 10'),
    make_cell('markdown', '### Fusion Granularity Ablations & G4 Stability\nModels: G1_early, G1_deep, G4'),
    make_cell('code', '!python train_stage2.py --models G1_early G1_deep --seeds 42 --epochs 50 --patience 10\n!python train_stage2.py --models G4 --seeds 123 2026 --epochs 50 --patience 10'),
    make_cell('markdown', '### Multi-Tier Evaluation & Statistical Hypothesis Testing\nIn-Domain (Kvasir-SEG) -> Generalization (ClinicDB, ColonDB, ETIS) -> Extreme Shift (ISIC 2016)'),
    make_cell('code', '!python evaluate_pipeline.py')
]

nb2 = {'cells': s2_cells, 'metadata': {'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 4}
with open('./notebooks/Stage2_Stage3_Full_Pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb2, f, indent=2)

print('Notebooks generated successfully in ./notebooks/')
