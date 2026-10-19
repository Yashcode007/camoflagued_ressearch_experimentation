import os
import torch
import torch.nn as nn
from .unet import UNet, AttentionUNet
from .fusion import FeatureFusion
from .guided_models import GuidedUNet, GuidedAttentionUNet
from .modern import UNetPlusPlus

MODEL_SPECS = {
    # Unguided Baselines
    "M1":           dict(family="baseline", student="unet",          teacher=None,       width=64,  levels="none"),
    "MUA1":         dict(family="baseline", student="attention",     teacher=None,       width=64,  levels="none"),
    "UNetPlusPlus": dict(family="modern",   student="unet_plus_plus", teacher=None,       width=64,  levels="none"),

    # Capacity & Architecture Controls
    "M1W":          dict(family="control",  student="unet",          teacher=None,       width=92,  levels="none"),
    "G1R":          dict(family="control",  student="unet",          teacher="RANDOM",   width=64,  levels="all"),

    # Pretraining Control (Reviewer Item 2)
    "G1_ImageNet":  dict(family="control",  student="unet",          teacher="IMAGENET", width=64,  levels="all"),

    # Guided Models (Camouflage Teachers)
    "G1":           dict(family="guided",   student="unet",          teacher="CU1",      width=64,  levels="all"),
    "G2":           dict(family="guided",   student="unet",          teacher="CUA1",     width=64,  levels="all"),
    "G3":           dict(family="guided",   student="attention",     teacher="CU1",      width=64,  levels="all"),
    "G4":           dict(family="guided",   student="attention",     teacher="CUA1",     width=64,  levels="all"),

    # Fusion Ablations (Reviewer Item 6)
    "G1_early":     dict(family="ablation", student="unet",          teacher="CU1",      width=64,  levels="early"),
    "G1_deep":      dict(family="ablation", student="unet",          teacher="CU1",      width=64,  levels="deep"),
}

LEVEL_MAPPINGS = {
    "all":   ["enc1", "enc2", "enc3", "enc4", "bottleneck"],
    "early": ["enc1", "enc2"],
    "deep":  ["enc3", "enc4", "bottleneck"],
}

def load_teacher_checkpoint(kind, path, base_channels=64, device="cuda"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Teacher checkpoint not found at: {path}")
    teacher = UNet(3, 1, base_channels) if kind == "unet" else AttentionUNet(3, 1, base_channels)
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher.to(device)

def make_random_teacher(kind="unet", base_channels=64, seed=43, device="cuda"):
    """Random frozen teacher for capacity control (G1R)."""
    torch.manual_seed(seed)
    teacher = UNet(3, 1, base_channels) if kind == "unet" else AttentionUNet(3, 1, base_channels)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher.to(device)

def make_imagenet_teacher(kind="unet", base_channels=64, device="cuda"):
    """
    ImageNet-pretrained teacher control (G1_ImageNet).
    Initializes encoder weights with ImageNet-pretrained representations, frozen.
    """
    import torchvision.models as models
    teacher = UNet(3, 1, base_channels) if kind == "unet" else AttentionUNet(3, 1, base_channels)
    
    # Load ImageNet-pretrained VGG-16 with batch norm as weights provider
    try:
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        vgg_layers = list(vgg.features.children())

        # Map VGG stages to UNet encoder stages
        # inc (DoubleConv 3 -> 64)
        teacher.inc.net[0].weight.data.copy_(vgg_layers[0].weight.data)
        teacher.inc.net[3].weight.data.copy_(vgg_layers[3].weight.data)
        # down1 (DoubleConv 64 -> 128)
        teacher.down1.net[1].net[0].weight.data.copy_(vgg_layers[7].weight.data)
        teacher.down1.net[1].net[3].weight.data.copy_(vgg_layers[10].weight.data)
        # down2 (DoubleConv 128 -> 256)
        teacher.down2.net[1].net[0].weight.data.copy_(vgg_layers[14].weight.data)
        teacher.down2.net[1].net[3].weight.data.copy_(vgg_layers[17].weight.data)
        # down3 (DoubleConv 256 -> 512)
        teacher.down3.net[1].net[0].weight.data.copy_(vgg_layers[24].weight.data)
        teacher.down3.net[1].net[3].weight.data.copy_(vgg_layers[27].weight.data)
    except Exception as e:
        print(f"Warning: Failed to load torchvision ImageNet weights ({e}), using default initialization.")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher.to(device)

def build_model(key, teacher_paths=None, base_channels=64, device="cuda"):
    """
    Unified model builder for all baselines, guided models, and controls.
    """
    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model key: {key}. Available: {list(MODEL_SPECS.keys())}")

    spec = MODEL_SPECS[key]
    w = spec["width"]
    teacher_type = spec["teacher"]
    levels_key = spec["levels"]
    fusion_levels = LEVEL_MAPPINGS.get(levels_key, LEVEL_MAPPINGS["all"])

    # 1. Plain baselines & modern architectures
    if spec["student"] == "unet_plus_plus":
        return UNetPlusPlus(3, 1, base_channels=w).to(device)
    if teacher_type is None:
        if spec["student"] == "unet":
            return UNet(3, 1, base_channels=w).to(device)
        elif spec["student"] == "attention":
            return AttentionUNet(3, 1, base_channels=w).to(device)

    # 2. Build teacher for guided models
    if teacher_type == "RANDOM":
        teacher = make_random_teacher("unet", base_channels=base_channels, device=device)
    elif teacher_type == "IMAGENET":
        teacher = make_imagenet_teacher("unet", base_channels=base_channels, device=device)
    else:
        if not teacher_paths or teacher_type not in teacher_paths:
            raise ValueError(f"Teacher path for '{teacher_type}' must be provided in teacher_paths dict.")
        t_path = teacher_paths[teacher_type]
        t_kind = "unet" if teacher_type == "CU1" else "attention"
        teacher = load_teacher_checkpoint(t_kind, t_path, base_channels=base_channels, device=device)

    # 3. Assemble Guided Model
    if spec["student"] == "unet":
        return GuidedUNet(teacher, base_channels=base_channels, fusion_levels=fusion_levels).to(device)
    else:
        return GuidedAttentionUNet(teacher, base_channels=base_channels, fusion_levels=fusion_levels).to(device)
