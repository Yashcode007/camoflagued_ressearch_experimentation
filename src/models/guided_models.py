import torch
import torch.nn as nn
from .unet import UNet, AttentionUNet
from .fusion import FeatureFusion

ALL_LEVELS = ("enc1", "enc2", "enc3", "enc4", "bottleneck")

class _GuidedBase(nn.Module):
    def __init__(self, teacher, student, base_channels=64, fusion_levels=ALL_LEVELS):
        super().__init__()
        self.teacher = teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        self.student = student
        self.fusion_levels = list(fusion_levels)

        chs = [base_channels * (2 ** i) for i in range(5)]
        level_names = ["enc1", "enc2", "enc3", "enc4", "bottleneck"]
        
        # Build fusion blocks for levels that require guidance
        self.fusions = nn.ModuleDict()
        for idx, lvl in enumerate(level_names):
            if lvl in self.fusion_levels:
                self.fusions[lvl] = FeatureFusion(chs[idx], chs[idx])

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()  # Teacher stays in eval mode (locks BatchNorm stats)
        return self

    def _encode(self, x):
        s1 = self.student.inc(x)
        s2 = self.student.down1(s1)
        s3 = self.student.down2(s2)
        s4 = self.student.down3(s3)
        s5 = self.student.down4(s4)
        return [s1, s2, s3, s4, s5]

    def forward(self, x, return_all=False):
        self.teacher.eval()
        with torch.no_grad():
            _, t = self.teacher(x, return_features=True)

        s = self._encode(x)
        level_names = ["enc1", "enc2", "enc3", "enc4", "bottleneck"]
        f = []
        for idx, lvl in enumerate(level_names):
            if lvl in self.fusions:
                f.append(self.fusions[lvl](s[idx], t[idx]))
            else:
                f.append(s[idx])

        # Decode using student decoder
        u = self.student.up1(f[4], f[3])
        u = self.student.up2(u, f[2])
        u = self.student.up3(u, f[1])
        u = self.student.up4(u, f[0])
        logits = self.student.outc(u)

        if return_all:
            return logits, t, s, f
        return logits

class GuidedUNet(_GuidedBase):
    def __init__(self, teacher, base_channels=64, out_channels=1, fusion_levels=ALL_LEVELS):
        super().__init__(
            teacher=teacher,
            student=UNet(3, out_channels, base_channels),
            base_channels=base_channels,
            fusion_levels=fusion_levels
        )

class GuidedAttentionUNet(_GuidedBase):
    def __init__(self, teacher, base_channels=64, out_channels=1, fusion_levels=ALL_LEVELS):
        super().__init__(
            teacher=teacher,
            student=AttentionUNet(3, out_channels, base_channels),
            base_channels=base_channels,
            fusion_levels=fusion_levels
        )
