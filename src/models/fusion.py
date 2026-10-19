import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
    def __init__(self, medical_ch, camo_ch, mode="concat"):
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.conv = nn.Conv2d(medical_ch + camo_ch, medical_ch, 1)
        elif mode == "add":
            self.conv = nn.Conv2d(camo_ch, medical_ch, 1)
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

    def forward(self, m, c):
        if c.shape[2:] != m.shape[2:]:
            c = F.interpolate(c, size=m.shape[2:], mode="bilinear", align_corners=True)
        if self.mode == "concat":
            return self.conv(torch.cat([m, c], dim=1))
        return m + self.conv(c)
