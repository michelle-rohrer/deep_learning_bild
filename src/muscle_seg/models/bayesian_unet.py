"""Bayesian 3D U-Net mit MC Dropout (Dropout bleibt in eval() aktiv)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BayesianUNet3D(nn.Module):
    """
    Kompaktes 3D U-Net (Encoder–Decoder + Skip Connections).
    Dropout nach jedem Conv-Block; zur Inferenz N× Forward mit train()-Modus.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 9,
        base_channels: int = 16,
        depth: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        ch = base_channels

        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        c_in = in_channels
        channels_per_level: list[int] = []
        for _ in range(depth):
            self.down_blocks.append(ConvBlock(c_in, ch, dropout))
            self.pools.append(nn.MaxPool3d(2))
            channels_per_level.append(ch)
            c_in = ch
            ch = min(ch * 2, 128)

        self.bottleneck = ConvBlock(c_in, ch, dropout)
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        rev = list(reversed(channels_per_level))
        c_in = ch
        for skip_ch in rev:
            self.up_convs.append(
                nn.ConvTranspose3d(c_in, skip_ch, kernel_size=2, stride=2)
            )
            self.up_blocks.append(ConvBlock(skip_ch * 2, skip_ch, dropout))
            c_in = skip_ch

        self.head = nn.Conv3d(c_in, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for down, pool in zip(self.down_blocks, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, block, skip in zip(self.up_convs, self.up_blocks, reversed(skips)):
            x = up(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = block(x)

        return self.head(x)


@torch.no_grad()
def mc_predict(
    model: BayesianUNet3D,
    x: torch.Tensor,
    n_samples: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Monte-Carlo-Inferenz: Mittelwert der Softmax-Wahrscheinlichkeiten + Varianz.
    Dropout bleibt aktiv (model.train()).
    """
    model.train()
    probs_stack = []
    for _ in range(n_samples):
        logits = model(x)
        probs_stack.append(torch.softmax(logits, dim=1))
    stacked = torch.stack(probs_stack, dim=0)
    mean_probs = stacked.mean(dim=0)
    variance = stacked.var(dim=0).mean(dim=1, keepdim=False)  # mittlere Unsicherheit über Klassen
    pred = mean_probs.argmax(dim=1)
    return pred, variance
