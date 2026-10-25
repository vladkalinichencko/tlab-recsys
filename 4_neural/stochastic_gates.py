"""Stochastic gates (Yamada et al. 2020, Feature Selection using Stochastic Gates): one gate per column.

Training: z = clamp(mu + eps, 0, 1) with eps ~ N(0, SIGMA²); evaluation: z = clamp(mu, 0, 1).
Every column of the table is multiplied by its gate, so z = 0 switches the column off.
open() is the expected number of open gates, sum Φ(mu / SIGMA); its penalty in the loss pushes
the gates of useless columns to 0 (notes §6, level 2). The gates start open (mu = 1), so the
pretrained reader first sees the table it was trained on.
"""
import torch
from torch import nn

SIGMA = 0.5


class Gates(nn.Module):
    def __init__(self, columns):
        super().__init__()
        self.mu = nn.Parameter(torch.ones(columns))

    def forward(self, X_BTH):
        noise = SIGMA * torch.randn_like(self.mu) if self.training else 0.0
        return X_BTH * (self.mu + noise).clamp(0, 1)

    def open(self):
        return (0.5 * (1 + torch.erf(self.mu / SIGMA / 2**0.5))).sum()
