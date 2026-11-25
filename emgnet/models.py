"""
Model definition.

``CNN1D`` - 1-D convolutional classifier on multi-channel EMG contraction segments.

Module/parameter names match the original training code so the released
checkpoints load with ``model.load_state_dict(ckpt["model"])``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    4 x [Conv1d(k=3, stride=2, pad=1) -> ReLU -> Dropout] with 16/32/64/128 filters,
    followed by an MLP (1024 -> 64 -> out_dim). Input (B, C, L), output logits (B, out_dim).
    """

    def __init__(self, in_dim: int = 64, out_dim: int = 1, window_size: int = 8000, dropout: float = 0.5):
        super().__init__()
        kernel_size, stride, padding = 3, 2, 1
        channels = [in_dim, 16, 32, 64, 128]
        self.layer1 = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for i in range(4)
        ])

        length = window_size
        for _ in range(4):
            length = (length + 2 * padding - kernel_size) // stride + 1

        self.out = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(length * channels[-1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.layer1(x))


def build_model(in_dim: int = 64, window_size: int = 8000, dropout: float = 0.5) -> nn.Module:
    return CNN1D(in_dim=in_dim, out_dim=1, window_size=window_size, dropout=dropout)
