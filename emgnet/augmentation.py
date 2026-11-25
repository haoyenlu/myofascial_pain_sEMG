"""
Optional training-time augmentations for multi-channel EMG segments (C, T).
Disabled by default (the reported experiments used no augmentation); enable with
``train.py --augment``.
"""
from __future__ import annotations

import random
from typing import List

import numpy as np
from scipy.interpolate import CubicSpline


class Augmentation:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return self.apply(signal) if random.random() < self.p else signal

    def apply(self, signal: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Compose:
    def __init__(self, augmentations: List[Augmentation]):
        self.augmentations = augmentations

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        for aug in self.augmentations:
            signal = aug(signal)
        return signal


class GaussianNoise(Augmentation):
    """Additive Gaussian noise with a std drawn uniformly from [min_std, max_std]."""

    def __init__(self, min_std: float = 0.05, max_std: float = 0.2, p: float = 0.7):
        super().__init__(p)
        self.min_std, self.max_std = min_std, max_std

    def apply(self, signal):
        return signal + np.random.normal(0, np.random.uniform(self.min_std, self.max_std), size=signal.shape)


class MagnitudeWarping(Augmentation):
    """Multiply each channel by a smooth random curve (cubic spline through `knot` points ~ N(1, sigma))."""

    def __init__(self, sigma: float = 0.5, knot: int = 4, p: float = 0.7):
        super().__init__(p)
        self.sigma, self.knot = sigma, knot

    def apply(self, signal):
        C, T = signal.shape
        x = np.linspace(0, T - 1, self.knot)
        t = np.arange(T)
        curves = np.stack([CubicSpline(x, np.random.normal(1.0, self.sigma, self.knot))(t) for _ in range(C)])
        return signal * curves


class ChannelPermutation(Augmentation):
    """Randomly permute the electrode channels."""

    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def apply(self, signal):
        return signal[np.random.permutation(signal.shape[0])]


def default_augmentation() -> Compose:
    """Augmentation pipeline used in the original experiments' augmentation ablation."""
    return Compose([
        ChannelPermutation(p=0.5),
        MagnitudeWarping(sigma=0.5, knot=4, p=0.7),
        GaussianNoise(min_std=0.05, max_std=0.2, p=0.7),
        ChannelPermutation(p=0.5),
    ])
