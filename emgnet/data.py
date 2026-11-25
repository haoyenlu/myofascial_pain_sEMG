"""
Dataset built on the extracted contraction segments.

Expected layout (produced by ``extract_segments.py``)::

    <segment_dir>/<subject>/segment-<i>.npy      # array (C, T_i)

Every contraction segment is one sample: it is (optionally augmented and)
Fourier-resampled to a fixed length of ``segment_length`` samples and labelled
with its subject's group (0 = healthy, 1 = latent/active MTrP).
"""
from __future__ import annotations

import glob
import os
import re
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

LABEL_MAP = {"Healthy": 0, "Latent": 1, "Active": 1}


def load_labels(label_csv: str, subject_col: str = "subject", type_col: str = "Type") -> Dict[str, int]:
    """Read a CSV with one row per subject and map its group to a binary label."""
    df = pd.read_csv(label_csv)
    labels = {}
    for subject, group in zip(df[subject_col], df[type_col]):
        if isinstance(group, str) and group in LABEL_MAP:
            labels[str(subject)] = LABEL_MAP[group]
    return labels


def list_subjects(segment_dir: str) -> List[str]:
    return sorted(d for d in os.listdir(segment_dir) if os.path.isdir(os.path.join(segment_dir, d)))


def segment_files(segment_dir: str, subject: str) -> List[str]:
    files = glob.glob(os.path.join(segment_dir, subject, "segment-*.npy"))
    return sorted(files, key=lambda f: int(re.findall(r"segment-(\d+)\.npy", f)[0]))


class ContractionSegmentDataset(Dataset):
    """
    Returns a dict ``{"segment": (C, segment_length) float tensor, "target": float tensor,
    "subject": str}`` per contraction segment.
    """

    def __init__(self, subjects: Sequence[str], labels: Dict[str, int], segment_dir: str,
                 segment_length: int = 8000, augmentation: Optional[Callable] = None,
                 preload: bool = True):
        self.labels = labels
        self.segment_length = segment_length
        self.augmentation = augmentation
        self.items = [(subject, f) for subject in subjects for f in segment_files(segment_dir, subject)]
        self.cache = [np.load(f) for _, f in self.items] if preload else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        subject, f = self.items[i]
        segment = self.cache[i] if self.cache is not None else np.load(f)
        if self.augmentation is not None:
            segment = self.augmentation(segment)
        segment = resample(segment, self.segment_length, axis=-1)
        return {
            "segment": torch.from_numpy(np.ascontiguousarray(segment)).float(),
            "target": torch.tensor(float(self.labels[subject]), dtype=torch.float32),
            "subject": subject,
        }
