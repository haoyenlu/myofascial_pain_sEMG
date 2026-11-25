from __future__ import annotations

import os
import random
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.model_selection import KFold


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subject_kfold(subjects: Sequence[str], n_splits: int = 5, seed: int = 121) -> Dict[int, Dict[str, List[str]]]:
    """Subject-level K-fold split (no subject appears in both train and test)."""
    subjects = list(subjects)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return {k: dict(train=[subjects[i] for i in tr], test=[subjects[i] for i in te])
            for k, (tr, te) in enumerate(kf.split(subjects))}


def save_checkpoint(model: torch.nn.Module, path: str, **extra) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(model=model.state_dict(), **extra), path)


def load_checkpoint(model: torch.nn.Module, path: str, device="cpu") -> torch.nn.Module:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    return model


def find_fold_checkpoint(root: str, fold: int) -> str:
    """Return the best checkpoint of a fold (supports both historical file names)."""
    for name in ("checkpoint_best.pth", "checkpoint_epoch_best.pth"):
        path = os.path.join(root, f"fold_{fold}", name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No best checkpoint for fold {fold} under {root}")


def get_device(name: str = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
