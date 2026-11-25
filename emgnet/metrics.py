from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.stats import t
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve


def aggregate_by_subject(y_true: Sequence[float], y_prob: Sequence[float], subjects: Sequence[str]):
    """Subject-level scores: mean predicted probability over each subject's segments."""
    y_true, y_prob, subjects = np.asarray(y_true), np.asarray(y_prob), np.asarray(subjects)
    ids = np.unique(subjects)
    return (np.array([y_true[subjects == s].mean() for s in ids]),
            np.array([y_prob[subjects == s].mean() for s in ids]),
            ids)


def youden_threshold(y_true, y_prob) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[int(np.argmax(tpr - fpr))])


def compute_metrics(y_true: Sequence[float], y_prob: Sequence[float]) -> Dict[str, float]:
    """
    AUC, plus accuracy / F1 / MCC at the Youden-optimal threshold (max TPR - FPR)
    of the same split. AUC is threshold-free; the thresholded metrics are optimistic
    because the threshold is chosen on the evaluated data.
    """
    y_true, y_prob = np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return dict(AUC=np.nan, ACC=np.nan, F1=np.nan, MCC=np.nan, threshold=np.nan)
    thr = youden_threshold(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    return dict(AUC=float(roc_auc_score(y_true, y_prob)),
                ACC=float(accuracy_score(y_true, y_pred)),
                F1=float(f1_score(y_true, y_pred, zero_division=0)),
                MCC=float(matthews_corrcoef(y_true, y_pred)),
                threshold=thr)


def interpolated_roc(y_true, y_prob, grid: np.ndarray = np.linspace(0, 1, 100)) -> np.ndarray:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    out = np.interp(grid, fpr, tpr)
    out[0] = 0.0
    return out


def ci_t(scores: Sequence[float], alpha: float = 0.05) -> Tuple[float, float, float]:
    """Mean and two-sided Student-t confidence interval."""
    x = np.asarray(scores, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float(np.mean(x)) if n else np.nan, np.nan, np.nan
    m = x.mean()
    margin = t.ppf(1 - alpha / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(m), float(m - margin), float(m + margin)


def ci_bootstrap(scores: Sequence[float], alpha: float = 0.05, n_boot: int = 50_000, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean."""
    x = np.asarray(scores, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
