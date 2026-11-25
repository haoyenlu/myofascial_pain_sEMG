"""
Signal preprocessing and automatic contraction-segment extraction for
high-density surface EMG recordings.

Pipeline (per subject):
    raw (C, T)  ->  band-pass  ->  sliding RMS  ->  percentile outlier removal
                ->  per-channel z-score  ->  [optional] pre/post-intervention cropping
                ->  channel-mean envelope  ->  sign-preserving power scaling  ->  moving average
                ->  peak detection + peak widths  ->  binary contraction mask
                ->  contiguous contraction segments (C, T_i)
"""
from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, fftconvolve, filtfilt, find_peaks, peak_widths


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessParam:
    fs: int = 4000                  # sampling rate (Hz)
    lowcut: float = 10.0            # band-pass low cut-off (Hz)
    highcut: float = 500.0          # band-pass high cut-off (Hz)
    filter_order: int = 4           # Butterworth order
    rms_window_size: int = 800      # sliding-RMS window (samples)
    outlier_percentile: float = 95  # time points above this per-channel percentile are removed
    # Subject-specific number of leading samples to discard (recording artefacts).
    manual_crop: Dict[str, int] = field(default_factory=lambda: {"S0013": 200000})


@dataclass
class ContractionParam:
    fs: int = 4000
    window_size: int = 20000        # moving-average window on the scaled envelope (samples)
    period: str = "pre"             # "pre" | "post" | "all": which part of the session to segment
    active_percentile: float = 90   # threshold used to locate the pre/post-intervention split
    active_extension: int = 4000    # samples added around the split point
    peak_percentile: float = 75     # minimum peak height, as a percentile of the envelope
    peak_rel_height: float = 0.5    # relative height at which a peak's width is measured
    peak_min_width: int = 12000     # minimum peak width (samples)
    peak_wlen: int = 400000         # window length for prominence / width computation
    exp_scale_gamma: float = 2.0    # power-scaling exponent


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_subject_signal(subject: str, data_dir: str) -> np.ndarray:
    """
    Load the raw multi-channel EMG of a subject as an array of shape (C, T).

    Accepts either ``<data_dir>/<subject>.npy`` or ``<data_dir>/<subject>-*.DATA_RAW.mat``
    (MATLAB variable ``data``).
    """
    npy = os.path.join(data_dir, f"{subject}.npy")
    if os.path.exists(npy):
        signal = np.load(npy)
    else:
        matches = sorted(glob.glob(os.path.join(data_dir, f"{subject}-*.DATA_RAW.mat")))
        if not matches:
            raise FileNotFoundError(f"Neither {subject}.npy nor {subject}-*.DATA_RAW.mat found in {data_dir}")
        signal = loadmat(matches[0])["data"]
    signal = np.asarray(signal, dtype=np.float64)
    if signal.shape[0] > signal.shape[1]:  # ensure (C, T)
        signal = signal.T
    return signal


# --------------------------------------------------------------------------- #
# Preprocessing primitives
# --------------------------------------------------------------------------- #
def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter along the time axis."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal, axis=-1)


def rms_sliding(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Centered sliding-window root-mean-square envelope (same alignment as np.convolve(..., 'same'))."""
    window = np.ones((1, window_size)) / window_size
    mean_sq = fftconvolve(signal ** 2, window, mode="same", axes=-1)
    return np.sqrt(np.maximum(mean_sq, 0.0))


def remove_outlier_percentile(signal: np.ndarray, percentile: float) -> np.ndarray:
    """
    Remove time points at which ANY channel reaches or exceeds its own
    `percentile`-th value. The returned signal is shorter than the input.
    """
    thresholds = np.percentile(signal, percentile, axis=1, keepdims=True)
    keep = np.all(signal < thresholds, axis=0)
    return signal[:, keep]


def zscore(signal: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return (signal - signal.mean(axis=-1, keepdims=True)) / (signal.std(axis=-1, keepdims=True) + eps)


def exp_scale(signal: np.ndarray, gamma: float) -> np.ndarray:
    """Sign-preserving power scaling sign(x)*|x|^gamma; emphasises high-amplitude (contraction) regions."""
    return np.sign(signal) * np.abs(signal) ** gamma


def preprocess(subject: str, data_dir: str, param: PreprocessParam) -> np.ndarray:
    """Full preprocessing of one subject. Returns a z-scored RMS envelope (C, T')."""
    signal = load_subject_signal(subject, data_dir)
    crop = param.manual_crop.get(subject, 0)
    if crop:
        signal = signal[:, crop:]
    signal = bandpass_filter(signal, param.lowcut, param.highcut, param.fs, param.filter_order)
    signal = rms_sliding(signal, param.rms_window_size)
    signal = remove_outlier_percentile(signal, param.outlier_percentile)
    return zscore(signal)


# --------------------------------------------------------------------------- #
# Contraction extraction
# --------------------------------------------------------------------------- #
def extract_active_period(signal: np.ndarray, percentile: float = 50, eps: int = 4000,
                          period: str = "pre") -> Tuple[int, int]:
    """
    Split the session into pre- and post-intervention parts.

    The channel-mean envelope is thresholded at `percentile`; the longest gap
    between two supra-threshold bursts is taken as the intervention. Returns
    the (start, end) sample indices of the requested `period`.
    """
    envelope = signal.mean(axis=0)
    T = envelope.shape[0]
    active = envelope >= np.percentile(envelope, percentile)

    diff = np.diff(active.astype(int))
    start_index = np.where(diff == 1)[0]   # 0 -> 1 transitions (burst onsets)
    end_index = np.where(diff == -1)[0]    # 1 -> 0 transitions (burst offsets)
    if len(end_index) == 0:
        return 0, T

    max_gap, gap_end, gap_start = 0, end_index[0], 0
    for j in range(len(end_index)):
        if j + 1 >= len(start_index):
            break
        gap = start_index[j + 1] - end_index[j]
        if gap > max_gap:
            max_gap, gap_end, gap_start = gap, end_index[j], start_index[j + 1]

    if period == "pre":
        return 0, min(T, gap_end + eps)
    if period == "post":
        return max(0, gap_start - eps), T
    raise ValueError(f"Unknown period '{period}'")


def extract_contraction_mask(signal: np.ndarray, param: ContractionParam,
                             return_debug: bool = False):
    """
    Detect contraction periods and return a binary mask with the same length as `signal`.

    If `return_debug` is True, also returns a dict with the intermediate envelope,
    detected peaks and their widths (useful for plotting).
    """
    T_full = signal.shape[1]
    start, end = 0, T_full
    if param.period in ("pre", "post"):
        start, end = extract_active_period(signal, param.active_percentile, param.active_extension, param.period)
    elif param.period != "all":
        raise ValueError(f"Unknown period '{param.period}'")

    envelope = signal[:, start:end].mean(axis=0)
    envelope = exp_scale(envelope, param.exp_scale_gamma)
    envelope = np.convolve(envelope, np.ones(param.window_size) / param.window_size, mode="same")

    min_height = np.percentile(envelope, param.peak_percentile)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "some peaks have a width of 0" is expected and harmless
        peaks, _ = find_peaks(envelope, height=min_height, width=param.peak_min_width)
    widths = peak_widths(envelope, peaks, rel_height=param.peak_rel_height, wlen=param.peak_wlen)
    left_ips, right_ips = widths[2], widths[3]

    mask = np.zeros(T_full, dtype=bool)
    for left, right in zip(left_ips, right_ips):
        mask[start + max(int(left) - 1, 0): start + int(right) + 1] = True

    if return_debug:
        return mask, dict(start=start, end=end, envelope=envelope, peaks=peaks,
                          widths=widths, min_height=min_height)
    return mask


def extract_segments_from_mask(signal: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
    """Cut `signal` (C, T) into the contiguous runs where `mask` is True."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    edges = np.diff(m)
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [signal[:, s:e] for s, e in zip(starts, ends)]


def plot_contraction_mask(signal: np.ndarray, mask: np.ndarray, debug: dict,
                          title: str = "", save_path: Optional[str] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s, e = debug["start"], debug["end"]
    mean_sig = signal[:, s:e].mean(axis=0)
    env = debug["envelope"]
    x = np.arange(len(mean_sig))
    seg_mask = mask[s:e]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(mean_sig, lw=0.5, label="Preprocessed signal (channel mean)")
    axs[0].fill_between(x, mean_sig.min(), mean_sig.max(), where=seg_mask, color="red", alpha=0.3, label="Contraction mask")
    axs[0].set_title("Preprocessed signal")
    axs[1].plot(env, lw=0.8, label="Scaled & smoothed envelope")
    axs[1].plot(debug["peaks"], env[debug["peaks"]], "x", label="Peaks")
    axs[1].hlines(debug["min_height"], 0, len(env), colors="k", linestyles="--", alpha=0.5, label="Min. peak height")
    axs[1].fill_between(x, env.min(), env.max(), where=seg_mask, color="red", alpha=0.3)
    axs[1].set_title("Automatic contraction segmentation")
    for ax in axs:
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("Sample")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
