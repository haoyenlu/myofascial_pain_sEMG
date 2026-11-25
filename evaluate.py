"""
Step 3 - Evaluate trained classifiers across folds and repeated runs.

Re-runs inference with each fold's best checkpoint on that fold's test subjects and
reports, at both the segment level and the subject level (mean segment probability
per subject):

  * AUC per (run, fold), with the mean, a Student-t 95 % CI and a bootstrap 95 % CI
  * accuracy / F1 / MCC at the Youden-optimal threshold
  * the mean ROC curve

Example
-------
    python evaluate.py --checkpoint_dir outputs/cnn1d_segments_pre_800_121_5_noaug_loss_pairwise \
                       --segment_dir data/segments_pre_800 --labels labels.csv

`checkpoint_dir` may contain `run_*/fold_*/checkpoint_best.pth` or `fold_*/checkpoint_best.pth`.
Settings are read from `<checkpoint_dir>/config.json` when present; otherwise pass
--segment_length / --seed / --n_splits / --subjects explicitly.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from emgnet.data import ContractionSegmentDataset, list_subjects, load_labels
from emgnet.metrics import aggregate_by_subject, ci_bootstrap, ci_t, compute_metrics, interpolated_roc
from emgnet.models import build_model
from emgnet.utils import find_fold_checkpoint, get_device, subject_kfold


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--segment_dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--segment_length", type=int, default=None)
    p.add_argument("--in_channels", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n_splits", type=int, default=None)
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None, help="Output folder (default: <checkpoint_dir>/evaluation)")
    return p.parse_args()


def resolve(args):
    cfg = {}
    cfg_path = os.path.join(args.checkpoint_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    defaults = dict(segment_length=8000, in_channels=64, seed=121, n_splits=5, subjects=None)
    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, cfg.get(key, default))
    return args


def plot_roc(tprs, grid, auc, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    for tpr in tprs:
        ax.plot(grid, tpr, color="tab:blue", alpha=0.25, lw=1)
    ax.plot(grid, np.mean(tprs, axis=0), color="black", lw=2, label=f"Mean ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = resolve(parse_args())
    device = get_device(args.device)
    labels = load_labels(args.labels)
    subjects = args.subjects or [s for s in list_subjects(args.segment_dir) if s in labels]
    folds = subject_kfold(subjects, args.n_splits, args.seed)

    run_dirs = sorted(glob.glob(os.path.join(args.checkpoint_dir, "run_*")),
                      key=lambda d: int(d.rsplit("_", 1)[-1])) or [args.checkpoint_dir]
    out_dir = args.output or os.path.join(args.checkpoint_dir, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    datasets = {k: ContractionSegmentDataset(split["test"], labels, args.segment_dir, args.segment_length)
                for k, split in folds.items()}
    grid = np.linspace(0, 1, 100)
    rows, tprs = [], {"segment": [], "subject": []}
    for run_dir in run_dirs:
        run = os.path.basename(run_dir) if run_dir != args.checkpoint_dir else "run_1"
        for k in folds:
            model = build_model(args.in_channels, args.segment_length)
            model.load_state_dict(torch.load(find_fold_checkpoint(run_dir, k), map_location="cpu")["model"])
            model.to(device).eval()

            probs, trues, subj = [], [], []
            with torch.no_grad():
                for batch in DataLoader(datasets[k], args.batch_size, shuffle=False, num_workers=args.num_workers):
                    probs.append(torch.sigmoid(model(batch["segment"].to(device)).squeeze(1)).cpu().numpy())
                    trues.append(batch["target"].numpy())
                    subj.extend(batch["subject"])
            y, p = np.concatenate(trues), np.concatenate(probs)
            ys, ps, _ = aggregate_by_subject(y, p, subj)

            for level, (yt, yp) in {"segment": (y, p), "subject": (ys, ps)}.items():
                m = compute_metrics(yt, yp)
                rows.append(dict(level=level, run=run, fold=k, n=len(yt), **m))
                if not np.isnan(m["AUC"]):
                    tprs[level].append(interpolated_roc(yt, yp, grid))
            print(f"{run} fold {k}: segment AUC={rows[-2]['AUC']:.3f} | subject AUC={rows[-1]['AUC']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "per_fold_metrics.csv"), index=False)

    summary = {}
    for level in ("segment", "subject"):
        d = df[df.level == level]
        summary[level] = {}
        print(f"\n{level.capitalize()}-level, {len(d)} (run, fold) splits: mean [t 95% CI] [bootstrap 95% CI]")
        for key in ("AUC", "ACC", "F1", "MCC"):
            mean, lo, hi = ci_t(d[key].values)
            blo, bhi = ci_bootstrap(d[key].values)
            summary[level][key] = dict(mean=mean, ci_t=[lo, hi], ci_bootstrap=[blo, bhi])
            print(f"  {key}: {mean:.3f} [{lo:.3f}, {hi:.3f}] [{blo:.3f}, {bhi:.3f}]")
        plot_roc(tprs[level], grid, summary[level]["AUC"]["mean"],
                 os.path.join(out_dir, f"roc_{level}.png"), f"{level.capitalize()}-level ROC")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
