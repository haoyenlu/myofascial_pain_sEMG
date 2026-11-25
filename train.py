"""
Step 2 - Subject-level K-fold training of the CNN1D classifier (healthy vs. MTrP).

Each contraction segment is resampled to a fixed length and classified; all
segments of a subject share the subject's label.

Example
-------
    python train.py --loss pairwise --segment_dir data/segments_pre_800 --labels labels.csv --runs 5
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from emgnet.augmentation import default_augmentation
from emgnet.data import ContractionSegmentDataset, list_subjects, load_labels
from emgnet.losses import get_criterion
from emgnet.metrics import aggregate_by_subject, compute_metrics
from emgnet.models import build_model
from emgnet.utils import get_device, save_checkpoint, set_seed, subject_kfold


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segment_dir", required=True)
    p.add_argument("--labels", required=True, help="CSV with columns `subject` and `Type` (Healthy / Latent / Active)")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--run_name", default=None, help="Default: derived from data / seed / folds / loss")
    p.add_argument("--subjects", nargs="*", default=None, help="Default: all subjects in segment_dir (with a label)")
    p.add_argument("--exclude", nargs="*", default=[])
    # data / model
    p.add_argument("--segment_length", type=int, default=8000, help="Each segment is resampled to this many samples")
    p.add_argument("--in_channels", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--augment", action="store_true", help="Enable training-time augmentation (off in the paper)")
    # optimisation
    p.add_argument("--loss", choices=["pairwise", "bce", "combine"], default="pairwise")
    p.add_argument("--loss_weights", type=float, nargs=2, default=[0.5, 0.5], help="(pairwise, bce) weights for `combine`")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=None)
    # cross-validation
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=121, help="Seed of the subject-level K-fold split")
    p.add_argument("--runs", type=int, default=5, help="Repeated trainings with different initialisations")
    p.add_argument("--start_run", type=int, default=1)
    p.add_argument("--folds", type=int, nargs="*", default=None, help="Train only these folds (0-based)")
    p.add_argument("--selection", choices=["inner_val", "test_fold"], default="test_fold",
                   help="Data used to pick the best epoch. `test_fold` reproduces the original study "
                        "(monitors the held-out fold); `inner_val` holds out a stratified subset of "
                        "training subjects instead (no test-set information used).")
    p.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of training subjects for `inner_val`")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def predict(model, loader, device):
    model.eval()
    probs, trues, subjects = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["segment"].to(device)).squeeze(1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            trues.append(batch["target"].numpy())
            subjects.extend(batch["subject"])
    return np.concatenate(trues), np.concatenate(probs), np.array(subjects)


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc=desc, leave=False):
            x, y = batch["segment"].to(device), batch["target"].to(device)
            loss = criterion(model(x).squeeze(1), y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)


def split_monitor(train_subjects, test_subjects, labels, args, seed):
    """Return (fit_subjects, monitor_subjects) according to --selection."""
    if args.selection == "test_fold":
        return train_subjects, test_subjects
    strat = [labels[s] for s in train_subjects]
    fit, val = train_test_split(train_subjects, test_size=args.val_fraction, random_state=seed, stratify=strat)
    return list(fit), list(val)


def main():
    args = parse_args()
    device = get_device(args.device)
    labels = load_labels(args.labels)

    subjects = args.subjects or list_subjects(args.segment_dir)
    subjects = [s for s in subjects if s not in set(args.exclude)]
    missing = [s for s in subjects if s not in labels]
    if missing:
        print(f"Skipping subjects without label: {missing}")
    subjects = [s for s in subjects if s in labels]
    print(f"{len(subjects)} subjects: {subjects}")

    seg_name = os.path.basename(os.path.normpath(args.segment_dir))
    run_name = args.run_name or (f"cnn1d_{seg_name}_{args.seed}_{args.n_splits}"
                                 f"_{'aug' if args.augment else 'noaug'}_loss_{args.loss}")
    root = os.path.join(args.output_dir, run_name)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(dict(vars(args), subjects=subjects), f, indent=2)
    print(f"Output: {root}")

    augmentation = default_augmentation() if args.augment else None

    def make_dataset(s, aug=None):
        return ContractionSegmentDataset(s, labels, args.segment_dir, args.segment_length, augmentation=aug)

    def make_loader(ds, shuffle):
        return DataLoader(ds, args.batch_size, shuffle=shuffle, num_workers=args.num_workers)

    folds = subject_kfold(subjects, args.n_splits, args.seed)
    for run in range(args.start_run, args.start_run + args.runs):
        for k, split in folds.items():
            if args.folds is not None and k not in args.folds:
                continue
            set_seed(args.seed * 1000 + run * 10 + k)
            fold_dir = os.path.join(root, f"run_{run}", f"fold_{k}")
            fit_subj, mon_subj = split_monitor(split["train"], split["test"], labels, args, seed=args.seed + k)
            print(f"\n=== run {run} | fold {k + 1}/{args.n_splits} ===")
            print(f"train: {fit_subj}\nmonitor ({args.selection}): {mon_subj}\ntest: {split['test']}")

            criterion = get_criterion(args.loss, *args.loss_weights)
            train_loader = make_loader(make_dataset(fit_subj, augmentation), shuffle=True)
            mon_loader = make_loader(make_dataset(mon_subj), shuffle=False)

            model = build_model(args.in_channels, args.segment_length, args.dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

            best_loss, best_auc, history = float("inf"), -1.0, []
            for epoch in range(args.epochs):
                tr = run_epoch(model, train_loader, criterion, device, optimizer, f"epoch {epoch} train")
                va = run_epoch(model, mon_loader, criterion, device, None, f"epoch {epoch} monitor")
                record = dict(epoch=epoch, train_loss=tr, monitor_loss=va)
                if va < best_loss:
                    best_loss = va
                    # Original criterion: when monitor loss improves, keep the model if monitor AUC improves.
                    y, p, _ = predict(model, mon_loader, device)
                    auc = compute_metrics(y, p)["AUC"]
                    record["monitor_auc"] = auc
                    if auc > best_auc:
                        best_auc = auc
                        save_checkpoint(model, os.path.join(fold_dir, "checkpoint_best.pth"), epoch=epoch)
                history.append(record)
                print(" | ".join(f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}" for a, b in record.items()))

            with open(os.path.join(fold_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)

            model.load_state_dict(torch.load(os.path.join(fold_dir, "checkpoint_best.pth"), map_location=device)["model"])
            y, p, subj = predict(model, make_loader(make_dataset(split["test"]), shuffle=False), device)
            np.savez(os.path.join(fold_dir, "predictions.npz"), y_true=y, y_prob=p, subject=subj)
            seg_m = compute_metrics(y, p)
            sub_m = compute_metrics(*aggregate_by_subject(y, p, subj)[:2])
            with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
                json.dump(dict(segment=seg_m, subject=sub_m), f, indent=2)
            print(f"fold {k} test: segment AUC={seg_m['AUC']:.3f} | subject AUC={sub_m['AUC']:.3f}")


if __name__ == "__main__":
    main()
