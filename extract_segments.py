"""
Step 1 - Preprocess raw HD-sEMG recordings and extract contraction segments.

Example
-------
    python extract_segments.py --data_dir data/raw --output_dir data/segments_pre_800 --plot

Input : <data_dir>/<subject>.npy  or  <data_dir>/<subject>-*.DATA_RAW.mat (variable `data`),
        shape (channels, samples)
Output: <output_dir>/<subject>/segment-<i>.npy (float64, shape (channels, samples_i))
        <output_dir>/extraction_summary.csv
        <output_dir>/figures/<subject>.png      (with --plot)
"""
import argparse
import glob
import json
import os
import re
from dataclasses import asdict

import numpy as np
import pandas as pd

from emgnet.signal import (ContractionParam, PreprocessParam, extract_contraction_mask,
                           extract_segments_from_mask, plot_contraction_mask, preprocess)


def discover_subjects(data_dir):
    subjects = set()
    for f in glob.glob(os.path.join(data_dir, "*.DATA_RAW.mat")) + glob.glob(os.path.join(data_dir, "*.npy")):
        m = re.match(r"(S\d+)(?:-|\.npy$)", os.path.basename(f))
        if m:
            subjects.add(m.group(1))
    return sorted(subjects)


def parse_args():
    pp, cp = PreprocessParam(), ContractionParam()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="Folder with <subject>.npy or <subject>-*.DATA_RAW.mat files")
    p.add_argument("--output_dir", required=True, help="Where segments are written")
    p.add_argument("--subjects", nargs="*", default=None, help="Subjects to process (default: all found)")
    p.add_argument("--exclude", nargs="*", default=[], help="Subjects to skip")
    p.add_argument("--plot", action="store_true", help="Save a diagnostic figure per subject")
    g = p.add_argument_group("preprocessing")
    g.add_argument("--fs", type=int, default=pp.fs)
    g.add_argument("--lowcut", type=float, default=pp.lowcut)
    g.add_argument("--highcut", type=float, default=pp.highcut)
    g.add_argument("--filter_order", type=int, default=pp.filter_order)
    g.add_argument("--rms_window", type=int, default=pp.rms_window_size)
    g.add_argument("--outlier_percentile", type=float, default=pp.outlier_percentile)
    g = p.add_argument_group("contraction detection")
    g.add_argument("--period", choices=["pre", "post", "all"], default=cp.period)
    g.add_argument("--smooth_window", type=int, default=cp.window_size)
    g.add_argument("--active_percentile", type=float, default=cp.active_percentile)
    g.add_argument("--active_extension", type=int, default=cp.active_extension)
    g.add_argument("--peak_percentile", type=float, default=cp.peak_percentile)
    g.add_argument("--peak_rel_height", type=float, default=cp.peak_rel_height)
    g.add_argument("--peak_min_width", type=int, default=cp.peak_min_width)
    g.add_argument("--peak_wlen", type=int, default=cp.peak_wlen)
    g.add_argument("--gamma", type=float, default=cp.exp_scale_gamma)
    return p.parse_args()


def main():
    args = parse_args()
    pparam = PreprocessParam(fs=args.fs, lowcut=args.lowcut, highcut=args.highcut,
                             filter_order=args.filter_order, rms_window_size=args.rms_window,
                             outlier_percentile=args.outlier_percentile)
    cparam = ContractionParam(fs=args.fs, window_size=args.smooth_window, period=args.period,
                              active_percentile=args.active_percentile, active_extension=args.active_extension,
                              peak_percentile=args.peak_percentile, peak_rel_height=args.peak_rel_height,
                              peak_min_width=args.peak_min_width, peak_wlen=args.peak_wlen,
                              exp_scale_gamma=args.gamma)

    subjects = args.subjects or discover_subjects(args.data_dir)
    subjects = [s for s in subjects if s not in set(args.exclude)]
    os.makedirs(args.output_dir, exist_ok=True)
    if args.plot:
        os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    with open(os.path.join(args.output_dir, "extraction_params.json"), "w") as f:
        json.dump(dict(preprocess=asdict(pparam), contraction=asdict(cparam)), f, indent=2)

    rows = []
    for i, subject in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] {subject}")
        signal = preprocess(subject, args.data_dir, pparam)
        mask, debug = extract_contraction_mask(signal, cparam, return_debug=True)
        segments = extract_segments_from_mask(signal, mask)

        subject_dir = os.path.join(args.output_dir, subject)
        os.makedirs(subject_dir, exist_ok=True)
        for old in glob.glob(os.path.join(subject_dir, "segment-*.npy")):
            os.remove(old)
        for j, seg in enumerate(segments):
            np.save(os.path.join(subject_dir, f"segment-{j}.npy"), seg)

        lengths = [s.shape[1] for s in segments]
        rows.append(dict(subject=subject, n_segments=len(segments),
                         total_seconds=sum(lengths) / args.fs,
                         mean_segment_seconds=(np.mean(lengths) / args.fs) if lengths else 0.0))
        print(f"    {len(segments)} segments, {sum(lengths) / args.fs:.1f} s of contraction")

        if args.plot:
            plot_contraction_mask(signal, mask, debug, title=subject,
                                  save_path=os.path.join(args.output_dir, "figures", f"{subject}.png"))

    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "extraction_summary.csv"), index=False)
    print(f"Done. Segments written to {args.output_dir}")


if __name__ == "__main__":
    main()
