"""Turn results/results.csv into the two answers the proposal asks for, plus the main figure.

  python src/plot_results.py            -> prints the gap and photos-needed, saves results/real_photos_curve.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import asl_data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18", help="resnet18 or keypoints-mlp")
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="'caught up' means within this much accuracy of the real-only model")
    args = ap.parse_args()

    df = pd.read_csv(D.RESULTS / "results.csv")
    df = df[(df.model == args.model) & (df.test == "real")].copy()
    df["n"] = pd.to_numeric(df.n_real, errors="coerce")  # "all" -> NaN

    synth_only = df[(df.train == "synth") & (df.augment == 0)].accuracy
    synth_aug = df[(df.train == "synth") & (df.augment == 1)].accuracy
    real_all = df[(df.train == "real") & df.n.isna()].accuracy
    mixed = df[df.train == "synth+real"].groupby("n").accuracy.agg(["mean", "std", "count"])
    real_few = df[(df.train == "real") & df.n.notna()].groupby("n").accuracy.agg(["mean", "std", "count"])

    def show(name, s):
        if len(s):
            print(f"{name:32} {s.mean():.3f} ± {s.std(ddof=0):.3f}  (n={len(s)} runs)")

    show("synthetic only -> real", synth_only)
    show("synthetic + augmentation -> real", synth_aug)
    show("real only (all photos) -> real", real_all)
    if len(synth_only) and len(real_all):
        print(f"\nAnswer 1, reality gap: {real_all.mean() - synth_only.mean():+.3f} accuracy")
        target = real_all.mean() - args.tolerance
        caught = mixed[mixed["mean"] >= target]
        if len(caught):
            print(f"Answer 2, real photos needed to catch up: {int(caught.index.min())}")
        else:
            print("Answer 2: synthetic + real never caught up within the tested sizes")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    if len(mixed):
        ax.errorbar(mixed.index, mixed["mean"], yerr=mixed["std"].fillna(0), marker="o",
                    capsize=3, label="synthetic + n real")
    if len(real_few):
        ax.errorbar(real_few.index, real_few["mean"], yerr=real_few["std"].fillna(0), marker="s",
                    capsize=3, linestyle="--", label="n real only")
    if len(real_all):
        ax.axhline(real_all.mean(), color="black", linewidth=1, label="all real photos")
    if len(synth_only):
        ax.axhline(synth_only.mean(), color="gray", linestyle=":", label="synthetic only")
    ax.set_xscale("log")
    ax.set_xlabel("real training photos (n)")
    ax.set_ylabel("accuracy on real test photos")
    ax.set_title(f"How many real photos close the gap? ({args.model})")
    ax.legend()
    fig.tight_layout()
    out = D.RESULTS / f"real_photos_curve_{args.model}.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
