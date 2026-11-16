"""Week 1 sanity check: show a few images of each number from each dataset side by side.

  python src/viz_samples.py  -> results/sample_grid.png

Look at it before training. Each row should show the SAME handshape in every column,
especially 6, 7, 8 and 9 (which finger touches the thumb). If the real photos for a number
use a different handshape than the synthetic ones, fix the label mapping first.
"""
from __future__ import annotations

import argparse
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import asl_data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="1-10")
    ap.add_argument("--per-source", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    classes = D.parse_classes(args.classes)
    sources = {
        "synthetic": D.scan(D.DEFAULT_ROOTS["lexset"], "lexset"),
        "mavi": D.scan(D.DEFAULT_ROOTS["mavi"], "mavi"),
        "own": D.scan(D.DEFAULT_ROOTS["own"], "own", group_from_top=True),
    }
    sources = {k: v for k, v in sources.items() if v}
    if not sources:
        raise SystemExit("No images found. Run scripts/download_data.sh first.")

    k = args.per_source
    rng = random.Random(args.seed)
    fig, axes = plt.subplots(len(classes), k * len(sources),
                             figsize=(1.3 * k * len(sources), 1.3 * len(classes)), squeeze=False)
    for row, c in enumerate(classes):
        for s_i, (name, recs) in enumerate(sources.items()):
            pool = [r for r in recs if r.label == c]
            picks = rng.sample(pool, min(k, len(pool)))
            for j in range(k):
                ax = axes[row][s_i * k + j]
                ax.axis("off")
                if j < len(picks):
                    ax.imshow(Image.open(picks[j].path).convert("RGB"))
                if row == 0 and j == 0:
                    ax.set_title(name, fontsize=9, loc="left")
        axes[row][0].text(-0.15, 0.5, str(c), transform=axes[row][0].transAxes,
                          ha="right", va="center", fontsize=12)
    fig.tight_layout()
    D.RESULTS.mkdir(parents=True, exist_ok=True)
    out = D.RESULTS / "sample_grid.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
