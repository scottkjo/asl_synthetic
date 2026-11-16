"""Index the raw image datasets into simple (path, label) records and split them.

No torch here on purpose, so this file runs and can be tested on any machine.
Labels are the ASL number itself (1..10), not a class index.

Run `python src/asl_data.py` to print how many images each source has per number.
Do this right after downloading, before training anything.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DEFAULT_ROOTS = {
    "lexset": DATA / "lexset",           # Kaggle: lexset/synthetic-asl-numbers
    "mavi": DATA / "mavi" / "Dataset",   # GitHub: ardamavi/Sign-Language-Digits-Dataset
    "own": DATA / "own",                 # data/own/<person>/<number>/*.jpg
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# The real test split must never change between runs, so it uses its own fixed seed.
SPLIT_SEED = 0


@dataclass(frozen=True)
class Record:
    path: str        # absolute path on this machine
    key: str         # "<source>/<path relative to the source root>", the same on every machine
    label: int       # the number being signed, 0..10
    source: str      # lexset | mavi | own
    group: str       # person id for your own photos, "" when unknown
    split_hint: str  # "train"/"test" if the folder names say so, else ""


def parse_classes(spec: str) -> list[int]:
    """'1-9' -> [1, ..., 9]. '1,2,6-9' also works."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def _label_from_name(name: str) -> int | None:
    n = name.strip().lower()
    if n.isdigit() and 0 <= int(n) <= 10:
        return int(n)
    return NUMBER_WORDS.get(n)


def scan(root: Path | str, source: str, group_from_top: bool = False) -> list[Record]:
    """Find every image under root whose folder path names a number (e.g. .../7/img.png).

    The label is the closest parent folder named 0-10 or zero-ten.
    group_from_top: use the first folder under root as the person id (for data/own).
    """
    root = Path(root)
    if not root.exists():
        return []
    records = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS or not p.is_file():
            continue
        rel = p.relative_to(root)
        folders = rel.parts[:-1]
        label = None
        for part in reversed(folders):
            label = _label_from_name(part)
            if label is not None:
                break
        if label is None:
            continue
        lowered = [f.lower() for f in folders]
        if any("test" in f for f in lowered):
            hint = "test"
        elif any("train" in f for f in lowered):
            hint = "train"
        else:
            hint = ""
        group = folders[0] if group_from_top and folders else ""
        records.append(Record(str(p), f"{source}/{rel.as_posix()}", label, source, group, hint))
    return records


def _unit_hash(key: str, seed: int = SPLIT_SEED) -> float:
    """Stable pseudo-random number in [0, 1) from a string. Same on every machine."""
    return int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0x100000000


def split_synthetic(records: list[Record], test_frac: float = 0.1):
    """Use the dataset's own train/test folders if it has them, otherwise a fixed hash split."""
    if any(r.split_hint == "test" for r in records):
        return ([r for r in records if r.split_hint != "test"],
                [r for r in records if r.split_hint == "test"])
    test = [r for r in records if _unit_hash(r.key) < test_frac]
    held = {r.key for r in test}
    return [r for r in records if r.key not in held], test


def split_real(records: list[Record], test_frac: float = 0.2, test_people: list[str] | None = None):
    """Fixed real-photo split, used as the one held-out real test set for every experiment.

    - Mavi photos: per-image split. The dataset has no signer ids, so the same person can
      appear in train and test. That inflates the real-only baseline; say so in the report.
    - Your own photos: split by person, so nobody is in both train and test.
      Pass test_people=["alex"] to choose who is held out (recommended with only 3-5 people).

    The split depends only on file names, so it is the same on your Mac and on Colab,
    and adding new photos later never moves existing photos between train and test.
    """
    train, test = [], []
    for r in records:
        if r.group:
            held = (r.group in test_people) if test_people else _unit_hash("person:" + r.group) < test_frac
        else:
            held = _unit_hash(r.key) < test_frac
        (test if held else train).append(r)
    return train, test


def load_synthetic(classes: list[int], root: Path | str | None = None):
    recs = [r for r in scan(root or DEFAULT_ROOTS["lexset"], "lexset") if r.label in classes]
    return split_synthetic(recs)


def load_real(classes: list[int], test_people: list[str] | None = None,
              mavi_root: Path | str | None = None, own_root: Path | str | None = None):
    recs = scan(mavi_root or DEFAULT_ROOTS["mavi"], "mavi")
    recs += scan(own_root or DEFAULT_ROOTS["own"], "own", group_from_top=True)
    return split_real([r for r in recs if r.label in classes], test_people=test_people)


def subsample(records: list[Record], n: int | None, seed: int) -> list[Record]:
    """Pick n images spread evenly over the numbers.

    Subsets are nested: with the same seed, the 10-image set is inside the 50-image set,
    which is inside the 100-image set. That makes the "how many real photos" curve smooth.
    """
    if n is None or n >= len(records):
        return list(records)
    by_label: dict[int, list[Record]] = defaultdict(list)
    for r in sorted(records, key=lambda r: r.key):
        by_label[r.label].append(r)
    rng = random.Random(seed)
    labels = sorted(by_label)
    for lab in labels:
        rng.shuffle(by_label[lab])
    out: list[Record] = []
    i = 0
    while len(out) < n:
        added = False
        for lab in labels:
            if i < len(by_label[lab]) and len(out) < n:
                out.append(by_label[lab][i])
                added = True
        if not added:
            break
        i += 1
    return out


RESULT_COLUMNS = ["timestamp", "run", "model", "train", "test", "n_real", "augment", "classes",
                  "seed", "epochs", "n_train", "n_test", "accuracy", "per_class"]


def append_result(row: dict, path: Path = RESULTS / "results.csv") -> None:
    """Every run adds one row here. The plots and report tables are made from this file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def summarize(classes: list[int], test_people: list[str] | None = None) -> None:
    synth_tr, synth_te = load_synthetic(classes)
    real_tr, real_te = load_real(classes, test_people=test_people)
    table = {
        "synthetic train": synth_tr, "synthetic test": synth_te,
        "real train": real_tr, "real test": real_te,
    }
    print(f"{'':16}" + "".join(f"{c:>6}" for c in classes) + f"{'total':>8}")
    for name, recs in table.items():
        counts = Counter(r.label for r in recs)
        print(f"{name:16}" + "".join(f"{counts.get(c, 0):>6}" for c in classes) + f"{len(recs):>8}")
    people = sorted({r.group for r in real_tr + real_te if r.group})
    if people:
        held = sorted({r.group for r in real_te if r.group})
        print(f"\nyour own photos: people={people}, held out for test={held}")
    for name, root in DEFAULT_ROOTS.items():
        if not Path(root).exists():
            print(f"(missing: {root})")
    if not synth_tr:
        print("\nNo synthetic images found. Check that data/lexset has folders named by number "
              "(e.g. .../7/...). If they are named differently, adjust _label_from_name().")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Count images per number in each split.")
    ap.add_argument("--classes", default="1-10")
    ap.add_argument("--test-people", nargs="*", default=None)
    args = ap.parse_args()
    summarize(parse_classes(args.classes), args.test_people)
