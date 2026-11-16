"""Train a ResNet-18 on one image set, test on another. Each run adds a row to results/results.csv.

Experiments from the proposal:
  1  python src/train.py --train synth      --test synth
  2  python src/train.py --train synth      --test real
  3  python src/train.py --train real       --test real
  4  python src/train.py --train synth+real --test real --n-real 50
  5  python src/train.py --train synth      --test real --augment
  (6 is src/keypoints.py)

Quick check on a laptop CPU before using Colab:
  python src/train.py --train synth --test synth --max-train 200 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import asl_data as D

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet statistics


def build_transforms(augment: bool):
    if augment:  # Experiment 5: rotate, blur, crop, recolor the synthetic images
        steps = [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),  # a left-handed signer makes the same number
            transforms.RandomRotation(25),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
            transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.1, 3.0))], p=0.5),
        ]
    else:
        steps = [transforms.Resize((224, 224))]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class ImageSet(Dataset):
    def __init__(self, records, classes, tf):
        self.records = records
        self.index = {c: i for i, c in enumerate(classes)}
        self.tf = tf

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return self.tf(Image.open(r.path).convert("RGB")), self.index[r.label]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(n_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


@torch.no_grad()
def confusion(model, loader, device, n_classes):
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=int)  # rows = true, cols = predicted
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for t, p in zip(y.tolist(), pred.tolist()):
            cm[t, p] += 1
    return cm


def select_data(args, classes):
    synth_tr, synth_te = D.load_synthetic(classes)
    real_tr, real_te = D.load_real(classes, test_people=args.test_people)
    if args.max_train:
        synth_tr = D.subsample(synth_tr, args.max_train, args.seed)

    if args.train == "synth":
        train = synth_tr
    elif args.train == "real":
        train = D.subsample(real_tr, args.n_real, args.seed)
    else:  # synth+real
        if args.n_real is None:
            raise SystemExit("--train synth+real needs --n-real")
        train = synth_tr + D.subsample(real_tr, args.n_real, args.seed) * args.real_repeat
    test = synth_te if args.test == "synth" else real_te

    if not train or not test:
        raise SystemExit(f"Empty data (train={len(train)}, test={len(test)}). "
                         "Run scripts/download_data.sh, then python src/asl_data.py to check.")
    return train, test


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", choices=["synth", "real", "synth+real"], required=True)
    ap.add_argument("--test", choices=["synth", "real"], required=True)
    ap.add_argument("--n-real", type=int, default=None, help="number of real training photos (default: all)")
    ap.add_argument("--real-repeat", type=int, default=1,
                    help="repeat the real photos this many times so a few dozen are not drowned out by 9,000 synthetic ones")
    ap.add_argument("--augment", action="store_true", help="Experiment 5: rotation/blur/crop/color augmentation")
    ap.add_argument("--classes", default="1-9",
                    help="'1-9' when testing on Mavi (it has no 10); '1-10' once your own photos are in")
    ap.add_argument("--test-people", nargs="*", default=None, help="which of your own photo folders to hold out")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=None, help="cap synthetic training images (for quick tests)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--run", default=None, help="name for this run (default: built from the options)")
    args = ap.parse_args()

    classes = D.parse_classes(args.classes)
    seed_everything(args.seed)
    device = pick_device()
    run = args.run or (f"{args.train}-to-{args.test}"
                       f"-n{args.n_real if args.n_real is not None else 'all'}"
                       f"{'-aug' if args.augment else ''}-s{args.seed}")

    train_recs, test_recs = select_data(args, classes)
    print(f"[{run}] device={device} train={len(train_recs)} test={len(test_recs)} classes={classes}")

    pin = device.type == "cuda"
    train_loader = DataLoader(ImageSet(train_recs, classes, build_transforms(args.augment)),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(ImageSet(test_recs, classes, build_transforms(False)),
                             batch_size=args.batch_size, num_workers=args.workers, pin_memory=pin)

    model = build_model(len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total_loss, correct, seen = time.time(), 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            seen += len(y)
        print(f"  epoch {epoch}/{args.epochs}  loss={total_loss / seen:.4f}  "
              f"train_acc={correct / seen:.3f}  ({time.time() - t0:.0f}s)")

    cm = confusion(model, test_loader, device, len(classes))
    acc = cm.trace() / cm.sum()
    per_class = {str(c): (round(float(cm[i, i] / cm[i].sum()), 4) if cm[i].sum() else None)
                 for i, c in enumerate(classes)}
    print(f"[{run}] test accuracy = {acc:.4f}")
    print("  per number:", per_class)

    conf_dir = D.RESULTS / "confusion"
    conf_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(conf_dir / f"{run}.csv", cm, fmt="%d", delimiter=",",
               header=",".join(map(str, classes)), comments="")
    D.append_result({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "run": run, "model": "resnet18",
        "train": args.train, "test": args.test,
        "n_real": args.n_real if args.n_real is not None else "all",
        "augment": int(args.augment), "classes": args.classes, "seed": args.seed,
        "epochs": args.epochs, "n_train": len(train_recs), "n_test": len(test_recs),
        "accuracy": round(float(acc), 4), "per_class": json.dumps(per_class),
    })


if __name__ == "__main__":
    main()
