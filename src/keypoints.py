"""Experiment 6: classify 21 hand keypoints (MediaPipe) instead of raw pixels.

Step 1, once per dataset (saves data/keypoints/<source>.npz):
  python src/keypoints.py extract --source lexset
  python src/keypoints.py extract --source mavi
  python src/keypoints.py extract --source own
Step 2, train a small classifier on the keypoints:
  python src/keypoints.py train --train synth      --test real
  python src/keypoints.py train --train synth+real --test real --n-real 50

Images where MediaPipe finds no hand count as wrong answers, so the accuracy is comparable
with the image model. How often detection fails on synthetic vs. real hands is worth reporting.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter

import numpy as np

import asl_data as D

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
KP_DIR = D.DATA / "keypoints"


def get_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    KP_DIR.mkdir(parents=True, exist_ok=True)
    model_path = KP_DIR / "hand_landmarker.task"
    if not model_path.exists():
        print("downloading MediaPipe hand model ...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )
    return mp, vision.HandLandmarker.create_from_options(opts)


def normalize(points: np.ndarray, handedness: str) -> np.ndarray:
    """Make keypoints independent of where the hand is, how big it is, and which hand it is."""
    pts = points.astype(np.float32).copy()  # (21, 3)
    pts -= pts[0]                            # wrist at the origin
    if handedness == "Left":
        pts[:, 0] *= -1                      # mirror left hands onto right hands
    scale = float(np.linalg.norm(pts[:, :2], axis=1).max()) or 1.0
    return (pts / scale).reshape(-1)         # 63 numbers


def extract(source: str):
    from PIL import Image

    recs = D.scan(D.DEFAULT_ROOTS[source], source, group_from_top=(source == "own"))
    if not recs:
        raise SystemExit(f"No images found for {source} under {D.DEFAULT_ROOTS[source]}")
    mp, landmarker = get_landmarker()
    X = np.zeros((len(recs), 63), dtype=np.float32)
    ok = np.zeros(len(recs), dtype=bool)
    for i, r in enumerate(recs):
        rgb = np.asarray(Image.open(r.path).convert("RGB"))
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if result.hand_landmarks:
            pts = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]])
            X[i] = normalize(pts, result.handedness[0][0].category_name)
            ok[i] = True
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(recs)}  detected so far: {ok[:i + 1].mean():.1%}")

    out = KP_DIR / f"{source}.npz"
    np.savez_compressed(
        out, X=X, ok=ok,
        label=np.array([r.label for r in recs]), key=np.array([r.key for r in recs]),
        path=np.array([r.path for r in recs]), group=np.array([r.group for r in recs]),
        split_hint=np.array([r.split_hint for r in recs]),
    )
    fails = Counter(r.label for r, good in zip(recs, ok) if not good)
    print(f"saved {out}: {ok.sum()}/{len(recs)} hands detected ({ok.mean():.1%})")
    print("  misses per number:", dict(sorted(fails.items())))


def load_kp(source: str):
    path = KP_DIR / f"{source}.npz"
    if not path.exists():
        return [], {}
    z = np.load(path)
    recs, feats = [], {}
    for i in range(len(z["label"])):
        r = D.Record(str(z["path"][i]), str(z["key"][i]), int(z["label"][i]), source,
                     str(z["group"][i]), str(z["split_hint"][i]))
        recs.append(r)
        feats[r.key] = z["X"][i] if z["ok"][i] else None
    return recs, feats


def train(args):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classes = D.parse_classes(args.classes)
    synth, f_synth = load_kp("lexset")
    mavi, f_mavi = load_kp("mavi")
    own, f_own = load_kp("own")
    feats = {**f_synth, **f_mavi, **f_own}

    synth_tr, synth_te = D.split_synthetic([r for r in synth if r.label in classes])
    real_tr, real_te = D.split_real([r for r in mavi + own if r.label in classes],
                                    test_people=args.test_people)
    if args.train == "synth":
        train_recs = synth_tr
    elif args.train == "real":
        train_recs = D.subsample(real_tr, args.n_real, args.seed)
    else:
        train_recs = synth_tr + D.subsample(real_tr, args.n_real, args.seed)
    test_recs = synth_te if args.test == "synth" else real_te
    if not train_recs or not test_recs:
        raise SystemExit("Missing keypoints. Run the extract step for each source first.")

    usable = [r for r in train_recs if feats[r.key] is not None]
    X = np.stack([feats[r.key] for r in usable])
    y = np.array([r.label for r in usable])
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=args.seed))
    clf.fit(X, y)

    correct = Counter()
    total = Counter(r.label for r in test_recs)
    detected = 0
    for r in test_recs:
        f = feats[r.key]
        if f is None:
            continue  # no hand found: counts as wrong
        detected += 1
        if clf.predict(f[None])[0] == r.label:
            correct[r.label] += 1
    acc = sum(correct.values()) / len(test_recs)
    acc_detected = sum(correct.values()) / max(detected, 1)
    per_class = {str(c): (round(correct[c] / total[c], 4) if total[c] else None) for c in classes}
    print(f"train={len(usable)} (of {len(train_recs)}), test={len(test_recs)}, hands found in test={detected}")
    print(f"accuracy (misses count as wrong) = {acc:.4f}; on detected hands only = {acc_detected:.4f}")
    print("per number:", per_class)

    run = args.run or f"kp-{args.train}-to-{args.test}-n{args.n_real if args.n_real is not None else 'all'}-s{args.seed}"
    D.append_result({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "run": run, "model": "keypoints-mlp",
        "train": args.train, "test": args.test,
        "n_real": args.n_real if args.n_real is not None else "all", "augment": 0,
        "classes": args.classes, "seed": args.seed, "epochs": "", "n_train": len(usable),
        "n_test": len(test_recs), "accuracy": round(acc, 4), "per_class": json.dumps(per_class),
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--source", choices=list(D.DEFAULT_ROOTS), required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", choices=["synth", "real", "synth+real"], required=True)
    t.add_argument("--test", choices=["synth", "real"], required=True)
    t.add_argument("--n-real", type=int, default=None)
    t.add_argument("--classes", default="1-9")
    t.add_argument("--test-people", nargs="*", default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--run", default=None)
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.source)
    else:
        if args.train == "synth+real" and args.n_real is None:
            raise SystemExit("--train synth+real needs --n-real")
        train(args)


if __name__ == "__main__":
    main()
