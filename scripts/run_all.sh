#!/usr/bin/env bash
# Run every image experiment with 3 seeds. On a Colab T4 this takes a few hours,
# so run one block at a time (e.g. `bash scripts/run_all.sh exp4`).
# Usage: bash scripts/run_all.sh [exp1|exp2|exp3|exp4|exp5|exp6|all]
set -euo pipefail
cd "$(dirname "$0")/.."

WHICH="${1:-all}"
SEEDS="0 1 2"
CLASSES="${CLASSES:-1-9}"   # set CLASSES=1-10 once your own photos (with 10s) are in data/own
COMMON="--classes $CLASSES"

run() { [ "$WHICH" = "all" ] || [ "$WHICH" = "$1" ]; }

if run exp1; then
  python src/train.py --train synth --test synth $COMMON
fi
if run exp2; then
  for s in $SEEDS; do python src/train.py --train synth --test real --seed "$s" $COMMON; done
fi
if run exp3; then
  for s in $SEEDS; do python src/train.py --train real --test real --seed "$s" $COMMON; done
  # also real-only with few photos, to compare against synthetic + few real
  for n in 10 25 50 100 250 500; do
    for s in $SEEDS; do python src/train.py --train real --test real --n-real "$n" --seed "$s" $COMMON; done
  done
fi
if run exp4; then
  for n in 10 25 50 100 250 500; do
    for s in $SEEDS; do
      python src/train.py --train synth+real --test real --n-real "$n" --real-repeat 5 --seed "$s" $COMMON
    done
  done
fi
if run exp5; then
  for s in $SEEDS; do python src/train.py --train synth --test real --augment --seed "$s" $COMMON; done
fi
if run exp6; then
  for src in lexset mavi own; do
    [ -f "data/keypoints/$src.npz" ] || python src/keypoints.py extract --source "$src" || true
  done
  for s in $SEEDS; do
    python src/keypoints.py train --train synth --test real --seed "$s" $COMMON
    python src/keypoints.py train --train real --test real --seed "$s" $COMMON
    for n in 10 50 100 500; do
      python src/keypoints.py train --train synth+real --test real --n-real "$n" --seed "$s" $COMMON
    done
  done
fi

python src/plot_results.py || true
