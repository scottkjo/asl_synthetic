#!/usr/bin/env bash
# Download both public datasets into data/, then print image counts per number.
# Needs a Kaggle API token at ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/own

# 1) Synthetic hands (Lexset), numbers 1-10
if [ ! -d data/lexset ]; then
  kaggle datasets download -d lexset/synthetic-asl-numbers -p data/lexset --unzip
fi

# 2) Real photos (Mavi), numbers 0-9.
#    Use the GitHub image folders, NOT the Kaggle X.npy/Y.npy version: its labels are out of order.
if [ ! -d data/mavi ]; then
  git clone --depth 1 https://github.com/ardamavi/Sign-Language-Digits-Dataset.git data/mavi
fi

python src/asl_data.py
