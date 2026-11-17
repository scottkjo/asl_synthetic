#!/usr/bin/env bash
# Download both public datasets into data/, then print image counts per number.
# Needs a Kaggle API token at ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/own

# 1) Synthetic hands (Lexset), numbers 1-10
if [ ! -d data/lexset ]; then
  if [ ! -f ~/.kaggle/kaggle.json ] && [ -z "${KAGGLE_USERNAME:-}" ]; then
    echo "No Kaggle token found. On kaggle.com: your picture -> Settings -> API -> Create New Token,"
    echo "then: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    exit 1
  fi
  kaggle datasets download -d lexset/synthetic-asl-numbers -p data/lexset --unzip
fi

# 2) Real photos (Mavi), numbers 0-9.
#    Use the GitHub image folders, NOT the Kaggle X.npy/Y.npy version: its labels are out of order.
if [ ! -d data/mavi ]; then
  git clone --depth 1 https://github.com/ardamavi/Sign-Language-Digits-Dataset.git data/mavi
fi

python src/asl_data.py
