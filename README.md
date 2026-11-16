# Can Synthetic Hands Teach a Model to Read ASL Numbers?

CS 6470 project. This project trains image classifiers on synthetic hands (Lexset) and tests them on real photos (Mavi 2020, plus my own photos) of the ASL numbers 1–10. It answers two questions:

1. **The reality gap:** how much accuracy is lost going from synthetic training to real test photos?
2. **The fix:** how many real photos must be added to catch up to a model trained on real photos only?

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh      # needs ~/.kaggle/kaggle.json
python src/viz_samples.py          # look at results/sample_grid.png before training
```

Put your own photos in `data/own/<person>/<number>/*.jpg`. For example: `data/own/kyungwoo/7/001.jpg`.

To train on a GPU, use `notebooks/colab.ipynb`.

## Experiments

| # | What | Command |
|---|------|---------|
| 1 | synthetic → synthetic (sanity check) | `python src/train.py --train synth --test synth` |
| 2 | synthetic → real (the gap) | `python src/train.py --train synth --test real` |
| 3 | real → real (target) | `python src/train.py --train real --test real` |
| 4 | synthetic + n real → real | `python src/train.py --train synth+real --n-real 50 --real-repeat 5 --test real` |
| 5 | augmented synthetic → real | `python src/train.py --train synth --test real --augment` |
| 6 | hand keypoints instead of pixels | `python src/keypoints.py extract --source lexset` (and `mavi`, `own`), then `python src/keypoints.py train --train synth --test real` |

`bash scripts/run_all.sh exp4` runs one experiment block with 3 seeds. Then run `python src/plot_results.py`. It prints the two answers and saves the main figure.

Every run adds one row to `results/results.csv` and saves a confusion matrix to `results/confusion/`.

## Data notes

- **Classes:** Mavi only covers 0–9, so the default is `--classes 1-9`. Number 10 is only tested on my own photos (`--classes 1-10`).
- **Fixed real test set:** the split is decided by file names alone, so it is the same on every machine and every run.
- **Splitting my own photos:** they are split **by person**. Use `--test-people <name>` to choose who is held out.
- **Signer overlap in Mavi:** Mavi has no signer ids, so the same signer can appear in both train and test. This makes the real-only baseline look better than it is, and it is reported as a limitation.
- **Mavi file format:** use Mavi's GitHub image folders, not the Kaggle `.npy` files, whose labels are out of order.

## Layout

```
src/asl_data.py      dataset indexing, fixed splits, nested subsampling, results CSV
src/train.py         ResNet-18 training/evaluation (experiments 1–5)
src/keypoints.py     MediaPipe keypoint extraction + MLP classifier (experiment 6)
src/viz_samples.py   side-by-side sample grid for checking handshapes
src/plot_results.py  gap / photos-needed summary + figure
scripts/             download and batch-run helpers
notebooks/colab.ipynb
```
