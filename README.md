# Galaxy ViT Project

Production-style PyTorch project for Galaxy Zoo morphological classification using real Galaxy Zoo CSV labels and image folders.

## Expected dataset layout

```text
DATA_ROOT/
├── training_solutions_rev1.csv
├── images_training_rev1/
│   ├── 100008.jpg
│   ├── 100023.jpg
│   └── ...
└── images_test_rev1/
```

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py --config configs/base.yaml
```

## Evaluate

```bash
python scripts/evaluate.py --config configs/base.yaml --checkpoint outputs/checkpoints/best.pt
```

## Visualize predictions

```bash
python scripts/visualize.py --config configs/base.yaml --checkpoint outputs/checkpoints/best.pt
```

## Notes

- Reads directly from `training_solutions_rev1.csv`.
- Supports multiclass labels derived from Galaxy Zoo decision-tree probabilities.
- Uses a Vision Transformer backbone from timm.
- Saves metrics, confusion matrix, ROC curves, example predictions, and checkpoints.
