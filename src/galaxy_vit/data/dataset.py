from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class GalaxyLabelMap:
    smooth_col: str = 'Class1.1'
    features_col: str = 'Class1.2'
    star_col: str = 'Class1.3'
    merger_col: str = 'Class11.1'


class GalaxyZooDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_dir: str | Path, transform=None, threshold: float = 0.5):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.threshold = threshold
        self.label_map = GalaxyLabelMap()
        self.df = pd.read_csv(self.csv_path)
        required = ['GalaxyID', self.label_map.smooth_col, self.label_map.features_col, self.label_map.star_col]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f'Missing required columns: {missing}')

    def __len__(self) -> int:
        return len(self.df)

    def _row_to_label(self, row: pd.Series) -> int:
        probs = {
            0: float(row.get(self.label_map.smooth_col, 0.0)),
            1: float(row.get(self.label_map.features_col, 0.0)),
            2: float(row.get(self.label_map.star_col, 0.0)),
            3: float(row.get(self.label_map.merger_col, 0.0)),
        }
        above = [k for k, v in probs.items() if v >= self.threshold]
        if above:
            return max(above, key=lambda k: probs[k])
        return max(probs, key=lambda k: probs[k])

    def __getitem__(self, index: int) -> Tuple[object, int]:
        row = self.df.iloc[index]
        image_path = self.image_dir / f"{int(row['GalaxyID'])}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f'Image not found: {image_path}')
        image = Image.open(image_path).convert('RGB')
        label = self._row_to_label(row)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    test_idx = idx[:test_n].tolist()
    val_idx = idx[test_n:test_n + val_n].tolist()
    train_idx = idx[test_n + val_n:].tolist()
    return train_idx, val_idx, test_idx
