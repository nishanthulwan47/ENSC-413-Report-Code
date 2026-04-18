from pathlib import Path

import pandas as pd
from PIL import Image

from galaxy_vit.data.dataset import GalaxyZooDataset


def test_dataset_reads_row(tmp_path: Path):
    img_dir = tmp_path / 'images'
    img_dir.mkdir()
    Image.new('RGB', (32, 32), color=(120, 80, 50)).save(img_dir / '1.jpg')
    df = pd.DataFrame([
        {'GalaxyID': 1, 'Class1.1': 0.9, 'Class1.2': 0.05, 'Class1.3': 0.03, 'Class11.1': 0.02}
    ])
    csv_path = tmp_path / 'labels.csv'
    df.to_csv(csv_path, index=False)
    ds = GalaxyZooDataset(csv_path, img_dir)
    _, label = ds[0]
    assert label == 0
