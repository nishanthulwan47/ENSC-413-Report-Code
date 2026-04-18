from __future__ import annotations
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from galaxy_vit.data.dataset import GalaxyZooDataset, split_indices
from galaxy_vit.data.transforms import build_eval_transform, build_train_transform


def build_dataloaders(cfg):
    ds_cfg = cfg['dataset']
    data_root = Path(ds_cfg['data_root'])
    csv_path = data_root / ds_cfg['csv_name']
    image_dir = data_root / ds_cfg['train_dir_name']

    full_train = GalaxyZooDataset(csv_path, image_dir, transform=build_train_transform(ds_cfg['image_size']), threshold=ds_cfg['threshold'])
    full_eval = GalaxyZooDataset(csv_path, image_dir, transform=build_eval_transform(ds_cfg['image_size']), threshold=ds_cfg['threshold'])

    train_idx, val_idx, test_idx = split_indices(len(full_train), ds_cfg['val_ratio'], ds_cfg['test_ratio'], cfg['seed'])

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_eval, val_idx)
    test_ds = Subset(full_eval, test_idx)

    common = dict(batch_size=ds_cfg['batch_size'], num_workers=ds_cfg['num_workers'], pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader
