from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from galaxy_vit.data.dataset import GalaxyZooDataset
from galaxy_vit.data.transforms import build_eval_transform
from galaxy_vit.models.vit import GalaxyViT
from galaxy_vit.training.engine import load_checkpoint
from galaxy_vit.utils.config import load_config
from galaxy_vit.utils.io import ensure_dir


def unnormalize(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = x.transpose(1, 2, 0)
    x = x * std + mean
    return np.clip(x, 0, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-samples', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ds_cfg = cfg['dataset']
    data_root = Path(ds_cfg['data_root'])
    dataset = GalaxyZooDataset(
        data_root / ds_cfg['csv_name'],
        data_root / ds_cfg['train_dir_name'],
        transform=build_eval_transform(ds_cfg['image_size']),
        threshold=ds_cfg['threshold'],
    )
    loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GalaxyViT(cfg['model']['name'], cfg['model']['num_classes'], pretrained=False, dropout=cfg['model']['dropout']).to(device)
    load_checkpoint(model, None, args.checkpoint, device)
    model.eval()

    images, labels = next(iter(loader))
    with torch.no_grad():
        preds = torch.argmax(model(images.to(device)), dim=1).cpu().numpy()

    out_dir = ensure_dir(Path(cfg['output_dir']) / 'figures')
    class_names = cfg['dataset']['classes']
    cols = 4
    rows = int(np.ceil(args.num_samples / cols))
    plt.figure(figsize=(14, 3 * rows))
    for i in range(args.num_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(unnormalize(images[i].numpy()))
        plt.title(f'T: {class_names[int(labels[i])] }\nP: {class_names[int(preds[i])]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / 'sample_predictions.png')
    plt.close()


if __name__ == '__main__':
    main()
