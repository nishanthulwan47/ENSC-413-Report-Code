from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from torch import nn
from torch.optim import AdamW

from galaxy_vit.data.datamodule import build_dataloaders
from galaxy_vit.models.vit import GalaxyViT
from galaxy_vit.training.engine import Trainer, save_checkpoint
from galaxy_vit.utils.config import load_config
from galaxy_vit.utils.io import ensure_dir, save_json
from galaxy_vit.utils.seed import set_seed
from galaxy_vit.visualization.plots import plot_history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg['seed'])

    out_dir = ensure_dir(cfg['output_dir'])
    ckpt_dir = ensure_dir(out_dir / 'checkpoints')
    fig_dir = ensure_dir(out_dir / 'figures')
    metrics_dir = ensure_dir(out_dir / 'metrics')

    train_loader, val_loader, _ = build_dataloaders(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GalaxyViT(
        name=cfg['model']['name'],
        num_classes=cfg['model']['num_classes'],
        pretrained=cfg['model']['pretrained'],
        dropout=cfg['model']['dropout'],
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device, amp=cfg['train']['amp'], grad_clip_norm=cfg['train']['gradient_clip_norm'])

    best_f1 = -1.0
    patience = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1_macro': []}

    for epoch in range(1, cfg['train']['epochs'] + 1):
        train_result = trainer.train_one_epoch(train_loader)
        val_result = trainer.evaluate(val_loader)

        history['train_loss'].append(train_result.loss)
        history['val_loss'].append(val_result.loss)
        history['val_f1_macro'].append(val_result.metrics['f1_macro'])

        epoch_metrics = {
            'epoch': epoch,
            'train': {'loss': train_result.loss, **train_result.metrics},
            'val': {'loss': val_result.loss, **val_result.metrics},
        }
        save_json(epoch_metrics, metrics_dir / f'epoch_{epoch:03d}.json')
        print(epoch_metrics)

        if val_result.metrics['f1_macro'] > best_f1:
            best_f1 = val_result.metrics['f1_macro']
            patience = 0
            save_checkpoint(model, optimizer, epoch, ckpt_dir / 'best.pt', best_f1)
        else:
            patience += 1
            if patience >= cfg['train']['early_stopping_patience']:
                print('Early stopping triggered.')
                break

    save_checkpoint(model, optimizer, epoch, ckpt_dir / 'last.pt', best_f1)
    plot_history(history, fig_dir / 'training_history.png')
    save_json(history, metrics_dir / 'history.json')


if __name__ == '__main__':
    main()
