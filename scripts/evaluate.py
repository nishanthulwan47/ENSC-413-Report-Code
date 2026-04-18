from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
from torch.optim import AdamW
from torch import nn

from galaxy_vit.data.datamodule import build_dataloaders
from galaxy_vit.models.vit import GalaxyViT
from galaxy_vit.training.engine import Trainer, load_checkpoint
from galaxy_vit.training.metrics import classification_details, confusion
from galaxy_vit.utils.config import load_config
from galaxy_vit.utils.io import ensure_dir, save_json
from galaxy_vit.visualization.plots import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    return parser.parse_args()


def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg['output_dir'])
    eval_dir = ensure_dir(out_dir / 'evaluation')

    _, _, test_loader = build_dataloaders(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GalaxyViT(
        name=cfg['model']['name'],
        num_classes=cfg['model']['num_classes'],
        pretrained=False,
        dropout=cfg['model']['dropout'],
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg['train']['lr'])
    load_checkpoint(model, optimizer, args.checkpoint, device)

    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device, amp=False)
    test_result = trainer.evaluate(test_loader)
    y_true, y_pred = collect_predictions(model, test_loader, device)

    details = classification_details(y_true, y_pred)
    cm = confusion(y_true, y_pred)

    save_json({'test': {'loss': test_result.loss, **test_result.metrics}, 'details': details}, eval_dir / 'test_metrics.json')
    plot_confusion_matrix(cm, cfg['dataset']['classes'], eval_dir / 'confusion_matrix.png')
    print({'loss': test_result.loss, **test_result.metrics})


if __name__ == '__main__':
    main()
