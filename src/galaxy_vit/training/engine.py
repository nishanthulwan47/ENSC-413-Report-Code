from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from galaxy_vit.training.metrics import compute_metrics


@dataclass
class EpochResult:
    loss: float
    metrics: Dict[str, float]


class Trainer:
    def __init__(self, model, optimizer, criterion, device, amp: bool = True, grad_clip_norm: float | None = None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.amp = amp and torch.cuda.is_available()
        self.grad_clip_norm = grad_clip_norm
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def _run_epoch(self, loader, train: bool) -> EpochResult:
        self.model.train(train)
        total_loss = 0.0
        y_true, y_pred = [], []
        pbar = tqdm(loader, leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=self.amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())
        avg_loss = total_loss / len(loader.dataset)
        metrics = compute_metrics(np.array(y_true), np.array(y_pred))
        return EpochResult(loss=avg_loss, metrics=metrics)

    def train_one_epoch(self, loader) -> EpochResult:
        return self._run_epoch(loader, train=True)

    def evaluate(self, loader) -> EpochResult:
        return self._run_epoch(loader, train=False)


def save_checkpoint(model, optimizer, epoch: int, path: str | Path, best_score: float) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_score': best_score,
    }, path)


def load_checkpoint(model, optimizer, path: str | Path, device: torch.device) -> Tuple[int, float]:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    return int(state.get('epoch', 0)), float(state.get('best_score', -1.0))
