from __future__ import annotations
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], out_path: str | Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_history(history: dict, out_path: str | Path) -> None:
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.plot(epochs, history['val_f1_macro'], label='val_f1_macro')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
