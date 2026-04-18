from __future__ import annotations
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }


def classification_details(y_true: np.ndarray, y_pred: np.ndarray):
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def confusion(y_true: np.ndarray, y_pred: np.ndarray):
    return confusion_matrix(y_true, y_pred)
