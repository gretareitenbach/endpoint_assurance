from typing import Dict, Iterable, Tuple

import numpy as np


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    y_true = y_true.astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

    beta = 0.1
    beta2 = beta * beta
    f01 = ((1 + beta2) * precision * recall) / max(1e-8, beta2 * precision + recall)

    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f01": float(f01),
    }


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Iterable[float] = tuple(np.linspace(0.5, 0.999, 100)),
) -> Tuple[float, Dict[str, float]]:
    best_thr = 0.98
    best = {"f01": -1.0}
    for thr in thresholds:
        cur = binary_metrics(y_true, y_prob, threshold=float(thr))
        if cur["f01"] > best["f01"]:
            best = cur
            best_thr = float(thr)
    return best_thr, best
