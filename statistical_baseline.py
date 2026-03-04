"""
M5 — Baseline estadístico sin ML.

Sirve para comparar contra IsolationForest y demostrar que el ML aporta valor.
Dos métodos clásicos: 3-sigma y IQR (Interquartile Range).

Uso:
  from statistical_baseline import score_3sigma, score_iqr, evaluate_baseline
"""

import numpy as np
import pandas as pd
from typing import Optional


def score_3sigma(values: np.ndarray, train_values: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Marca como anomalía todo lo que está a más de 3 desviaciones estándar.

    Args:
        values:       array de valores a evaluar
        train_values: si se proporciona, calcula media/std sobre estos (train set)

    Returns:
        Array booleano: True = anomalía
    """
    ref = train_values if train_values is not None else values
    mean = np.nanmean(ref)
    std = np.nanstd(ref)
    if std == 0:
        return np.zeros(len(values), dtype=bool)
    return np.abs(values - mean) > 3 * std


def score_iqr(values: np.ndarray, train_values: Optional[np.ndarray] = None,
              multiplier: float = 1.5) -> np.ndarray:
    """
    Marca como anomalía usando IQR (Interquartile Range).
    Más robusto que 3-sigma frente a distribuciones no normales.

    Args:
        values:       array de valores a evaluar
        train_values: si se proporciona, calcula IQR sobre estos
        multiplier:   factor de IQR (1.5=estándar, 3.0=conservador)

    Returns:
        Array booleano: True = anomalía
    """
    ref = train_values if train_values is not None else values
    q1 = np.nanpercentile(ref, 25)
    q3 = np.nanpercentile(ref, 75)
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros(len(values), dtype=bool)
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (values < lower) | (values > upper)


def evaluate_baseline(train_values: np.ndarray, test_values: np.ndarray,
                      true_anomaly_mask: np.ndarray,
                      iqr_multiplier: float = 1.5) -> dict:
    """
    Evalúa ambos métodos (3-sigma e IQR) contra un ground truth.

    Args:
        train_values:      valores del train set (para calcular estadísticas)
        test_values:       valores del test set (donde están las anomalías inyectadas)
        true_anomaly_mask: booleano indicando posiciones de anomalías reales
        iqr_multiplier:    factor IQR (1.5=estándar, 2.0+= conservador)

    Returns:
        dict con métricas para cada método
    """
    results = {}
    for name, method in [("3sigma", score_3sigma), ("iqr", score_iqr)]:
        if name == "iqr":
            detected = method(test_values, train_values, multiplier=iqr_multiplier)
        else:
            detected = method(test_values, train_values)
        tp = int(np.sum(detected & true_anomaly_mask))
        fp = int(np.sum(detected & ~true_anomaly_mask))
        fn = int(np.sum(~detected & true_anomaly_mask))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
    return results
