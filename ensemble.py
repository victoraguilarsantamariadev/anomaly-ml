"""
Ensemble voting para combinar predicciones de M1 y M2.

M1 (temporal):       Precision=100%, Recall=40%  → cuando dice anomalía, siempre acierta
M2 (cross-sectional): Precision=56%, Recall=100% → detecta todo pero tiene falsos positivos

Combinación:
  Ambos detectan  → CRITICAL (máxima confianza)
  Solo M1 detecta → HIGH     (M1 tiene zero FPs)
  Solo M2 detecta → MEDIUM   (M2 podría ser FP)
  Ninguno detecta → NORMAL

Uso:
  from ensemble import ensemble_vote, ensemble_evaluate
"""

import numpy as np
from typing import Optional


def ensemble_vote(m1_anomaly: bool, m2_anomaly: bool,
                  m1_score: Optional[float] = None,
                  m2_score: Optional[float] = None) -> dict:
    """
    Combina las predicciones de M1 y M2 en un resultado con severidad.

    Args:
        m1_anomaly: True si M1 (temporal) detecta anomalía
        m2_anomaly: True si M2 (cross-sectional) detecta anomalía
        m1_score:   score raw de M1 (más negativo = más anómalo)
        m2_score:   score raw de M2

    Returns:
        dict con is_anomaly, severity, y scores individuales
    """
    if m1_anomaly and m2_anomaly:
        severity = "CRITICAL"
    elif m1_anomaly:
        severity = "HIGH"
    elif m2_anomaly:
        severity = "MEDIUM"
    else:
        severity = "NORMAL"

    return {
        "is_anomaly": m1_anomaly or m2_anomaly,
        "severity": severity,
        "m1_anomaly": m1_anomaly,
        "m2_anomaly": m2_anomaly,
        "m1_score": m1_score,
        "m2_score": m2_score,
    }


def ensemble_evaluate(m1_detected: np.ndarray, m2_detected: np.ndarray,
                      true_labels: np.ndarray,
                      mode: str = "or") -> dict:
    """
    Evalúa el ensemble contra ground truth.

    Args:
        m1_detected: booleano, predicciones de M1
        m2_detected: booleano, predicciones de M2
        true_labels: booleano, anomalías reales
        mode:        "or" (alerta si cualquiera detecta) o "and" (solo si ambos)

    Returns:
        dict con precision, recall, F1 del ensemble
    """
    if mode == "or":
        detected = m1_detected | m2_detected
    elif mode == "and":
        detected = m1_detected & m2_detected
    else:
        raise ValueError(f"mode must be 'or' or 'and', got '{mode}'")

    tp = int(np.sum(detected & true_labels))
    fp = int(np.sum(detected & ~true_labels))
    fn = int(np.sum(~detected & true_labels))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "mode": mode,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }
