"""
M6 — Amazon Chronos: detección de anomalías basada en forecasting.

Chronos es un foundation model pre-entrenado para series temporales (Hugging Face).
Enfoque: predice el próximo valor → compara predicción vs real → si la desviación
excede el intervalo de confianza → anomalía.

Ventajas:
  - Zero-shot: no necesita entrenamiento con nuestros datos
  - Entiende patrones estacionales automáticamente
  - Forecast probabilístico (distribución, no solo un punto)

Limitaciones:
  - Más lento que IsolationForest (transformer inference)
  - Diseñado para forecasting, adaptado aquí para anomaly detection
  - Necesita ~15+ puntos de contexto para buen resultado

Uso:
  from chronos_detector import score_chronos, evaluate_chronos, run_validation_chronos
  anomalies = score_chronos(train_values, test_values)
"""

import numpy as np
import pandas as pd
from typing import Optional

# Lazy import de torch y chronos para no fallar si no están instalados
_pipeline = None


def _get_pipeline():
    """Carga el modelo Chronos una sola vez (singleton)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError:
        raise ImportError(
            "Chronos no instalado. Ejecuta: pip install chronos-forecasting torch"
        )

    print("  Cargando modelo amazon/chronos-t5-small...")
    _pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        dtype=torch.float32,
    )
    print("  Modelo cargado.")
    return _pipeline


def score_chronos(
    train_values: np.ndarray,
    test_values: np.ndarray,
    threshold_sigma: float = 2.0,
    num_samples: int = 50,
) -> np.ndarray:
    """
    Detecta anomalías usando Chronos forecast probabilístico.

    Para cada punto del test set:
    1. Usa todo el histórico previo como contexto
    2. Chronos genera `num_samples` predicciones del próximo valor
    3. Si el valor real cae fuera de `threshold_sigma` * std de las predicciones → anomalía

    Args:
        train_values:    array con valores de entrenamiento (contexto base)
        test_values:     array con valores de test a evaluar
        threshold_sigma: número de desviaciones estándar para marcar anomalía
        num_samples:     muestras del forecast probabilístico (más = más preciso pero lento)

    Returns:
        Array booleano: True = anomalía detectada
    """
    import torch

    pipeline = _get_pipeline()
    anomalies = np.zeros(len(test_values), dtype=bool)

    for i in range(len(test_values)):
        # Contexto: todo lo anterior al punto actual
        if i == 0:
            context = train_values.copy()
        else:
            context = np.concatenate([train_values, test_values[:i]])

        # Chronos espera tensor 2D: (batch, sequence)
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        # Forecast: (batch, num_samples, prediction_length)
        forecast = pipeline.predict(
            context_tensor,
            prediction_length=1,
            num_samples=num_samples,
        )

        # Extraer estadísticas de la distribución predicha
        samples = forecast[0, :, 0].numpy()  # (num_samples,)
        median = np.median(samples)
        std = np.std(samples)

        actual = test_values[i]

        # Anomalía si el valor real está fuera del intervalo de confianza
        if std > 0:
            anomalies[i] = abs(actual - median) > threshold_sigma * std
        else:
            # Si std == 0, Chronos predice un valor exacto
            # Marcar como anomalía si difiere significativamente
            anomalies[i] = abs(actual - median) > abs(median) * 0.5 if median != 0 else actual != 0

    return anomalies


def score_chronos_raw(
    train_values: np.ndarray,
    test_values: np.ndarray,
    num_samples: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ejecuta inferencia Chronos y devuelve (medians, stds) sin aplicar threshold.
    Permite probar múltiples thresholds sin re-ejecutar la inferencia costosa.
    """
    import torch

    pipeline = _get_pipeline()
    medians = np.zeros(len(test_values))
    stds = np.zeros(len(test_values))

    for i in range(len(test_values)):
        if i == 0:
            context = train_values.copy()
        else:
            context = np.concatenate([train_values, test_values[:i]])

        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        forecast = pipeline.predict(context_tensor, prediction_length=1, num_samples=num_samples)
        samples = forecast[0, :, 0].numpy()
        medians[i] = np.median(samples)
        stds[i] = np.std(samples)

    return medians, stds


def apply_chronos_threshold(
    medians: np.ndarray,
    stds: np.ndarray,
    test_values: np.ndarray,
    threshold_sigma: float = 2.0,
) -> np.ndarray:
    """Aplica threshold sobre resultados pre-computados de score_chronos_raw()."""
    anomalies = np.zeros(len(test_values), dtype=bool)
    for i in range(len(test_values)):
        if stds[i] > 0:
            anomalies[i] = abs(test_values[i] - medians[i]) > threshold_sigma * stds[i]
        else:
            anomalies[i] = abs(test_values[i] - medians[i]) > abs(medians[i]) * 0.5 if medians[i] != 0 else test_values[i] != 0
    return anomalies


def evaluate_chronos(
    train_values: np.ndarray,
    test_values: np.ndarray,
    true_anomaly_mask: np.ndarray,
    threshold_sigma: float = 2.0,
    num_samples: int = 50,
) -> dict:
    """
    Evalúa Chronos contra un ground truth.

    Returns:
        dict con precision, recall, F1
    """
    detected = score_chronos(train_values, test_values,
                             threshold_sigma=threshold_sigma,
                             num_samples=num_samples)

    tp = int(np.sum(detected & true_anomaly_mask))
    fp = int(np.sum(detected & ~true_anomaly_mask))
    fn = int(np.sum(~detected & true_anomaly_mask))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def run_validation_chronos(
    df_all: pd.DataFrame,
    barrio: str,
    uso: str,
    anomalies: list | None = None,
    threshold_sigma: float = 2.0,
    num_samples: int = 50,
) -> dict | None:
    """
    Valida Chronos para un barrio del hackathon.
    Compatible con validate_model.py — misma interfaz que run_validation_m5.
    """
    from validate_model import _load_barrio, inject_anomalies, SYNTHETIC_ANOMALIES

    anom_list = anomalies if anomalies is not None else SYNTHETIC_ANOMALIES

    df = _load_barrio(df_all, barrio, uso)
    if df is None:
        return None

    n_train = min(25, int(len(df) * 0.7))
    n_test = len(df) - n_train

    if n_test < max(a["offset"] for a in anom_list) + 1:
        return None

    df_injected, anomaly_indices = inject_anomalies(df, n_train, anomalies=anom_list, verbose=False)

    train_vals = df_injected.iloc[:n_train]["consumption"].values.astype(float)
    test_vals = df_injected.iloc[n_train:]["consumption"].values.astype(float)

    true_labels = np.zeros(len(test_vals), dtype=bool)
    for abs_idx in anomaly_indices:
        rel = abs_idx - n_train
        if 0 <= rel < len(test_vals):
            true_labels[rel] = True

    return evaluate_chronos(train_vals, test_vals, true_labels,
                            threshold_sigma=threshold_sigma,
                            num_samples=num_samples)


# ─────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    from train_local import load_hackathon_amaem
    from validate_model import _load_barrio, inject_anomalies, SYNTHETIC_ANOMALIES_EXTREME

    DATA_FILE = "data/datos-hackathon-amaem.xlsx-set-de-datos-.csv"
    if not Path(DATA_FILE).exists():
        print(f"ERROR: No se encuentra {DATA_FILE}")
        exit(1)

    print("Cargando datos del hackathon...")
    df_all = load_hackathon_amaem(DATA_FILE)

    barrio = "10-FLORIDA BAJA"
    uso = "DOMESTICO"
    print(f"\nValidando M6 Chronos para {barrio} ({uso})...")

    result = run_validation_chronos(df_all, barrio, uso, anomalies=SYNTHETIC_ANOMALIES_EXTREME)
    if result:
        print(f"\nResultados M6 Chronos:")
        print(f"  Precision: {result['precision']:.1%}")
        print(f"  Recall:    {result['recall']:.1%}")
        print(f"  F1:        {result['f1']:.1%}")
        print(f"  TP={result['tp']} FP={result['fp']} FN={result['fn']}")
    else:
        print("No hay datos suficientes.")
