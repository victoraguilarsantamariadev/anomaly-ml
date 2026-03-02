"""
M7 — Prophet: detección de anomalías basada en descomposición temporal.

Prophet (Facebook/Meta) descompone la serie en:
  tendencia + estacionalidad anual + estacionalidad semanal + residual

Si el valor real cae fuera del intervalo de confianza [yhat_lower, yhat_upper] → anomalía.

Ventajas:
  - Diseñado para datos mensuales/diarios (nuestro caso exacto)
  - Maneja estacionalidad y tendencia nativamente
  - Rápido, sin GPU
  - Detecta puntos de cambio (changepoints) automáticamente

Limitaciones:
  - Necesita mínimo ~2 períodos estacionales para capturar estacionalidad (24 meses para anual)
  - Puede ser ruidoso con pocos datos
  - Requiere columnas específicas: ds (datetime) e y (valor)

Uso:
  from prophet_detector import score_prophet, evaluate_prophet, run_validation_prophet
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

# Suprimir logs verbosos de Prophet (cmdstanpy/pystan)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def score_prophet(
    train_values: np.ndarray,
    test_values: np.ndarray,
    train_dates: pd.DatetimeIndex | np.ndarray,
    test_dates: pd.DatetimeIndex | np.ndarray,
    interval_width: float = 0.95,
    changepoint_prior_scale: float = 0.05,
) -> np.ndarray:
    """
    Detecta anomalías usando Prophet forecast + intervalo de confianza.

    1. Entrena Prophet con datos de entrenamiento
    2. Predice para el período de test (con intervalos de confianza)
    3. Si valor real cae fuera de [yhat_lower, yhat_upper] → anomalía

    Args:
        train_values:   array de valores de entrenamiento
        test_values:    array de valores de test
        train_dates:    fechas correspondientes al train set
        test_dates:     fechas correspondientes al test set
        interval_width: ancho del intervalo de confianza (0.95 = 95%, 0.97 = 97%)
        changepoint_prior_scale: flexibilidad para detectar cambios de tendencia
                                 (0.05 = conservador, 0.15 = mas adaptativo)

    Returns:
        Array booleano: True = anomalía detectada
    """
    from prophet import Prophet

    # Preparar DataFrame en formato Prophet
    df_train = pd.DataFrame({
        "ds": pd.to_datetime(train_dates),
        "y": train_values.astype(float),
    })

    # Configurar Prophet
    model = Prophet(
        interval_width=interval_width,
        yearly_seasonality=True,
        weekly_seasonality=False,  # datos mensuales, no hay semanas
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
    )

    # Suprimir stdout de Prophet
    model.fit(df_train)

    # Predecir para las fechas de test
    df_future = pd.DataFrame({"ds": pd.to_datetime(test_dates)})
    forecast = model.predict(df_future)

    # Anomalía si el valor real está fuera del intervalo
    anomalies = (
        (test_values < forecast["yhat_lower"].values) |
        (test_values > forecast["yhat_upper"].values)
    )
    return anomalies.astype(bool)


def evaluate_prophet(
    train_values: np.ndarray,
    test_values: np.ndarray,
    train_dates: pd.DatetimeIndex | np.ndarray,
    test_dates: pd.DatetimeIndex | np.ndarray,
    true_anomaly_mask: np.ndarray,
    interval_width: float = 0.95,
) -> dict:
    """
    Evalúa Prophet contra un ground truth.

    Returns:
        dict con precision, recall, F1
    """
    detected = score_prophet(
        train_values, test_values, train_dates, test_dates,
        interval_width=interval_width,
    )

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


def run_validation_prophet(
    df_all: pd.DataFrame,
    barrio: str,
    uso: str,
    anomalies: list | None = None,
) -> dict | None:
    """
    Valida Prophet para un barrio del hackathon.
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
    train_dates = df_injected.iloc[:n_train]["timestamp"].values
    test_dates = df_injected.iloc[n_train:]["timestamp"].values

    true_labels = np.zeros(len(test_vals), dtype=bool)
    for abs_idx in anomaly_indices:
        rel = abs_idx - n_train
        if 0 <= rel < len(test_vals):
            true_labels[rel] = True

    return evaluate_prophet(train_vals, test_vals, train_dates, test_dates, true_labels)


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
    print(f"\nValidando M7 Prophet para {barrio} ({uso})...")

    result = run_validation_prophet(df_all, barrio, uso, anomalies=SYNTHETIC_ANOMALIES_EXTREME)
    if result:
        print(f"\nResultados M7 Prophet:")
        print(f"  Precision: {result['precision']:.1%}")
        print(f"  Recall:    {result['recall']:.1%}")
        print(f"  F1:        {result['f1']:.1%}")
        print(f"  TP={result['tp']} FP={result['fp']} FN={result['fn']}")
    else:
        print("No hay datos suficientes.")
