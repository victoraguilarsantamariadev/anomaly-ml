"""
Lógica del modelo de detección de anomalías.

Usa Isolation Forest de scikit-learn.
Cada process_point_id tiene su propio modelo entrenado (.pkl en models/).

Este módulo es independiente de FastAPI — puede usarse desde train_local.py
para pruebas locales o desde main.py como microservicio.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Optional

from feature_engineering import FEATURE_COLUMNS, prepare_training_matrix
from monthly_features import MONTHLY_FEATURE_COLUMNS, prepare_monthly_matrix

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train(meter_id: str, daily_df: pd.DataFrame,
          contamination: float = 0.05, threshold: float = -0.1) -> dict:
    """
    Entrena un Isolation Forest para un contador específico.
    Guarda el modelo en models/{meter_id}.pkl

    Args:
        meter_id:      identificador único del contador / process point
        daily_df:      DataFrame con features diarios (salida de add_rolling_features)
        contamination: fracción de datos asumida anómala (0.01-0.5)
        threshold:     umbral de score para clasificar anomalías (más negativo = más estricto)

    Returns:
        dict con resultado del entrenamiento
    """
    X = prepare_training_matrix(daily_df)

    if X is None:
        return {
            "meter_id": meter_id,
            "status": "error",
            "reason": "Not enough data (minimum 10 days required)",
            "samples_used": 0,
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Guardar estadísticas del training set para explicabilidad
    training_means = dict(zip(FEATURE_COLUMNS, np.mean(X, axis=0)))
    training_stds = dict(zip(FEATURE_COLUMNS, np.std(X, axis=0)))

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "contamination": contamination,
        "threshold": threshold,
        "model_type": "temporal",
        "training_means": training_means,
        "training_stds": training_stds,
    }
    joblib.dump(artifact, MODELS_DIR / f"{meter_id}.pkl")

    return {
        "meter_id": meter_id,
        "status": "trained",
        "samples_used": len(X),
        "features": FEATURE_COLUMNS,
        "contamination": contamination,
        "threshold": threshold,
    }


def score(meter_id: str, features: dict, threshold: Optional[float] = None) -> dict:
    """
    Puntúa un nuevo punto de datos.

    Args:
        meter_id:   identificador del contador
        features:   dict con los features del día a puntuar
        threshold:  umbral de anomalía (si None, usa el guardado en el modelo)

    Returns:
        dict con score, is_anomaly y severity
    """
    artifact = _load_model(meter_id)
    if artifact is None:
        return {
            "meter_id": meter_id,
            "score": None,
            "is_anomaly": False,
            "severity": "UNKNOWN",
            "reason": "Model not trained yet",
        }

    model: IsolationForest = artifact["model"]
    scaler: StandardScaler = artifact["scaler"]
    thr = threshold if threshold is not None else artifact.get("threshold", -0.1)

    feature_vector = [float(features.get(col, 0.0) or 0.0) for col in FEATURE_COLUMNS]
    X = np.array([feature_vector])
    X_scaled = scaler.transform(X)

    raw_score = float(model.score_samples(X_scaled)[0])
    is_anomaly = raw_score < thr

    return {
        "meter_id": meter_id,
        "score": raw_score,
        "is_anomaly": is_anomaly,
        "severity": _classify_severity(raw_score),
    }


def score_batch(meter_id: str, daily_df: pd.DataFrame,
                threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Puntúa varios días a la vez. Útil para análisis histórico.

    Args:
        meter_id:   identificador del contador
        daily_df:   DataFrame con features diarios
        threshold:  umbral de anomalía (si None, usa el guardado en el modelo)

    Returns:
        DataFrame con columnas adicionales: anomaly_score, is_anomaly, severity
    """
    artifact = _load_model(meter_id)
    if artifact is None:
        daily_df["anomaly_score"] = None
        daily_df["is_anomaly"] = False
        daily_df["severity"] = "UNKNOWN"
        return daily_df

    model: IsolationForest = artifact["model"]
    scaler: StandardScaler = artifact["scaler"]
    thr = threshold if threshold is not None else artifact.get("threshold", -0.1)

    subset = daily_df[FEATURE_COLUMNS].fillna(0)
    X_scaled = scaler.transform(subset.values)

    scores = model.score_samples(X_scaled)

    result = daily_df.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = scores < thr
    result["severity"] = [_classify_severity(s) for s in scores]

    return result


def train_monthly(meter_id: str, df_features: pd.DataFrame,
                   contamination: float = 0.10, threshold: float = -0.1) -> dict:
    """
    Entrena un Isolation Forest con features cross-seccionales (M2).
    Usa los 7 features de monthly_features.py que comparan barrios entre sí.

    Args:
        meter_id:      identificador del barrio (ej: "10-FLORIDA BAJA__DOMESTICO")
        df_features:   DataFrame con features mensuales (salida de compute_monthly_features)
        contamination: fracción de datos asumida anómala (M2 usa 0.10 por defecto)
        threshold:     umbral de score para clasificar anomalías

    Returns:
        dict con resultado del entrenamiento
    """
    X = prepare_monthly_matrix(df_features)

    if X is None:
        return {
            "meter_id": meter_id,
            "status": "error",
            "reason": "Not enough monthly data (minimum 10 months required)",
            "samples_used": 0,
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Guardar medias del training set para explicabilidad
    training_means = dict(zip(MONTHLY_FEATURE_COLUMNS, np.mean(X, axis=0)))
    training_stds = dict(zip(MONTHLY_FEATURE_COLUMNS, np.std(X, axis=0)))

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": MONTHLY_FEATURE_COLUMNS,
        "contamination": contamination,
        "threshold": threshold,
        "model_type": "monthly",
        "training_means": training_means,
        "training_stds": training_stds,
    }
    joblib.dump(artifact, MODELS_DIR / f"{meter_id}_monthly.pkl")

    return {
        "meter_id": meter_id,
        "status": "trained",
        "samples_used": len(X),
        "features": MONTHLY_FEATURE_COLUMNS,
        "contamination": contamination,
        "threshold": threshold,
    }


def score_monthly(meter_id: str, features: dict,
                  threshold: Optional[float] = None) -> dict:
    """
    Puntúa un mes usando el modelo cross-seccional (M2).

    Args:
        meter_id:   identificador del barrio
        features:   dict con los monthly features del mes a puntuar
        threshold:  umbral de anomalía (si None, usa el guardado en el modelo)

    Returns:
        dict con score, is_anomaly y severity
    """
    artifact = _load_model_monthly(meter_id)
    if artifact is None:
        return {
            "meter_id": meter_id,
            "score": None,
            "is_anomaly": False,
            "severity": "UNKNOWN",
            "reason": "Monthly model not trained yet",
        }

    model: IsolationForest = artifact["model"]
    scaler: StandardScaler = artifact["scaler"]
    thr = threshold if threshold is not None else artifact.get("threshold", -0.1)

    feature_vector = [float(features.get(col, 0.0) or 0.0) for col in MONTHLY_FEATURE_COLUMNS]
    X = np.array([feature_vector])
    X_scaled = scaler.transform(X)

    raw_score = float(model.score_samples(X_scaled)[0])
    is_anomaly = raw_score < thr

    return {
        "meter_id": meter_id,
        "score": raw_score,
        "is_anomaly": is_anomaly,
        "severity": _classify_severity(raw_score),
    }


def score_monthly_batch(meter_id: str, df_features: pd.DataFrame,
                        threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Puntúa varios meses a la vez usando el modelo cross-seccional (M2).

    Args:
        meter_id:     identificador del barrio
        df_features:  DataFrame con features mensuales
        threshold:    umbral de anomalía

    Returns:
        DataFrame con columnas adicionales: anomaly_score, is_anomaly, severity
    """
    artifact = _load_model_monthly(meter_id)
    if artifact is None:
        df_features = df_features.copy()
        df_features["anomaly_score"] = None
        df_features["is_anomaly"] = False
        df_features["severity"] = "UNKNOWN"
        return df_features

    model: IsolationForest = artifact["model"]
    scaler: StandardScaler = artifact["scaler"]
    thr = threshold if threshold is not None else artifact.get("threshold", -0.1)

    subset = df_features[MONTHLY_FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)
    X_scaled = scaler.transform(subset.values)

    scores = model.score_samples(X_scaled)

    result = df_features.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = scores < thr
    result["severity"] = [_classify_severity(s) for s in scores]

    return result


def _load_model_monthly(meter_id: str) -> Optional[dict]:
    """Carga el modelo monthly del disco."""
    path = MODELS_DIR / f"{meter_id}_monthly.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


# =============================================================
# FEEDBACK LOOP — ajuste automático con etiquetas del usuario
# =============================================================

LABELS_DIR = Path("labels")
LABELS_DIR.mkdir(exist_ok=True)


def adjust_threshold(meter_id: str) -> dict:
    """
    Ajusta el threshold del modelo basándose en las etiquetas del usuario.

    Lógica:
      - Si >50% son FALSE_POSITIVE: el modelo es muy agresivo → subir threshold
      - Si >50% son CONFIRMED: el modelo detecta bien → bajar threshold (más estricto)
      - Ajuste proporcional a la tasa de FP

    Requiere >=10 etiquetas para actuar.

    Returns:
        dict con old_threshold, new_threshold y stats
    """
    import json

    labels_file = LABELS_DIR / f"{meter_id}_labels.json"
    if not labels_file.exists():
        return {"status": "no_labels", "meter_id": meter_id}

    labels = json.loads(labels_file.read_text())
    if len(labels) < 10:
        return {
            "status": "insufficient_labels",
            "meter_id": meter_id,
            "labels_count": len(labels),
            "min_required": 10,
        }

    n_fp = sum(1 for l in labels if l["label"] == "FALSE_POSITIVE")
    n_confirmed = sum(1 for l in labels if l["label"] == "CONFIRMED")
    fp_rate = n_fp / len(labels)

    # Cargar modelo actual
    artifact = _load_model(meter_id)
    if artifact is None:
        return {"status": "no_model", "meter_id": meter_id}

    old_threshold = artifact.get("threshold", -0.1)

    # Ajustar threshold
    if fp_rate > 0.5:
        # Demasiados falsos positivos → hacer menos estricto (threshold más negativo)
        adjustment = -0.05 * (fp_rate - 0.3)  # proporcional a cuánto pasa de 30%
        new_threshold = max(old_threshold + adjustment, -0.5)  # no pasar de -0.5
    elif fp_rate < 0.2:
        # Pocos FP → puede ser más estricto (threshold menos negativo)
        adjustment = 0.02 * (0.3 - fp_rate)
        new_threshold = min(old_threshold + adjustment, -0.02)  # no pasar de -0.02
    else:
        new_threshold = old_threshold  # zona equilibrada, no tocar

    # Guardar modelo actualizado
    artifact["threshold"] = new_threshold
    joblib.dump(artifact, MODELS_DIR / f"{meter_id}.pkl")

    return {
        "status": "adjusted",
        "meter_id": meter_id,
        "old_threshold": old_threshold,
        "new_threshold": new_threshold,
        "labels_count": len(labels),
        "false_positives": n_fp,
        "confirmed": n_confirmed,
        "fp_rate": fp_rate,
    }


def model_exists(meter_id: str, model_type: str = "temporal") -> bool:
    """Comprueba si existe un modelo entrenado para este contador."""
    if model_type == "monthly":
        return (MODELS_DIR / f"{meter_id}_monthly.pkl").exists()
    return (MODELS_DIR / f"{meter_id}.pkl").exists()


def delete_model(meter_id: str, model_type: str = "temporal") -> bool:
    """Elimina el modelo de un contador (útil para forzar re-entrenamiento)."""
    if model_type == "monthly":
        path = MODELS_DIR / f"{meter_id}_monthly.pkl"
    else:
        path = MODELS_DIR / f"{meter_id}.pkl"
    if path.exists():
        path.unlink()
        return True
    return False


def explain_anomaly(meter_id: str, features: dict) -> Optional[str]:
    """
    Genera una explicación legible de por qué un punto es anómalo.
    Compara cada feature contra las estadísticas del training set.

    Args:
        meter_id:   identificador del contador
        features:   dict con los features del punto a explicar

    Returns:
        String con la explicación, o None si no hay modelo o no es anómalo
    """
    artifact = _load_model(meter_id)
    if artifact is None:
        return None

    training_means = artifact.get("training_means")
    training_stds = artifact.get("training_stds")
    if not training_means or not training_stds:
        return None

    # Calcular desviaciones de cada feature respecto al training set
    deviations = []
    feature_labels = {
        "daily_total": "consumo total diario",
        "nocturnal_min": "consumo mínimo nocturno (2-5AM)",
        "nocturnal_mean": "consumo medio nocturno",
        "diurnal_mean": "consumo medio diurno",
        "night_day_ratio": "ratio noche/día",
        "active_hours": "horas activas",
        "is_weekend": "fin de semana",
        "zscore": "z-score (desviación de tendencia)",
    }

    for col in FEATURE_COLUMNS:
        val = float(features.get(col, 0.0) or 0.0)
        mean = training_means.get(col, 0.0)
        std = training_stds.get(col, 1.0)

        if std > 0:
            deviation = abs(val - mean) / std
        else:
            deviation = 0.0

        if deviation > 1.5:  # solo reportar features que se desvían significativamente
            direction = "por encima" if val > mean else "por debajo"
            label = feature_labels.get(col, col)
            deviations.append((deviation, label, val, mean, direction))

    if not deviations:
        return "Combinación inusual de features (ningún feature individual destaca)"

    deviations.sort(reverse=True)
    parts = []
    for dev, label, val, mean, direction in deviations[:3]:  # top 3
        parts.append(f"{label} {direction} de lo normal ({val:.1f} vs media {mean:.1f})")

    return "; ".join(parts)


def _load_model(meter_id: str) -> Optional[dict]:
    """Carga el modelo y scaler del disco."""
    path = MODELS_DIR / f"{meter_id}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def _classify_severity(score: float) -> str:
    """
    Clasifica la severidad de la anomalía según el score.

    Isolation Forest score_samples:
    - Valores cercanos a 0 o positivos → normal
    - Valores más negativos → más anómalo
    """
    if score < -0.5:
        return "CRITICAL"
    if score < -0.3:
        return "HIGH"
    if score < -0.1:
        return "MEDIUM"
    return "NORMAL"
