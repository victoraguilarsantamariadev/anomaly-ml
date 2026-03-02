"""
Microservicio FastAPI de detección de anomalías.

Endpoints:
  POST /train           → entrena modelo para un contador
  POST /score           → puntúa un nuevo punto de datos
  POST /score/batch     → puntúa varios días a la vez
  POST /feedback        → recibe feedback humano (confirmar/descartar anomalía)
  GET  /models/{id}     → comprueba si existe modelo para ese contador
  DELETE /models/{id}   → elimina modelo (fuerza re-entrenamiento)
  GET  /health          → health check

Spring Boot Java llama a este servicio cuando:
  - Hay que entrenar el modelo de un nuevo process point
  - Hay que puntuar las lecturas de las últimas horas
  - El usuario confirma o descarta una anomalía via Telegram
"""

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
import logging

from feature_engineering import compute_daily_features, add_rolling_features, FEATURE_COLUMNS
from model import (
    train, score, score_batch, model_exists, delete_model,
    train_monthly, score_monthly, score_monthly_batch,
    adjust_threshold, explain_anomaly,
)
from monthly_features import compute_monthly_features, MONTHLY_FEATURE_COLUMNS
from ensemble import ensemble_vote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABELS_DIR = Path("labels")
LABELS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="AnomalyML Service",
    description="Detección de anomalías en contadores de agua, electricidad, gas y calor",
    version="1.1.0",
)


# =============================================================
# MODELOS DE REQUEST/RESPONSE
# =============================================================

class ReadingItem(BaseModel):
    """Una lectura individual de un contador."""
    timestamp: str = Field(..., example="2024-01-15T03:00:00")
    consumption: float = Field(..., ge=0, example=0.3)


class TrainRequest(BaseModel):
    """Request para entrenar el modelo de un contador."""
    meter_id: str = Field(..., example="process_point_12345")
    readings: list[ReadingItem] = Field(
        ...,
        min_length=10,
        description="Lecturas históricas del contador (mínimo 10 días de datos)"
    )
    contamination: float = Field(0.05, ge=0.01, le=0.5,
        description="Fracción de datos asumida anómala (0.01=conservador, 0.10=agresivo)")
    threshold: float = Field(-0.1, le=0,
        description="Umbral de score para anomalía (más negativo = más estricto)")


class ScoreRequest(BaseModel):
    """Request para puntuar un punto de datos nuevo. Features pre-calculados."""
    meter_id: str
    features: dict = Field(
        ...,
        description=f"Features calculados. Claves esperadas: {FEATURE_COLUMNS}"
    )


class BatchScoreRequest(BaseModel):
    """Request para puntuar varios días a la vez."""
    meter_id: str
    readings: list[ReadingItem]
    include_all: bool = Field(False,
        description="Si True, devuelve todos los días (no solo anomalías)")


class FeedbackRequest(BaseModel):
    """Feedback humano sobre una anomalía detectada."""
    meter_id: str
    timestamp: str = Field(..., description="Timestamp de la anomalía")
    label: str = Field(..., description="CONFIRMED o FALSE_POSITIVE")


class MonthlyReadingItem(BaseModel):
    """Una lectura mensual de un barrio."""
    barrio: str = Field(..., example="10-FLORIDA BAJA")
    uso: str = Field(..., example="DOMESTICO")
    fecha: str = Field(..., example="2024/01/31")
    consumo_litros: float = Field(..., ge=0, example=29205005)
    num_contratos: float = Field(..., ge=0, example=4665)


class TrainMonthlyRequest(BaseModel):
    """Request para entrenar modelo cross-seccional (M2)."""
    meter_id: str = Field(..., example="10-FLORIDA BAJA__DOMESTICO",
        description="Barrio + uso como identificador")
    readings: list[MonthlyReadingItem] = Field(
        ..., min_length=10,
        description="Lecturas mensuales de TODOS los barrios (necesario para cross-sectional)")
    contamination: float = Field(0.10, ge=0.01, le=0.5)
    threshold: float = Field(-0.1, le=0)


class ScoreMonthlyRequest(BaseModel):
    """Request para puntuar un mes con modelo cross-seccional."""
    meter_id: str
    features: dict = Field(
        ..., description=f"Monthly features. Claves esperadas: {MONTHLY_FEATURE_COLUMNS}")


class EnsembleScoreRequest(BaseModel):
    """Request para puntuar con ensemble M1+M2."""
    meter_id: str = Field(..., description="Identificador del barrio (ej: 10-FLORIDA BAJA__DOMESTICO)")
    temporal_features: dict = Field(
        ..., description=f"Features temporales (M1): {FEATURE_COLUMNS}")
    monthly_features: dict = Field(
        ..., description=f"Features cross-seccionales (M2): {MONTHLY_FEATURE_COLUMNS}")


class AllModelsScoreRequest(BaseModel):
    """Request para puntuar con todos los modelos (M2+M5+M7)."""
    readings: list[MonthlyReadingItem] = Field(
        ..., min_length=10,
        description="Lecturas mensuales de TODOS los barrios")
    target_barrio: str = Field(..., example="10-FLORIDA BAJA")
    target_uso: str = Field(..., example="DOMESTICO")
    target_fecha: str = Field(..., example="2024/06/30",
        description="Mes a evaluar")
    with_external: bool = Field(False,
        description="Incluir datos externos (temperatura, turismo)")


class AnomalyResult(BaseModel):
    meter_id: str
    score: Optional[float]
    is_anomaly: bool
    severity: str  # NORMAL, MEDIUM, HIGH, CRITICAL
    reason: Optional[str] = None


class TrainResult(BaseModel):
    meter_id: str
    status: str
    samples_used: int
    features: Optional[list[str]] = None
    contamination: Optional[float] = None
    threshold: Optional[float] = None
    reason: Optional[str] = None


# =============================================================
# ENDPOINTS
# =============================================================

@app.post("/train", response_model=TrainResult, summary="Entrenar modelo para un contador")
def train_endpoint(req: TrainRequest):
    """
    Entrena un Isolation Forest con el histórico de lecturas del contador.
    Guarda el modelo en models/{meter_id}.pkl

    Llamar cuando:
    - Se añade un contador nuevo al sistema
    - Se quiere actualizar el modelo con datos más recientes (reentrenamiento semanal)
    - Se detectaron muchos falsos positivos (ajuste manual)
    """
    logger.info(f"Training model for meter: {req.meter_id} ({len(req.readings)} readings, "
                f"contamination={req.contamination}, threshold={req.threshold})")

    df = pd.DataFrame([r.model_dump() for r in req.readings])

    try:
        daily = compute_daily_features(df)
        daily = add_rolling_features(daily)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error computing features: {str(e)}")

    result = train(req.meter_id, daily,
                   contamination=req.contamination, threshold=req.threshold)

    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("reason"))

    logger.info(f"Model trained: {result}")
    return TrainResult(**result)


@app.post("/score", response_model=AnomalyResult, summary="Puntuar un punto de datos")
def score_endpoint(req: ScoreRequest):
    """
    Puntúa un nuevo punto de datos usando el modelo entrenado.
    Los features vienen ya calculados desde el backend Java.
    Si detecta anomalía, incluye una explicación de qué features causan la alerta.
    """
    if not model_exists(req.meter_id):
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for meter '{req.meter_id}'. Call /train first."
        )

    result = score(req.meter_id, req.features)

    # Añadir explicación si es anomalía
    if result.get("is_anomaly"):
        reason = explain_anomaly(req.meter_id, req.features)
        if reason:
            result["reason"] = reason

    return AnomalyResult(**result)


@app.post("/score/batch", summary="Puntuar varios días a la vez (análisis histórico)")
def score_batch_endpoint(req: BatchScoreRequest):
    """
    Puntúa múltiples días de lecturas a la vez.
    Útil para análisis retroactivo o cuando arranca el sistema.
    """
    if not model_exists(req.meter_id):
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for meter '{req.meter_id}'. Call /train first."
        )

    df = pd.DataFrame([r.model_dump() for r in req.readings])
    daily = compute_daily_features(df)
    daily = add_rolling_features(daily)

    scored = score_batch(req.meter_id, daily)

    # Filtrar según include_all
    if req.include_all:
        rows_to_return = scored
    else:
        rows_to_return = scored[scored["is_anomaly"] == True]

    results = []
    for _, row in rows_to_return.iterrows():
        results.append({
            "date": str(row["date"]),
            "score": float(row["anomaly_score"]) if pd.notna(row.get("anomaly_score")) else None,
            "is_anomaly": bool(row.get("is_anomaly", False)),
            "severity": row.get("severity", "UNKNOWN"),
            "daily_total": float(row["daily_total"]),
            "nocturnal_min": float(row.get("nocturnal_min", 0)),
        })

    return {
        "meter_id": req.meter_id,
        "days_analyzed": len(daily),
        "anomalies_found": int(scored["is_anomaly"].sum()),
        "results": results,
    }


@app.post("/train/monthly", response_model=TrainResult,
          summary="Entrenar modelo cross-seccional (M2) para un barrio")
def train_monthly_endpoint(req: TrainMonthlyRequest):
    """
    Entrena un Isolation Forest con features cross-seccionales.
    Compara el barrio objetivo contra todos los demás del mismo tipo.
    Necesita datos de TODOS los barrios para calcular z-scores relativos.
    """
    logger.info(f"Training monthly model for: {req.meter_id} ({len(req.readings)} readings)")

    df = pd.DataFrame([r.model_dump() for r in req.readings])
    df["fecha"] = pd.to_datetime(df["fecha"])

    try:
        df_features = compute_monthly_features(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error computing monthly features: {str(e)}")

    # Filtrar solo el barrio objetivo para entrenar su modelo
    barrio_data = df_features[df_features["barrio_key"] == req.meter_id]
    if len(barrio_data) < 10:
        raise HTTPException(status_code=422,
            detail=f"Not enough data for '{req.meter_id}' ({len(barrio_data)} months, need 10)")

    result = train_monthly(req.meter_id, barrio_data,
                           contamination=req.contamination, threshold=req.threshold)

    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("reason"))

    logger.info(f"Monthly model trained: {result}")
    return TrainResult(**result)


@app.post("/score/monthly", response_model=AnomalyResult,
          summary="Puntuar un mes con modelo cross-seccional (M2)")
def score_monthly_endpoint(req: ScoreMonthlyRequest):
    """Puntúa un mes usando el modelo cross-seccional entrenado."""
    if not model_exists(req.meter_id, model_type="monthly"):
        raise HTTPException(
            status_code=404,
            detail=f"No monthly model for '{req.meter_id}'. Call /train/monthly first."
        )

    result = score_monthly(req.meter_id, req.features)
    return AnomalyResult(**result)


@app.post("/score/ensemble", summary="Puntuar con ensemble M1+M2 (máxima precisión)")
def score_ensemble_endpoint(req: EnsembleScoreRequest):
    """
    Combina predicciones de M1 (temporal) y M2 (cross-seccional) con voting.

    Severidades del ensemble:
      - CRITICAL: ambos modelos detectan anomalía (máxima confianza)
      - HIGH: solo M1 detecta (M1 tiene ~0 falsos positivos)
      - MEDIUM: solo M2 detecta (M2 tiene más falsos positivos)
      - NORMAL: ninguno detecta
    """
    has_temporal = model_exists(req.meter_id, model_type="temporal")
    has_monthly = model_exists(req.meter_id, model_type="monthly")

    if not has_temporal and not has_monthly:
        raise HTTPException(
            status_code=404,
            detail=f"No models for '{req.meter_id}'. Train at least one model first."
        )

    # Ejecutar M1 si existe
    m1_result = None
    if has_temporal:
        m1_result = score(req.meter_id, req.temporal_features)

    # Ejecutar M2 si existe
    m2_result = None
    if has_monthly:
        m2_result = score_monthly(req.meter_id, req.monthly_features)

    # Si solo hay un modelo, devolver su resultado directamente
    if m1_result and not m2_result:
        return AnomalyResult(**m1_result)
    if m2_result and not m1_result:
        return AnomalyResult(**m2_result)

    # Ensemble voting
    result = ensemble_vote(
        m1_anomaly=m1_result["is_anomaly"],
        m2_anomaly=m2_result["is_anomaly"],
        m1_score=m1_result.get("score"),
        m2_score=m2_result.get("score"),
    )

    return {
        "meter_id": req.meter_id,
        "score": m1_result.get("score"),  # M1 score como principal
        "is_anomaly": result["is_anomaly"],
        "severity": result["severity"],
        "reason": f"M1={'ANOMALY' if result['m1_anomaly'] else 'OK'}, "
                  f"M2={'ANOMALY' if result['m2_anomaly'] else 'OK'}",
    }


@app.post("/score/all-models",
          summary="Puntuar con todos los modelos (M2 + M5 + M7)")
def score_all_models_endpoint(req: AllModelsScoreRequest):
    """
    Ejecuta M2, M5 (3-sigma + IQR), y M7 (Prophet) para un barrio/mes concreto.
    Devuelve el resultado de cada modelo por separado.

    M2 necesita datos de todos los barrios (cross-sectional).
    M5 y M7 solo necesitan el historico del barrio objetivo.
    """
    from statistical_baseline import score_3sigma, score_iqr

    df = pd.DataFrame([r.model_dump() for r in req.readings])
    df["fecha"] = pd.to_datetime(df["fecha"])
    target_fecha = pd.to_datetime(req.target_fecha)

    # Opcional: datos externos
    external_df = None
    if req.with_external:
        from external_data import load_external_data
        min_date = df["fecha"].min()
        max_date = df["fecha"].max()
        external_df = load_external_data(str(min_date.date()), str(max_date.date()))

    # Calcular monthly features
    try:
        df_features = compute_monthly_features(df, external_df=external_df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error computing features: {str(e)}")

    barrio_key = f"{req.target_barrio.strip()}__{req.target_uso.strip()}"
    barrio_data = df_features[df_features["barrio_key"] == barrio_key].sort_values("fecha")

    if len(barrio_data) == 0:
        raise HTTPException(status_code=404,
            detail=f"No data for barrio '{barrio_key}'")

    # Buscar el mes objetivo
    target_row = barrio_data[
        barrio_data["fecha"].dt.to_period("M") == target_fecha.to_period("M")
    ]
    if len(target_row) == 0:
        raise HTTPException(status_code=404,
            detail=f"No data for fecha '{req.target_fecha}' in barrio '{barrio_key}'")

    results = {"meter_id": barrio_key}
    models_detecting = []

    # --- M2: IsolationForest cross-sectional (features relativos) ---
    try:
        from sklearn.ensemble import IsolationForest as IF
        from sklearn.preprocessing import StandardScaler as SS
        from monthly_features import RELATIVE_FEATURE_COLUMNS, RELATIVE_EXTENDED_FEATURE_COLUMNS

        feature_cols = RELATIVE_FEATURE_COLUMNS
        if external_df is not None:
            available_ext = [c for c in RELATIVE_EXTENDED_FEATURE_COLUMNS if c in df_features.columns]
            if len(available_ext) > len(RELATIVE_FEATURE_COLUMNS):
                feature_cols = available_ext

        # Entrenar sobre todos los datos del mismo uso
        uso_data = df_features[df_features["uso"].str.strip() == req.target_uso.strip()]
        X_all = uso_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

        if len(X_all) >= 10:
            scaler = SS()
            X_scaled = scaler.fit_transform(X_all)
            model_m2 = IF(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
            model_m2.fit(X_scaled)

            target_vector = target_row[feature_cols].replace(
                [np.inf, -np.inf], np.nan).fillna(0).values
            m2_score = float(model_m2.score_samples(scaler.transform(target_vector))[0])
            m2_anomaly = m2_score < -0.1

            results["m2"] = {"is_anomaly": m2_anomaly, "score": round(m2_score, 4)}
            if m2_anomaly:
                models_detecting.append("M2")
        else:
            results["m2"] = {"is_anomaly": False, "error": "insufficient data"}
    except Exception as e:
        results["m2"] = {"is_anomaly": False, "error": str(e)}

    # --- M5: 3-sigma + IQR sobre deviation_from_group_trend ---
    try:
        all_devs = barrio_data["deviation_from_group_trend"].fillna(0).values.astype(float)
        target_dev = float(target_row["deviation_from_group_trend"].fillna(0).iloc[0])

        if len(all_devs) >= 6:
            ref_devs = all_devs[:-1] if len(all_devs) > 1 else all_devs

            sigma_flag = bool(score_3sigma(np.array([target_dev]), ref_devs)[0])
            iqr_flag = bool(score_iqr(np.array([target_dev]), ref_devs)[0])

            results["m5_3sigma"] = {"is_anomaly": sigma_flag}
            results["m5_iqr"] = {"is_anomaly": iqr_flag}
            if sigma_flag:
                models_detecting.append("M5_3sigma")
            if iqr_flag:
                models_detecting.append("M5_IQR")
        else:
            results["m5_3sigma"] = {"is_anomaly": False, "error": "insufficient data"}
            results["m5_iqr"] = {"is_anomaly": False, "error": "insufficient data"}
    except Exception as e:
        results["m5_3sigma"] = {"is_anomaly": False, "error": str(e)}
        results["m5_iqr"] = {"is_anomaly": False, "error": str(e)}

    # --- M7: Prophet ---
    try:
        from prophet_detector import score_prophet

        vals = barrio_data["consumption_per_contract"].values.astype(float)
        dates = barrio_data["fecha"].values

        if len(vals) >= 12:
            # Entrenar con todo menos el punto objetivo, testear el punto
            train_vals = vals[:-1]
            test_vals = vals[-1:]
            train_dates = dates[:-1]
            test_dates = dates[-1:]

            flags = score_prophet(train_vals, test_vals, train_dates, test_dates,
                                  interval_width=0.97,
                                  changepoint_prior_scale=0.15)
            prophet_flag = bool(flags[0]) if len(flags) > 0 else False

            results["m7_prophet"] = {"is_anomaly": prophet_flag}
            if prophet_flag:
                models_detecting.append("M7_Prophet")
        else:
            results["m7_prophet"] = {"is_anomaly": False, "error": "insufficient data"}
    except ImportError:
        results["m7_prophet"] = {"is_anomaly": False, "error": "prophet not installed"}
    except Exception as e:
        results["m7_prophet"] = {"is_anomaly": False, "error": str(e)}

    results["n_models_detecting"] = len(models_detecting)
    results["models_detecting"] = models_detecting

    # Contexto: desviacion del grupo y confianza
    dev = 0.0
    if "deviation_from_group_trend" in target_row.columns:
        dev = float(target_row["deviation_from_group_trend"].fillna(0).iloc[0])
        results["deviation_from_group_trend"] = round(dev, 4)
        results["deviation_pct"] = f"{dev*100:+.1f}%"
    if "group_yoy_median" in target_row.columns:
        results["group_yoy_median"] = round(float(target_row["group_yoy_median"].fillna(1).iloc[0]), 4)
    if "yoy_ratio" in target_row.columns:
        results["yoy_ratio"] = round(float(target_row["yoy_ratio"].fillna(1).iloc[0]), 4)

    # Nivel de confianza
    abs_dev = abs(dev)
    if len(models_detecting) == 0:
        results["confidence"] = "NONE"
    elif abs_dev > 0.30:
        results["confidence"] = "HIGH"
    elif abs_dev > 0.10:
        results["confidence"] = "MEDIUM"
    else:
        results["confidence"] = "LOW"

    return results


@app.post("/feedback", summary="Feedback humano sobre anomalía detectada")
def feedback_endpoint(req: FeedbackRequest):
    """
    Recibe feedback del usuario (via Telegram o frontend).

    Almacena las etiquetas en labels/{meter_id}_labels.json.
    Cuando hay suficientes etiquetas, se usan para:
    - ≥10 labels: ajustar threshold automáticamente
    - ≥50 labels: re-entrenar con supervisión (XGBoost)
    """
    if req.label not in ("CONFIRMED", "FALSE_POSITIVE"):
        raise HTTPException(status_code=400, detail="label must be CONFIRMED or FALSE_POSITIVE")

    labels_file = LABELS_DIR / f"{req.meter_id}_labels.json"

    # Cargar labels existentes
    if labels_file.exists():
        labels = json.loads(labels_file.read_text())
    else:
        labels = []

    labels.append({
        "timestamp": req.timestamp,
        "label": req.label,
        "received_at": pd.Timestamp.now().isoformat(),
    })

    labels_file.write_text(json.dumps(labels, indent=2))

    n_labels = len(labels)
    n_false_positives = sum(1 for l in labels if l["label"] == "FALSE_POSITIVE")
    n_confirmed = sum(1 for l in labels if l["label"] == "CONFIRMED")

    logger.info(f"Feedback for {req.meter_id}: {req.label} "
                f"(total: {n_labels}, FP: {n_false_positives}, confirmed: {n_confirmed})")

    # Auto-ajustar threshold si hay suficientes etiquetas
    threshold_adjustment = None
    if n_labels >= 10 and n_labels % 5 == 0:  # cada 5 etiquetas nuevas
        threshold_adjustment = adjust_threshold(req.meter_id)
        if threshold_adjustment.get("status") == "adjusted":
            logger.info(f"Auto-adjusted threshold for {req.meter_id}: "
                        f"{threshold_adjustment['old_threshold']:.3f} → "
                        f"{threshold_adjustment['new_threshold']:.3f}")

    return {
        "meter_id": req.meter_id,
        "status": "recorded",
        "total_labels": n_labels,
        "false_positives": n_false_positives,
        "confirmed": n_confirmed,
        "recommendation": _feedback_recommendation(n_labels, n_false_positives),
        "threshold_adjustment": threshold_adjustment,
    }


def _feedback_recommendation(n_labels: int, n_false_positives: int) -> str:
    """Recomienda acción según cantidad y tipo de feedback."""
    if n_labels < 10:
        return f"Collecting labels ({n_labels}/10 for threshold tuning)"
    if n_false_positives > n_labels * 0.5:
        return "High FP rate — threshold auto-adjusting"
    if n_labels >= 50:
        return "Enough labels for supervised retraining (POST /retrain-with-labels)"
    return f"Good progress — {n_labels} labels collected"


@app.post("/adjust-threshold/{meter_id}", summary="Ajustar threshold manualmente con feedback")
def adjust_threshold_endpoint(meter_id: str):
    """
    Fuerza un ajuste del threshold basado en las etiquetas acumuladas.
    Normalmente se ejecuta automáticamente cada 5 etiquetas nuevas.
    """
    result = adjust_threshold(meter_id)
    if result.get("status") == "no_model":
        raise HTTPException(status_code=404, detail=f"No model for '{meter_id}'")
    return result


@app.get("/models/{meter_id}", summary="Comprobar si existe modelo para un contador")
def get_model_status(meter_id: str):
    """Devuelve si hay un modelo entrenado para este contador."""
    exists_temporal = model_exists(meter_id, model_type="temporal")
    exists_monthly = model_exists(meter_id, model_type="monthly")

    # Contar labels si existen
    labels_file = LABELS_DIR / f"{meter_id}_labels.json"
    n_labels = 0
    if labels_file.exists():
        n_labels = len(json.loads(labels_file.read_text()))

    return {
        "meter_id": meter_id,
        "model_exists": exists_temporal,
        "monthly_model_exists": exists_monthly,
        "labels_count": n_labels,
    }


@app.delete("/models/{meter_id}", summary="Eliminar modelo (fuerza re-entrenamiento)")
def delete_model_endpoint(meter_id: str):
    """Elimina el modelo de un contador para forzar un re-entrenamiento."""
    deleted = delete_model(meter_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No model found for '{meter_id}'")
    return {"meter_id": meter_id, "status": "deleted"}


@app.get("/health", summary="Health check")
def health():
    """Comprueba que el servicio está funcionando."""
    models_count = len(list(Path("models").glob("*.pkl"))) if Path("models").exists() else 0
    labels_count = len(list(LABELS_DIR.glob("*_labels.json")))
    return {
        "status": "ok",
        "models_trained": models_count,
        "meters_with_labels": labels_count,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
