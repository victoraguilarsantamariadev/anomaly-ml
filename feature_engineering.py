"""
Feature Engineering para detección de anomalías en contadores.

Funciona con cualquier tipo de suministro: agua, electricidad, gas, calor.
La granularidad mínima recomendada es horaria para mejores resultados,
pero también funciona con datos diarios.

Input esperado: DataFrame con columnas ['timestamp', 'consumption']
Output: DataFrame con una fila por día y todos los features calculados
"""

import pandas as pd
import numpy as np
from typing import Optional

# Features que se pasan al modelo (orden fijo, no cambiar)
FEATURE_COLUMNS = [
    "daily_total",
    "nocturnal_min",
    "nocturnal_mean",
    "diurnal_mean",
    "night_day_ratio",
    "active_hours",
    "is_weekend",
    "zscore",
]


def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features diarios a partir de lecturas horarias.

    Args:
        df: DataFrame con columnas ['timestamp', 'consumption']
            - timestamp: datetime con zona horaria o naive
            - consumption: consumo en la unidad del suministro (litros, kWh, m³)

    Returns:
        DataFrame con una fila por día y los features calculados.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek  # 0=Lunes, 6=Domingo
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    daily = (
        df.groupby("date")
        .apply(_compute_day_features, include_groups=False)
        .reset_index()
    )

    # Añadir is_weekend desde el día
    daily["date"] = pd.to_datetime(daily["date"])
    daily["is_weekend"] = (daily["date"].dt.dayofweek >= 5).astype(int)

    return daily


def _compute_day_features(group: pd.DataFrame) -> pd.Series:
    """Calcula features para un día concreto."""
    # Horas nocturnas (02:00 - 05:00) — indicador principal de fugas de agua
    nocturnal_mask = group["hour"].between(2, 5)
    nocturnal = group.loc[nocturnal_mask, "consumption"]

    # Horas diurnas (08:00 - 22:00) — consumo activo normal
    diurnal_mask = group["hour"].between(8, 22)
    diurnal = group.loc[diurnal_mask, "consumption"]

    diurnal_mean = diurnal.mean() if len(diurnal) > 0 else 0.0
    nocturnal_mean = nocturnal.mean() if len(nocturnal) > 0 else 0.0

    return pd.Series(
        {
            # Volumen total del día
            "daily_total": group["consumption"].sum(),
            # Indicadores nocturnos (fugas de agua fluyen 24h, incluso de noche)
            "nocturnal_min": nocturnal.min() if len(nocturnal) > 0 else 0.0,
            "nocturnal_mean": nocturnal_mean,
            # Consumo diurno (patrón normal de actividad)
            "diurnal_mean": diurnal_mean,
            # Ratio noche/día — si es alto y no debería serlo, posible fuga
            "night_day_ratio": (
                nocturnal_mean / diurnal_mean if diurnal_mean > 0.001 else 0.0
            ),
            # Horas con consumo > 0 (perfil de actividad del día)
            "active_hours": (group["consumption"] > 0).sum(),
        }
    )


def add_rolling_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Añade features de tendencia temporal (requiere varios días de histórico).
    Llama a esta función DESPUÉS de compute_daily_features.
    """
    daily = daily.copy()
    daily = daily.sort_values("date").reset_index(drop=True)

    # Media móvil 7 días — tendencia reciente
    daily["moving_avg_7d"] = (
        daily["daily_total"].rolling(7, min_periods=3).mean()
    )

    # Desviación estándar móvil 7 días — variabilidad normal
    daily["moving_std_7d"] = (
        daily["daily_total"].rolling(7, min_periods=3).std()
    )

    # Z-score: ¿cuántas desviaciones estándar se aleja hoy de su media reciente?
    # zscore > 2 = raro | > 3 = muy raro | > 4 = casi seguro anomalía
    daily["zscore"] = np.where(
        daily["moving_std_7d"] > 0,
        (daily["daily_total"] - daily["moving_avg_7d"]) / daily["moving_std_7d"],
        0.0
    )

    return daily


def features_to_vector(row: pd.Series) -> list:
    """
    Convierte una fila del DataFrame de features en un vector para el modelo.
    El orden debe coincidir con FEATURE_COLUMNS.
    """
    return [float(row.get(col, 0.0) or 0.0) for col in FEATURE_COLUMNS]


def prepare_training_matrix(daily: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Prepara la matriz de entrenamiento para el modelo.
    Elimina filas con NaN y devuelve numpy array.
    """
    subset = daily[FEATURE_COLUMNS].dropna()
    if len(subset) < 10:
        return None  # Muy pocos datos para entrenar de forma fiable
    return subset.values
