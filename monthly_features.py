"""
Feature engineering para datos mensuales por barrio/zona.

Estrategia A — Cross-sectional: compara barrios entre sí en el mismo período.
No requiere granularidad horaria. Funciona directamente con los datos del hackathon.

Features calculadas:
  - consumption_per_contract  : normaliza por tamaño del barrio
  - yoy_ratio                 : comparación año contra año (mismo mes)
  - seasonal_zscore           : zscore dentro del mismo mes a lo largo de los años
  - cross_sectional_zscore    : zscore comparando barrios del mismo tipo en ese mes
  - type_percentile           : percentil dentro de la categoría (DOMESTICO/INDUSTRIAL)
  - trend_3m                  : pendiente de los últimos 3 meses (regresión lineal)
  - months_above_mean         : racha de meses seguidos por encima de la media histórica

Input esperado: DataFrame con columnas:
  barrio, uso, fecha (datetime), consumo_litros (float), num_contratos (float)

Output: DataFrame con una fila por (barrio, uso, mes) con los features calculados.
"""

import numpy as np
import pandas as pd
from typing import Optional

MONTHLY_FEATURE_COLUMNS = [
    "consumption_per_contract",
    "yoy_ratio",
    "seasonal_zscore",
    "cross_sectional_zscore",
    "type_percentile",
    "trend_3m",
    "months_above_mean",
    "deviation_from_group_trend",
    "relative_consumption",
    "group_yoy_median",
]

# Features exclusivamente relativos (sin valores absolutos)
# Para M2 IsolationForest: evita que la tendencia global contamine la deteccion
RELATIVE_FEATURE_COLUMNS = [
    "deviation_from_group_trend",
    "relative_consumption",
    "seasonal_zscore",
    "cross_sectional_zscore",
    "type_percentile",
    "trend_3m",
    "months_above_mean",
]

# Features extendidos con datos externos (temperatura, turismo, sequia)
EXTENDED_FEATURE_COLUMNS = MONTHLY_FEATURE_COLUMNS + [
    "temp_adjusted_consumption",
    "tourism_adjusted_consumption",
    "spei_index",
]

# Features relativos + datos externos
RELATIVE_EXTENDED_FEATURE_COLUMNS = RELATIVE_FEATURE_COLUMNS + [
    "temp_adjusted_consumption",
    "tourism_adjusted_consumption",
    "spei_index",
]


def compute_monthly_features(df: pd.DataFrame,
                              external_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calcula features cross-seccionales y temporales para todos los barrios a la vez.

    Args:
        df:          DataFrame con columnas [barrio, uso, fecha, consumo_litros, num_contratos]
        external_df: DataFrame con datos externos (salida de load_external_data).
                     Si se pasa, calcula 3 features extra: temp_adjusted_consumption,
                     tourism_adjusted_consumption, spei_index.

    Returns:
        DataFrame con una fila por (barrio, uso, mes) y las features calculadas.
        Incluye columna 'barrio_key' = f"{barrio}__{uso}" como identificador.
    """
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["year"]  = df["fecha"].dt.year
    df["month"] = df["fecha"].dt.month
    df["barrio_key"] = df["barrio"].str.strip() + "__" + df["uso"].str.strip()

    # Normalizar consumo por número de contratos
    df["consumption_per_contract"] = np.where(
        df["num_contratos"] > 0,
        df["consumo_litros"] / df["num_contratos"],
        df["consumo_litros"]
    )

    df = _add_yoy_ratio(df)
    df = _add_seasonal_zscore(df)
    df = _add_cross_sectional_zscore(df)
    df = _add_type_percentile(df)
    df = _add_trend_3m(df)
    df = _add_months_above_mean(df)
    df = _add_group_trend_deviation(df)
    df = _add_relative_consumption(df)

    # Features extendidos con datos externos
    if external_df is not None:
        df = _add_external_features(df, external_df)

    return df


def _add_yoy_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio actual / mismo mes año anterior.
    yoy_ratio > 1.5 → consumo 50% superior al año pasado (sospechoso)
    yoy_ratio < 0.5 → consumo 50% inferior al año pasado (también sospechoso)
    """
    df = df.copy()
    # Consumo del mismo barrio, mismo mes, año anterior
    prev_year = df[["barrio_key", "year", "month", "consumo_litros"]].copy()
    prev_year["year"] = prev_year["year"] + 1  # desplazar para hacer join
    prev_year = prev_year.rename(columns={"consumo_litros": "consumo_prev_year"})

    df = df.merge(prev_year[["barrio_key", "year", "month", "consumo_prev_year"]],
                  on=["barrio_key", "year", "month"], how="left")

    df["yoy_ratio"] = np.where(
        (df["consumo_prev_year"] > 0) & df["consumo_prev_year"].notna(),
        df["consumo_litros"] / df["consumo_prev_year"],
        1.0  # sin histórico previo → ratio neutro
    )
    df = df.drop(columns=["consumo_prev_year"])
    return df


def _add_seasonal_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score del barrio respecto a todos los valores históricos del mismo mes.
    Ej: ¿cómo se compara enero 2024 con todos los eneros históricos de ese barrio?
    """
    df = df.copy()
    seasonal_stats = (
        df.groupby(["barrio_key", "month"])["consumo_litros"]
        .agg(seasonal_mean="mean", seasonal_std="std")
        .reset_index()
    )
    df = df.merge(seasonal_stats, on=["barrio_key", "month"], how="left")
    df["seasonal_zscore"] = np.where(
        df["seasonal_std"] > 0,
        (df["consumo_litros"] - df["seasonal_mean"]) / df["seasonal_std"],
        0.0
    )
    df = df.drop(columns=["seasonal_mean", "seasonal_std"])
    return df


def _add_cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score del barrio respecto a TODOS los barrios del mismo tipo en el mismo mes-año.
    Detecta: "este barrio consume 3x más que la media de barrios DOMESTICO en ese mes".
    """
    df = df.copy()
    cross_stats = (
        df.groupby(["uso", "year", "month"])["consumption_per_contract"]
        .agg(cross_mean="mean", cross_std="std")
        .reset_index()
    )
    df = df.merge(cross_stats, on=["uso", "year", "month"], how="left")
    df["cross_sectional_zscore"] = np.where(
        df["cross_std"] > 0,
        (df["consumption_per_contract"] - df["cross_mean"]) / df["cross_std"],
        0.0
    )
    df = df.drop(columns=["cross_mean", "cross_std"])
    return df


def _add_type_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Percentil del barrio dentro de su categoría (uso) en cada mes-año.
    type_percentile = 0.99 → consume más que el 99% de barrios similares ese mes.
    """
    def _percentile_rank(group):
        vals = group["consumption_per_contract"].values
        ranks = np.array([
            np.sum(vals <= v) / len(vals) for v in vals
        ])
        return pd.Series(ranks, index=group.index)

    df = df.copy()
    df["type_percentile"] = (
        df.groupby(["uso", "year", "month"], group_keys=False)
        .apply(_percentile_rank, include_groups=False)
    )
    return df


def _add_trend_3m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pendiente de regresión lineal sobre los últimos 3 meses de consumo/contrato.
    trend_3m > 0 → tendencia creciente
    trend_3m >> 0 → crecimiento acelerado (sospechoso)
    """
    df = df.copy().sort_values(["barrio_key", "fecha"])

    def _slope(values):
        """Pendiente normalizada de los últimos N valores."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values), dtype=float)
        if np.std(values) == 0:
            return 0.0
        # Pendiente normalizada por la media para que sea comparable entre barrios
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0
        slope = np.polyfit(x, values, 1)[0]
        return slope / mean_val  # pendiente relativa

    trends = []
    for barrio_key, group in df.groupby("barrio_key"):
        vals = group["consumption_per_contract"].values
        for i in range(len(vals)):
            window = vals[max(0, i - 2): i + 1]  # últimos 3 meses
            trends.append(_slope(window))

    df["trend_3m"] = trends
    return df


def _add_months_above_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Racha de meses consecutivos en que el consumo está por encima de la media histórica.
    months_above_mean = 6 → lleva 6 meses seguidos por encima de su media histórica.
    Una racha larga puede indicar fuga no detectada acumulada.
    """
    df = df.copy().sort_values(["barrio_key", "fecha"])

    # Media histórica por barrio (toda la serie)
    hist_mean = (
        df.groupby("barrio_key")["consumo_litros"]
        .mean()
        .rename("hist_mean")
    )
    df = df.join(hist_mean, on="barrio_key")
    df["above_mean"] = (df["consumo_litros"] > df["hist_mean"]).astype(int)

    streaks = []
    for _, group in df.groupby("barrio_key"):
        streak = 0
        group_streaks = []
        for val in group["above_mean"].values:
            streak = streak + 1 if val else 0
            group_streaks.append(streak)
        streaks.extend(group_streaks)

    df["months_above_mean"] = streaks
    df = df.drop(columns=["hist_mean", "above_mean"])
    return df


def _add_group_trend_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mide cuanto se desvia el cambio YoY de un barrio respecto al grupo.

    Si todos los barrios del mismo tipo suben un 4% en julio 2024,
    eso NO es anomalo. Pero si uno sube un 22%, su deviation sera +0.18.

    deviation_from_group_trend = yoy_ratio_barrio - median(yoy_ratio_grupo)
    group_yoy_median = la mediana del grupo (para contexto)
    """
    df = df.copy()

    # Calcular la mediana de yoy_ratio por (uso, year, month)
    group_medians = (
        df.groupby(["uso", "year", "month"])["yoy_ratio"]
        .median()
        .rename("group_yoy_median")
        .reset_index()
    )
    df = df.merge(group_medians, on=["uso", "year", "month"], how="left")

    # Desviacion respecto al grupo
    df["deviation_from_group_trend"] = df["yoy_ratio"] - df["group_yoy_median"].fillna(1.0)

    return df


def _add_relative_consumption(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio del consumo per contract del barrio vs la mediana del grupo.

    relative_consumption = 1.0 → consume igual que la mediana del grupo
    relative_consumption = 2.0 → consume el doble que la mediana
    relative_consumption = 0.5 → consume la mitad

    Mas util que el valor absoluto porque normaliza por el nivel del grupo.
    """
    df = df.copy()

    group_median_cpc = (
        df.groupby(["uso", "year", "month"])["consumption_per_contract"]
        .median()
        .rename("_group_median_cpc")
        .reset_index()
    )
    df = df.merge(group_median_cpc, on=["uso", "year", "month"], how="left")

    df["relative_consumption"] = np.where(
        df["_group_median_cpc"] > 0,
        df["consumption_per_contract"] / df["_group_median_cpc"],
        1.0,
    )
    df = df.drop(columns=["_group_median_cpc"])
    return df


def prepare_monthly_matrix(df: pd.DataFrame,
                           extended: bool = False) -> Optional[np.ndarray]:
    """
    Prepara la matriz numpy para entrenar el modelo.
    Elimina filas con NaN.

    Args:
        df:       DataFrame con features calculados
        extended: si True, incluye los 3 features de datos externos
    """
    cols = EXTENDED_FEATURE_COLUMNS if extended else MONTHLY_FEATURE_COLUMNS
    available = [c for c in cols if c in df.columns]
    subset = df[available].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 10:
        return None
    return subset.values


def monthly_features_to_vector(row: pd.Series) -> list:
    """Convierte una fila en vector para el modelo (orden fijo)."""
    return [float(row.get(col, 0.0) or 0.0) for col in MONTHLY_FEATURE_COLUMNS]


def _add_external_features(df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge datos externos y calcula features ajustados por temperatura y turismo.

    - temp_adjusted_consumption: consumo/contrato eliminando el efecto de la temperatura.
      Formula: cpc / (1 + alpha * (temp - baseline)). Si hace calor, el consumo alto
      es "esperado", asi que lo normalizamos. Valores altos DESPUES de ajustar = sospechoso.

    - tourism_adjusted_consumption: consumo/contrato eliminando el efecto turistico.
      Formula: cpc / (1 + beta * occupancy). Barrios con hoteles suben en verano.

    - spei_index: indice de sequia (pass-through desde datos externos).
    """
    df = df.copy()
    ext = external_df.copy()

    # Normalizar fechas a primer dia del mes para el merge
    df["_merge_month"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    ext["_merge_month"] = pd.to_datetime(ext["fecha"]).dt.to_period("M").dt.to_timestamp()
    ext = ext.drop(columns=["fecha"])

    df = df.merge(ext, on="_merge_month", how="left")
    df = df.drop(columns=["_merge_month"])

    # Consumo ajustado por temperatura
    # alpha=0.03 → por cada grado por encima de 18C, se espera un 3% mas de consumo
    baseline_temp = 18.0
    alpha = 0.03
    if "avg_temp" in df.columns:
        temp_factor = 1 + alpha * (df["avg_temp"].fillna(baseline_temp) - baseline_temp)
        temp_factor = temp_factor.clip(lower=0.5)  # evitar divisiones extremas
        df["temp_adjusted_consumption"] = df["consumption_per_contract"] / temp_factor

    # Consumo ajustado por turismo
    # beta=0.002 → ocupacion hotelera del 80% sube el consumo esperado un 16%
    beta = 0.002
    if "tourist_occupancy_pct" in df.columns:
        tourism_factor = 1 + beta * df["tourist_occupancy_pct"].fillna(0)
        df["tourism_adjusted_consumption"] = df["consumption_per_contract"] / tourism_factor

    # SPEI index (ya viene del merge, rellenar NaN con 0.0 = neutral)
    if "spei_index" not in df.columns:
        df["spei_index"] = 0.0
    else:
        df["spei_index"] = df["spei_index"].fillna(0.0)

    return df


# ─────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from train_local import load_hackathon_amaem

    data_file = "data/datos-hackathon-amaem.xlsx-set-de-datos-.csv"
    if not __import__("pathlib").Path(data_file).exists():
        print(f"ERROR: No se encuentra {data_file}")
        sys.exit(1)

    print(f"Cargando datos...")
    df_raw = load_hackathon_amaem(data_file)

    print(f"\nCalculando features mensuales cross-seccionales...")
    df_features = compute_monthly_features(df_raw)

    print(f"\nFeatures calculadas para {df_features['barrio_key'].nunique()} barrios")
    print(f"Total filas: {len(df_features)}")

    # Mostrar muestra de un barrio
    barrio_ejemplo = "10-FLORIDA BAJA__DOMESTICO"
    sample = df_features[df_features["barrio_key"] == barrio_ejemplo].tail(6)
    print(f"\nEjemplo — {barrio_ejemplo} (últimos 6 meses):")
    cols = ["fecha", "consumo_litros", "consumption_per_contract",
            "yoy_ratio", "seasonal_zscore", "cross_sectional_zscore",
            "type_percentile", "trend_3m", "months_above_mean"]
    available = [c for c in cols if c in sample.columns]
    print(sample[available].to_string(index=False))

    # Top 10 anomalías por cross_sectional_zscore
    print(f"\nTop 10 barrios con mayor cross_sectional_zscore (posibles anomalías):")
    top = (df_features
           .nlargest(10, "cross_sectional_zscore")
           [["barrio_key", "fecha", "consumo_litros", "consumption_per_contract",
             "cross_sectional_zscore", "yoy_ratio"]]
           )
    print(top.to_string(index=False))
