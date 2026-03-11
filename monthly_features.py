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
    "zscore_rolling_3m",
    "above_mean_streak",
    "trend_accel",
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
    "zscore_rolling_3m",
    "above_mean_streak",
    "trend_accel",
]

# Features avanzados (lag, rolling, log-transform)
ADVANCED_FEATURE_COLUMNS = [
    "log_consumption",
    "consumption_volatility",
    "momentum",
    "yoy_acceleration",
]

# Features Fourier + interaccion
FOURIER_INTERACTION_COLUMNS = [
    "sin_month",
    "cos_month",
    "sin_quarter",
    "cos_quarter",
    "consumption_x_contracts",
    "volatility_x_trend",
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

# Features auxiliares de datasets adicionales (contadores, regenerada)
AUXILIARY_FEATURE_COLUMNS = [
    "smart_meter_ratio",
    "avg_calibre",
    "regenerada_ratio",
]

# Features de infraestructura GIS (estaticos por barrio)
INFRASTRUCTURE_FEATURE_COLUMNS = [
    "pipe_density_km_per_km2",
    "avg_pipe_diameter_mm",
    "sewer_density_km_per_km2",
    "pct_sewer_unitaria",
    "hydrant_density_per_km2",
    "elevation_mean",
    "elevation_range",
    "imbornal_density_per_km2",
    "colector_coverage_pct",
    "infrastructure_risk_score",
]

# Features temporales de datos auxiliares (nivel ciudad, variantes por mes)
TEMPORAL_AUX_FEATURE_COLUMNS = [
    "net_new_contracts",
    "growth_momentum",
    "inspection_coverage_pct",
]

# Features de EDAR (depuradora, nivel ciudad, temporal)
EDAR_FEATURE_COLUMNS = [
    "treated_m3",
    "reuse_ratio",
]

# Features demograficos del Padron Municipal de Alicante 2025
# Fuente: alicante.es/estadisticas-poblacion (barrios_2025.xls, miembros_barrio_2025.xls)
DEMOGRAPHIC_FEATURE_COLUMNS = [
    "pct_elderly_65plus",       # % poblacion >= 65 (Padron 2025)
    "pct_elderly_80plus",       # % poblacion >= 80 (muy mayor)
    "pct_elderly_alone",        # % mayores 65+ que viven SOLOS
    "population_density",       # poblacion total del barrio (proxy densidad)
    "elderly_consumption_ratio",  # consumo / esperado_por_demografia
    "elderly_x_drop",           # interaccion: pct_elderly * caida_consumo
    "alone_x_volatility",       # interaccion: pct_solos * volatilidad_consumo
]


def compute_monthly_features(df: pd.DataFrame,
                              external_df: Optional[pd.DataFrame] = None,
                              cutoff_date=None) -> pd.DataFrame:
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
    df = _add_seasonal_zscore(df, cutoff_date=cutoff_date)
    df = _add_cross_sectional_zscore(df)  # point-in-time (groups by year+month) — no leak
    df = _add_type_percentile(df)         # point-in-time — no leak
    df = _add_trend_3m(df)
    df = _add_months_above_mean(df, cutoff_date=cutoff_date)
    df = _add_group_trend_deviation(df)
    df = _add_relative_consumption(df)
    df = _add_persistence_features(df)
    df = _add_advanced_features(df)
    df = _add_fourier_interaction_features(df)

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


def _add_seasonal_zscore(df: pd.DataFrame, cutoff_date=None) -> pd.DataFrame:
    """
    Z-score del barrio respecto a todos los valores históricos del mismo mes.
    Ej: ¿cómo se compara enero 2024 con todos los eneros históricos de ese barrio?
    If cutoff_date is provided, stats are computed ONLY from data before cutoff
    to prevent look-ahead bias.
    """
    df = df.copy()
    stats_df = df if cutoff_date is None else df[df["fecha"] < cutoff_date]
    seasonal_stats = (
        stats_df.groupby(["barrio_key", "month"])["consumo_litros"]
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


def _add_months_above_mean(df: pd.DataFrame, cutoff_date=None) -> pd.DataFrame:
    """
    Racha de meses consecutivos en que el consumo está por encima de la media histórica.
    months_above_mean = 6 → lleva 6 meses seguidos por encima de su media histórica.
    Una racha larga puede indicar fuga no detectada acumulada.
    If cutoff_date is provided, hist_mean is computed ONLY from pre-cutoff data.
    """
    df = df.copy().sort_values(["barrio_key", "fecha"])

    # Media histórica por barrio (only pre-cutoff to avoid look-ahead bias)
    stats_df = df if cutoff_date is None else df[df["fecha"] < cutoff_date]
    hist_mean = (
        stats_df.groupby("barrio_key")["consumo_litros"]
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


def _add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features avanzados para mejorar la deteccion:
      - log_consumption: log-transform del consumo (normaliza distribuciones sesgadas)
      - lag_1, lag_2, lag_12: consumo de hace 1, 2, 12 meses
      - rolling_mean_3, rolling_std_3: media y volatilidad ventana 3 meses
      - rolling_mean_6: media ventana 6 meses
      - consumption_volatility: rolling_std_3 / rolling_mean_3 (CV)
      - momentum: cambio relativo mes actual vs lag_1
      - yoy_acceleration: cambio del yoy_ratio respecto al mes anterior
    """
    df = df.copy()
    df = df.sort_values(["barrio_key", "fecha"])

    # Log-transform (evitar log(0) con clip)
    df["log_consumption"] = np.log1p(df["consumption_per_contract"].clip(lower=0))

    # Lag features y rolling stats por barrio
    for col_name in ["lag_1", "lag_2", "lag_12",
                     "rolling_mean_3", "rolling_std_3", "rolling_mean_6",
                     "consumption_volatility", "momentum", "yoy_acceleration"]:
        df[col_name] = np.nan

    for barrio_key, grp in df.groupby("barrio_key"):
        idx = grp.index
        cpc = grp["consumption_per_contract"]

        df.loc[idx, "lag_1"] = cpc.shift(1)
        df.loc[idx, "lag_2"] = cpc.shift(2)
        df.loc[idx, "lag_12"] = cpc.shift(12)

        df.loc[idx, "rolling_mean_3"] = cpc.rolling(3, min_periods=1).mean()
        df.loc[idx, "rolling_std_3"] = cpc.rolling(3, min_periods=2).std()
        df.loc[idx, "rolling_mean_6"] = cpc.rolling(6, min_periods=1).mean()

        # Coeficiente de variacion (volatilidad relativa)
        rm3 = df.loc[idx, "rolling_mean_3"]
        rs3 = df.loc[idx, "rolling_std_3"]
        df.loc[idx, "consumption_volatility"] = np.where(rm3 > 0, rs3 / rm3, 0)

        # Momentum: cambio relativo vs mes anterior
        lag1 = cpc.shift(1)
        df.loc[idx, "momentum"] = np.where(lag1 > 0, (cpc - lag1) / lag1, 0)

        # Aceleracion del YoY ratio
        if "yoy_ratio" in grp.columns:
            yoy = grp["yoy_ratio"]
            df.loc[idx, "yoy_acceleration"] = yoy - yoy.shift(1)

    # Fillna con valores neutrales
    df["lag_1"] = df["lag_1"].fillna(df["consumption_per_contract"])
    df["lag_2"] = df["lag_2"].fillna(df["consumption_per_contract"])
    df["lag_12"] = df["lag_12"].fillna(df["consumption_per_contract"])
    df["rolling_std_3"] = df["rolling_std_3"].fillna(0)
    df["consumption_volatility"] = df["consumption_volatility"].fillna(0)
    df["momentum"] = df["momentum"].fillna(0)
    df["yoy_acceleration"] = df["yoy_acceleration"].fillna(0)

    return df


def _add_fourier_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de estacionalidad Fourier + interacciones entre features.

    Fourier (sin/cos): capturan patrones ciclicos sin asumir forma lineal.
      - sin_month/cos_month: ciclo anual (12 meses)
      - sin_quarter/cos_quarter: ciclo trimestral (alta freq, captura picos verano/invierno)

    Interacciones:
      - consumption_x_contracts: consumo absoluto * num_contratos (escala del barrio)
      - volatility_x_trend: volatilidad * tendencia (barrios con variabilidad creciente)
    """
    df = df.copy()

    # Fourier: ciclo anual
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # Fourier: ciclo trimestral (segundo armonico)
    df["sin_quarter"] = np.sin(2 * np.pi * df["month"] / 6)
    df["cos_quarter"] = np.cos(2 * np.pi * df["month"] / 6)

    # Interaccion: escala del barrio (consumo * contratos normalizado)
    cpc = df["consumption_per_contract"].clip(lower=0)
    contracts = df["num_contratos"].clip(lower=1)
    raw = np.log1p(cpc) * np.log1p(contracts)
    df["consumption_x_contracts"] = raw

    # Interaccion: volatilidad * tendencia (alerta temprana de deterioro)
    vol = df.get("consumption_volatility", pd.Series(0, index=df.index))
    trend = df.get("trend_3m", pd.Series(0, index=df.index))
    df["volatility_x_trend"] = vol * trend

    return df


def _add_persistence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de persistencia temporal — separan ruido puntual de señal real.

    Una anomalía que persiste 3+ meses es probablemente una fuga activa.
    Una anomalía puntual (1 mes) es más probablemente ruido estacional.
    """
    df = df.copy().sort_values(["barrio_key", "fecha"])

    # 1. Rolling z-score (3 meses): ¿el consumo lleva 3+ meses desviado?
    if "seasonal_zscore" in df.columns:
        df["zscore_rolling_3m"] = df.groupby("barrio_key")["seasonal_zscore"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
    else:
        df["zscore_rolling_3m"] = 0.0

    # 2. Streak above mean: ¿cuántos de los últimos 6 meses estuvo por encima?
    if "months_above_mean" in df.columns:
        df["above_mean_streak"] = df.groupby("barrio_key")["months_above_mean"].transform(
            lambda x: x.rolling(6, min_periods=1).sum()
        )
    else:
        df["above_mean_streak"] = 0.0

    # 3. Trend acceleration: 2nd derivative del consumo per contract
    df["trend_accel"] = df.groupby("barrio_key")["consumption_per_contract"].transform(
        lambda x: x.diff().diff()
    )
    df["trend_accel"] = df["trend_accel"].fillna(0)

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
# Enrichment con datos auxiliares
# ─────────────────────────────────────────────────────────────────

def enrich_with_telelectura(df: pd.DataFrame, contadores_path: str) -> pd.DataFrame:
    """
    Enriquece features con datos de contadores-telelectura.

    Añade por barrio (static, cross-sectional):
      - smart_meter_ratio: % de contadores con telelectura (vs manual)
      - avg_calibre: calibre medio de contadores (13=doméstico, 20+=comercial)

    Args:
        df: DataFrame con features (output de compute_monthly_features)
        contadores_path: ruta al CSV de contadores-telelectura
    """
    import os
    if not os.path.exists(contadores_path):
        return df

    cont = pd.read_csv(contadores_path)

    # Normalizar USO: quitar acentos para match con hackathon
    uso_map = {
        "DOMÉSTICO": "DOMESTICO",
        "COMERCIAL": "COMERCIAL",
        "NO DOMÉSTICO": "NO DOMESTICO",
        "INDUSTRIAL": "INDUSTRIAL",
        "COMUNIDAD PROPIETARIOS": "DOMESTICO",  # agrupar con doméstico
    }
    cont["uso_norm"] = cont["USO"].map(uso_map).fillna("OTRO")

    # Agrupar por barrio + uso normalizado
    stats = cont.groupby(["BARRIO", "uso_norm"]).agg(
        total_meters=("CALIBRE", "size"),
        smart_count=("SISTEMA", lambda x: (x == "Leer por telelectura").sum()),
        avg_calibre=("CALIBRE", "mean"),
    ).reset_index()

    stats["smart_meter_ratio"] = stats["smart_count"] / stats["total_meters"]
    stats = stats.rename(columns={"BARRIO": "barrio", "uso_norm": "uso"})
    stats = stats[["barrio", "uso", "smart_meter_ratio", "avg_calibre"]]

    # Merge con features (barrio + uso)
    df = df.copy()
    df["_barrio_clean"] = df["barrio"].str.strip()
    df["_uso_clean"] = df["uso"].str.strip()

    df = df.merge(
        stats,
        left_on=["_barrio_clean", "_uso_clean"],
        right_on=["barrio", "uso"],
        how="left",
        suffixes=("", "_cont"),
    )
    df = df.drop(columns=["barrio_cont", "uso_cont", "_barrio_clean", "_uso_clean"],
                 errors="ignore")

    # Rellenar NaN con valores neutros
    df["smart_meter_ratio"] = df["smart_meter_ratio"].fillna(0.5)
    df["avg_calibre"] = df["avg_calibre"].fillna(13.0)

    return df


def enrich_with_regenerada(df: pd.DataFrame, regenerada_path: str) -> pd.DataFrame:
    """
    Enriquece features con datos de consumo de agua regenerada por barrio.

    Añade:
      - regenerada_ratio: consumo_regenerada / consumo_potable (solo 2024)
        Barrios con agua regenerada son típicamente industriales/agrícolas.

    Args:
        df: DataFrame con features (output de compute_monthly_features)
        regenerada_path: ruta al CSV de consumos regenerada
    """
    import os
    if not os.path.exists(regenerada_path):
        df["regenerada_ratio"] = 0.0
        return df

    reg = pd.read_csv(regenerada_path)
    # Columnas: LOCALIDAD, BARRIO, MES, CONSUMO_2024

    # Calcular total anual regenerada por barrio
    reg_annual = reg.groupby("BARRIO")["CONSUMO_2024"].sum().reset_index()
    reg_annual = reg_annual.rename(columns={"BARRIO": "barrio", "CONSUMO_2024": "regenerada_total"})

    # Calcular total anual potable por barrio (del dataset principal, solo 2024)
    df = df.copy()
    df_2024 = df[df["fecha"].dt.year == 2024]
    potable_annual = (
        df_2024.groupby(df_2024["barrio"].str.strip())["consumo_litros"]
        .sum()
        .reset_index()
        .rename(columns={"barrio": "barrio", "consumo_litros": "potable_total"})
    )

    # Merge y calcular ratio
    ratios = reg_annual.merge(potable_annual, on="barrio", how="outer")
    ratios["regenerada_ratio"] = np.where(
        ratios["potable_total"] > 0,
        ratios["regenerada_total"].fillna(0) / ratios["potable_total"],
        0.0,
    )
    ratios = ratios[["barrio", "regenerada_ratio"]]

    # Merge con df principal
    df["_barrio_clean"] = df["barrio"].str.strip()
    df = df.merge(ratios, left_on="_barrio_clean", right_on="barrio",
                  how="left", suffixes=("", "_reg"))
    df = df.drop(columns=["barrio_reg", "_barrio_clean"], errors="ignore")
    df["regenerada_ratio"] = df["regenerada_ratio"].fillna(0.0)

    return df


# ─────────────────────────────────────────────────────────────────
# Enrichment con infraestructura GIS y datos auxiliares
# ─────────────────────────────────────────────────────────────────

def enrich_with_infrastructure(df: pd.DataFrame,
                                infra_df: pd.DataFrame,
                                risk_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Enriquece features con datos de infraestructura GIS.

    Añade por barrio (estatico):
      - pipe_density_km_per_km2: densidad de red de abastecimiento
      - avg_pipe_diameter_mm: diametro medio de tuberias
      - sewer_density_km_per_km2: densidad de red de saneamiento
      - pct_sewer_unitaria: % de red unitaria (vs separativa)
      - hydrant_density_per_km2: hidrantes + bocas por km²
      - elevation_mean: elevacion media (m)
      - elevation_range: rango de elevacion (m)
      - infrastructure_risk_score: indice de riesgo (0-5)

    Args:
        df: DataFrame con features (output de compute_monthly_features)
        infra_df: DataFrame de load_infrastructure_features()
        risk_df: DataFrame de compute_infrastructure_risk() (opcional)
    """
    if infra_df.empty:
        for col in INFRASTRUCTURE_FEATURE_COLUMNS:
            df[col] = 0.0
        return df

    df = df.copy()

    # Merge infraestructura por barrio
    infra_cols = [c for c in INFRASTRUCTURE_FEATURE_COLUMNS
                  if c in infra_df.columns and c != "infrastructure_risk_score"]
    infra_merge = infra_df[["barrio"] + infra_cols].copy()

    df["_barrio_clean"] = df["barrio"].str.strip()
    df = df.merge(infra_merge, left_on="_barrio_clean", right_on="barrio",
                  how="left", suffixes=("", "_infra"))
    df = df.drop(columns=["barrio_infra", "_barrio_clean"], errors="ignore")

    # Rellenar NaN con medianas
    for col in infra_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    # Merge riesgo si disponible
    if risk_df is not None and not risk_df.empty:
        df["_barrio_clean"] = df["barrio"].str.strip()
        df = df.merge(risk_df, left_on="_barrio_clean", right_on="barrio",
                      how="left", suffixes=("", "_risk"))
        df = df.drop(columns=["barrio_risk", "_barrio_clean"], errors="ignore")
        df["infrastructure_risk_score"] = df["infrastructure_risk_score"].fillna(
            df["infrastructure_risk_score"].median() if df["infrastructure_risk_score"].notna().any() else 2.5
        )
    elif "infrastructure_risk_score" not in df.columns:
        df["infrastructure_risk_score"] = 2.5  # neutral

    return df


def enrich_with_contract_growth(df: pd.DataFrame,
                                 growth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquece features con datos de crecimiento de contratos.

    Añade por mes (nivel ciudad, temporal):
      - net_new_contracts: altas netas en ese mes
      - growth_momentum: media movil 3 meses de altas netas

    Args:
        df: DataFrame con features
        growth_df: DataFrame de load_contract_growth()
    """
    if growth_df.empty:
        df["net_new_contracts"] = 0
        df["growth_momentum"] = 0
        return df

    df = df.copy()
    growth = growth_df[["fecha", "net_new_contracts", "growth_momentum"]].copy()

    # Normalizar fechas para merge
    df["_merge_month"] = pd.to_datetime(df["fecha"]).dt.to_period("M").dt.to_timestamp()
    growth["_merge_month"] = pd.to_datetime(growth["fecha"]).dt.to_period("M").dt.to_timestamp()
    growth = growth.drop(columns=["fecha"])

    df = df.merge(growth, on="_merge_month", how="left")
    df = df.drop(columns=["_merge_month"])

    df["net_new_contracts"] = df["net_new_contracts"].fillna(0)
    df["growth_momentum"] = df["growth_momentum"].fillna(0)

    return df


def enrich_with_network_stats(df: pd.DataFrame,
                               network_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquece features con estadisticas de red de abastecimiento.

    Añade por año (nivel ciudad):
      - inspection_coverage_pct: % de red inspeccionada con buscafugas

    Args:
        df: DataFrame con features
        network_df: DataFrame de load_network_stats()
    """
    if network_df.empty:
        df["inspection_coverage_pct"] = 100.0
        return df

    df = df.copy()
    df["_year_net"] = pd.to_datetime(df["fecha"]).dt.year

    net = network_df[["year", "inspection_coverage_pct"]].copy()
    df = df.merge(net, left_on="_year_net", right_on="year", how="left")
    df = df.drop(columns=["year", "_year_net"], errors="ignore")

    df["inspection_coverage_pct"] = df["inspection_coverage_pct"].fillna(100.0)

    return df


def enrich_with_edar(df: pd.DataFrame,
                      edar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquece features con datos de la depuradora EDAR Rincon de Leon.

    Añade por mes (nivel ciudad, temporal):
      - treated_m3: caudal tratado en ese mes (proxy de consumo total ciudad)
      - reuse_ratio: % agua reutilizada (indicador de estres hidrico)

    Un mes con reuse_ratio alto + anomalia de consumo puede indicar
    restricciones o sequia, no fraude.

    Args:
        df: DataFrame con features
        edar_df: DataFrame de load_edar_data()
    """
    if edar_df.empty:
        df["treated_m3"] = 0.0
        df["reuse_ratio"] = 0.0
        return df

    df = df.copy()
    edar = edar_df[["fecha", "treated_m3", "reuse_ratio"]].copy()

    df["_merge_month"] = pd.to_datetime(df["fecha"]).dt.to_period("M").dt.to_timestamp()
    edar["_merge_month"] = pd.to_datetime(edar["fecha"]).dt.to_period("M").dt.to_timestamp()
    edar = edar.drop(columns=["fecha"])

    df = df.merge(edar, on="_merge_month", how="left")
    df = df.drop(columns=["_merge_month"])

    df["treated_m3"] = df["treated_m3"].fillna(df["treated_m3"].median() if df["treated_m3"].notna().any() else 0)
    df["reuse_ratio"] = df["reuse_ratio"].fillna(df["reuse_ratio"].median() if df["reuse_ratio"].notna().any() else 0)

    return df


# ─────────────────────────────────────────────────────────────────
# Enriquecimiento con datos demograficos del Padron Municipal 2025
# Fuente: alicante.es/estadisticas-poblacion (barrios_2025.xls)
# ─────────────────────────────────────────────────────────────────

# Mapping AMAEM barrio name -> Padron barrio name
_AMAEM_TO_PADRON = {
    "1-BENALUA": "BENALUA",
    "10-FLORIDA BAJA": "FLORIDA BAJA",
    "11-CIUDAD DE ASIS": "CIUDAD DE ASIS",
    "12-POLIGONO BABEL": "POLIGONO BABEL",
    "13-SAN GABRIEL": "SAN GABRIEL",
    "14-ENSANCHE DIPUTACION": "ENSANCHE DIPUTACION",
    "15-POLIGONO SAN BLAS": "POLIGONO SAN BLAS",
    "16-PLA DEL BON REPOS": "PLA DEL BON REPOS",
    "17-CAROLINAS ALTAS": "CAROLINAS ALTAS",
    "18-CAROLINAS BAJAS": "CAROLINAS BAJAS",
    "19-GARBINET": "GARBINET",
    "2-SAN ANTON": "SANANTON",
    "20-RABASA": "RABASA",
    "21-TOMBOLA": "TOMBOLA",
    "22-CASCO ANTIGUO - SANTA CRUZ": "CASCO ANTIGUO - SANTA CRUZ - AYUNTAMIENTO",
    "23-RAVAL ROIG -V. DEL SOCORRO": "RAVAL ROIG - VIRGEN DEL SOCORRO",
    "24-SAN BLAS - SANTO DOMINGO": "SAN BLAS - SANTO DOMINGO",
    "25-ALTOZANO - CONDE LUMIARES": "ALTOZANO - CONDE LUMIARES",
    "26-SIDI IFNI - NOU ALACANT": "SIDI IFNI - NOU ALACANT",
    "27-SAN FERNANDO-PRIN. MERCEDES": "SAN FERNANDO - PRINCESA MERCEDES",
    "28-EL PALMERAL": "EL PALMERAL - URBANOVA - TABARCA",
    "29-URBANOVA": "EL PALMERAL - URBANOVA - TABARCA",
    "3-CENTRO": "CENTRO",
    "30-DIVINA PASTORA": "DIVINA PASTORA",
    "31-CIUDAD JARDIN": "CIUDAD JARDIN",
    "32-VIRGEN DEL REMEDIO": "VIRGEN DEL REMEDIO",
    "33- MORANT -SAN NICOLAS BARI": "LO MORANT - SAN NICOLAS DE BARI",
    "34-COLONIA REQUENA": "COLONIA REQUENA",
    "35-VIRGEN DEL CARMEN": "VIRGEN DEL CARMEN",
    "36-CUATROCIENTAS VIVIENDAS": "CUATROCIENTAS VIVIENDAS",
    "37-JUAN XXIII": "JUAN XXIII",
    "38-VISTAHERMOSA": "VISTAHERMOSA",
    "39-ALBUFERETA": "ALBUFERETA",
    "4-MERCADO": "MERCADO",
    "40-CABO DE LAS HUERTAS": "CABO DE LAS HUERTAS",
    "41-PLAYA DE SAN JUAN": "PLAYA DE SAN JUAN",
    "5-CAMPOAMOR": "CAMPOAMOR",
    "56-DISPERSOS": "DISPERSO PARTIDAS",
    "6-LOS ANGELES": "LOS ANGELES",
    "7-SAN AGUSTIN": "SAN AGUSTIN",
    "8-ALIPARK": "ALIPARK",
    "9-FLORIDA ALTA": "FLORIDA ALTA",
    "TABARCA": "EL PALMERAL - URBANOVA - TABARCA",
    "VILLAFRANQUEZA": "VILLAFRANQUEZA - SANTA FAZ",
    "SANTA FAZ": "VILLAFRANQUEZA - SANTA FAZ",
    "BACAROT": "DISPERSO PARTIDAS",
    "FONTCALENT": "DISPERSO PARTIDAS",
    "LA ALCORAYA": "DISPERSO PARTIDAS",
    "LA CAÑADA": "DISPERSO PARTIDAS",
    "MONNEGRE": "DISPERSO PARTIDAS",
    "MORALET": "DISPERSO PARTIDAS",
    "PDA VALLONGA": "DISPERSO PARTIDAS",
    "REBOLLEDO": "DISPERSO PARTIDAS",
    "VERDEGAS": "DISPERSO PARTIDAS",
}


def enrich_with_demographics(df: pd.DataFrame,
                             padron_path: str = "data/padron_elderly_barrios_2025.csv") -> pd.DataFrame:
    """
    Enriquece features con datos REALES del Padron Municipal de Alicante 2025.

    Anade por barrio (estatico):
      - pct_elderly_65plus: % poblacion >= 65 anos
      - pct_elderly_80plus: % poblacion >= 80 anos
      - pct_elderly_alone: % de mayores 65+ que viven SOLOS
      - population_density: log(poblacion_total) normalizado

    Anade features de interaccion (dinamicos, por barrio-mes):
      - elderly_consumption_ratio: consumo vs esperado por demografia
      - elderly_x_drop: pct_elderly * |desviacion_del_grupo| — amplifica anomalias en barrios vulnerables
      - alone_x_volatility: pct_solos * volatilidad_consumo

    Fuente: alicante.es/estadisticas-poblacion (barrios_2025.xls, miembros_barrio_2025.xls)
    """
    padron_file = __import__("pathlib").Path(padron_path)
    if not padron_file.exists():
        for col in DEMOGRAPHIC_FEATURE_COLUMNS:
            df[col] = 0.0
        return df

    df = df.copy()
    padron = pd.read_csv(padron_path)

    # Build lookup: AMAEM barrio_clean -> padron row
    padron_lookup = {}
    for padron_name, row in padron.set_index("barrio_padron").iterrows():
        padron_lookup[padron_name] = row

    # Map each barrio_clean in df to padron data
    df["_barrio_clean"] = df["barrio"].str.strip() if "barrio" in df.columns else df["barrio_key"].str.split("__").str[0]

    def _get_padron_field(barrio_clean, field, default=0.0):
        padron_name = _AMAEM_TO_PADRON.get(barrio_clean, "")
        if padron_name in padron_lookup:
            val = padron_lookup[padron_name].get(field, default)
            return float(val) if pd.notna(val) else default
        return default

    # Static features (per barrio, same every month)
    df["pct_elderly_65plus"] = df["_barrio_clean"].apply(
        lambda b: _get_padron_field(b, "pct_65plus"))
    df["pct_elderly_80plus"] = df["_barrio_clean"].apply(
        lambda b: _get_padron_field(b, "pct_80plus"))
    df["pct_elderly_alone"] = df["_barrio_clean"].apply(
        lambda b: _get_padron_field(b, "pct_65plus_solos"))

    # Population density: log-normalized (range ~4.6 to ~10.4)
    df["population_density"] = df["_barrio_clean"].apply(
        lambda b: np.log1p(_get_padron_field(b, "poblacion_total")))
    # Normalize to 0-1 range
    pop_max = df["population_density"].max()
    if pop_max > 0:
        df["population_density"] = df["population_density"] / pop_max

    # Interaction features (require existing consumption features)
    # 1. elderly_consumption_ratio: actual consumption vs median for similar demographic group
    if "consumption_per_contract" in df.columns:
        # Group barrios by elderly % terciles and compute group median
        df["_elderly_group"] = pd.cut(df["pct_elderly_65plus"], bins=3, labels=["young", "mid", "elderly"])
        group_medians = df.groupby(["_elderly_group", "fecha"])["consumption_per_contract"].transform("median")
        df["elderly_consumption_ratio"] = np.where(
            group_medians > 0,
            df["consumption_per_contract"] / group_medians,
            1.0
        )
        df = df.drop(columns=["_elderly_group"])
    else:
        df["elderly_consumption_ratio"] = 1.0

    # 2. elderly_x_drop: amplifies deviation signal in elderly barrios
    if "deviation_from_group_trend" in df.columns:
        df["elderly_x_drop"] = (
            (df["pct_elderly_65plus"] / 100.0) *
            df["deviation_from_group_trend"].abs()
        )
    else:
        df["elderly_x_drop"] = 0.0

    # 3. alone_x_volatility: elderly alone have stable patterns — high volatility is suspicious
    if "consumption_volatility" in df.columns:
        df["alone_x_volatility"] = (
            (df["pct_elderly_alone"] / 100.0) *
            df["consumption_volatility"]
        )
    else:
        df["alone_x_volatility"] = 0.0

    # Clean up
    df = df.drop(columns=["_barrio_clean"], errors="ignore")

    # Fill any remaining NaN
    for col in DEMOGRAPHIC_FEATURE_COLUMNS:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0.0)

    n_with_data = (df["pct_elderly_65plus"] > 0).sum()
    print(f"    + Features demograficos (Padron 2025): {n_with_data}/{len(df)} filas con datos")

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
