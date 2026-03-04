"""
Pipeline unificado: ejecuta todos los modelos sobre el dataset del hackathon.

Modelos:
  M2  — IsolationForest cross-sectional (compara barrios entre si)
  M5  — 3-sigma + IQR (outliers estadisticos puros)
  M6  — Amazon Chronos (transformer pre-entrenado, forecasting)
  M7  — Facebook Prophet (descomposicion estacional)
  M8  — Agua No Registrada (caudal inyectado vs consumo facturado)
  M9  — Nighttime Minimum Flow (consumo nocturno anomalo)
  M10 — Anomalias en lecturas individuales de contadores

Uso:
  python run_all_models.py                         # todos los barrios DOMESTICO
  python run_all_models.py --barrios 5             # primeros 5
  python run_all_models.py --with-external         # con datos temperatura/turismo
  python run_all_models.py --skip-chronos          # sin M6 (rapido)
  python run_all_models.py --output results.csv    # guardar CSV
  python run_all_models.py --uso COMERCIAL         # otro tipo de uso
  python run_all_models.py --contamination 0.03    # M2 mas conservador
  python run_all_models.py --prophet-interval 0.97 # Prophet 97% CI
  python run_all_models.py --chronos-sigma 2.5     # Chronos mas conservador
  python run_all_models.py --iqr-multiplier 2.0    # IQR fences mas amplias
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from train_local import load_hackathon_amaem
from monthly_features import (
    compute_monthly_features,
    prepare_monthly_matrix,
    MONTHLY_FEATURE_COLUMNS,
    EXTENDED_FEATURE_COLUMNS,
    RELATIVE_FEATURE_COLUMNS,
    RELATIVE_EXTENDED_FEATURE_COLUMNS,
    AUXILIARY_FEATURE_COLUMNS,
    enrich_with_telelectura,
    enrich_with_regenerada,
)
from statistical_baseline import score_3sigma, score_iqr
from anr_detector import load_caudal_monthly, load_consumo_monthly, compute_anr, detect_anr_anomalies
from nightflow_detector import load_hourly_data, compute_night_day_ratios, detect_nmf_anomalies
from meter_reading_detector import (
    load_all_readings, preprocess_readings, detect_reading_anomalies,
    compute_monthly_stats,
)

DATA_FILE = "data/datos-hackathon-amaem.xlsx-set-de-datos-.csv"


def load_data(csv_path: str, with_external: bool = False):
    """Carga el dataset del hackathon y opcionalmente datos externos."""
    df = load_hackathon_amaem(csv_path)

    external_df = None
    if with_external:
        from external_data import load_external_data
        min_date = pd.to_datetime(df["fecha"]).min()
        max_date = pd.to_datetime(df["fecha"]).max()
        external_df = load_external_data(
            str(min_date.date()), str(max_date.date())
        )
        print(f"  Datos externos cargados: {len(external_df)} meses")

    return df, external_df


CONTADORES_PATH = "data/contadores-telelectura-instalados-solo-alicante_hackaton-dataart-contadores-telelectura-instalad.csv"
REGENERADA_PATH = "data/_consumos_alicante_regenerada_barrio_mes-2024_-consumos_alicante_regenerada_barrio_mes-2024.csv.csv"


def run_m2(df_all: pd.DataFrame, external_df=None,
           uso_filter: str = "DOMESTICO", contamination: float = 0.05):
    """
    M2 — IsolationForest cross-sectional.
    Entrena sobre los primeros 24 meses de TODOS los barrios del mismo tipo,
    puntua los ultimos 12 meses.

    Detecta: barrios que se comportan raro comparados con otros del mismo tipo.
    """
    print(f"\n  [M2] IsolationForest cross-sectional (contamination={contamination})...")

    # Calcular features (con o sin datos externos)
    df_features = compute_monthly_features(df_all, external_df=external_df)

    # Enriquecer con datos auxiliares (contadores telelectura, agua regenerada)
    df_features = enrich_with_telelectura(df_features, CONTADORES_PATH)
    df_features = enrich_with_regenerada(df_features, REGENERADA_PATH)

    # Filtrar por tipo de uso
    df_uso = df_features[df_features["uso"].str.strip() == uso_filter].copy()
    df_uso = df_uso.sort_values(["barrio_key", "fecha"]).reset_index(drop=True)

    # Usar features relativos + auxiliares
    use_extended = external_df is not None
    feature_cols = RELATIVE_EXTENDED_FEATURE_COLUMNS if use_extended else RELATIVE_FEATURE_COLUMNS
    feature_cols = feature_cols + AUXILIARY_FEATURE_COLUMNS
    available_cols = [c for c in feature_cols if c in df_uso.columns]

    # Split temporal: primeros 24 meses para train, ultimos 12 para test
    all_dates = sorted(df_uso["fecha"].unique())
    n_dates = len(all_dates)
    n_train_dates = min(24, int(n_dates * 0.7))
    train_dates = set(all_dates[:n_train_dates])
    test_dates = set(all_dates[n_train_dates:])

    train_data = df_uso[df_uso["fecha"].isin(train_dates)]
    test_data = df_uso[df_uso["fecha"].isin(test_dates)]

    # Preparar matrices
    X_train = train_data[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_test = test_data[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

    if len(X_train) < 10:
        print(f"    No hay suficientes datos de entrenamiento ({len(X_train)} filas)")
        return pd.DataFrame()

    # Entrenar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = IsolationForest(
        n_estimators=100, contamination=contamination,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train_scaled)

    # Puntuar test set
    scores = model.score_samples(X_test_scaled)
    predictions = model.predict(X_test_scaled)

    # Incluir features de contexto para el resumen
    context_cols = ["barrio_key", "fecha", "consumo_litros",
                    "consumption_per_contract"]
    for extra in ["yoy_ratio", "group_yoy_median", "deviation_from_group_trend",
                   "relative_consumption"]:
        if extra in test_data.columns:
            context_cols.append(extra)

    result = test_data[context_cols].copy()
    result["is_anomaly_m2"] = predictions == -1
    result["score_m2"] = scores

    n_anomalies = result["is_anomaly_m2"].sum()
    n_barrios = result["barrio_key"].nunique()
    features_used = "extendidos" if use_extended else "base"
    print(f"    {n_anomalies} anomalias en {len(result)} puntos "
          f"({n_barrios} barrios, {len(test_dates)} meses, features {features_used})")

    return result


def run_m5(df_all: pd.DataFrame, uso_filter: str = "DOMESTICO",
           iqr_multiplier: float = 2.0, min_deviation: float = 0.10):
    """
    M5 — 3-sigma + IQR sobre deviation_from_group_trend.
    Para cada barrio, calcula estadisticas de desviacion sobre los primeros 24 meses
    y detecta outliers en los ultimos 12.

    Filtro de significancia practica: solo flag si |desviacion| > min_deviation.
    Esto evita falsos positivos en barrios estables con varianza baja.

    Detecta: barrios cuya desviacion del grupo es extrema.
    """
    print(f"\n  [M5] 3-sigma + IQR sobre desviacion del grupo "
          f"(iqr_multiplier={iqr_multiplier}, min_dev={min_deviation:.0%})...")

    df_features = compute_monthly_features(df_all)
    df_uso = df_features[df_features["uso"].str.strip() == uso_filter].copy()
    df_uso = df_uso.sort_values(["barrio_key", "fecha"]).reset_index(drop=True)

    all_dates = sorted(df_uso["fecha"].unique())
    n_train_dates = min(24, int(len(all_dates) * 0.7))
    train_dates = set(all_dates[:n_train_dates])
    test_dates = set(all_dates[n_train_dates:])

    results = []
    for barrio_key, group in df_uso.groupby("barrio_key"):
        train = group[group["fecha"].isin(train_dates)]
        test = group[group["fecha"].isin(test_dates)]

        if len(train) < 6 or len(test) == 0:
            continue

        # Usar deviation_from_group_trend en vez de consumption_per_contract
        train_vals = train["deviation_from_group_trend"].values.astype(float)
        test_vals = test["deviation_from_group_trend"].values.astype(float)

        # Reemplazar NaN (primer ano sin yoy) con 0 (neutral)
        train_vals = np.nan_to_num(train_vals, nan=0.0)
        test_vals = np.nan_to_num(test_vals, nan=0.0)

        sigma_flags = score_3sigma(test_vals, train_vals)
        iqr_flags = score_iqr(test_vals, train_vals, multiplier=iqr_multiplier)

        # Filtro de significancia practica: requiere desviacion minima
        # Evita flagear barrios con 3% de desviacion solo porque su varianza es baja
        practical_flags = np.abs(test_vals) >= min_deviation

        for i, (_, row) in enumerate(test.iterrows()):
            results.append({
                "barrio_key": barrio_key,
                "fecha": row["fecha"],
                "is_anomaly_3sigma": bool(sigma_flags[i] and practical_flags[i]),
                "is_anomaly_iqr": bool(iqr_flags[i] and practical_flags[i]),
            })

    result = pd.DataFrame(results)
    n_sigma = result["is_anomaly_3sigma"].sum() if len(result) > 0 else 0
    n_iqr = result["is_anomaly_iqr"].sum() if len(result) > 0 else 0
    print(f"    3-sigma: {n_sigma} anomalias, IQR: {n_iqr} anomalias "
          f"en {len(result)} puntos")

    return result


def run_m6(df_all: pd.DataFrame, uso_filter: str = "DOMESTICO",
           max_barrios: int = 0, threshold_sigma: float = 2.5):
    """
    M6 — Amazon Chronos.
    Transformer pre-entrenado que predice el siguiente valor y compara con el real.
    LENTO: ~1 min por barrio.

    Detecta: meses que rompen la prediccion de un modelo de deep learning.
    """
    print(f"\n  [M6] Amazon Chronos (threshold_sigma={threshold_sigma})...")

    try:
        from chronos_detector import score_chronos
    except ImportError:
        print("    SKIP: chronos-forecasting no instalado")
        return pd.DataFrame()

    df_features = compute_monthly_features(df_all)
    df_uso = df_features[df_features["uso"].str.strip() == uso_filter].copy()

    all_dates = sorted(df_uso["fecha"].unique())
    n_train_dates = min(24, int(len(all_dates) * 0.7))
    train_dates_set = set(all_dates[:n_train_dates])
    test_dates_set = set(all_dates[n_train_dates:])

    barrios = sorted(df_uso["barrio_key"].unique())
    if max_barrios > 0:
        barrios = barrios[:max_barrios]

    results = []
    for idx, barrio_key in enumerate(barrios):
        group = df_uso[df_uso["barrio_key"] == barrio_key].sort_values("fecha")
        train = group[group["fecha"].isin(train_dates_set)]
        test = group[group["fecha"].isin(test_dates_set)]

        if len(train) < 12 or len(test) == 0:
            continue

        train_vals = train["consumption_per_contract"].values.astype(float)
        test_vals = test["consumption_per_contract"].values.astype(float)

        t0 = time.time()
        try:
            flags = score_chronos(train_vals, test_vals,
                                  threshold_sigma=threshold_sigma, num_samples=30)
        except Exception as e:
            print(f"    ERROR en {barrio_key}: {e}")
            continue
        elapsed = time.time() - t0

        for i, (_, row) in enumerate(test.iterrows()):
            results.append({
                "barrio_key": barrio_key,
                "fecha": row["fecha"],
                "is_anomaly_chronos": bool(flags[i]) if i < len(flags) else False,
            })

        n_det = sum(flags) if len(flags) > 0 else 0
        print(f"    [{idx+1}/{len(barrios)}] {barrio_key}: "
              f"{n_det} anomalias ({elapsed:.1f}s)")

    result = pd.DataFrame(results)
    n_total = result["is_anomaly_chronos"].sum() if len(result) > 0 else 0
    print(f"    Total: {n_total} anomalias en {len(result)} puntos")
    return result


def run_m7(df_all: pd.DataFrame, uso_filter: str = "DOMESTICO",
           max_barrios: int = 0, interval_width: float = 0.97,
           changepoint_prior_scale: float = 0.15):
    """
    M7 — Facebook Prophet.
    Descompone la serie en tendencia + estacionalidad, predice y compara.

    Detecta: meses que rompen la estacionalidad/tendencia del barrio.
    """
    print(f"\n  [M7] Facebook Prophet (interval={interval_width}, "
          f"changepoint_scale={changepoint_prior_scale})...")

    try:
        from prophet_detector import score_prophet
    except ImportError:
        print("    SKIP: prophet no instalado")
        return pd.DataFrame()

    df_features = compute_monthly_features(df_all)
    df_uso = df_features[df_features["uso"].str.strip() == uso_filter].copy()

    all_dates = sorted(df_uso["fecha"].unique())
    n_train_dates = min(24, int(len(all_dates) * 0.7))
    train_dates_set = set(all_dates[:n_train_dates])
    test_dates_set = set(all_dates[n_train_dates:])

    barrios = sorted(df_uso["barrio_key"].unique())
    if max_barrios > 0:
        barrios = barrios[:max_barrios]

    results = []
    for idx, barrio_key in enumerate(barrios):
        group = df_uso[df_uso["barrio_key"] == barrio_key].sort_values("fecha")
        train = group[group["fecha"].isin(train_dates_set)]
        test = group[group["fecha"].isin(test_dates_set)]

        if len(train) < 12 or len(test) == 0:
            continue

        train_vals = train["consumption_per_contract"].values.astype(float)
        test_vals = test["consumption_per_contract"].values.astype(float)
        train_dates = train["fecha"].values
        test_dates = test["fecha"].values

        try:
            flags = score_prophet(
                train_vals, test_vals, train_dates, test_dates,
                interval_width=interval_width,
                changepoint_prior_scale=changepoint_prior_scale,
            )
        except Exception as e:
            print(f"    ERROR en {barrio_key}: {e}")
            continue

        for i, (_, row) in enumerate(test.iterrows()):
            results.append({
                "barrio_key": barrio_key,
                "fecha": row["fecha"],
                "is_anomaly_prophet": bool(flags[i]) if i < len(flags) else False,
            })

        if (idx + 1) % 10 == 0 or idx == len(barrios) - 1:
            print(f"    [{idx+1}/{len(barrios)}] barrios procesados")

    result = pd.DataFrame(results)
    n_total = result["is_anomaly_prophet"].sum() if len(result) > 0 else 0
    print(f"    Total: {n_total} anomalias en {len(result)} puntos")
    return result


def run_m8(df_all: pd.DataFrame, uso_filter: str = "DOMESTICO",
           anr_threshold: float = 2.0) -> pd.DataFrame:
    """
    M8 — Agua No Registrada (ANR).
    Compara caudal inyectado por sector vs consumo facturado por barrio.
    ANR alto = perdidas tecnicas (fugas) o comerciales (fraude).

    Solo 2024 (unico año con datos de caudal horario).
    """
    caudal_path = "data/_caudal_medio_sector_hidraulico_hora_2024_-caudal_medio_sector_hidraulico_hora_2024.csv"
    if not Path(caudal_path).exists():
        print(f"\n  [M8] ANR — SKIP (no se encuentra {caudal_path})")
        return pd.DataFrame()

    print(f"\n  [M8] Agua No Registrada — ANR (threshold_ratio={anr_threshold})...")

    caudal_monthly = load_caudal_monthly(caudal_path)
    consumo_monthly = load_consumo_monthly(DATA_FILE, uso_filter=None)  # todos los usos para ANR
    anr = compute_anr(caudal_monthly, consumo_monthly)
    barrio_stats = detect_anr_anomalies(anr, threshold_ratio=anr_threshold)

    n_anomalies = barrio_stats["is_anomaly_anr"].sum()
    n_barrios = len(barrio_stats)
    print(f"    {n_anomalies} barrios con ANR anomalo de {n_barrios} mapeados")

    # Expandir a formato por (barrio_key, fecha) para merge con otros modelos
    # M8 es anual por barrio → marcamos todos los meses 2024 del barrio como anomalia
    anomalous_barrios = set(
        barrio_stats[barrio_stats["is_anomaly_anr"]]["barrio"].values
    )

    # Preparar resultado con formato compatible
    df_2024 = df_all[pd.to_datetime(df_all["fecha"]).dt.year == 2024].copy()
    if uso_filter:
        df_2024 = df_2024[df_2024["uso"].str.strip() == uso_filter]

    df_2024["barrio_clean"] = df_2024["barrio"].str.strip()
    df_2024["barrio_key"] = df_2024["barrio_clean"] + "__" + df_2024["uso"].str.strip()
    df_2024["fecha"] = pd.to_datetime(df_2024["fecha"])

    result = df_2024[["barrio_key", "fecha"]].copy()
    result["is_anomaly_anr"] = df_2024["barrio_clean"].isin(anomalous_barrios)

    # Añadir ANR ratio como score
    barrio_ratio = barrio_stats.set_index("barrio")["avg_anr_ratio"].to_dict()
    result["anr_ratio"] = df_2024["barrio_clean"].map(barrio_ratio).fillna(0)

    if len(anomalous_barrios) > 0:
        for b in sorted(anomalous_barrios):
            ratio = barrio_ratio.get(b, 0)
            print(f"      {b}: ANR ratio={ratio:.1f}")

    return result


def run_m9(df_all: pd.DataFrame, uso_filter: str = "DOMESTICO",
           zscore_threshold: float = 2.0) -> pd.DataFrame:
    """
    M9 — Nighttime Minimum Flow (NMF).
    Analiza caudal horario 2am-5am vs 10am-18h por sector hidraulico.
    Ratio nocturno alto = fugas o uso no autorizado.
    """
    caudal_path = "data/_caudal_medio_sector_hidraulico_hora_2024_-caudal_medio_sector_hidraulico_hora_2024.csv"
    if not Path(caudal_path).exists():
        print(f"\n  [M9] NMF — SKIP (no se encuentra {caudal_path})")
        return pd.DataFrame()

    print(f"\n  [M9] Nighttime Minimum Flow — NMF (zscore={zscore_threshold})...")

    df_hourly = load_hourly_data(caudal_path)
    daily = compute_night_day_ratios(df_hourly)
    sector_stats = detect_nmf_anomalies(daily, zscore_threshold=zscore_threshold)

    n_anomalies = sector_stats["is_anomaly_nmf"].sum()
    n_sectors = len(sector_stats)
    print(f"    {n_anomalies} sectores con NMF anomalo de {n_sectors} analizados")

    # Mapear sectores anomalos a barrios
    from sector_mapping import get_mapped_sectors
    mapping = get_mapped_sectors()
    anomalous_sectors = set(sector_stats[sector_stats["is_anomaly_nmf"]]["SECTOR"].values)
    anomalous_barrios = set()
    for sector in anomalous_sectors:
        barrio = mapping.get(sector)
        if barrio:
            anomalous_barrios.add(barrio)
            print(f"      {sector} → {barrio} (ratio={sector_stats[sector_stats['SECTOR']==sector]['avg_ratio'].values[0]:.1f})")

    # Crear resultado por (barrio_key, fecha) — marcar todos los meses 2024
    df_2024 = df_all[pd.to_datetime(df_all["fecha"]).dt.year == 2024].copy()
    if uso_filter:
        df_2024 = df_2024[df_2024["uso"].str.strip() == uso_filter]

    df_2024["barrio_clean"] = df_2024["barrio"].str.strip()
    df_2024["barrio_key"] = df_2024["barrio_clean"] + "__" + df_2024["uso"].str.strip()
    df_2024["fecha"] = pd.to_datetime(df_2024["fecha"])

    result = df_2024[["barrio_key", "fecha"]].copy()
    result["is_anomaly_nmf"] = df_2024["barrio_clean"].isin(anomalous_barrios)

    return result


def run_m10(df_all: pd.DataFrame, uso_filter: str = "DOMESTICO",
            iqr_multiplier: float = 3.0) -> pd.DataFrame:
    """
    M10 — Anomalias en lecturas individuales de contadores.
    Analiza ~4M lecturas de m3 registrados/facturados.
    Detecta meses con tasa anomala de lecturas extremas, zeros, o retrasos.

    Como no hay barrio en los datos de lecturas, marca los meses donde
    la tasa de anomalia es significativamente alta (afecta a todos los barrios).
    """
    import glob as _glob
    pattern = "data/m3-registrados_facturados-tll_*-solo-alicante-*.csv"
    files = sorted(_glob.glob(pattern))
    if not files:
        print(f"\n  [M10] Lecturas — SKIP (no se encuentran archivos m3)")
        return pd.DataFrame()

    print(f"\n  [M10] Anomalias en lecturas individuales (iqr={iqr_multiplier})...")

    readings = load_all_readings(pattern)
    if readings.empty:
        return pd.DataFrame()

    readings = preprocess_readings(readings)
    readings = detect_reading_anomalies(readings, iqr_multiplier=iqr_multiplier)
    monthly_stats = compute_monthly_stats(readings)

    n_anom_months = monthly_stats["is_anomalous_month"].sum()
    total_readings = len(readings)
    total_anomalies = readings["is_anomaly_reading"].sum()
    print(f"    {total_readings:,} lecturas, {total_anomalies:,} anomalas ({total_anomalies/total_readings*100:.1f}%)")
    print(f"    {n_anom_months} meses con tasa anomala elevada")

    # Mapear a formato (barrio_key, fecha): marcar meses anomalos para todos los barrios
    anomalous_periods = set(
        monthly_stats[monthly_stats["is_anomalous_month"]]["year_month"].astype(str).values
    )

    if anomalous_periods:
        for p in sorted(anomalous_periods):
            rate = monthly_stats[monthly_stats["year_month"].astype(str) == p]["anomaly_rate"].values[0]
            print(f"      {p}: tasa anomala {rate:.1f}%")

    df_tmp = df_all.copy()
    df_tmp["fecha"] = pd.to_datetime(df_tmp["fecha"])
    if uso_filter:
        df_tmp = df_tmp[df_tmp["uso"].str.strip() == uso_filter]
    df_tmp["barrio_key"] = df_tmp["barrio"].str.strip() + "__" + df_tmp["uso"].str.strip()
    df_tmp["year_month_str"] = df_tmp["fecha"].dt.to_period("M").astype(str)

    result = df_tmp[["barrio_key", "fecha"]].copy()
    result["is_anomaly_readings"] = df_tmp["year_month_str"].isin(anomalous_periods)

    return result


def collect_results(m2_results: pd.DataFrame, m5_results: pd.DataFrame,
                    m6_results: pd.DataFrame, m7_results: pd.DataFrame,
                    m8_results: pd.DataFrame = None,
                    m9_results: pd.DataFrame = None,
                    m10_results: pd.DataFrame = None) -> pd.DataFrame:
    """
    Merge todos los resultados por (barrio_key, fecha).
    Anade columna con cuantos modelos detectan anomalia.
    """
    # Empezar con M2 como base (tiene consumo_litros y consumption_per_contract)
    if len(m2_results) == 0:
        print("  WARNING: M2 no produjo resultados")
        return pd.DataFrame()

    result = m2_results.copy()

    # Merge M5
    if len(m5_results) > 0:
        result = result.merge(
            m5_results[["barrio_key", "fecha", "is_anomaly_3sigma", "is_anomaly_iqr"]],
            on=["barrio_key", "fecha"], how="left",
        )
    else:
        result["is_anomaly_3sigma"] = False
        result["is_anomaly_iqr"] = False

    # Merge M6
    if len(m6_results) > 0:
        result = result.merge(
            m6_results[["barrio_key", "fecha", "is_anomaly_chronos"]],
            on=["barrio_key", "fecha"], how="left",
        )
    else:
        result["is_anomaly_chronos"] = np.nan  # NaN = no ejecutado

    # Merge M7
    if len(m7_results) > 0:
        result = result.merge(
            m7_results[["barrio_key", "fecha", "is_anomaly_prophet"]],
            on=["barrio_key", "fecha"], how="left",
        )
    else:
        result["is_anomaly_prophet"] = np.nan  # NaN = no ejecutado

    # Merge M8 (ANR)
    if m8_results is not None and len(m8_results) > 0:
        result = result.merge(
            m8_results[["barrio_key", "fecha", "is_anomaly_anr", "anr_ratio"]],
            on=["barrio_key", "fecha"], how="left",
        )
    else:
        result["is_anomaly_anr"] = np.nan
        result["anr_ratio"] = np.nan

    # Merge M9 (NMF)
    if m9_results is not None and len(m9_results) > 0:
        result = result.merge(
            m9_results[["barrio_key", "fecha", "is_anomaly_nmf"]],
            on=["barrio_key", "fecha"], how="left",
        )
    else:
        result["is_anomaly_nmf"] = np.nan

    # Merge M10 (Lecturas)
    if m10_results is not None and len(m10_results) > 0:
        result = result.merge(
            m10_results[["barrio_key", "fecha", "is_anomaly_readings"]],
            on=["barrio_key", "fecha"], how="left",
        )
    else:
        result["is_anomaly_readings"] = np.nan

    # Rellenar NaN en columnas booleanas
    for col in ["is_anomaly_3sigma", "is_anomaly_iqr"]:
        result[col] = result[col].fillna(False).astype(bool)

    # Contar cuantos modelos detectan anomalia (ignorar NaN = modelo no ejecutado)
    model_cols = ["is_anomaly_m2", "is_anomaly_3sigma", "is_anomaly_iqr",
                  "is_anomaly_chronos", "is_anomaly_prophet", "is_anomaly_anr",
                  "is_anomaly_nmf", "is_anomaly_readings"]
    available = [c for c in model_cols if c in result.columns]

    def _count_detecting(row):
        detecting = []
        model_names = {
            "is_anomaly_m2": "M2",
            "is_anomaly_3sigma": "M5_3sigma",
            "is_anomaly_iqr": "M5_IQR",
            "is_anomaly_chronos": "M6_Chronos",
            "is_anomaly_prophet": "M7_Prophet",
            "is_anomaly_anr": "M8_ANR",
            "is_anomaly_nmf": "M9_NMF",
            "is_anomaly_readings": "M10_Lecturas",
        }
        for col in available:
            val = row.get(col)
            if pd.notna(val) and val:
                detecting.append(model_names.get(col, col))
        return detecting

    result["models_detecting"] = result.apply(_count_detecting, axis=1)
    result["n_models_detecting"] = result["models_detecting"].apply(len)

    # Nivel de confianza basado en desviacion del grupo
    def _confidence_level(row):
        dev = abs(row.get("deviation_from_group_trend", 0) or 0)
        n_mod = row.get("n_models_detecting", 0)
        if n_mod == 0:
            return "NONE"
        if dev > 0.30:
            return "HIGH"
        elif dev > 0.10:
            return "MEDIUM"
        else:
            return "LOW"

    result["confidence"] = result.apply(_confidence_level, axis=1)

    return result


def print_summary(results: pd.DataFrame):
    """Imprime resumen inteligente con ranking por confianza y magnitud."""
    if len(results) == 0:
        print("\n  No hay resultados.")
        return

    print(f"\n{'='*80}")
    print(f"  RESUMEN DE DETECCION — {results['barrio_key'].nunique()} barrios")
    print(f"{'='*80}")

    # Resumen por modelo
    model_cols = {
        "is_anomaly_m2": "M2 (IF relativo)",
        "is_anomaly_3sigma": "M5 (3-sigma desv)",
        "is_anomaly_iqr": "M5 (IQR desv)",
        "is_anomaly_chronos": "M6 (Chronos)",
        "is_anomaly_prophet": "M7 (Prophet)",
        "is_anomaly_anr": "M8 (ANR perdidas)",
        "is_anomaly_nmf": "M9 (NMF nocturno)",
        "is_anomaly_readings": "M10 (Lecturas indiv)",
    }

    print(f"\n  {'Modelo':<22}  {'Anomalias':>10}  {'% del total':>12}  {'Estado':>10}")
    print(f"  {'─'*60}")
    for col, name in model_cols.items():
        if col in results.columns:
            valid = results[col].dropna()
            if len(valid) > 0:
                n_anom = valid.sum()
                pct = n_anom / len(valid) * 100
                print(f"  {name:<22}  {int(n_anom):>10}  {pct:>11.1f}%  {'OK':>10}")
            else:
                print(f"  {name:<22}  {'—':>10}  {'—':>12}  {'skip':>10}")
        else:
            print(f"  {name:<22}  {'—':>10}  {'—':>12}  {'N/A':>10}")

    # Resumen por confianza
    has_confidence = "confidence" in results.columns
    if has_confidence:
        flagged = results[results["n_models_detecting"] >= 1]
        n_high = len(flagged[flagged["confidence"] == "HIGH"])
        n_med = len(flagged[flagged["confidence"] == "MEDIUM"])
        n_low = len(flagged[flagged["confidence"] == "LOW"])
        print(f"\n  Confianza:  HIGH={n_high} (>30% desv)  "
              f"MEDIUM={n_med} (10-30%)  LOW={n_low} (<10% ruido)")

    # --- TOP ANOMALIAS (solo MEDIUM y HIGH) ---
    has_deviation = "deviation_from_group_trend" in results.columns
    if has_deviation and has_confidence:
        print(f"\n  {'─'*95}")
        print(f"  ANOMALIAS CONFIRMADAS (confianza MEDIUM/HIGH)")
        print(f"  {'─'*95}")

        confirmed = results[
            (results["n_models_detecting"] >= 1) &
            (results["confidence"].isin(["HIGH", "MEDIUM"]))
        ].copy()

        if len(confirmed) > 0:
            confirmed["abs_deviation"] = confirmed["deviation_from_group_trend"].abs()
            top = confirmed.nlargest(15, "abs_deviation")

            print(f"\n  {'Barrio':<32}  {'Mes':>7}  {'YoY%':>7}  {'Grupo%':>7}  "
                  f"{'Desv':>7}  {'Conf':>6}  {'#Mod':>5}  {'Modelos'}")
            print(f"  {'─'*100}")
            for _, row in top.iterrows():
                fecha_str = row["fecha"].strftime("%Y-%m") if hasattr(row["fecha"], "strftime") else str(row["fecha"])[:7]
                yoy_pct = (row.get("yoy_ratio", 1.0) - 1) * 100
                group_pct = (row.get("group_yoy_median", 1.0) - 1) * 100
                dev = row["deviation_from_group_trend"] * 100
                models = ", ".join(row["models_detecting"])
                print(f"  {row['barrio_key']:<32}  {fecha_str:>7}  "
                      f"{yoy_pct:>+6.1f}%  {group_pct:>+6.1f}%  "
                      f"{dev:>+6.1f}%  {row['confidence']:>6}  "
                      f"{row['n_models_detecting']:>5}  {models}")
        else:
            print("\n  Ninguna anomalia con confianza MEDIUM o HIGH")

        if n_low > 0:
            print(f"\n  ({n_low} alertas LOW descartadas — desviacion < 10% del grupo)")

    # --- TOP BARRIOS ---
    print(f"\n  {'─'*95}")
    print(f"  RANKING DE BARRIOS ANOMALOS")
    print(f"  {'─'*95}")

    if has_confidence:
        significant = results[
            (results["n_models_detecting"] >= 1) &
            (results["confidence"].isin(["HIGH", "MEDIUM"]))
        ]
    else:
        significant = results[results["n_models_detecting"] >= 1]

    if len(significant) > 0:
        barrio_summary = (
            significant.groupby("barrio_key")
            .agg(
                n_alerts=("n_models_detecting", "count"),
                max_models=("n_models_detecting", "max"),
                max_deviation=(
                    "deviation_from_group_trend" if has_deviation else "n_models_detecting",
                    lambda x: x.abs().max() if has_deviation else x.max()
                ),
                n_high=("confidence", lambda x: (x == "HIGH").sum()) if has_confidence else ("n_models_detecting", "count"),
                meses=("fecha", lambda x: ", ".join(
                    sorted(set(d.strftime("%Y-%m") for d in x))[:4]
                )),
            )
            .sort_values("max_deviation", ascending=False)
            .head(10)
        )

        print(f"\n  {'Barrio':<35}  {'Alertas':>8}  {'HIGH':>5}  "
              f"{'Max desv':>9}  {'Meses'}")
        print(f"  {'─'*85}")
        for barrio_key, row in barrio_summary.iterrows():
            dev_str = f"{row['max_deviation']*100:+.1f}%" if has_deviation else f"{row['max_deviation']:.0f}"
            print(f"  {barrio_key:<35}  {row['n_alerts']:>8}  "
                  f"{int(row['n_high']):>5}  {dev_str:>9}  {row['meses']}")
    else:
        print("\n  Ningun barrio con alertas significativas")


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar 4 modelos de deteccion de anomalias sobre el dataset del hackathon"
    )
    parser.add_argument("--file", default=DATA_FILE, help="Ruta al CSV")
    parser.add_argument("--barrios", type=int, default=0,
                        help="Limitar a N barrios (0=todos)")
    parser.add_argument("--uso", default="DOMESTICO",
                        help="Tipo de uso (DOMESTICO, COMERCIAL, NO DOMESTICO)")
    parser.add_argument("--with-external", action="store_true",
                        help="Incluir datos externos (temperatura, turismo)")
    parser.add_argument("--skip-chronos", action="store_true",
                        help="Saltar M6 Chronos (lento)")
    parser.add_argument("--skip-prophet", action="store_true",
                        help="Saltar M7 Prophet")
    parser.add_argument("--skip-anr", action="store_true",
                        help="Saltar M8 ANR (Agua No Registrada)")
    parser.add_argument("--anr-threshold", type=float, default=2.0,
                        help="M8 ANR ratio threshold (default: 2.0)")
    parser.add_argument("--skip-nmf", action="store_true",
                        help="Saltar M9 NMF (Nighttime Minimum Flow)")
    parser.add_argument("--nmf-zscore", type=float, default=2.0,
                        help="M9 NMF z-score threshold (default: 2.0)")
    parser.add_argument("--skip-readings", action="store_true",
                        help="Saltar M10 Lecturas individuales (lento, ~4M filas)")
    parser.add_argument("--output", type=str, default=None,
                        help="Guardar resultados en CSV")
    # Tuning parameters (optimized via tune_models.py grid search)
    parser.add_argument("--contamination", type=float, default=0.01,
                        help="M2 contamination rate (default: 0.01, tuned)")
    parser.add_argument("--prophet-interval", type=float, default=0.99,
                        help="Prophet interval width (default: 0.99, tuned)")
    parser.add_argument("--prophet-changepoint", type=float, default=0.30,
                        help="Prophet changepoint_prior_scale (default: 0.30, tuned)")
    parser.add_argument("--chronos-sigma", type=float, default=3.5,
                        help="Chronos threshold sigma (default: 3.5, tuned)")
    parser.add_argument("--iqr-multiplier", type=float, default=3.0,
                        help="IQR multiplier for fences (default: 3.0, tuned)")
    parser.add_argument("--min-deviation", type=float, default=0.10,
                        help="M5 minimum practical deviation to flag (default: 0.10 = 10%%)")
    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"ERROR: No se encuentra {csv_path}")
        sys.exit(1)

    print(f"{'='*80}")
    print(f"  AQUAGUARD AI — Pipeline Multi-Modelo")
    print(f"{'='*80}")
    print(f"  Dataset:         {csv_path}")
    print(f"  Uso:             {args.uso}")
    print(f"  Barrios:         {'todos' if args.barrios == 0 else args.barrios}")
    print(f"  Ext. data:       {'SI' if args.with_external else 'NO'}")
    print(f"  Modelos:         M2, M5" +
          ("" if args.skip_chronos else ", M6") +
          ("" if args.skip_prophet else ", M7") +
          ("" if args.skip_anr else ", M8") +
          ("" if args.skip_nmf else ", M9") +
          ("" if args.skip_readings else ", M10"))
    print(f"  M2 contamination: {args.contamination}")
    print(f"  M5 IQR mult:     {args.iqr_multiplier}")
    if not args.skip_chronos:
        print(f"  M6 sigma:        {args.chronos_sigma}")
    if not args.skip_prophet:
        print(f"  M7 interval:     {args.prophet_interval} "
              f"(changepoint={args.prophet_changepoint})")

    # Cargar datos
    print(f"\n  Cargando datos...")
    t_start = time.time()
    df_all, external_df = load_data(str(csv_path), with_external=args.with_external)

    # Limitar barrios si se pide
    if args.barrios > 0:
        barrios_unicos = df_all["barrio"].unique()[:args.barrios]
        df_all = df_all[df_all["barrio"].isin(barrios_unicos)]
        print(f"  Limitado a {len(barrios_unicos)} barrios")

    # Ejecutar modelos
    m2_results = run_m2(df_all, external_df=external_df, uso_filter=args.uso,
                        contamination=args.contamination)
    m5_results = run_m5(df_all, uso_filter=args.uso,
                        iqr_multiplier=args.iqr_multiplier,
                        min_deviation=args.min_deviation)

    m6_results = pd.DataFrame()
    if not args.skip_chronos:
        m6_results = run_m6(df_all, uso_filter=args.uso,
                            max_barrios=args.barrios if args.barrios > 0 else 0,
                            threshold_sigma=args.chronos_sigma)

    m7_results = pd.DataFrame()
    if not args.skip_prophet:
        m7_results = run_m7(df_all, uso_filter=args.uso,
                            max_barrios=args.barrios if args.barrios > 0 else 0,
                            interval_width=args.prophet_interval,
                            changepoint_prior_scale=args.prophet_changepoint)

    m8_results = pd.DataFrame()
    if not args.skip_anr:
        m8_results = run_m8(df_all, uso_filter=args.uso,
                            anr_threshold=args.anr_threshold)

    m9_results = pd.DataFrame()
    if not args.skip_nmf:
        m9_results = run_m9(df_all, uso_filter=args.uso,
                            zscore_threshold=args.nmf_zscore)

    m10_results = pd.DataFrame()
    if not args.skip_readings:
        m10_results = run_m10(df_all, uso_filter=args.uso,
                              iqr_multiplier=args.iqr_multiplier)

    # Combinar
    print(f"\n  Combinando resultados...")
    results = collect_results(m2_results, m5_results, m6_results, m7_results,
                              m8_results, m9_results, m10_results)

    elapsed = time.time() - t_start
    print(f"  Tiempo total: {elapsed:.1f}s")

    # Mostrar resumen
    print_summary(results)

    # Guardar CSV si se pide
    if args.output and len(results) > 0:
        # Convertir lista de modelos a string para CSV
        results_csv = results.copy()
        results_csv["models_detecting"] = results_csv["models_detecting"].apply(
            lambda x: ";".join(x) if x else ""
        )
        results_csv.to_csv(args.output, index=False)
        print(f"\n  Resultados guardados en: {args.output}")
        print(f"  Filas: {len(results_csv)}, Columnas: {len(results_csv.columns)}")


if __name__ == "__main__":
    main()
