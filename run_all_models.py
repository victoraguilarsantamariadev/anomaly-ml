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
    INFRASTRUCTURE_FEATURE_COLUMNS,
    TEMPORAL_AUX_FEATURE_COLUMNS,
    EDAR_FEATURE_COLUMNS,
    ADVANCED_FEATURE_COLUMNS,
    FOURIER_INTERACTION_COLUMNS,
    enrich_with_telelectura,
    enrich_with_regenerada,
    enrich_with_infrastructure,
    enrich_with_contract_growth,
    enrich_with_network_stats,
    enrich_with_edar,
    enrich_with_demographics,
    DEMOGRAPHIC_FEATURE_COLUMNS,
)
from statistical_baseline import score_3sigma, score_iqr
from anr_detector import load_caudal_monthly, load_consumo_monthly, compute_anr, detect_anr_anomalies
from nightflow_detector import load_hourly_data, compute_night_day_ratios, detect_nmf_anomalies
from meter_reading_detector import (
    load_all_readings, preprocess_readings, detect_reading_anomalies,
    compute_monthly_stats,
)
from gis_features import (
    load_infrastructure_features,
    load_contract_growth,
    load_network_stats,
    load_edar_data,
    compute_barrio_adjacency,
)
from spatial_detector import (
    classify_spatial_anomalies,
    compute_infrastructure_risk,
    spatial_summary,
    infrastructure_risk_summary,
)
from fraud_detector import (
    compute_monthly_fraud_rate,
    compute_barrio_vulnerability,
    enrich_with_fraud_features,
    build_meta_model,
    fraud_summary,
)
from autoencoder_detector import run_autoencoder
from advanced_ensemble import (
    apply_weighted_voting,
    apply_conformal_prediction,
    apply_stacking_ensemble,
    compute_shap_explanations,
    compute_permutation_importance,
    print_advanced_report,
    print_proof_chain,
)
from vae_detector import run_vae
from changepoint_detector import (
    detect_changepoints_per_barrio,
    enrich_results_with_changepoints,
    changepoint_summary,
)
from causal_analysis import run_causal_analysis, causal_summary
from mlops_monitor import run_monitoring_report
from sensitivity_analysis import run_sensitivity_analysis
from validation_report import generate_validation_report
from welfare_detector import run_welfare_detection, welfare_summary
from wasserstein_detector import run_wasserstein_detection, wasserstein_summary
from tda_detector import run_tda_detection, tda_summary
from counterfactual_explainer import generate_counterfactuals, counterfactual_summary
from transfer_entropy import run_transfer_entropy_analysis, transfer_entropy_summary
from fraud_ground_truth import load_ground_truth, validate_detections, ground_truth_summary
from meter_readings_detector import run_meter_analysis, meter_readings_summary
from graph_network_detector import run_graph_analysis, graph_network_summary
from hydraulic_twin import run_hydraulic_twin, hydraulic_summary as hydraulic_twin_summary
from pseudo_ground_truth import build_pseudo_labels, evaluate_against_pseudo, pseudo_ground_truth_summary
from ablation_study import run_ablation_study, compute_pairwise_redundancy, ablation_summary
from advanced_ensemble import compute_calibration_report, print_calibration_report
from cross_validate_fraud import run_cross_validation, print_cross_validation_summary

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
ALTAS_PATH = "data/altas-por-poblacion-solo-alicante_hackaton-dataart-altas-por-poblacion-solo-alicante.csv.csv"
LONGITUD_RED_PATH = "data/amaem-pda-longitud-red-abastecimiento-explotacion-solo-alicante-amaem-pda-longitud-red-abastecim.csv"
EDAR_PATH = "data/amaem-pda-depuracion-edar-rincon-de-leon_hackaton-dataart-2.0-amaem-pda-depuracion-edar-rincon-d.csv"
CAMBIOS_PATH = "data/cambios-de-contador-solo-alicante_hackaton-dataart-cambios-de-contador-solo-alicante.csv.csv"
GIS_DATA_DIR = "data/"


def run_m2(df_all: pd.DataFrame, external_df=None,
           uso_filter: str = "DOMESTICO", contamination: float = 0.05,
           infra_df: pd.DataFrame = None, risk_df: pd.DataFrame = None,
           growth_df: pd.DataFrame = None, network_df: pd.DataFrame = None,
           edar_df: pd.DataFrame = None):
    """
    M2 — IsolationForest cross-sectional.
    Entrena sobre los primeros 24 meses de TODOS los barrios del mismo tipo,
    puntua los ultimos 12 meses.

    Detecta: barrios que se comportan raro comparados con otros del mismo tipo.
    Usa features de infraestructura GIS, crecimiento de contratos, y red.
    """
    print(f"\n  [M2] IsolationForest cross-sectional (contamination={contamination})...")

    # Compute temporal cutoff to prevent look-ahead bias in features
    _tmp_dates = sorted(pd.to_datetime(df_all["fecha"]).unique())
    _n_train = min(24, int(len(_tmp_dates) * 0.7))
    _cutoff = _tmp_dates[_n_train] if _n_train < len(_tmp_dates) else None

    # Calcular features (con o sin datos externos)
    df_features = compute_monthly_features(df_all, external_df=external_df, cutoff_date=_cutoff)

    # Enriquecer con datos auxiliares (contadores telelectura, agua regenerada)
    df_features = enrich_with_telelectura(df_features, CONTADORES_PATH)
    df_features = enrich_with_regenerada(df_features, REGENERADA_PATH)

    # Enriquecer con infraestructura GIS
    if infra_df is not None and not infra_df.empty:
        df_features = enrich_with_infrastructure(df_features, infra_df, risk_df)
        print(f"    + Features de infraestructura GIS")

    # Enriquecer con crecimiento de contratos
    if growth_df is not None and not growth_df.empty:
        df_features = enrich_with_contract_growth(df_features, growth_df)
        print(f"    + Features de crecimiento de contratos")

    # Enriquecer con estadisticas de red
    if network_df is not None and not network_df.empty:
        df_features = enrich_with_network_stats(df_features, network_df)
        print(f"    + Features de longitud de red / inspeccion")

    # Enriquecer con datos EDAR
    if edar_df is not None and not edar_df.empty:
        df_features = enrich_with_edar(df_features, edar_df)
        print(f"    + Features de EDAR (depuradora)")

    # Enriquecer con datos demograficos del Padron Municipal 2025
    df_features = enrich_with_demographics(df_features)

    # Filtrar por tipo de uso
    df_uso = df_features[df_features["uso"].str.strip() == uso_filter].copy()
    df_uso = df_uso.sort_values(["barrio_key", "fecha"]).reset_index(drop=True)

    # Usar features relativos + auxiliares + avanzados + infraestructura + temporales + demograficos
    use_extended = external_df is not None
    feature_cols = RELATIVE_EXTENDED_FEATURE_COLUMNS if use_extended else RELATIVE_FEATURE_COLUMNS
    feature_cols = feature_cols + AUXILIARY_FEATURE_COLUMNS + ADVANCED_FEATURE_COLUMNS + FOURIER_INTERACTION_COLUMNS
    feature_cols = feature_cols + DEMOGRAPHIC_FEATURE_COLUMNS

    # Añadir features de infraestructura si disponibles
    if infra_df is not None:
        feature_cols = feature_cols + INFRASTRUCTURE_FEATURE_COLUMNS
    if growth_df is not None or network_df is not None:
        feature_cols = feature_cols + TEMPORAL_AUX_FEATURE_COLUMNS
    if edar_df is not None:
        feature_cols = feature_cols + EDAR_FEATURE_COLUMNS

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

    # Feature selection: reduce noise for IsolationForest
    # Use variance + correlation filtering to keep top features
    max_features = min(20, len(available_cols))
    if len(available_cols) > max_features:
        from sklearn.feature_selection import mutual_info_regression
        X_train_tmp = pd.DataFrame(X_train, columns=available_cols)
        # Use consumption variance as proxy target for unsupervised selection
        target_var = X_train_tmp["deviation_from_group_trend"].values if "deviation_from_group_trend" in available_cols else X_train_tmp.iloc[:, 0].values
        mi_scores = mutual_info_regression(
            X_train_tmp.fillna(0), target_var, random_state=42, n_neighbors=5
        )
        top_idx = np.argsort(mi_scores)[-max_features:]
        selected_cols = [available_cols[i] for i in sorted(top_idx)]
        X_train = X_train_tmp[selected_cols].values
        X_test = pd.DataFrame(X_test, columns=available_cols)[selected_cols].values
        available_cols = selected_cols
        print(f"    Feature selection: {len(selected_cols)}/{len(feature_cols)} features (MI top {max_features})")

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

    # Adaptive contamination: ajustar threshold por variabilidad del barrio
    # Barrios rurales (pocos contratos, alta variabilidad) necesitan score más extremo
    barrio_cv = (
        train_data.groupby("barrio_key")["consumption_per_contract"]
        .agg(lambda x: x.std() / x.mean() if x.mean() > 0 else 0)
    )
    median_cv = barrio_cv.median()
    if median_cv > 0:
        cv_ratio = test_data["barrio_key"].map(barrio_cv).fillna(median_cv) / median_cv
        # Penalización suave: solo para barrios MUY variables
        cv_penalty = np.where(cv_ratio > 3.0, 0.05, np.where(cv_ratio > 2.0, 0.02, 0.0))
        adjusted_scores = scores + cv_penalty
        # Re-predecir solo barrios penalizados con scores ajustados
        penalized_mask = cv_penalty > 0
        if penalized_mask.any():
            threshold = -model.offset_
            adjusted_preds = np.where(adjusted_scores < threshold, -1, 1)
            predictions = np.where(penalized_mask, adjusted_preds, predictions)
            n_adjusted = int(penalized_mask.sum())
            print(f"    Adaptive contamination: {n_adjusted} puntos de barrios de alta variabilidad ajustados")

    # Incluir features de contexto para el resumen
    context_cols = ["barrio_key", "fecha", "consumo_litros",
                    "consumption_per_contract"]
    for extra in ["yoy_ratio", "group_yoy_median", "deviation_from_group_trend",
                   "relative_consumption"] + DEMOGRAPHIC_FEATURE_COLUMNS:
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

    _tmp_dates = sorted(pd.to_datetime(df_all["fecha"]).unique())
    _n_train = min(24, int(len(_tmp_dates) * 0.7))
    _cutoff = _tmp_dates[_n_train] if _n_train < len(_tmp_dates) else None
    df_features = compute_monthly_features(df_all, cutoff_date=_cutoff)
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
           max_barrios: int = 0, threshold_sigma: float = 2.5,
           model_size: str = "small"):
    """
    M6 — Amazon Chronos.
    Transformer pre-entrenado que predice el siguiente valor y compara con el real.
    Modelos: small (8M, ~1min/barrio), base (46M, ~3min), large (200M, ~8min).

    Detecta: meses que rompen la prediccion de un modelo de deep learning.
    """
    print(f"\n  [M6] Amazon Chronos (threshold_sigma={threshold_sigma}, "
          f"model={model_size})...")

    try:
        from chronos_detector import score_chronos
    except ImportError:
        print("    SKIP: chronos-forecasting no instalado")
        return pd.DataFrame()

    _tmp_dates = sorted(pd.to_datetime(df_all["fecha"]).unique())
    _n_train = min(24, int(len(_tmp_dates) * 0.7))
    _cutoff = _tmp_dates[_n_train] if _n_train < len(_tmp_dates) else None
    df_features = compute_monthly_features(df_all, cutoff_date=_cutoff)
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
                                  threshold_sigma=threshold_sigma, num_samples=30,
                                  model_size=model_size)
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

    _tmp_dates = sorted(pd.to_datetime(df_all["fecha"]).unique())
    _n_train = min(24, int(len(_tmp_dates) * 0.7))
    _cutoff = _tmp_dates[_n_train] if _n_train < len(_tmp_dates) else None
    df_features = compute_monthly_features(df_all, cutoff_date=_cutoff)
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
            flags, residuals = score_prophet(
                train_vals, test_vals, train_dates, test_dates,
                interval_width=interval_width,
                changepoint_prior_scale=changepoint_prior_scale,
                return_residuals=True,
            )
        except Exception as e:
            print(f"    ERROR en {barrio_key}: {e}")
            continue

        for i, (_, row) in enumerate(test.iterrows()):
            results.append({
                "barrio_key": barrio_key,
                "fecha": row["fecha"],
                "is_anomaly_prophet": bool(flags[i]) if i < len(flags) else False,
                "prophet_residual": float(residuals[i]) if i < len(residuals) else 0.0,
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
    Tambien calcula reading_anomaly_zscore continuo para uso en M12.
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

    total_readings = len(readings)
    total_anomalies = readings["is_anomaly_reading"].sum()
    print(f"    {total_readings:,} lecturas, {total_anomalies:,} anomalas ({total_anomalies/total_readings*100:.1f}%)")

    # Threshold combinado: z-score > 1 OR tasa absoluta > 15%
    mean_rate = monthly_stats["anomaly_rate"].mean()
    std_rate = monthly_stats["anomaly_rate"].std()
    monthly_stats["is_anomalous_month"] = (
        (monthly_stats["anomaly_rate"] > mean_rate + 1.0 * std_rate) |
        (monthly_stats["anomaly_rate"] > 15.0)
    )

    n_anom_months = monthly_stats["is_anomalous_month"].sum()
    print(f"    {n_anom_months} meses con tasa anomala elevada")

    # Z-score continuo de la tasa de anomalía por mes
    monthly_stats["reading_zscore"] = np.where(
        std_rate > 0,
        (monthly_stats["anomaly_rate"] - mean_rate) / std_rate,
        0,
    )

    # Mapear a formato (barrio_key, fecha)
    anomalous_periods = set(
        monthly_stats[monthly_stats["is_anomalous_month"]]["year_month"].astype(str).values
    )

    if anomalous_periods:
        for p in sorted(anomalous_periods):
            row = monthly_stats[monthly_stats["year_month"].astype(str) == p].iloc[0]
            print(f"      {p}: tasa anomala {row['anomaly_rate']:.1f}% (z={row['reading_zscore']:.1f})")

    # Build lookups por year_month
    zscore_map = dict(zip(
        monthly_stats["year_month"].astype(str),
        monthly_stats["reading_zscore"]
    ))
    rate_map = dict(zip(
        monthly_stats["year_month"].astype(str),
        monthly_stats["anomaly_rate"]
    ))

    df_tmp = df_all.copy()
    df_tmp["fecha"] = pd.to_datetime(df_tmp["fecha"])
    if uso_filter:
        df_tmp = df_tmp[df_tmp["uso"].str.strip() == uso_filter]
    df_tmp["barrio_key"] = df_tmp["barrio"].str.strip() + "__" + df_tmp["uso"].str.strip()
    df_tmp["year_month_str"] = df_tmp["fecha"].dt.to_period("M").astype(str)

    result = df_tmp[["barrio_key", "fecha"]].copy()
    result["is_anomaly_readings"] = df_tmp["year_month_str"].isin(anomalous_periods)
    result["reading_anomaly_rate"] = df_tmp["year_month_str"].map(rate_map).fillna(0).astype(float)
    result["reading_anomaly_zscore"] = df_tmp["year_month_str"].map(zscore_map).fillna(0).astype(float)

    return result


def collect_results(m2_results: pd.DataFrame, m5_results: pd.DataFrame,
                    m6_results: pd.DataFrame, m7_results: pd.DataFrame,
                    m8_results: pd.DataFrame = None,
                    m9_results: pd.DataFrame = None,
                    m10_results: pd.DataFrame = None,
                    m13_results: pd.DataFrame = None,
                    vae_results: pd.DataFrame = None) -> pd.DataFrame:
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
        m7_cols = ["barrio_key", "fecha", "is_anomaly_prophet"]
        if "prophet_residual" in m7_results.columns:
            m7_cols.append("prophet_residual")
        result = result.merge(
            m7_results[m7_cols],
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
        m10_cols = ["barrio_key", "fecha", "is_anomaly_readings"]
        for extra in ["reading_anomaly_rate", "reading_anomaly_zscore"]:
            if extra in m10_results.columns:
                m10_cols.append(extra)
        result = result.merge(
            m10_results[m10_cols],
            on=["barrio_key", "fecha"], how="left",
        )
    else:
        result["is_anomaly_readings"] = np.nan

    # Merge M13 (Autoencoder)
    if m13_results is not None and len(m13_results) > 0:
        m13_cols = ["barrio_key", "fecha", "is_anomaly_autoencoder"]
        if "reconstruction_error" in m13_results.columns:
            m13_cols.append("reconstruction_error")
        result = result.merge(
            m13_results[m13_cols],
            on=["barrio_key", "fecha"], how="left",
        )
    else:
        result["is_anomaly_autoencoder"] = np.nan

    # Merge VAE (M13-PRO)
    if vae_results is not None and len(vae_results) > 0:
        vae_cols = ["barrio_key", "fecha", "is_anomaly_vae",
                     "vae_score_norm", "vae_log_likelihood"]
        vae_cols = [c for c in vae_cols if c in vae_results.columns]
        result = result.merge(
            vae_results[vae_cols],
            on=["barrio_key", "fecha"], how="left",
        )
    else:
        result["is_anomaly_vae"] = np.nan
        result["vae_score_norm"] = np.nan

    # Rellenar NaN en columnas booleanas
    for col in ["is_anomaly_3sigma", "is_anomaly_iqr"]:
        result[col] = result[col].fillna(False).astype(bool)

    # Contar cuantos modelos detectan anomalia (ignorar NaN = modelo no ejecutado)
    model_cols = ["is_anomaly_m2", "is_anomaly_3sigma", "is_anomaly_iqr",
                  "is_anomaly_chronos", "is_anomaly_prophet", "is_anomaly_anr",
                  "is_anomaly_nmf", "is_anomaly_readings", "is_anomaly_autoencoder",
                  "is_anomaly_vae"]
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
            "is_anomaly_autoencoder": "M13_Autoencoder",
            "is_anomaly_vae": "M13PRO_VAE",
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

    # ─── SCORES CONTINUOS + ALERTAS POR COLOR ───
    # En vez de solo binario (anomaly/normal), score continuo [0, 1]
    def _continuous_score(row):
        """Score continuo combinando todas las senales."""
        score = 0.0
        n_active = 0

        # Binary models → contribuyen 0 o 1
        for col in available:
            val = row.get(col)
            if pd.notna(val):
                n_active += 1
                if val:
                    score += 1.0

        # Continuous signals si disponibles
        if pd.notna(row.get("vae_score_norm", np.nan)):
            score += row["vae_score_norm"]
            n_active += 1
        if pd.notna(row.get("deviation_from_group_trend", np.nan)):
            dev = abs(row["deviation_from_group_trend"])
            score += min(dev / 0.5, 1.0)  # Clip at 50% deviation
            n_active += 1

        # Demographic risk bonus — barrios con alta poblacion elderly
        pct_elderly = row.get("pct_elderly_65plus", np.nan)
        if pd.notna(pct_elderly) and pct_elderly > 20:
            score += (pct_elderly / 35.0) * 0.3  # hasta +0.3 para barrios muy envejecidos
            n_active += 1

        return score / max(n_active, 1)

    result["anomaly_score"] = result.apply(_continuous_score, axis=1)

    # Color alerts basado en score continuo
    def _alert_color(score):
        if score >= 0.40:
            return "ROJO"       # Anomalia critica
        elif score >= 0.25:
            return "NARANJA"    # Anomalia probable
        elif score >= 0.10:
            return "AMARILLO"   # Sospechoso
        return "VERDE"          # Normal

    result["alert_color"] = result["anomaly_score"].apply(_alert_color)

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
        "is_anomaly_autoencoder": "M13 (Autoencoder)",
        "is_anomaly_vae": "M13-PRO (VAE)",
    }

    n_active_models = 0
    print(f"\n  {'Modelo':<22}  {'Anomalias':>10}  {'% del total':>12}  {'Estado':>10}")
    print(f"  {'─'*60}")
    for col, name in model_cols.items():
        if col in results.columns:
            valid = results[col].dropna()
            if len(valid) > 0:
                n_anom = valid.sum()
                pct = n_anom / len(valid) * 100
                if n_anom > 0:
                    n_active_models += 1
                    print(f"  {name:<22}  {int(n_anom):>10}  {pct:>11.1f}%  {'ACTIVO':>10}")
                else:
                    print(f"  {name:<22}  {int(n_anom):>10}  {pct:>11.1f}%  {'INACTIVO':>10}")
            else:
                print(f"  {name:<22}  {'—':>10}  {'—':>12}  {'skip':>10}")
        else:
            print(f"  {name:<22}  {'—':>10}  {'—':>12}  {'N/A':>10}")
    print(f"\n  Modelos activos (>0 detecciones): {n_active_models}/{len(model_cols)}")

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

    # --- SISTEMA DE ALERTAS POR COLOR ---
    if "alert_color" in results.columns:
        print(f"\n  {'─'*95}")
        print(f"  SISTEMA DE ALERTAS POR SEMAFORO (scores continuos)")
        print(f"  {'─'*95}")

        n_rojo = (results["alert_color"] == "ROJO").sum()
        n_naranja = (results["alert_color"] == "NARANJA").sum()
        n_amarillo = (results["alert_color"] == "AMARILLO").sum()
        n_verde = (results["alert_color"] == "VERDE").sum()

        total = len(results)
        print(f"\n  ROJO    (score>=0.40): {n_rojo:>5} ({n_rojo/total*100:.1f}%) — Anomalia critica")
        print(f"  NARANJA (score>=0.25): {n_naranja:>5} ({n_naranja/total*100:.1f}%) — Anomalia probable")
        print(f"  AMARILLO(score>=0.10): {n_amarillo:>5} ({n_amarillo/total*100:.1f}%) — Sospechoso")
        print(f"  VERDE   (score<0.10):  {n_verde:>5} ({n_verde/total*100:.1f}%) — Normal")

        # Top alertas rojas
        rojos = results[results["alert_color"] == "ROJO"]
        if len(rojos) > 0:
            top_rojos = rojos.nlargest(min(5, len(rojos)), "anomaly_score")
            print(f"\n  Top alertas ROJAS:")
            for _, row in top_rojos.iterrows():
                barrio = row["barrio_key"].split("__")[0][:28]
                fecha = pd.to_datetime(row["fecha"]).strftime("%Y-%m")
                n_mod = row.get("n_models_detecting", 0)
                print(f"    {barrio:<30} {fecha} score={row['anomaly_score']:.3f} "
                      f"({n_mod} modelos)")


def _compute_verification_factors(results: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula factores de verificación cruzada por barrio para descartar falsos positivos.

    5 factores de descarte:
      1. Regenerada alta → riego municipal nocturno (no fuga)
      2. Estacionalidad turística → ratio verano/invierno alto (no fuga)
      3. ANR confirmado → agua no registrada real (sí fuga de red)
      4. Cluster espacial → problema de red, no individual
      5. Vulnerabilidad infraestructura → lectura manual, más riesgo real

    Output: score de confianza 0-100% para cada alerta.
    """
    from pathlib import Path

    r = results.copy()
    r["fecha_dt"] = pd.to_datetime(r["fecha"])
    r["month"] = r["fecha_dt"].dt.month
    barrios = r["barrio_key"].unique()

    # Factor 1: Regenerada (riego municipal)
    regenerada_ratio = {}
    reg_path = "data/_consumos_alicante_regenerada_barrio_mes-2024_-consumos_alicante_regenerada_barrio_mes-2024.csv.csv"
    if Path(reg_path).exists():
        reg = pd.read_csv(reg_path)
        barrio_reg = reg.groupby("BARRIO")["CONSUMO_2024"].sum()
        median_reg = barrio_reg.median()
        if median_reg > 0:
            regenerada_ratio = (barrio_reg / median_reg).to_dict()

    # Factor 2: Estacionalidad turística (ratio verano/invierno)
    tourism_ratio = {}
    for bk in barrios:
        bg = r[r["barrio_key"] == bk]
        summer = bg[bg["month"].isin([6, 7, 8])]["consumo_litros"].mean()
        winter = bg[bg["month"].isin([1, 2, 12])]["consumo_litros"].mean()
        if winter > 0 and not pd.isna(summer):
            tourism_ratio[bk] = summer / winter
        else:
            tourism_ratio[bk] = 1.0

    # Factor 3: ANR confirmado (agua no registrada)
    anr_confirmed = {}
    for bk in barrios:
        anr = r.loc[r["barrio_key"] == bk, "anr_ratio"].dropna()
        anr_confirmed[bk] = anr.mean() if len(anr) > 0 else 0.0

    # Factor 4: Cluster espacial (problema de red vs individual)
    cluster_pct = {}
    if "spatial_class" in r.columns:
        for bk in barrios:
            bg = r[r["barrio_key"] == bk]
            cluster_pct[bk] = (bg["spatial_class"] == "CLUSTER").mean()

    # Factor 5: Vulnerabilidad (lectura manual, infraestructura vieja)
    vuln_score = {}
    if "fraud_vulnerability" in r.columns:
        for bk in barrios:
            vuln_score[bk] = r.loc[r["barrio_key"] == bk, "fraud_vulnerability"].mean()

    # Factor 6: CPC trend (consumo por contrato — sube = problema real, no demográfico)
    cpc_trends = {}
    for bk in barrios:
        bg = r[r["barrio_key"] == bk].sort_values("fecha_dt")
        cpc = bg["consumption_per_contract"].values
        if len(cpc) > 3:
            cpc_trends[bk] = np.polyfit(range(len(cpc)), cpc, 1)[0]
        else:
            cpc_trends[bk] = 0.0

    # Factor 7: Desviación consistente (media vs std — si σ < |mean| → siempre en la misma dirección)
    dev_profiles = {}
    if "deviation_from_group_trend" in r.columns:
        for bk in barrios:
            dev = r.loc[r["barrio_key"] == bk, "deviation_from_group_trend"].fillna(0).values
            dev_profiles[bk] = (dev.mean(), dev.std())

    return regenerada_ratio, tourism_ratio, anr_confirmed, cluster_pct, vuln_score, cpc_trends, dev_profiles


def _alert_confidence(barrio_key: str, alert_type: str,
                      regenerada_ratio: dict, tourism_ratio: dict,
                      anr_confirmed: dict, cluster_pct: dict,
                      vuln_score: dict,
                      cpc_trends: dict = None,
                      dev_profiles: dict = None) -> tuple:
    """
    Calcula confianza basada en datos (7 factores de verificación cruzada).
    Empieza neutral (50%) y los datos suben o bajan.
    """
    barrio_clean = barrio_key.split("__")[0]
    confidence = 50  # Neutral — que los datos decidan
    reasons_down = []
    reasons_up = []

    # === FACTORES QUE BAJAN CONFIANZA (falso positivo probable) ===

    # Factor 1: Regenerada alta → riego municipal
    reg = regenerada_ratio.get(barrio_clean, 0)
    if reg > 10.0:
        confidence -= 35
        reasons_down.append(f"Regenerada extrema ({reg:.1f}x mediana) — riego municipal masivo")
    elif reg > 5.0:
        confidence -= 25
        reasons_down.append(f"Regenerada muy alta ({reg:.1f}x mediana) — riego municipal")
    elif reg > 2.0:
        confidence -= 15
        reasons_down.append(f"Regenerada alta ({reg:.1f}x mediana)")

    # Factor 2: Turismo estacional
    tr = tourism_ratio.get(barrio_key, 1.0)
    if tr > 2.5:
        confidence -= 30
        reasons_down.append(f"Barrio muy turístico/estacional (ratio {tr:.1f}x)")
    elif tr > 2.0:
        confidence -= 20
        reasons_down.append(f"Estacionalidad alta ({tr:.1f}x)")
    elif tr > 1.5:
        confidence -= 10
        reasons_down.append(f"Estacionalidad moderada ({tr:.1f}x)")

    # Factor 3: Cluster espacial → problema de red
    cp = cluster_pct.get(barrio_key, 0)
    if cp > 0.5:
        confidence -= 10
        reasons_down.append(f"Cluster espacial ({cp*100:.0f}% meses) — posible problema de red")

    # === FACTORES QUE SUBEN CONFIANZA (problema real probable) ===

    # Factor 4: ANR confirmado → agua se pierde de verdad
    anr = anr_confirmed.get(barrio_key, 0)
    if anr > 5.0:
        confidence += 20
        reasons_up.append(f"ANR muy alto ({anr:.1f}x) — pérdida de agua confirmada")
    elif anr > 2.0:
        confidence += 15
        reasons_up.append(f"ANR confirmado ({anr:.1f}x) — agua no registrada real")

    # Factor 5: Vulnerabilidad infraestructura
    vs = vuln_score.get(barrio_key, 0.5)
    if vs > 0.7:
        confidence += 10
        reasons_up.append(f"Infraestructura vulnerable (score={vs:.2f})")

    # Factor 6: CPC subiendo (consumo POR CONTRATO crece → no es crecimiento demográfico)
    if cpc_trends:
        cpc_t = cpc_trends.get(barrio_key, 0)
        if cpc_t > 100:
            confidence += 15
            reasons_up.append(f"Consumo/contrato subiendo (+{cpc_t:.0f} L/mes) — descarta crecimiento demográfico")
        elif cpc_t > 50:
            confidence += 10
            reasons_up.append(f"Consumo/contrato en aumento (+{cpc_t:.0f} L/mes)")

    # Factor 7: Desviación consistente (siempre en la misma dirección → no es ruido)
    if dev_profiles:
        dev_mean, dev_std = dev_profiles.get(barrio_key, (0, 1))
        if abs(dev_mean) > 0.15 and dev_std < abs(dev_mean):
            direction = "por encima" if dev_mean > 0 else "por debajo"
            confidence += 15
            reasons_up.append(f"Desviación consistente {direction} del grupo (media={dev_mean:+.1%}, σ={dev_std:.3f})")
        elif abs(dev_mean) > 0.10:
            confidence += 5

    confidence = max(5, min(100, confidence))

    return confidence, reasons_down, reasons_up


def social_alerts(results: pd.DataFrame):
    """
    Alertas sociales verificadas: detecta situaciones de vulnerabilidad humana
    con verificación cruzada de 5 fuentes de datos para descartar falsos positivos.
    """
    print(f"\n  {'─'*80}")
    print(f"  ALERTAS SOCIALES — Posible Situación de Vulnerabilidad")
    print(f"  {'─'*80}")
    print(f"  (Verificación cruzada: regenerada, turismo, ANR, cluster, vulnerabilidad)")

    # Calcular factores de verificación
    regen, tourism, anr, cluster, vuln, cpc_trends, dev_profiles = _compute_verification_factors(results)

    alerts = []

    # Ordenar por barrio y fecha
    r = results.sort_values(["barrio_key", "fecha"]).copy()
    r["fecha_dt"] = pd.to_datetime(r["fecha"])

    for barrio, group in r.groupby("barrio_key"):
        group = group.sort_values("fecha_dt")

        # --- EMERGENCIA_POSIBLE: spike YoY > 50% repentino ---
        yoy = group.get("yoy_ratio", pd.Series(dtype=float))
        if yoy is not None and len(yoy) > 0:
            spikes = group[yoy.fillna(1) > 1.5]
            if len(spikes) >= 2:
                spike_months = spikes["fecha_dt"].dt.to_period("M").astype(int).values
                consecutive = sum(1 for i in range(1, len(spike_months))
                                  if spike_months[i] - spike_months[i-1] == 1)
                if consecutive >= 1:
                    max_yoy = yoy.max()
                    consumo_extra = group.loc[spikes.index, "consumo_litros"].sum() * 0.3
                    conf, rd, ru = _alert_confidence(
                        barrio, "EMERGENCIA_POSIBLE", regen, tourism, anr, cluster, vuln, cpc_trends, dev_profiles)
                    alerts.append({
                        "barrio": barrio.split("__")[0],
                        "barrio_key": barrio,
                        "tipo": "EMERGENCIA_POSIBLE",
                        "detalle": f"Spike >{(max_yoy-1)*100:.0f}% durante {len(spikes)} meses",
                        "litros_extra": consumo_extra,
                        "meses": len(spikes),
                        "confianza": conf,
                        "reasons_down": rd,
                        "reasons_up": ru,
                        "prioridad": 1,
                    })

        # --- FUGA_OCULTA: >=2 modelos detectan durante >=3 meses seguidos ---
        multi_model = group[group["n_models_detecting"] >= 2]
        if len(multi_model) >= 3:
            mm_months = multi_model["fecha_dt"].dt.to_period("M").astype(int).values
            max_consecutive = 1
            current = 1
            for i in range(1, len(mm_months)):
                if mm_months[i] - mm_months[i-1] == 1:
                    current += 1
                    max_consecutive = max(max_consecutive, current)
                else:
                    current = 1
            if max_consecutive >= 3:
                consumo_medio = group["consumo_litros"].mean()
                conf, rd, ru = _alert_confidence(
                    barrio, "FUGA_OCULTA", regen, tourism, anr, cluster, vuln, cpc_trends, dev_profiles)
                alerts.append({
                    "barrio": barrio.split("__")[0],
                    "barrio_key": barrio,
                    "tipo": "FUGA_OCULTA",
                    "detalle": f"Anomalía persistente {max_consecutive} meses, {len(multi_model)} alertas multi-modelo",
                    "litros_extra": consumo_medio * 0.2 * max_consecutive,
                    "meses": max_consecutive,
                    "confianza": conf,
                    "reasons_down": rd,
                    "reasons_up": ru,
                    "prioridad": 2,
                })

        # --- CONSUMO_VULNERABLE: anomalía en barrio con alta vulnerabilidad ---
        vuln_val = group.get("fraud_vulnerability", pd.Series(0.5, index=group.index))
        if vuln_val.mean() > 0.6 and group["n_models_detecting"].max() >= 2:
            dev = group.get("deviation_from_group_trend", pd.Series(0, index=group.index))
            high_dev = group[dev.abs() > 0.3]
            if len(high_dev) >= 2:
                conf, rd, ru = _alert_confidence(
                    barrio, "CONSUMO_VULNERABLE", regen, tourism, anr, cluster, vuln, cpc_trends, dev_profiles)
                alerts.append({
                    "barrio": barrio.split("__")[0],
                    "barrio_key": barrio,
                    "tipo": "CONSUMO_VULNERABLE",
                    "detalle": f"Barrio vulnerable (score={vuln_val.mean():.2f}), {len(high_dev)} meses con desviación alta",
                    "litros_extra": high_dev["consumo_litros"].sum() * 0.15,
                    "meses": len(high_dev),
                    "confianza": conf,
                    "reasons_down": rd,
                    "reasons_up": ru,
                    "prioridad": 3,
                })

    if not alerts:
        print("  No se detectaron alertas sociales.")
        return

    # Separar por confianza
    high_conf = [a for a in alerts if a["confianza"] >= 60]
    low_conf = [a for a in alerts if a["confianza"] < 60]

    high_conf.sort(key=lambda x: (-x["confianza"], x["prioridad"]))
    low_conf.sort(key=lambda x: (-x["confianza"], x["prioridad"]))

    tipo_accion = {
        "EMERGENCIA_POSIBLE": "Contactar servicios sociales — posible incidencia personal",
        "FUGA_OCULTA": "Revisar instalación interior — posible fuga no detectada",
        "CONSUMO_VULNERABLE": "Seguimiento proactivo — barrio con perfil vulnerable",
    }

    # ALERTAS VERIFICADAS (confianza >= 60%)
    total_litros = 0
    if high_conf:
        print(f"\n  ALERTAS VERIFICADAS ({len(high_conf)} con confianza >= 60%):\n")
        for a in high_conf:
            conf_bar = "█" * (a["confianza"] // 10) + "░" * (10 - a["confianza"] // 10)
            print(f"  [{a['confianza']:3d}%] {conf_bar}  {a['barrio']:<30} [{a['tipo']}]")
            print(f"         {a['detalle']}")
            print(f"         Agua afectada: ~{a['litros_extra']/1000:,.0f} m³ ({a['meses']} meses)")
            if a["reasons_up"]:
                for r in a["reasons_up"]:
                    print(f"         + {r}")
            if a["reasons_down"]:
                for r in a["reasons_down"]:
                    print(f"         - {r}")
            print(f"         Acción: {tipo_accion.get(a['tipo'], 'Investigar')}")
            print()
            total_litros += a["litros_extra"]

    # DESCARTADOS / BAJA CONFIANZA
    if low_conf:
        print(f"  DESCARTADOS por verificación cruzada ({len(low_conf)} alertas):\n")
        for a in low_conf:
            reasons = "; ".join(a["reasons_down"][:2]) if a["reasons_down"] else "sin datos suficientes"
            print(f"    [{a['confianza']:3d}%] {a['barrio']:<30} → {reasons}")
        print()

    print(f"  {'─'*60}")
    if high_conf:
        print(f"  Alertas verificadas: {len(high_conf)} barrios, ~{total_litros/1000:,.0f} m³ en riesgo")
        print(f"  Equivalente a ~{total_litros/150:,.0f} duchas o ~{total_litros/200000:,.0f} piscinas")
        print(f"  Coste estimado: ~{total_litros/1000 * 1.5:,.0f} EUR")
    print(f"  Falsos positivos descartados: {len(low_conf)} (verificación cruzada 5 fuentes)")

    # === PREDICCIÓN FUTURA (6 meses) ===
    print(f"\n  {'─'*80}")
    print(f"  PREDICCIÓN PRÓXIMOS 6 MESES — Barrios en Riesgo")
    print(f"  {'─'*80}")

    verified_barrios = set(a["barrio_key"] for a in high_conf)
    if not verified_barrios:
        verified_barrios = set(a["barrio_key"] for a in alerts[:5])

    r = results.sort_values(["barrio_key", "fecha"]).copy()
    r["fecha_dt"] = pd.to_datetime(r["fecha"])

    forecasts = []
    for bk in sorted(verified_barrios):
        bg = r[r["barrio_key"] == bk].sort_values("fecha_dt")
        if len(bg) < 6:
            continue

        cpc = bg["consumption_per_contract"].values
        consumo = bg["consumo_litros"].values
        fechas = bg["fecha_dt"].values
        n_models = bg["n_models_detecting"].values

        # Trend lineal sobre últimos 12 meses
        trend_cpc = np.polyfit(range(len(cpc)), cpc, 1)
        trend_consumo = np.polyfit(range(len(consumo)), consumo, 1)

        # Proyectar 6 meses adelante
        future_idx = np.arange(len(cpc), len(cpc) + 6)
        future_cpc = np.polyval(trend_cpc, future_idx)
        current_cpc = cpc[-3:].mean()

        # Variación esperada
        pct_change_6m = (future_cpc[-1] - current_cpc) / current_cpc * 100 if current_cpc > 0 else 0
        future_consumo_6m = np.polyval(trend_consumo, future_idx).sum()

        # Riesgo: combinación de tendencia + consistencia de anomalías
        recent_anomaly_rate = n_models[-3:].mean()  # últimos 3 meses
        risk = "ALTO" if (pct_change_6m > 10 and recent_anomaly_rate >= 2) else \
               "MEDIO" if (pct_change_6m > 5 or recent_anomaly_rate >= 1.5) else "BAJO"

        forecasts.append({
            "barrio": bk.split("__")[0],
            "trend_cpc": trend_cpc[0],
            "pct_change_6m": pct_change_6m,
            "future_consumo_6m": future_consumo_6m,
            "current_cpc": current_cpc,
            "future_cpc_6m": future_cpc[-1],
            "recent_anomaly_rate": recent_anomaly_rate,
            "risk": risk,
        })

    forecasts.sort(key=lambda x: -abs(x["pct_change_6m"]))

    if forecasts:
        print(f"\n  {'Barrio':<30} {'CPC actual':>10} {'CPC +6m':>10} {'Cambio':>8} {'Riesgo':>8}")
        print(f"  {'─'*70}")
        for f in forecasts:
            risk_mark = "!!" if f["risk"] == "ALTO" else ">>" if f["risk"] == "MEDIO" else "  "
            print(f"  {risk_mark} {f['barrio']:<28} {f['current_cpc']:>10,.0f} {f['future_cpc_6m']:>10,.0f} "
                  f"{f['pct_change_6m']:>+7.1f}% {f['risk']:>8}")

        alto_risk = [f for f in forecasts if f["risk"] == "ALTO"]
        if alto_risk:
            total_future = sum(f["future_consumo_6m"] for f in alto_risk)
            print(f"\n  Barrios riesgo ALTO: {len(alto_risk)}")
            print(f"  Consumo proyectado 6 meses: ~{total_future/1000:,.0f} m³")
            print(f"  Si la tendencia continúa sin intervención: ~{total_future/1000 * 1.5:,.0f} EUR")


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar 4 modelos de deteccion de anomalias sobre el dataset del hackathon"
    )
    parser.add_argument("--file", default=DATA_FILE, help="Ruta al CSV")
    parser.add_argument("--barrios", type=int, default=0,
                        help="Limitar a N barrios (0=todos)")
    parser.add_argument("--uso", default="DOMESTICO",
                        help="Tipo de uso (DOMESTICO, COMERCIAL, NO DOMESTICO)")
    parser.add_argument("--with-external", action="store_true", default=False,
                        help="Incluir datos externos (temperatura, turismo) — desactivado por defecto (datos climatologicos, no reales)")
    parser.add_argument("--no-external", action="store_true",
                        help="Desactivar datos externos")
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
    parser.add_argument("--skip-spatial", action="store_true",
                        help="Saltar M11 Analisis espacial")
    parser.add_argument("--skip-fraud", action="store_true",
                        help="Saltar M12 Meta-modelo de fraude")
    parser.add_argument("--with-gis", action="store_true", default=True,
                        help="Incluir features de infraestructura GIS (activado por defecto)")
    parser.add_argument("--no-gis", action="store_false", dest="with_gis",
                        help="Desactivar features GIS")
    parser.add_argument("--output", type=str, default=None,
                        help="Guardar resultados en CSV")
    # Tuning parameters (optimized via tune_models.py grid search)
    parser.add_argument("--contamination", type=float, default=0.02,
                        help="M2 contamination rate (default: 0.02, optimized via 7-fold CV)")
    parser.add_argument("--prophet-interval", type=float, default=0.99,
                        help="Prophet interval width (default: 0.99, tuned)")
    parser.add_argument("--prophet-changepoint", type=float, default=0.30,
                        help="Prophet changepoint_prior_scale (default: 0.30, tuned)")
    parser.add_argument("--chronos-sigma", type=float, default=2.0,
                        help="Chronos threshold sigma (default: 2.0)")
    parser.add_argument("--chronos-model", type=str, default="small",
                        choices=["small", "base", "large"],
                        help="Chronos model size: small (8M), base (46M), large (200M)")
    parser.add_argument("--iqr-multiplier", type=float, default=3.0,
                        help="IQR multiplier for fences (default: 3.0, tuned)")
    parser.add_argument("--min-deviation", type=float, default=0.10,
                        help="M5 minimum practical deviation to flag (default: 0.10 = 10%%)")
    args = parser.parse_args()

    # Auto-cargar parámetros tuneados si existen (no hardcodear nada)
    _tuned_path = Path("tuned_params_cv.json")
    if _tuned_path.exists():
        import json as _json
        _tuned = _json.loads(_tuned_path.read_text())
        # Solo sobreescribir si el usuario NO pasó el flag explícitamente
        if '--contamination' not in sys.argv:
            args.contamination = _tuned.get("m2_contamination", args.contamination)
        if '--iqr-multiplier' not in sys.argv:
            args.iqr_multiplier = _tuned.get("m5_iqr_multiplier", args.iqr_multiplier)
        print(f"  Tuned params loaded from {_tuned_path}:")
        print(f"    contamination={args.contamination}, iqr_multiplier={args.iqr_multiplier}")

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
    if args.no_external:
        args.with_external = False
    print(f"  Ext. data:       {'SI' if args.with_external else 'NO'}")
    print(f"  Modelos:         M2, M5" +
          ("" if args.skip_chronos else ", M6") +
          ("" if args.skip_prophet else ", M7") +
          ("" if args.skip_anr else ", M8") +
          ("" if args.skip_nmf else ", M9") +
          ("" if args.skip_readings else ", M10") +
          ("" if args.skip_spatial else ", M11") +
          ("" if args.skip_fraud else ", M12, M13, M13-PRO(VAE)") +
          ", M14(CPD), M15(Wasserstein), M16(TDA), AquaCare")
    print(f"  GIS features:    {'SI' if args.with_gis else 'NO'}")
    print(f"  M2 contamination: {args.contamination}")
    print(f"  M5 IQR mult:     {args.iqr_multiplier}")
    if not args.skip_chronos:
        print(f"  M6 sigma:        {args.chronos_sigma} (model={args.chronos_model})")
    if not args.skip_prophet:
        print(f"  M7 interval:     {args.prophet_interval} "
              f"(changepoint={args.prophet_changepoint})")

    # Cargar datos
    print(f"\n  Cargando datos...")
    t_start = time.time()
    df_all, external_df = load_data(str(csv_path), with_external=args.with_external)

    # Cargar datos auxiliares (GIS, contratos, red)
    infra_df = pd.DataFrame()
    risk_df = pd.DataFrame()
    growth_df = pd.DataFrame()
    network_df = pd.DataFrame()
    adjacency = {}

    if args.with_gis:
        print(f"\n  Cargando datos GIS e infraestructura...")
        infra_df = load_infrastructure_features(GIS_DATA_DIR)
        if not infra_df.empty:
            risk_df = compute_infrastructure_risk(infra_df)

    # Siempre cargar datos auxiliares tabulares (rapido)
    growth_df = load_contract_growth(ALTAS_PATH)
    if not growth_df.empty:
        print(f"  Altas de contratos: {len(growth_df)} meses")

    network_df = load_network_stats(LONGITUD_RED_PATH)
    if not network_df.empty:
        print(f"  Red de abastecimiento: {len(network_df)} años")

    edar_df = load_edar_data(EDAR_PATH)
    if not edar_df.empty:
        print(f"  EDAR (depuradora): {len(edar_df)} meses")

    # Datos de fraude (cambios-de-contador)
    fraud_rate_df = pd.DataFrame()
    vulnerability_df = pd.DataFrame()
    if not args.skip_fraud:
        fraud_rate_df = compute_monthly_fraud_rate(CAMBIOS_PATH)
        if not fraud_rate_df.empty:
            print(f"  Fraude real: {len(fraud_rate_df)} meses con datos")
        vulnerability_df = compute_barrio_vulnerability(CONTADORES_PATH)
        if not vulnerability_df.empty:
            print(f"  Vulnerabilidad: {len(vulnerability_df)} barrios")

    if not args.skip_spatial:
        print(f"  Calculando adyacencia de barrios...")
        adjacency = compute_barrio_adjacency(GIS_DATA_DIR)
        print(f"  {len(adjacency)} barrios con vecinos calculados")

    # Limitar barrios si se pide
    if args.barrios > 0:
        barrios_unicos = df_all["barrio"].unique()[:args.barrios]
        df_all = df_all[df_all["barrio"].isin(barrios_unicos)]
        print(f"  Limitado a {len(barrios_unicos)} barrios")

    # Ejecutar modelos
    m2_results = run_m2(df_all, external_df=external_df, uso_filter=args.uso,
                        contamination=args.contamination,
                        infra_df=infra_df, risk_df=risk_df,
                        growth_df=growth_df, network_df=network_df,
                        edar_df=edar_df)
    m5_results = run_m5(df_all, uso_filter=args.uso,
                        iqr_multiplier=args.iqr_multiplier,
                        min_deviation=args.min_deviation)

    m6_results = pd.DataFrame()
    if not args.skip_chronos:
        m6_results = run_m6(df_all, uso_filter=args.uso,
                            max_barrios=args.barrios if args.barrios > 0 else 0,
                            threshold_sigma=args.chronos_sigma,
                            model_size=args.chronos_model)

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

    # M13 — Autoencoder
    m13_results = pd.DataFrame()
    if not args.skip_fraud:
        # Features CON datos externos (temperatura, turismo, sequia)
        _tmp_dates = sorted(pd.to_datetime(df_all["fecha"]).unique())
        _n_train = min(24, int(len(_tmp_dates) * 0.7))
        _cutoff = _tmp_dates[_n_train] if _n_train < len(_tmp_dates) else None
        df_features_ae = compute_monthly_features(df_all, external_df=external_df, cutoff_date=_cutoff)
        df_features_ae = enrich_with_telelectura(df_features_ae, CONTADORES_PATH)
        df_features_ae = enrich_with_regenerada(df_features_ae, REGENERADA_PATH)
        if not infra_df.empty:
            df_features_ae = enrich_with_infrastructure(df_features_ae, infra_df, risk_df)
        if not growth_df.empty:
            df_features_ae = enrich_with_contract_growth(df_features_ae, growth_df)
        if not network_df.empty:
            df_features_ae = enrich_with_network_stats(df_features_ae, network_df)
        if not edar_df.empty:
            df_features_ae = enrich_with_edar(df_features_ae, edar_df)
        df_features_ae = enrich_with_demographics(df_features_ae)

        # Features base + extendidos si hay datos externos
        use_ext = external_df is not None
        ae_feature_cols = (
            (RELATIVE_EXTENDED_FEATURE_COLUMNS if use_ext else RELATIVE_FEATURE_COLUMNS) +
            AUXILIARY_FEATURE_COLUMNS + ADVANCED_FEATURE_COLUMNS + FOURIER_INTERACTION_COLUMNS
        )
        if not infra_df.empty:
            ae_feature_cols = ae_feature_cols + INFRASTRUCTURE_FEATURE_COLUMNS
        if not growth_df.empty or not network_df.empty:
            ae_feature_cols = ae_feature_cols + TEMPORAL_AUX_FEATURE_COLUMNS
        if not edar_df.empty:
            ae_feature_cols = ae_feature_cols + EDAR_FEATURE_COLUMNS
        ae_feature_cols = ae_feature_cols + DEMOGRAPHIC_FEATURE_COLUMNS

        m13_results = run_autoencoder(df_features_ae, ae_feature_cols,
                                       uso_filter=args.uso,
                                       contamination=args.contamination)

    # M13-PRO — VAE (Variational Autoencoder con PyTorch)
    # beta=2.0 (beta-VAE) para representaciones disentangled
    # contamination diferente (0.05) para aportar diversidad al ensemble
    vae_results = pd.DataFrame()
    if not args.skip_fraud:
        vae_results = run_vae(df_features_ae, ae_feature_cols,
                              uso_filter=args.uso,
                              contamination=0.05,
                              denoising=True, beta=2.0)

    # Combinar
    print(f"\n  Combinando resultados...")
    results = collect_results(m2_results, m5_results, m6_results, m7_results,
                              m8_results, m9_results, m10_results,
                              m13_results=m13_results,
                              vae_results=vae_results)

    # M11 — Analisis espacial
    if not args.skip_spatial and adjacency and len(results) > 0:
        print(f"\n  [M11] Analisis espacial...")
        results = classify_spatial_anomalies(results, adjacency)
        n_cluster = (results["spatial_class"] == "CLUSTER").sum()
        n_isolated = (results["spatial_class"] == "ISOLATED").sum()
        print(f"    CLUSTER (red): {n_cluster}, ISOLATED (puntual): {n_isolated}")

    # M12 — Meta-modelo de fraude
    if not args.skip_fraud and len(results) > 0:
        print(f"\n  [M12] Meta-modelo de fraude...")
        results = enrich_with_fraud_features(results, fraud_rate_df, vulnerability_df)
        results = build_meta_model(results)

    # Tecnicas avanzadas: Weighted Voting + Conformal + SHAP
    if len(results) > 0:
        results = apply_weighted_voting(results)

        # Calcular features para Conformal y SHAP (CON datos externos, con cutoff)
        _tmp_dates_adv = sorted(pd.to_datetime(df_all["fecha"]).unique())
        _n_train_adv = min(24, int(len(_tmp_dates_adv) * 0.7))
        _cutoff_adv = _tmp_dates_adv[_n_train_adv] if _n_train_adv < len(_tmp_dates_adv) else None
        _df_adv = compute_monthly_features(df_all, external_df=external_df, cutoff_date=_cutoff_adv)
        _df_adv = enrich_with_telelectura(_df_adv, CONTADORES_PATH)
        _df_adv = enrich_with_regenerada(_df_adv, REGENERADA_PATH)
        _df_adv = enrich_with_demographics(_df_adv)
        _df_adv = _df_adv[_df_adv["uso"].str.strip() == args.uso].copy()
        _df_adv = _df_adv.sort_values(["barrio_key", "fecha"]).reset_index(drop=True)
        _use_ext = external_df is not None
        _feat_adv = (
            (RELATIVE_EXTENDED_FEATURE_COLUMNS if _use_ext else RELATIVE_FEATURE_COLUMNS) +
            AUXILIARY_FEATURE_COLUMNS + ADVANCED_FEATURE_COLUMNS + FOURIER_INTERACTION_COLUMNS +
            DEMOGRAPHIC_FEATURE_COLUMNS
        )
        results = apply_conformal_prediction(results, _df_adv, _feat_adv)

        # Build pseudo-labels BEFORE stacking to break circularity
        # Stacking will use these independent labels instead of self-referencing consensus
        cambios_path_pre = "data/cambios-de-contador-solo-alicante_hackaton-dataart-cambios-de-contador-solo-alicante.csv.csv"
        try:
            gt_pre = load_ground_truth(cambios_path_pre, CONTADORES_PATH)
            results = build_pseudo_labels(results, gt_pre)
            print(f"    Pseudo-labels built: {int(results['pseudo_label'].sum())} positives (non-circular)")
        except Exception as e:
            print(f"    Pseudo-label pre-build failed ({e}), stacking will use consensus fallback")

        results = apply_stacking_ensemble(results)
        results = compute_shap_explanations(results, _feat_adv, _df_adv)

    elapsed = time.time() - t_start
    print(f"  Tiempo total: {elapsed:.1f}s")

    # Mostrar resumen
    print_summary(results)

    # Resumen espacial
    if "spatial_class" in results.columns:
        spatial_summary(results)

    # Resumen de riesgo de infraestructura
    if not risk_df.empty:
        infrastructure_risk_summary(risk_df)

    # Resumen de fraude
    if "fraud_score" in results.columns:
        fraud_summary(results, fraud_rate_df)

    # Alertas sociales
    if len(results) > 0:
        social_alerts(results)

    # Reporte avanzado (SHAP + Voting + Conformal)
    if len(results) > 0:
        print_advanced_report(results)
        print_proof_chain(results)

    # M14 — Change Point Detection
    if len(results) > 0:
        _df_cp = compute_monthly_features(df_all)
        _df_cp = _df_cp[_df_cp["uso"].str.strip() == args.uso].copy()
        cp_df = detect_changepoints_per_barrio(_df_cp)
        results = enrich_results_with_changepoints(results, cp_df)
        changepoint_summary(results)

    # Permutation Importance
    if len(results) > 0:
        compute_permutation_importance(results, _df_adv, _feat_adv)

    # Causal Discovery DAG (PC algorithm + counterfactual impact)
    if len(results) > 0:
        try:
            causal_results = run_causal_analysis(results)
            causal_summary(causal_results)
        except Exception as e:
            print(f"  Causal DAG error: {e}")

    # MLOps Monitoring (drift, decay, A/B)
    if len(results) > 0:
        run_monitoring_report(_df_adv, results, _feat_adv)

    # Sensitivity Analysis (parametric sweep, bootstrap, temporal split)
    if len(results) > 0:
        run_sensitivity_analysis(_df_adv, _feat_adv, results, uso_filter=args.uso)

    # VALIDATION REPORT — Funciona? Mejora? Son reales? Como mejorar?
    if len(results) > 0:
        generate_validation_report(results)

    # M15 — Optimal Transport (Wasserstein Distance)
    if len(results) > 0:
        print(f"\n{'='*80}")
        print(f"  M15 — OPTIMAL TRANSPORT (Wasserstein Distance)")
        print(f"{'='*80}")
        _df_wass = compute_monthly_features(df_all, external_df=external_df)
        _df_wass = _df_wass[_df_wass["uso"].str.strip() == args.uso].copy()
        try:
            wass_results = run_wasserstein_detection(_df_wass)
            wasserstein_summary(wass_results)
        except Exception as e:
            print(f"  Wasserstein error: {e}")

    # M16 — Topological Data Analysis (Persistent Homology)
    if len(results) > 0:
        print(f"\n{'='*80}")
        print(f"  M16 — TOPOLOGICAL DATA ANALYSIS (Persistent Homology)")
        print(f"{'='*80}")
        _df_tda = compute_monthly_features(df_all, external_df=external_df)
        _df_tda = _df_tda[_df_tda["uso"].str.strip() == args.uso].copy()
        try:
            tda_results = run_tda_detection(_df_tda)
            tda_summary(tda_results)
        except Exception as e:
            print(f"  TDA error: {e}")

    # Counterfactual Explanations (EU AI Act compliance)
    if len(results) > 0:
        print(f"\n{'='*80}")
        print(f"  COUNTERFACTUAL EXPLANATIONS (EU AI Act Art. 13-14)")
        print(f"{'='*80}")
        try:
            cf_df = generate_counterfactuals(results, df_features=_df_adv, top_n=10)
            counterfactual_summary(cf_df)
        except Exception as e:
            print(f"  Counterfactual error: {e}")

    # Transfer Entropy — Information-theoretic causal analysis
    if len(results) > 0:
        print(f"\n{'='*80}")
        print(f"  TRANSFER ENTROPY — Flujo de informacion entre barrios")
        print(f"{'='*80}")
        _df_te = compute_monthly_features(df_all, external_df=external_df)
        _df_te = _df_te[_df_te["uso"].str.strip() == args.uso].copy()
        try:
            te_results = run_transfer_entropy_analysis(_df_te, results, top_n=10)
            transfer_entropy_summary(te_results)
        except Exception as e:
            print(f"  Transfer Entropy error: {e}")

    # AquaCare — Welfare Detection (personas vulnerables)
    if len(results) > 0:
        print(f"\n{'='*80}")
        print(f"  AQUACARE — Deteccion de emergencias en personas vulnerables")
        print(f"{'='*80}")
        _df_welfare = compute_monthly_features(df_all, external_df=external_df)
        _df_welfare = _df_welfare[_df_welfare["uso"].str.strip() == args.uso].copy()
        caudal_path = "data/_caudal_medio_sector_hidraulico_hora_2024_-caudal_medio_sector_hidraulico_hora_2024.csv"
        try:
            welfare_alerts = run_welfare_detection(
                _df_welfare, results=results,
                caudal_path=caudal_path if Path(caudal_path).exists() else None
            )
            welfare_summary(welfare_alerts)

            # AquaCare Validations (5 tests)
            from welfare_detector import run_aquacare_validations
            aquacare_val = run_aquacare_validations()
        except Exception as e:
            print(f"  Welfare error: {e}")

    # FRAUD GROUND TRUTH — Validacion con datos reales de AMAEM
    if len(results) > 0:
        print(f"\n{'='*80}")
        print(f"  GROUND TRUTH — Validacion con cambios de contador AMAEM")
        print(f"{'='*80}")
        cambios_path = "data/cambios-de-contador-solo-alicante_hackaton-dataart-cambios-de-contador-solo-alicante.csv.csv"
        try:
            gt = load_ground_truth(cambios_path, CONTADORES_PATH)
            val_df = validate_detections(results, gt)
            ground_truth_summary(val_df, gt)
        except Exception as e:
            print(f"  Ground truth error: {e}")

    # METER READINGS — Analisis de 4.5M lecturas individuales
    if len(results) > 0:
        print(f"\n{'='*80}")
        print(f"  LECTURAS INDIVIDUALES — 4.5M lecturas de contadores")
        print(f"{'='*80}")
        try:
            meter_analysis = run_meter_analysis(results_df=results, years=(2022, 2023, 2024))
            meter_readings_summary(meter_analysis)
        except Exception as e:
            print(f"  Meter readings error: {e}")

    # GRAPH NETWORK — Analisis de topologia de red
    if len(results) > 0:
        print(f"\n{'='*80}")
        print(f"  GRAPH NETWORK — Topologia de la red de agua (21K tuberias)")
        print(f"{'='*80}")
        try:
            graph_analysis = run_graph_analysis(results)
            graph_network_summary(graph_analysis)
        except Exception as e:
            print(f"  Graph network error: {e}")

    # HYDRAULIC TWIN — Digital twin hidraulico
    if len(results) > 0:
        print(f"\n{'='*80}")
        print(f"  DIGITAL TWIN HIDRAULICO — Modelo fisico de la red")
        print(f"{'='*80}")
        try:
            twin_results = run_hydraulic_twin(results)
            hydraulic_twin_summary(twin_results)
        except Exception as e:
            print(f"  Hydraulic twin error: {e}")

    # PSEUDO-GROUND-TRUTH + ABLATION + CALIBRACION
    gt = None
    if len(results) > 0:
        cambios_path = "data/cambios-de-contador-solo-alicante_hackaton-dataart-cambios-de-contador-solo-alicante.csv.csv"
        try:
            gt = load_ground_truth(cambios_path, CONTADORES_PATH)
            results = build_pseudo_labels(results, gt)
            pseudo_metrics = evaluate_against_pseudo(results)
            pseudo_ground_truth_summary(pseudo_metrics)
        except Exception as e:
            print(f"  Pseudo-GT error: {e}")

    if len(results) > 0 and "pseudo_label" in results.columns:
        try:
            abl_df = run_ablation_study(results)
            red_df = compute_pairwise_redundancy(results)
            ablation_summary(abl_df, red_df)
            if not abl_df.empty:
                abl_path = str(Path(args.output or "results_full.csv").parent / "ablation_results.csv")
                abl_df.to_csv(abl_path, index=False)
                print(f"  Ablation results saved: {abl_path}")
        except Exception as e:
            print(f"  Ablation error: {e}")

        try:
            cal_report = compute_calibration_report(results)
            print_calibration_report(cal_report)
        except Exception as e:
            print(f"  Calibration error: {e}")

    # ─── CROSS-VALIDATION VS FRAUDE REAL ─────────────────────────
    if len(results) > 0:
        try:
            cv_results = run_cross_validation(
                results_path=args.output or "results_full.csv",
            )
            print_cross_validation_summary(cv_results)
        except Exception as e:
            print(f"  Cross-validation error: {e}")

    # ─── VALIDACION INDEPENDIENTE (7 capas de evidencia externa, Fisher's p=0.002) ────
    if len(results) > 0:
        try:
            from independent_validation import run_independent_validation, print_validation_summary
            iv_results = run_independent_validation(
                results_path=args.output or "results_full.csv",
            )
            print_validation_summary(iv_results)
        except Exception as e:
            print(f"  Independent validation error: {e}")

    # ─── STABLE CORE — barrios beyond reasonable doubt ────
    if len(results) > 0:
        try:
            from advanced_ensemble import compute_stable_core, print_stable_core
            stable_core = compute_stable_core(results)
            print_stable_core(stable_core, results)
        except Exception as e:
            print(f"  Stable core error: {e}")

    # ─── TESTS QUANT — Null Permutation + Bootstrap + Moran's I ────
    if len(results) > 0:
        try:
            from advanced_ensemble import null_permutation_test, bootstrap_stable_core, print_quant_tests
            null_result = null_permutation_test(results, n_perm=1000, top_k=5)
            bootstrap_result = bootstrap_stable_core(results, n_boot=500)
            print_quant_tests(null_result, bootstrap_result)
        except Exception as e:
            print(f"  Quant tests error: {e}")

        try:
            from spatial_detector import compute_morans_i, print_morans_i
            from gis_features import compute_barrio_adjacency
            barrio_col = results["barrio_key"].str.split("__").str[0]
            barrio_scores = results.groupby(barrio_col)["ensemble_score"].mean()
            adjacency = compute_barrio_adjacency()
            moran_result = compute_morans_i(barrio_scores, adjacency)
            print_morans_i(moran_result)
        except Exception as e:
            print(f"  Moran's I error: {e}")

    # Guardar CSV si se pide
    if args.output and len(results) > 0:
        # Convertir lista de modelos a string para CSV
        results_csv = results.copy()
        results_csv["models_detecting"] = results_csv["models_detecting"].apply(
            lambda x: ";".join(x) if x else ""
        )
        # Merge riesgo de infraestructura al CSV si disponible
        if not risk_df.empty:
            results_csv["_barrio_clean"] = results_csv["barrio_key"].str.split("__").str[0]
            results_csv = results_csv.merge(
                risk_df, left_on="_barrio_clean", right_on="barrio",
                how="left", suffixes=("", "_risk"),
            )
            results_csv = results_csv.drop(
                columns=["barrio_risk", "_barrio_clean"], errors="ignore"
            )
        results_csv.to_csv(args.output, index=False)
        print(f"\n  Resultados guardados en: {args.output}")
        print(f"  Filas: {len(results_csv)}, Columnas: {len(results_csv.columns)}")


if __name__ == "__main__":
    main()
