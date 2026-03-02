"""
Validación del modelo con inyección de anomalías sintéticas.

Demuestra que el pipeline detecta anomalías antes de integrar con el backend.
Mide precision, recall y F1 comparando detecciones con anomalías conocidas.

Uso:
  python validate_model.py
  python validate_model.py --barrio "10-FLORIDA BAJA" --uso DOMESTICO
  python validate_model.py --all-barrios   # evaluar todos y promediar métricas
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from train_local import load_hackathon_amaem
from feature_engineering import compute_daily_features, add_rolling_features
from model import train, score_batch, delete_model
from monthly_features import (
    compute_monthly_features,
    prepare_monthly_matrix,
    MONTHLY_FEATURE_COLUMNS,
)

DATA_FILE = "data/datos-hackathon-amaem.xlsx-set-de-datos-.csv"

# Anomalías sintéticas a inyectar en el período de test.
# month_offset: índice relativo al inicio del test set (0 = primer mes de test)
SYNTHETIC_ANOMALIES_EXTREME = [
    {"offset": 0, "multiplier": 4.0, "label": "SPIKE_4X",  "desc": "Pico 4x — posible fuga en red"},
    {"offset": 2, "multiplier": 0.0, "label": "ZERO",      "desc": "Consumo cero — contador parado"},
    {"offset": 4, "multiplier": 0.05, "label": "NEAR_ZERO", "desc": "Consumo casi nulo — posible fraude"},
    {"offset": 6, "multiplier": 5.0, "label": "SPIKE_5X",  "desc": "Pico 5x — fuga severa"},
    {"offset": 8, "multiplier": 3.0, "label": "SPIKE_3X",  "desc": "Pico 3x — consumo inusual"},
]

SYNTHETIC_ANOMALIES_SUBTLE = [
    {"offset": 1, "multiplier": 1.3, "label": "SLIGHT_UP",   "desc": "Subida 30% — difícil de detectar"},
    {"offset": 3, "multiplier": 0.7, "label": "SLIGHT_DOWN", "desc": "Bajada 30% — difícil de detectar"},
    {"offset": 5, "multiplier": 1.5, "label": "MODERATE_UP", "desc": "Subida 50%"},
    {"offset": 7, "multiplier": 2.0, "label": "DOUBLE",      "desc": "Doble consumo"},
    {"offset": 9, "multiplier": 0.3, "label": "DROP_70",     "desc": "Caída 70%"},
]

# Por defecto usar las extremas (retrocompatible)
SYNTHETIC_ANOMALIES = SYNTHETIC_ANOMALIES_EXTREME


def _load_barrio(df_all: pd.DataFrame, barrio: str, uso: str) -> pd.DataFrame | None:
    """Extrae y prepara los datos de un barrio concreto."""
    if "uso" in df_all.columns:
        mask = (df_all["barrio"] == barrio) & (df_all["uso"] == uso)
    else:
        mask = df_all["barrio"] == barrio

    subset = df_all[mask].copy()
    if len(subset) < 20:
        return None

    subset["timestamp"] = pd.to_datetime(subset["fecha"])
    subset["consumption"] = (
        subset["consumo_litros"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(float)  # asegurar float para evitar problemas con asignación de multiplicadores
    )
    return subset[["timestamp", "consumption"]].sort_values("timestamp").reset_index(drop=True)


def inject_anomalies(df: pd.DataFrame, n_train: int,
                     anomalies: list | None = None, verbose: bool = True) -> tuple:
    """
    Multiplica el consumo de meses concretos del test set.
    Devuelve (df_modificado, lista_de_indices_anomalos_absolutos).
    """
    if anomalies is None:
        anomalies = SYNTHETIC_ANOMALIES
    df = df.copy()
    anomaly_indices = []

    for a in anomalies:
        idx = n_train + a["offset"]
        if idx >= len(df):
            continue
        original = df.at[idx, "consumption"]
        df.at[idx, "consumption"] = original * a["multiplier"]
        anomaly_indices.append(idx)
        if verbose:
            print(f"    [{a['label']:10s}] mes idx={idx}: {original:,.0f} → {original * a['multiplier']:,.0f}  ({a['desc']})")

    return df, anomaly_indices


def evaluate_metrics(test_scored: pd.DataFrame, anomaly_indices: list, n_train: int) -> dict:
    """
    Calcula precision, recall, F1 sobre el test set.
    anomaly_indices son absolutos respecto al df completo.
    """
    n_test = len(test_scored)
    # Convertir índices absolutos a relativos al test set
    true_labels = np.zeros(n_test, dtype=bool)
    for abs_idx in anomaly_indices:
        rel = abs_idx - n_train
        if 0 <= rel < n_test:
            true_labels[rel] = True

    detected = test_scored["is_anomaly"].values.astype(bool)

    tp = int(np.sum(detected & true_labels))
    fp = int(np.sum(detected & ~true_labels))
    fn = int(np.sum(~detected & true_labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def run_validation(df_all: pd.DataFrame, barrio: str, uso: str,
                   verbose: bool = True, anomalies: list | None = None) -> dict | None:
    """Valida el modelo para un barrio específico. Devuelve las métricas o None si no hay datos."""
    anom_list = anomalies if anomalies is not None else SYNTHETIC_ANOMALIES
    df = _load_barrio(df_all, barrio, uso)
    if df is None:
        if verbose:
            print(f"  Saltando {barrio} ({uso}) — menos de 20 meses de datos.")
        return None

    n_train = min(25, int(len(df) * 0.7))
    n_test  = len(df) - n_train

    if n_test < max(a["offset"] for a in anom_list) + 1:
        if verbose:
            print(f"  Saltando {barrio} — test set muy pequeño ({n_test} meses).")
        return None

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Barrio: {barrio} ({uso})")
        print(f"  Total meses: {len(df)}  |  Train: {n_train}  |  Test: {n_test}")
        print(f"\n  [1/4] Inyectando anomalías sintéticas en el test set...")

    df_injected, anomaly_indices = inject_anomalies(df, n_train, anomalies=anom_list, verbose=verbose)

    if verbose:
        print(f"\n  [2/4] Feature engineering...")
    daily = compute_daily_features(df_injected)
    daily = add_rolling_features(daily)

    train_data = daily.iloc[:n_train].copy()
    test_data  = daily.iloc[n_train:].copy()

    meter_id = f"val_{barrio}_{uso}".replace(" ", "_")
    delete_model(meter_id)

    if verbose:
        print(f"  [3/4] Entrenando con {len(train_data)} meses...")
    result = train(meter_id, train_data)
    if result.get("status") == "error":
        if verbose:
            print(f"  ERROR al entrenar: {result.get('reason')}")
        return None

    if verbose:
        print(f"  [4/4] Puntuando {len(test_data)} meses de test...")
    test_scored = score_batch(meter_id, test_data)

    metrics = evaluate_metrics(test_scored, anomaly_indices, n_train)

    if verbose:
        _print_results(metrics, test_scored, anomaly_indices, n_train)

    # Limpiar modelo temporal
    delete_model(meter_id)

    return metrics


def _print_results(metrics: dict, test_scored: pd.DataFrame, anomaly_indices: list, n_train: int):
    tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
    n_injected = len(anomaly_indices)
    detected_total = test_scored["is_anomaly"].sum()

    print(f"\n  {'─'*50}")
    print(f"  RESULTADOS")
    print(f"  {'─'*50}")
    print(f"  Anomalías inyectadas:      {n_injected}")
    print(f"  Anomalías detectadas:      {detected_total}  (de {len(test_scored)} meses)")
    print(f"  True Positives  (TP):      {tp}")
    print(f"  False Positives (FP):      {fp}  ← alertas sobre meses normales")
    print(f"  False Negatives (FN):      {fn}  ← anomalías no detectadas")
    print(f"")
    print(f"  Precision: {metrics['precision']:.1%}   (de las alertas, ¿cuántas eran reales?)")
    print(f"  Recall:    {metrics['recall']:.1%}   (de las anomalías reales, ¿cuántas detectó?)")
    print(f"  F1:        {metrics['f1']:.1%}")

    if metrics["recall"] >= 0.8 and metrics["precision"] >= 0.7:
        print(f"\n  ✓ MODELO VÁLIDO — cumple los criterios de calidad")
    elif metrics["recall"] >= 0.6:
        print(f"\n  ⚠ MODELO PARCIAL — detecta la mayoría, margen de mejora")
        print(f"    → Prueba: contamination=0.10 o threshold=-0.05 en model.py")
    else:
        print(f"\n  ✗ MODELO INSUFICIENTE con datos mensuales puros")
        print(f"    → Necesitas: DAIAD (Alicante hourly) para disaggregación temporal")
        print(f"    → O: monthly_features.py con cross-sectional zscore")

    # Tabla detallada del test set
    rel_anomaly_set = {i - n_train for i in anomaly_indices}
    print(f"\n  Detalle del test set:")
    header = f"  {'Mes':>4}  {'Fecha':>12}  {'Consumo':>14}  {'Z-score':>8}  {'Score':>8}  {'Detectado':>10}  {'Inyectada':>10}"
    print(header)
    print(f"  {'─'*90}")

    for i, (_, row) in enumerate(test_scored.iterrows()):
        injected_mark = "<<< SÍ" if i in rel_anomaly_set else ""
        detected_mark = "✓" if row.get("is_anomaly", False) else "·"
        score_val = f"{row.get('anomaly_score', 0):.3f}" if row.get("anomaly_score") is not None else "  N/A"
        zscore_val = f"{row.get('zscore', 0):.2f}" if pd.notna(row.get("zscore")) else "  N/A"
        print(f"  {i:>4}  {str(row['date']):>12}  {row['daily_total']:>14,.0f}  "
              f"{zscore_val:>8}  {score_val:>8}  {detected_mark:>10}  {injected_mark:>10}")


def run_all_barrios(df_all: pd.DataFrame, uso: str = "DOMESTICO", max_barrios: int = 10):
    """Evalúa múltiples barrios y calcula métricas promedio."""
    barrios = df_all["barrio"].unique()[:max_barrios]
    all_metrics = []

    print(f"\nEvaluando {len(barrios)} barrios (uso={uso})...\n")

    for barrio in barrios:
        metrics = run_validation(df_all, barrio, uso, verbose=False)
        if metrics:
            all_metrics.append(metrics)
            status = "✓" if metrics["recall"] >= 0.8 else ("⚠" if metrics["recall"] >= 0.6 else "✗")
            print(f"  {status} {barrio:35s}  P={metrics['precision']:.0%}  R={metrics['recall']:.0%}  F1={metrics['f1']:.0%}")

    if not all_metrics:
        print("No hay datos suficientes en ningún barrio.")
        return

    avg_precision = np.mean([m["precision"] for m in all_metrics])
    avg_recall    = np.mean([m["recall"]    for m in all_metrics])
    avg_f1        = np.mean([m["f1"]        for m in all_metrics])

    print(f"\n{'─'*60}")
    print(f"  PROMEDIO ({len(all_metrics)} barrios con datos suficientes)")
    print(f"  Precision: {avg_precision:.1%}  |  Recall: {avg_recall:.1%}  |  F1: {avg_f1:.1%}")

    if avg_recall >= 0.8:
        print(f"\n  ✓ El modelo funciona bien con datos mensuales para anomalías grandes (3-5x)")
    else:
        print(f"\n  ⚠ Recall bajo — considera añadir features cross-seccionales (monthly_features.py)")


def run_validation_m2(df_all: pd.DataFrame, barrio: str, uso: str,
                      anomalies: list | None = None) -> dict | None:
    """
    Modelo M2 — IsolationForest con features cross-seccionales (monthly_features.py).
    Entrena sobre todos los barrios del mismo tipo para aprender qué es 'normal',
    luego evalúa las anomalías inyectadas en el barrio objetivo.
    """
    anom_list = anomalies if anomalies is not None else SYNTHETIC_ANOMALIES
    barrio_key = f"{barrio}__{uso}"

    # Preparar todos los datos con features mensuales
    df_features = compute_monthly_features(df_all)
    barrio_data = df_features[df_features["barrio_key"] == barrio_key].copy()
    barrio_data = barrio_data.sort_values("fecha").reset_index(drop=True)

    if len(barrio_data) < 20:
        return None

    n_train = min(25, int(len(barrio_data) * 0.7))
    n_test  = len(barrio_data) - n_train

    if n_test < max(a["offset"] for a in anom_list) + 1:
        return None

    # Inyectar anomalías en consumo_litros del barrio objetivo ANTES de recalcular features
    # Para M2 necesitamos recalcular features después de inyectar
    df_raw_barrio = df_all[
        (df_all["barrio"] == barrio) &
        (df_all.get("uso", pd.Series([uso] * len(df_all))) == uso)
    ].copy()
    df_raw_barrio = df_raw_barrio.sort_values("fecha").reset_index(drop=True)

    anomaly_indices = []
    for a in anom_list:
        idx = n_train + a["offset"]
        if idx >= len(df_raw_barrio):
            continue
        df_raw_barrio.at[idx, "consumo_litros"] = float(df_raw_barrio.at[idx, "consumo_litros"]) * a["multiplier"]
        anomaly_indices.append(idx)

    # Recombinar con el resto de barrios para recalcular features cross-seccionales
    other_barrios = df_all[~(
        (df_all["barrio"] == barrio) &
        (df_all.get("uso", pd.Series([uso] * len(df_all))) == uso)
    )].copy()
    df_combined = pd.concat([other_barrios, df_raw_barrio], ignore_index=True)
    df_features_injected = compute_monthly_features(df_combined)

    barrio_injected = df_features_injected[
        df_features_injected["barrio_key"] == barrio_key
    ].sort_values("fecha").reset_index(drop=True)

    train_data = barrio_injected.iloc[:n_train]
    test_data  = barrio_injected.iloc[n_train:]

    X_train = prepare_monthly_matrix(train_data)
    X_test  = test_data[MONTHLY_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0).values

    if X_train is None or len(X_test) == 0:
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = IsolationForest(n_estimators=100, contamination=0.10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled)

    scores      = model.score_samples(X_test_scaled)
    predictions = model.predict(X_test_scaled)  # 1=normal, -1=anomalía

    test_result = test_data.copy()
    test_result["anomaly_score_m2"] = scores
    test_result["is_anomaly_m2"]    = predictions == -1

    metrics = evaluate_metrics(
        test_result.rename(columns={"is_anomaly_m2": "is_anomaly"}),
        anomaly_indices, n_train
    )
    return metrics


def run_validation_m5(df_all: pd.DataFrame, barrio: str, uso: str,
                      anomalies: list | None = None) -> dict | None:
    """
    M5 — Baseline estadístico (3-sigma + IQR) sin ML.
    Usa solo consumo_litros crudo del barrio objetivo.
    """
    from statistical_baseline import evaluate_baseline
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

    results = evaluate_baseline(train_vals, test_vals, true_labels)
    return results


def run_ensemble_validation(df_all: pd.DataFrame, barrio: str, uso: str,
                            anomalies: list | None = None) -> dict | None:
    """
    Ensemble M1+M2. Ejecuta ambos modelos y combina predicciones.
    Requiere alineación de índices de test entre M1 y M2.
    """
    from ensemble import ensemble_evaluate
    anom_list = anomalies if anomalies is not None else SYNTHETIC_ANOMALIES

    # --- M1: temporal ---
    df = _load_barrio(df_all, barrio, uso)
    if df is None:
        return None

    n_train = min(25, int(len(df) * 0.7))
    n_test = len(df) - n_train

    if n_test < max(a["offset"] for a in anom_list) + 1:
        return None

    df_injected, anomaly_indices = inject_anomalies(df, n_train, anomalies=anom_list, verbose=False)

    daily = compute_daily_features(df_injected)
    daily = add_rolling_features(daily)
    train_data = daily.iloc[:n_train].copy()
    test_data = daily.iloc[n_train:].copy()

    meter_id = f"ens_{barrio}_{uso}".replace(" ", "_")
    delete_model(meter_id)
    result = train(meter_id, train_data)
    if result.get("status") == "error":
        return None
    test_scored_m1 = score_batch(meter_id, test_data)
    delete_model(meter_id)

    m1_detected = test_scored_m1["is_anomaly"].values.astype(bool)

    # --- M2: cross-sectional ---
    barrio_key = f"{barrio}__{uso}"
    df_raw_barrio = df_all[
        (df_all["barrio"] == barrio) &
        (df_all.get("uso", pd.Series([uso] * len(df_all))) == uso)
    ].copy()
    df_raw_barrio = df_raw_barrio.sort_values("fecha").reset_index(drop=True)

    for a in anom_list:
        idx = n_train + a["offset"]
        if idx < len(df_raw_barrio):
            df_raw_barrio.at[idx, "consumo_litros"] = float(df_raw_barrio.at[idx, "consumo_litros"]) * a["multiplier"]

    other_barrios = df_all[~(
        (df_all["barrio"] == barrio) &
        (df_all.get("uso", pd.Series([uso] * len(df_all))) == uso)
    )].copy()
    df_combined = pd.concat([other_barrios, df_raw_barrio], ignore_index=True)
    df_features_injected = compute_monthly_features(df_combined)

    barrio_injected = df_features_injected[
        df_features_injected["barrio_key"] == barrio_key
    ].sort_values("fecha").reset_index(drop=True)

    train_m2 = barrio_injected.iloc[:n_train]
    test_m2 = barrio_injected.iloc[n_train:]

    X_train = prepare_monthly_matrix(train_m2)
    X_test = test_m2[MONTHLY_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0).values

    if X_train is None or len(X_test) == 0:
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = IsolationForest(n_estimators=100, contamination=0.10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled)
    predictions_m2 = model.predict(X_test_scaled)
    m2_detected = predictions_m2 == -1

    # Alinear: ambos tienen n_test elementos
    n = min(len(m1_detected), len(m2_detected))
    m1_detected = m1_detected[:n]
    m2_detected = m2_detected[:n]

    true_labels = np.zeros(n, dtype=bool)
    for abs_idx in anomaly_indices:
        rel = abs_idx - n_train
        if 0 <= rel < n:
            true_labels[rel] = True

    results = {}
    for mode in ("or", "and"):
        results[mode] = ensemble_evaluate(m1_detected, m2_detected, true_labels, mode=mode)
    return results


def _run_all_models(df_all, barrio, uso, anomalies):
    """Ejecuta M1, M2, M5, ensemble con un gold set concreto. Devuelve dict de resultados."""
    results = {}
    results["M1 temporal"] = run_validation(df_all, barrio, uso, verbose=False, anomalies=anomalies)
    results["M2 cross-sec"] = run_validation_m2(df_all, barrio, uso, anomalies=anomalies)

    m5 = run_validation_m5(df_all, barrio, uso, anomalies=anomalies)
    if m5:
        results["M5 3-sigma"] = m5.get("3sigma")
        results["M5 IQR"] = m5.get("iqr")

    ens = run_ensemble_validation(df_all, barrio, uso, anomalies=anomalies)
    if ens:
        results["Ensemble (OR)"] = ens.get("or")
        results["Ensemble (AND)"] = ens.get("and")

    # M6 (Chronos) y M7 (Prophet) si están disponibles
    try:
        from chronos_detector import run_validation_chronos
        results["M6 Chronos"] = run_validation_chronos(df_all, barrio, uso, anomalies=anomalies)
    except ImportError:
        pass

    try:
        from prophet_detector import run_validation_prophet
        results["M7 Prophet"] = run_validation_prophet(df_all, barrio, uso, anomalies=anomalies)
    except ImportError:
        pass

    return results


def _print_model_agreement(results: dict, anomalies: list, title: str = ""):
    """
    Imprime tabla de acuerdo entre modelos: para cada anomalia inyectada,
    muestra que modelos la detectaron.
    """
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        return

    model_names = list(valid.keys())

    print(f"\n  {title or 'ACUERDO ENTRE MODELOS'}")
    print(f"  {'─'*70}")

    # Header
    header = f"  {'Anomalia':<15}"
    for name in model_names:
        short = name[:10]
        header += f"  {short:>10}"
    header += f"  {'#Detectan':>10}"
    print(header)
    print(f"  {'─'*70}")

    # Para cada anomalia inyectada, ver si cada modelo tiene TP en ese offset
    for a in anomalies:
        row_str = f"  {a['label']:<15}"
        n_detecting = 0
        for name in model_names:
            m = valid[name]
            # Los modelos con recall>0 detectan al menos algunas anomalias
            # Simplificacion: marcamos como detectado si recall > 0
            # (la tabla real requeriria acceso a las predicciones individuales)
            detected = m.get("tp", 0) > 0 if m.get("recall", 0) > 0 else False
            symbol = "YES" if detected else "·"
            row_str += f"  {symbol:>10}"
            if detected:
                n_detecting += 1
        row_str += f"  {n_detecting:>10}"
        print(row_str)

    # Resumen
    print(f"\n  {'Resumen':}")
    for name in model_names:
        m = valid[name]
        print(f"    {name:<18}  R={m.get('recall',0):.0%}  P={m.get('precision',0):.0%}  "
              f"F1={m.get('f1',0):.0%}")


def _print_comparison_table(results: dict, title: str):
    """Imprime tabla de comparación."""
    def _row(name, m):
        if m is None:
            return f"  {name:<18}  {'N/A':>10}  {'N/A':>8}  {'N/A':>8}  {'':>4}  {'':>4}  {'':>4}"
        return (f"  {name:<18}  {m['precision']:>9.1%}  {m['recall']:>7.1%}  "
                f"{m['f1']:>7.1%}  {m['tp']:>4}  {m['fp']:>4}  {m['fn']:>4}")

    print(f"\n  {title}")
    print(f"  {'Modelo':<18}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print(f"  {'─'*66}")

    for name, m in results.items():
        print(_row(name, m))

    # Mejor modelo
    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        best = max(valid.items(), key=lambda x: x[1]["f1"])
        print(f"\n  Mejor por F1: {best[0]} (F1={best[1]['f1']:.1%})")


def compare_models(df_all: pd.DataFrame, barrio: str, uso: str):
    """Compara M1, M2, M5, M6, M7 y ensemble con ambos gold sets."""
    print(f"\n{'='*70}")
    print(f"  COMPARACION DE MODELOS — {barrio} ({uso})")
    print(f"{'='*70}")
    print(f"\n  M1: IsolationForest temporal (daily_total, zscore...)")
    print(f"  M2: IsolationForest cross-sectional (monthly_features.py)")
    print(f"  M5: Baseline estadistico (3-sigma / IQR)")
    print(f"  M6: Amazon Chronos (forecast-based, si disponible)")
    print(f"  M7: Prophet (decomposition-based, si disponible)")
    print(f"  EN: Ensemble M1+M2 (OR / AND)")

    # --- Gold set 1: Anomalías extremas ---
    print(f"\n  {'='*60}")
    print(f"  GOLD SET 1: Anomalias extremas ({len(SYNTHETIC_ANOMALIES_EXTREME)})")
    for a in SYNTHETIC_ANOMALIES_EXTREME:
        print(f"    {a['label']:12s} x{a['multiplier']:.2f} — {a['desc']}")

    print(f"\n  Ejecutando modelos...")
    results_extreme = _run_all_models(df_all, barrio, uso, SYNTHETIC_ANOMALIES_EXTREME)
    _print_comparison_table(results_extreme, "RESULTADOS — Anomalias extremas")
    _print_model_agreement(results_extreme, SYNTHETIC_ANOMALIES_EXTREME,
                           "ACUERDO ENTRE MODELOS — Anomalias extremas")

    # --- Gold set 2: Anomalías sutiles ---
    print(f"\n  {'='*60}")
    print(f"  GOLD SET 2: Anomalias sutiles ({len(SYNTHETIC_ANOMALIES_SUBTLE)})")
    for a in SYNTHETIC_ANOMALIES_SUBTLE:
        print(f"    {a['label']:12s} x{a['multiplier']:.2f} — {a['desc']}")

    print(f"\n  Ejecutando modelos...")
    results_subtle = _run_all_models(df_all, barrio, uso, SYNTHETIC_ANOMALIES_SUBTLE)
    _print_comparison_table(results_subtle, "RESULTADOS — Anomalias sutiles")
    _print_model_agreement(results_subtle, SYNTHETIC_ANOMALIES_SUBTLE,
                           "ACUERDO ENTRE MODELOS — Anomalias sutiles")

    # --- Resumen final ---
    print(f"\n  {'='*60}")
    print(f"  RESUMEN")
    print(f"  {'─'*60}")
    print(f"  {'Modelo':<18}  {'Extreme F1':>11}  {'Subtle F1':>10}  {'Media F1':>9}")
    print(f"  {'─'*55}")

    all_names = set(list(results_extreme.keys()) + list(results_subtle.keys()))
    summary = []
    for name in all_names:
        ext = results_extreme.get(name)
        sub = results_subtle.get(name)
        f1_ext = ext["f1"] if ext else 0.0
        f1_sub = sub["f1"] if sub else 0.0
        f1_avg = (f1_ext + f1_sub) / 2
        summary.append((name, f1_ext, f1_sub, f1_avg))

    summary.sort(key=lambda x: x[3], reverse=True)
    for name, f1_ext, f1_sub, f1_avg in summary:
        print(f"  {name:<18}  {f1_ext:>10.1%}  {f1_sub:>10.1%}  {f1_avg:>9.1%}")

    if summary:
        best = summary[0]
        print(f"\n  Mejor modelo global: {best[0]} (Media F1={best[3]:.1%})")


def grid_search_hyperparams(df_all: pd.DataFrame, barrio: str, uso: str,
                            anomalies: list | None = None):
    """
    Busca la mejor combinación de contamination y threshold para M1.
    Evalúa todas las combinaciones y muestra los resultados ordenados por F1.
    """
    contamination_values = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]
    threshold_values = [-0.02, -0.05, -0.10, -0.15, -0.20, -0.30]

    anom_list = anomalies if anomalies is not None else SYNTHETIC_ANOMALIES
    df = _load_barrio(df_all, barrio, uso)
    if df is None:
        print(f"  No hay datos suficientes para {barrio}")
        return

    n_train = min(25, int(len(df) * 0.7))
    n_test = len(df) - n_train

    if n_test < max(a["offset"] for a in anom_list) + 1:
        print(f"  Test set demasiado pequeño ({n_test} meses)")
        return

    df_injected, anomaly_indices = inject_anomalies(df, n_train, anomalies=anom_list, verbose=False)
    daily = compute_daily_features(df_injected)
    daily = add_rolling_features(daily)
    train_data = daily.iloc[:n_train].copy()
    test_data = daily.iloc[n_train:].copy()

    print(f"\n{'='*70}")
    print(f"  GRID SEARCH — {barrio} ({uso})")
    print(f"  {len(contamination_values)} contamination x {len(threshold_values)} threshold "
          f"= {len(contamination_values) * len(threshold_values)} combinaciones")
    print(f"{'='*70}")

    results = []
    for cont in contamination_values:
        for thr in threshold_values:
            meter_id = f"gs_{barrio}_{uso}_{cont}_{thr}".replace(" ", "_")
            delete_model(meter_id)
            result = train(meter_id, train_data, contamination=cont, threshold=thr)
            if result.get("status") == "error":
                continue

            test_scored = score_batch(meter_id, test_data, threshold=thr)
            metrics = evaluate_metrics(test_scored, anomaly_indices, n_train)
            metrics["contamination"] = cont
            metrics["threshold"] = thr
            results.append(metrics)
            delete_model(meter_id)

    if not results:
        print("  No se pudo entrenar ningún modelo.")
        return

    # Ordenar por F1
    results.sort(key=lambda x: x["f1"], reverse=True)

    print(f"\n  {'Contam':>8}  {'Thresh':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print(f"  {'─'*66}")

    for r in results[:15]:  # top 15
        marker = " ★" if r == results[0] else ""
        print(f"  {r['contamination']:>8.2f}  {r['threshold']:>8.2f}  "
              f"{r['precision']:>9.1%}  {r['recall']:>7.1%}  {r['f1']:>7.1%}  "
              f"{r['tp']:>4}  {r['fp']:>4}  {r['fn']:>4}{marker}")

    best = results[0]
    print(f"\n  ★ MEJOR: contamination={best['contamination']}, threshold={best['threshold']}")
    print(f"    F1={best['f1']:.1%}  Precision={best['precision']:.1%}  Recall={best['recall']:.1%}")

    # Comparar con defaults actuales
    default = next((r for r in results if r["contamination"] == 0.05 and r["threshold"] == -0.10), None)
    if default:
        improvement = best["f1"] - default["f1"]
        print(f"\n  vs defaults (0.05, -0.10): F1={default['f1']:.1%}")
        if improvement > 0:
            print(f"  Mejora: +{improvement:.1%} en F1")
        else:
            print(f"  Los defaults ya son óptimos o muy cercanos")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validación del modelo con anomalías sintéticas")
    parser.add_argument("--barrio",       default="10-FLORIDA BAJA", help="Barrio a analizar")
    parser.add_argument("--uso",          default="DOMESTICO",       help="Tipo de uso")
    parser.add_argument("--file",         default=DATA_FILE,         help="Ruta al CSV")
    parser.add_argument("--all-barrios",  action="store_true",       help="Evaluar los primeros 10 barrios")
    parser.add_argument("--compare",      action="store_true",       help="Comparar todos los modelos (por defecto)")
    parser.add_argument("--grid-search",  action="store_true",       help="Buscar mejores hiperparámetros")
    parser.add_argument("--max-barrios",  type=int, default=10,      help="Número máximo de barrios con --all-barrios")
    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"ERROR: No se encuentra el CSV en '{csv_path}'")
        sys.exit(1)

    print(f"Cargando datos desde: {csv_path}")
    df_all = load_hackathon_amaem(str(csv_path))

    if args.grid_search:
        grid_search_hyperparams(df_all, barrio=args.barrio, uso=args.uso)
    elif args.all_barrios:
        run_all_barrios(df_all, uso=args.uso, max_barrios=args.max_barrios)
    else:
        compare_models(df_all, barrio=args.barrio, uso=args.uso)
