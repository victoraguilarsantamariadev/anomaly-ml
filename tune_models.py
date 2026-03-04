"""
Grid search de hiperparametros para los 4 modelos de deteccion de anomalias.

Inyecta anomalias sinteticas (EXTREME y SUBTLE), mide precision/recall/F1,
y encuentra los parametros optimos para cada modelo.

Uso:
  python tune_models.py                        # todos los modelos
  python tune_models.py --models m2 m5         # solo M2 y M5 (rapido)
  python tune_models.py --models m6            # solo Chronos
  python tune_models.py --barrios 15           # mas barrios (mas robusto)
  python tune_models.py --chronos-barrios 3    # menos barrios para Chronos (rapido)
"""

import argparse
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from train_local import load_hackathon_amaem
from monthly_features import (
    compute_monthly_features,
    RELATIVE_FEATURE_COLUMNS,
)
from statistical_baseline import score_iqr
from validate_model import (
    _load_barrio,
    inject_anomalies,
    SYNTHETIC_ANOMALIES_EXTREME,
    SYNTHETIC_ANOMALIES_SUBTLE,
)

DATA_FILE = "data/datos-hackathon-amaem.xlsx-set-de-datos-.csv"

# ─── Grids de parametros ──────────────────────────────────────────
PARAM_GRID_M2 = {"contamination": [0.01, 0.03, 0.05, 0.08, 0.10]}
PARAM_GRID_M5 = {"iqr_multiplier": [1.5, 2.0, 2.5, 3.0]}
PARAM_GRID_M6 = {"threshold_sigma": [1.5, 2.0, 2.5, 3.0, 3.5]}
PARAM_GRID_M7 = {
    "interval_width": [0.90, 0.95, 0.97, 0.99],
    "changepoint_prior_scale": [0.05, 0.10, 0.15, 0.30],
}


def _compute_prf(detected: np.ndarray, true_labels: np.ndarray) -> dict:
    """Calcula precision, recall, F1."""
    tp = int(np.sum(detected & true_labels))
    fp = int(np.sum(detected & ~true_labels))
    fn = int(np.sum(~detected & true_labels))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


# ─── M2 Grid Search ──────────────────────────────────────────────

def grid_search_m2(df_all, barrios, uso, verbose=True):
    """Grid search de contamination para M2 IsolationForest."""
    all_results = []

    for anom_name, anom_list in [("extreme", SYNTHETIC_ANOMALIES_EXTREME),
                                  ("subtle", SYNTHETIC_ANOMALIES_SUBTLE)]:
        for barrio in barrios:
            barrio_key = f"{barrio}__{uso}"

            # Preparar features sin anomalia para obtener split info
            df_features_clean = compute_monthly_features(df_all)
            barrio_data = df_features_clean[
                df_features_clean["barrio_key"] == barrio_key
            ].sort_values("fecha").reset_index(drop=True)

            if len(barrio_data) < 20:
                continue

            n_train = min(25, int(len(barrio_data) * 0.7))
            n_test = len(barrio_data) - n_train
            if n_test < max(a["offset"] for a in anom_list) + 1:
                continue

            # Inyectar anomalias en raw y recalcular features
            df_raw_barrio = df_all[
                (df_all["barrio"] == barrio) & (df_all["uso"] == uso)
            ].copy().sort_values("fecha").reset_index(drop=True)

            anomaly_indices = []
            for a in anom_list:
                idx = n_train + a["offset"]
                if idx < len(df_raw_barrio):
                    df_raw_barrio.at[idx, "consumo_litros"] = (
                        float(df_raw_barrio.at[idx, "consumo_litros"]) * a["multiplier"]
                    )
                    anomaly_indices.append(idx)

            other = df_all[~((df_all["barrio"] == barrio) & (df_all["uso"] == uso))].copy()
            df_combined = pd.concat([other, df_raw_barrio], ignore_index=True)
            df_features = compute_monthly_features(df_combined)

            barrio_injected = df_features[
                df_features["barrio_key"] == barrio_key
            ].sort_values("fecha").reset_index(drop=True)

            train_data = barrio_injected.iloc[:n_train]
            test_data = barrio_injected.iloc[n_train:]

            available_cols = [c for c in RELATIVE_FEATURE_COLUMNS if c in barrio_injected.columns]
            X_train = train_data[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
            X_test = test_data[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

            if len(X_train) < 5 or len(X_test) == 0:
                continue

            true_labels = np.zeros(len(X_test), dtype=bool)
            for abs_idx in anomaly_indices:
                rel = abs_idx - n_train
                if 0 <= rel < len(true_labels):
                    true_labels[rel] = True

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Probar cada contamination
            for cont in PARAM_GRID_M2["contamination"]:
                model = IsolationForest(
                    n_estimators=100, contamination=cont, random_state=42, n_jobs=-1
                )
                model.fit(X_train_s)
                preds = model.predict(X_test_s)
                detected = preds == -1

                metrics = _compute_prf(detected, true_labels)
                metrics.update({
                    "model": "M2", "barrio": barrio, "anomaly_set": anom_name,
                    "contamination": cont,
                })
                all_results.append(metrics)

            if verbose:
                print(f"    M2 — {barrio} ({anom_name}): {len(PARAM_GRID_M2['contamination'])} configs")

    return all_results


# ─── M5 Grid Search ──────────────────────────────────────────────

def grid_search_m5(df_all, barrios, uso, verbose=True):
    """Grid search de iqr_multiplier para M5 sobre deviation_from_group_trend."""
    all_results = []

    for anom_name, anom_list in [("extreme", SYNTHETIC_ANOMALIES_EXTREME),
                                  ("subtle", SYNTHETIC_ANOMALIES_SUBTLE)]:
        for barrio in barrios:
            barrio_key = f"{barrio}__{uso}"

            # Inyectar en raw y recalcular features
            df_raw_barrio = df_all[
                (df_all["barrio"] == barrio) & (df_all["uso"] == uso)
            ].copy().sort_values("fecha").reset_index(drop=True)

            if len(df_raw_barrio) < 20:
                continue

            n_train = min(25, int(len(df_raw_barrio) * 0.7))
            n_test = len(df_raw_barrio) - n_train
            if n_test < max(a["offset"] for a in anom_list) + 1:
                continue

            anomaly_indices = []
            df_raw_mod = df_raw_barrio.copy()
            for a in anom_list:
                idx = n_train + a["offset"]
                if idx < len(df_raw_mod):
                    df_raw_mod.at[idx, "consumo_litros"] = (
                        float(df_raw_mod.at[idx, "consumo_litros"]) * a["multiplier"]
                    )
                    anomaly_indices.append(idx)

            other = df_all[~((df_all["barrio"] == barrio) & (df_all["uso"] == uso))].copy()
            df_combined = pd.concat([other, df_raw_mod], ignore_index=True)
            df_features = compute_monthly_features(df_combined)

            barrio_data = df_features[
                df_features["barrio_key"] == barrio_key
            ].sort_values("fecha").reset_index(drop=True)

            train_vals = barrio_data.iloc[:n_train]["deviation_from_group_trend"].values.astype(float)
            test_vals = barrio_data.iloc[n_train:]["deviation_from_group_trend"].values.astype(float)
            train_vals = np.nan_to_num(train_vals, nan=0.0)
            test_vals = np.nan_to_num(test_vals, nan=0.0)

            true_labels = np.zeros(len(test_vals), dtype=bool)
            for abs_idx in anomaly_indices:
                rel = abs_idx - n_train
                if 0 <= rel < len(true_labels):
                    true_labels[rel] = True

            for mult in PARAM_GRID_M5["iqr_multiplier"]:
                detected = score_iqr(test_vals, train_vals, multiplier=mult)
                metrics = _compute_prf(detected, true_labels)
                metrics.update({
                    "model": "M5", "barrio": barrio, "anomaly_set": anom_name,
                    "iqr_multiplier": mult,
                })
                all_results.append(metrics)

            if verbose:
                print(f"    M5 — {barrio} ({anom_name}): {len(PARAM_GRID_M5['iqr_multiplier'])} configs")

    return all_results


# ─── M6 Grid Search ──────────────────────────────────────────────

def grid_search_m6(df_all, barrios, uso, verbose=True):
    """Grid search de threshold_sigma para M6 Chronos (optimizado: inferencia 1 vez)."""
    from chronos_detector import score_chronos_raw, apply_chronos_threshold
    all_results = []

    for anom_name, anom_list in [("extreme", SYNTHETIC_ANOMALIES_EXTREME),
                                  ("subtle", SYNTHETIC_ANOMALIES_SUBTLE)]:
        for barrio in barrios:
            df = _load_barrio(df_all, barrio, uso)
            if df is None:
                continue

            n_train = min(25, int(len(df) * 0.7))
            n_test = len(df) - n_train
            if n_test < max(a["offset"] for a in anom_list) + 1:
                continue

            df_injected, anomaly_indices = inject_anomalies(
                df, n_train, anomalies=anom_list, verbose=False
            )

            train_vals = df_injected.iloc[:n_train]["consumption"].values.astype(float)
            test_vals = df_injected.iloc[n_train:]["consumption"].values.astype(float)

            true_labels = np.zeros(len(test_vals), dtype=bool)
            for abs_idx in anomaly_indices:
                rel = abs_idx - n_train
                if 0 <= rel < len(true_labels):
                    true_labels[rel] = True

            # Inferencia UNA vez
            t0 = time.time()
            medians, stds = score_chronos_raw(train_vals, test_vals, num_samples=20)
            elapsed = time.time() - t0

            # Probar cada threshold (instantaneo)
            for sigma in PARAM_GRID_M6["threshold_sigma"]:
                detected = apply_chronos_threshold(medians, stds, test_vals, sigma)
                metrics = _compute_prf(detected, true_labels)
                metrics.update({
                    "model": "M6", "barrio": barrio, "anomaly_set": anom_name,
                    "threshold_sigma": sigma,
                })
                all_results.append(metrics)

            if verbose:
                print(f"    M6 — {barrio} ({anom_name}): inferencia {elapsed:.1f}s, "
                      f"{len(PARAM_GRID_M6['threshold_sigma'])} thresholds")

    return all_results


# ─── M7 Grid Search ──────────────────────────────────────────────

def grid_search_m7(df_all, barrios, uso, verbose=True):
    """Grid search de interval_width x changepoint_prior_scale para M7 Prophet."""
    from prophet_detector import score_prophet
    all_results = []

    for anom_name, anom_list in [("extreme", SYNTHETIC_ANOMALIES_EXTREME),
                                  ("subtle", SYNTHETIC_ANOMALIES_SUBTLE)]:
        for barrio in barrios:
            df = _load_barrio(df_all, barrio, uso)
            if df is None:
                continue

            n_train = min(25, int(len(df) * 0.7))
            n_test = len(df) - n_train
            if n_test < max(a["offset"] for a in anom_list) + 1:
                continue

            df_injected, anomaly_indices = inject_anomalies(
                df, n_train, anomalies=anom_list, verbose=False
            )

            train_vals = df_injected.iloc[:n_train]["consumption"].values.astype(float)
            test_vals = df_injected.iloc[n_train:]["consumption"].values.astype(float)
            train_dates = df_injected.iloc[:n_train]["timestamp"].values
            test_dates = df_injected.iloc[n_train:]["timestamp"].values

            true_labels = np.zeros(len(test_vals), dtype=bool)
            for abs_idx in anomaly_indices:
                rel = abs_idx - n_train
                if 0 <= rel < len(true_labels):
                    true_labels[rel] = True

            for iw in PARAM_GRID_M7["interval_width"]:
                for cps in PARAM_GRID_M7["changepoint_prior_scale"]:
                    try:
                        detected = score_prophet(
                            train_vals, test_vals, train_dates, test_dates,
                            interval_width=iw, changepoint_prior_scale=cps,
                        )
                        metrics = _compute_prf(detected, true_labels)
                    except Exception:
                        metrics = {"tp": 0, "fp": 0, "fn": 5, "precision": 0, "recall": 0, "f1": 0}

                    metrics.update({
                        "model": "M7", "barrio": barrio, "anomaly_set": anom_name,
                        "interval_width": iw, "changepoint_prior_scale": cps,
                    })
                    all_results.append(metrics)

            if verbose:
                n_combos = len(PARAM_GRID_M7["interval_width"]) * len(PARAM_GRID_M7["changepoint_prior_scale"])
                print(f"    M7 — {barrio} ({anom_name}): {n_combos} configs")

    return all_results


# ─── Agregacion y seleccion ───────────────────────────────────────

def aggregate_and_select(results: list, param_cols: list, min_precision: float = 0.80):
    """
    Agrega resultados por parametros (media sobre barrios),
    calcula F1 ponderado (50% extreme + 50% subtle),
    selecciona mejor con restriccion de precision.
    """
    if not results:
        return None, None

    df = pd.DataFrame(results)

    # Agregar por (parametros, anomaly_set) — media sobre barrios
    group_cols = param_cols + ["anomaly_set"]
    agg = df.groupby(group_cols).agg(
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        n_barrios=("barrio", "nunique"),
    ).reset_index()

    # Separar extreme y subtle
    extreme = agg[agg["anomaly_set"] == "extreme"][param_cols + ["mean_f1", "mean_precision", "mean_recall"]].copy()
    subtle = agg[agg["anomaly_set"] == "subtle"][param_cols + ["mean_f1", "mean_precision", "mean_recall"]].copy()

    extreme = extreme.rename(columns={"mean_f1": "f1_extreme", "mean_precision": "p_extreme", "mean_recall": "r_extreme"})
    subtle = subtle.rename(columns={"mean_f1": "f1_subtle", "mean_precision": "p_subtle", "mean_recall": "r_subtle"})

    combined = extreme.merge(subtle, on=param_cols, how="outer").fillna(0)
    combined["f1_weighted"] = 0.5 * combined["f1_extreme"] + 0.5 * combined["f1_subtle"]
    combined["p_avg"] = 0.5 * combined["p_extreme"] + 0.5 * combined["p_subtle"]

    # Filtrar por precision minima
    valid = combined[combined["p_avg"] >= min_precision]
    if len(valid) == 0:
        # Relajar restriccion
        valid = combined[combined["p_avg"] >= min_precision - 0.10]
        if len(valid) == 0:
            valid = combined

    best = valid.nlargest(1, "f1_weighted").iloc[0]
    return combined.sort_values("f1_weighted", ascending=False), best


def print_report(model_name, combined, best, param_cols):
    """Imprime reporte de tuning para un modelo."""
    print(f"\n  {model_name}")
    print(f"  {'─'*70}")

    # Header
    param_header = "  ".join(f"{p:>12}" for p in param_cols)
    print(f"  {param_header}  {'P(ext)':>7}  {'R(ext)':>7}  {'F1(ext)':>8}  "
          f"{'P(sub)':>7}  {'R(sub)':>7}  {'F1(sub)':>8}  {'F1(avg)':>8}")
    print(f"  {'─'*95}")

    for _, row in combined.iterrows():
        params_str = "  ".join(f"{row[p]:>12.3f}" if isinstance(row[p], float) else f"{row[p]:>12}" for p in param_cols)
        is_best = all(row[p] == best[p] for p in param_cols)
        marker = " ★" if is_best else ""
        print(f"  {params_str}  {row['p_extreme']:>6.0%}  {row['r_extreme']:>6.0%}  "
              f"{row['f1_extreme']:>7.0%}  {row['p_subtle']:>6.0%}  {row['r_subtle']:>6.0%}  "
              f"{row['f1_subtle']:>7.0%}  {row['f1_weighted']:>7.0%}{marker}")

    # Mejor
    params_str = ", ".join(f"{p}={best[p]}" for p in param_cols)
    print(f"\n  ★ MEJOR: {params_str}")
    print(f"    F1 ponderado={best['f1_weighted']:.1%}  "
          f"P(avg)={best['p_avg']:.1%}  "
          f"F1(extreme)={best['f1_extreme']:.1%}  F1(subtle)={best['f1_subtle']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Grid search de hiperparametros")
    parser.add_argument("--file", default=DATA_FILE)
    parser.add_argument("--models", nargs="+", default=["m2", "m5", "m6", "m7"],
                        choices=["m2", "m5", "m6", "m7"])
    parser.add_argument("--barrios", type=int, default=10,
                        help="Num barrios para M2/M5/M7 (default: 10)")
    parser.add_argument("--chronos-barrios", type=int, default=5,
                        help="Num barrios para M6 Chronos (default: 5)")
    parser.add_argument("--uso", default="DOMESTICO")
    parser.add_argument("--min-precision", type=float, default=0.80)
    parser.add_argument("--output", type=str, default=None,
                        help="Guardar resultados en JSON")
    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"ERROR: No se encuentra {csv_path}")
        sys.exit(1)

    print(f"{'='*80}")
    print(f"  AQUAGUARD AI — Grid Search de Hiperparametros")
    print(f"{'='*80}")
    print(f"  Modelos:  {', '.join(args.models)}")
    print(f"  Barrios:  {args.barrios} (Chronos: {args.chronos_barrios})")
    print(f"  Min precision: {args.min_precision:.0%}")
    print(f"  Anomaly sets: EXTREME ({len(SYNTHETIC_ANOMALIES_EXTREME)}) + "
          f"SUBTLE ({len(SYNTHETIC_ANOMALIES_SUBTLE)})")

    print(f"\n  Cargando datos...")
    df_all = load_hackathon_amaem(str(csv_path))

    # Seleccionar barrios con datos suficientes
    all_barrios = sorted(df_all[df_all["uso"] == args.uso]["barrio"].unique())
    barrios = all_barrios[:args.barrios]
    barrios_chronos = all_barrios[:args.chronos_barrios]

    print(f"  Barrios seleccionados: {len(barrios)} "
          f"(Chronos: {len(barrios_chronos)})")

    t_start = time.time()
    best_params = {}

    # ─── M2 ───
    if "m2" in args.models:
        print(f"\n{'─'*60}")
        print(f"  [M2] IsolationForest — grid: contamination {PARAM_GRID_M2['contamination']}")
        results_m2 = grid_search_m2(df_all, barrios, args.uso)
        combined, best = aggregate_and_select(results_m2, ["contamination"], args.min_precision)
        if combined is not None:
            print_report("M2 (IsolationForest, RELATIVE features)", combined, best, ["contamination"])
            best_params["m2"] = {"contamination": float(best["contamination"])}

    # ─── M5 ───
    if "m5" in args.models:
        print(f"\n{'─'*60}")
        print(f"  [M5] IQR — grid: iqr_multiplier {PARAM_GRID_M5['iqr_multiplier']}")
        results_m5 = grid_search_m5(df_all, barrios, args.uso)
        combined, best = aggregate_and_select(results_m5, ["iqr_multiplier"], args.min_precision)
        if combined is not None:
            print_report("M5 (IQR sobre deviation_from_group_trend)", combined, best, ["iqr_multiplier"])
            best_params["m5"] = {"iqr_multiplier": float(best["iqr_multiplier"])}

    # ─── M6 ───
    if "m6" in args.models:
        print(f"\n{'─'*60}")
        print(f"  [M6] Chronos — grid: threshold_sigma {PARAM_GRID_M6['threshold_sigma']}")
        print(f"  (Solo {len(barrios_chronos)} barrios para Chronos)")
        results_m6 = grid_search_m6(df_all, barrios_chronos, args.uso)
        combined, best = aggregate_and_select(results_m6, ["threshold_sigma"], args.min_precision)
        if combined is not None:
            print_report("M6 (Amazon Chronos)", combined, best, ["threshold_sigma"])
            best_params["m6"] = {"threshold_sigma": float(best["threshold_sigma"])}

    # ─── M7 ───
    if "m7" in args.models:
        print(f"\n{'─'*60}")
        print(f"  [M7] Prophet — grid: interval_width x changepoint_prior_scale")
        results_m7 = grid_search_m7(df_all, barrios, args.uso)
        combined, best = aggregate_and_select(
            results_m7, ["interval_width", "changepoint_prior_scale"], args.min_precision
        )
        if combined is not None:
            print_report("M7 (Facebook Prophet)", combined, best,
                        ["interval_width", "changepoint_prior_scale"])
            best_params["m7"] = {
                "interval_width": float(best["interval_width"]),
                "changepoint_prior_scale": float(best["changepoint_prior_scale"]),
            }

    # ─── Resumen final ───
    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"  RESUMEN — Parametros optimos (tiempo total: {elapsed:.0f}s)")
    print(f"{'='*80}")

    for model, params in best_params.items():
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"  {model.upper()}: {params_str}")

    # Comando recomendado
    if best_params:
        cmd_parts = ["venv/bin/python run_all_models.py"]
        if "m2" in best_params:
            cmd_parts.append(f"--contamination {best_params['m2']['contamination']}")
        if "m5" in best_params:
            cmd_parts.append(f"--iqr-multiplier {best_params['m5']['iqr_multiplier']}")
        if "m6" in best_params:
            cmd_parts.append(f"--chronos-sigma {best_params['m6']['threshold_sigma']}")
        if "m7" in best_params:
            cmd_parts.append(f"--prophet-interval {best_params['m7']['interval_width']}")
            cmd_parts.append(f"--prophet-changepoint {best_params['m7']['changepoint_prior_scale']}")

        print(f"\n  Comando recomendado:")
        print(f"  {' '.join(cmd_parts)}")

    # Guardar JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\n  Parametros guardados en: {args.output}")


if __name__ == "__main__":
    main()
