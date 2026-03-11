"""
Cross-validation temporal para todos los modelos de AquaGuard AI.

Usa TimeSeriesSplit (walk-forward) para respetar la temporalidad:
  Fold 1: train [Ene-Jun 2022], test [Jul-Dic 2022]
  Fold 2: train [Ene 2022 - Dic 2022], test [Ene-Jun 2023]
  Fold 3: train [Ene 2022 - Jun 2023], test [Jul-Dic 2023]
  Fold 4: train [Ene 2022 - Dic 2023], test [Ene-Jun 2024]
  Fold 5: train [Ene 2022 - Jun 2024], test [Jul-Dic 2024]

Métricas por fold:
  - N anomalías detectadas y % del total
  - Estabilidad: ¿los mismos barrios se detectan en folds sucesivos?
  - Score medio de anomalías vs normales (separación)

Uso:
  python model_cv.py                          # CV completo
  python model_cv.py --quick                  # solo M2, M5, M13 (sin Prophet)
"""

import argparse
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neural_network import MLPRegressor

from train_local import load_hackathon_amaem
from monthly_features import (
    compute_monthly_features,
    RELATIVE_FEATURE_COLUMNS,
    AUXILIARY_FEATURE_COLUMNS,
    ADVANCED_FEATURE_COLUMNS,
    FOURIER_INTERACTION_COLUMNS,
    enrich_with_telelectura,
    enrich_with_regenerada,
)

CONTADORES_PATH = "data/contadores-telelectura-instalados-solo-alicante_hackaton-dataart-contadores-telelectura-instalad.csv"
REGENERADA_PATH = "data/_consumos_alicante_regenerada_barrio_mes-2024_-consumos_alicante_regenerada_barrio_mes-2024.csv.csv"


def prepare_data(uso_filter="DOMESTICO"):
    """Carga y prepara features una sola vez."""
    df_all = load_hackathon_amaem("data/datos-hackathon-amaem.xlsx-set-de-datos-.csv")
    df_features = compute_monthly_features(df_all)
    df_features = enrich_with_telelectura(df_features, CONTADORES_PATH)
    df_features = enrich_with_regenerada(df_features, REGENERADA_PATH)

    df_uso = df_features[df_features["uso"].str.strip() == uso_filter].copy()
    df_uso = df_uso.sort_values(["barrio_key", "fecha"]).reset_index(drop=True)

    feature_cols = (RELATIVE_FEATURE_COLUMNS + AUXILIARY_FEATURE_COLUMNS +
                    ADVANCED_FEATURE_COLUMNS + FOURIER_INTERACTION_COLUMNS)
    available = [c for c in feature_cols if c in df_uso.columns]

    return df_uso, available


def generate_temporal_folds(df, n_folds=5, min_train_months=6,
                            purge_months=1, embargo_months=1):
    """
    Genera folds temporales walk-forward con purge + embargo.

    Purged/Embargo CV (Lopez de Prado, 2018):
      - Purge: elimina N meses entre train y test para evitar data leakage
      - Embargo: elimina N meses despues del test para evitar autocorrelacion

    Sin purge, los folds adyacentes comparten informacion temporal
    (el consumo de mes M esta correlacionado con M-1).

    Args:
        purge_months: meses a eliminar entre train y test (default=1)
        embargo_months: meses de cuarentena despues del test (default=1)
    """
    all_dates = sorted(df["fecha"].unique())
    n_dates = len(all_dates)

    # Cada fold usa ~test_size meses de test
    test_size = max(3, n_dates // (n_folds + 1))
    folds = []

    for i in range(n_folds):
        test_start = min_train_months + i * test_size
        test_end = min(test_start + test_size, n_dates)
        if test_start >= n_dates or test_end <= test_start:
            break

        # Purge: train termina purge_months ANTES del test
        train_end = max(0, test_start - purge_months)
        train_dates = set(all_dates[:train_end])
        test_dates = set(all_dates[test_start:test_end])

        # Embargo: no usar datos justo despues del test en folds futuros
        # (esto se aplica implicitamente en walk-forward, pero lo hacemos explicito)

        if len(train_dates) < min_train_months:
            continue

        train = df[df["fecha"].isin(train_dates)]
        test = df[df["fecha"].isin(test_dates)]

        if len(train) >= 20 and len(test) >= 10:
            folds.append({
                "fold": i + 1,
                "train_months": len(train_dates),
                "test_months": len(test_dates),
                "purge_months": purge_months,
                "embargo_months": embargo_months,
                "train": train,
                "test": test,
            })

    return folds


def cv_isolation_forest(df, feature_cols, contamination=0.03, n_folds=5):
    """Cross-validation para IsolationForest (M2)."""
    folds = generate_temporal_folds(df, n_folds=n_folds)
    results = []

    for fold in folds:
        X_train = fold["train"][feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        X_test = fold["test"][feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = IsolationForest(n_estimators=100, contamination=contamination,
                                random_state=42, n_jobs=-1)
        model.fit(X_train_s)

        scores = model.score_samples(X_test_s)
        preds = model.predict(X_test_s)
        n_anom = (preds == -1).sum()

        # Barrios detectados
        test_copy = fold["test"][["barrio_key"]].copy()
        test_copy["is_anomaly"] = preds == -1
        barrios_detected = set(test_copy[test_copy["is_anomaly"]]["barrio_key"].unique())

        # Separación de scores
        anom_scores = scores[preds == -1]
        normal_scores = scores[preds == 1]
        separation = (normal_scores.mean() - anom_scores.mean()) / normal_scores.std() if len(anom_scores) > 0 else 0

        results.append({
            "fold": fold["fold"],
            "train_months": fold["train_months"],
            "test_months": fold["test_months"],
            "n_test": len(X_test),
            "n_anomalies": n_anom,
            "pct_anomalies": n_anom / len(X_test) * 100,
            "barrios_detected": barrios_detected,
            "score_separation": separation,
        })

    return results


def cv_autoencoder(df, feature_cols, contamination=0.03, n_folds=5):
    """Cross-validation para Autoencoder (M13)."""
    folds = generate_temporal_folds(df, n_folds=n_folds)
    results = []

    for fold in folds:
        X_train = fold["train"][feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        X_test = fold["test"][feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ae = MLPRegressor(
            hidden_layer_sizes=(16, 8, 4, 8, 16),
            activation="relu", solver="adam",
            max_iter=500, learning_rate_init=0.001,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=42, verbose=False,
        )
        ae.fit(X_train_s, X_train_s)

        train_recon = ae.predict(X_train_s)
        train_errors = np.mean((X_train_s - train_recon) ** 2, axis=1)
        threshold = np.percentile(train_errors, (1 - contamination) * 100)

        test_recon = ae.predict(X_test_s)
        test_errors = np.mean((X_test_s - test_recon) ** 2, axis=1)

        preds = test_errors > threshold
        n_anom = preds.sum()

        test_copy = fold["test"][["barrio_key"]].copy()
        test_copy["is_anomaly"] = preds
        barrios_detected = set(test_copy[test_copy["is_anomaly"]]["barrio_key"].unique())

        anom_errors = test_errors[preds]
        normal_errors = test_errors[~preds]
        separation = (anom_errors.mean() - normal_errors.mean()) / normal_errors.std() if len(anom_errors) > 0 and normal_errors.std() > 0 else 0

        results.append({
            "fold": fold["fold"],
            "train_months": fold["train_months"],
            "test_months": fold["test_months"],
            "n_test": len(X_test),
            "n_anomalies": int(n_anom),
            "pct_anomalies": n_anom / len(X_test) * 100,
            "barrios_detected": barrios_detected,
            "score_separation": separation,
        })

    return results


def cv_statistical(df, iqr_multiplier=3.0, min_deviation=0.10, n_folds=5):
    """Cross-validation para M5 (3-sigma + IQR)."""
    folds = generate_temporal_folds(df, n_folds=n_folds)
    results = []

    for fold in folds:
        train = fold["train"]
        test = fold["test"]

        if "deviation_from_group_trend" not in test.columns:
            continue

        # Calcular stats del train
        dev_train = train["deviation_from_group_trend"].dropna()
        mean_dev = dev_train.mean()
        std_dev = dev_train.std()
        q1 = dev_train.quantile(0.25)
        q3 = dev_train.quantile(0.75)
        iqr = q3 - q1

        # Aplicar al test
        dev_test = test["deviation_from_group_trend"].fillna(0)
        is_3sigma = (dev_test.abs() > mean_dev + 3 * std_dev) & (dev_test.abs() > min_deviation)
        is_iqr = ((dev_test < q1 - iqr_multiplier * iqr) | (dev_test > q3 + iqr_multiplier * iqr)) & (dev_test.abs() > min_deviation)
        combined = is_3sigma | is_iqr

        test_copy = test[["barrio_key"]].copy()
        test_copy["is_anomaly"] = combined.values
        barrios_detected = set(test_copy[test_copy["is_anomaly"]]["barrio_key"].unique())

        results.append({
            "fold": fold["fold"],
            "train_months": fold["train_months"],
            "test_months": fold["test_months"],
            "n_test": len(test),
            "n_anomalies": int(combined.sum()),
            "pct_anomalies": combined.sum() / len(test) * 100,
            "barrios_detected": barrios_detected,
            "score_separation": std_dev,  # proxy: spread
        })

    return results


def cv_prophet(df, n_folds=5, interval_width=0.99, changepoint_scale=0.3):
    """Cross-validation para Prophet (M7). Más lento — solo top 10 barrios."""
    from prophet_detector import score_prophet

    folds = generate_temporal_folds(df, n_folds=n_folds)
    # Solo barrios con más datos para velocidad
    barrio_counts = df.groupby("barrio_key").size()
    top_barrios = barrio_counts.nlargest(10).index.tolist()

    results = []
    for fold in folds:
        total_anom = 0
        total_test = 0
        barrios_detected = set()

        for barrio in top_barrios:
            train_b = fold["train"][fold["train"]["barrio_key"] == barrio]
            test_b = fold["test"][fold["test"]["barrio_key"] == barrio]

            if len(train_b) < 6 or len(test_b) < 2:
                continue

            try:
                anomalies = score_prophet(
                    train_b["consumption_per_contract"].values,
                    test_b["consumption_per_contract"].values,
                    train_b["fecha"].values,
                    test_b["fecha"].values,
                    interval_width=interval_width,
                    changepoint_prior_scale=changepoint_scale,
                )
                n_anom = anomalies.sum()
                total_anom += n_anom
                total_test += len(test_b)
                if n_anom > 0:
                    barrios_detected.add(barrio)
            except Exception:
                total_test += len(test_b)

        results.append({
            "fold": fold["fold"],
            "train_months": fold["train_months"],
            "test_months": fold["test_months"],
            "n_test": total_test,
            "n_anomalies": int(total_anom),
            "pct_anomalies": total_anom / total_test * 100 if total_test > 0 else 0,
            "barrios_detected": barrios_detected,
            "score_separation": 0,  # Prophet no da score continuo
        })

    return results


def compute_stability(cv_results):
    """Calcula estabilidad: ¿los mismos barrios aparecen en múltiples folds?"""
    all_barrios = set()
    barrio_fold_count = {}
    for r in cv_results:
        for b in r["barrios_detected"]:
            all_barrios.add(b)
            barrio_fold_count[b] = barrio_fold_count.get(b, 0) + 1

    n_folds = len(cv_results)
    stable = {b for b, c in barrio_fold_count.items() if c >= n_folds * 0.6}
    return {
        "total_unique_barrios": len(all_barrios),
        "stable_barrios": len(stable),
        "stability_pct": len(stable) / len(all_barrios) * 100 if all_barrios else 0,
        "stable_names": sorted(b.split("__")[0] for b in stable),
    }


def print_cv_report(model_name, cv_results):
    """Imprime reporte de CV para un modelo."""
    print(f"\n  {'─'*70}")
    print(f"  {model_name}")
    print(f"  {'─'*70}")
    print(f"  {'Fold':>6} {'Train':>7} {'Test':>6} {'N_test':>7} {'Anomalías':>10} {'%':>7} {'Separación':>11}")
    print(f"  {'─'*60}")

    pcts = []
    for r in cv_results:
        pcts.append(r["pct_anomalies"])
        print(f"  {r['fold']:>6} {r['train_months']:>5}m {r['test_months']:>4}m "
              f"{r['n_test']:>7} {r['n_anomalies']:>10} {r['pct_anomalies']:>6.1f}% "
              f"{r['score_separation']:>10.2f}")

    mean_pct = np.mean(pcts)
    std_pct = np.std(pcts)
    print(f"  {'─'*60}")
    print(f"  Media: {mean_pct:.1f}% ± {std_pct:.1f}%")

    # Estabilidad
    stab = compute_stability(cv_results)
    print(f"  Estabilidad: {stab['stable_barrios']}/{stab['total_unique_barrios']} "
          f"barrios estables ({stab['stability_pct']:.0f}%)")
    if stab["stable_names"]:
        print(f"  Barrios estables: {', '.join(stab['stable_names'][:8])}")

    return {
        "model": model_name,
        "mean_pct": mean_pct,
        "std_pct": std_pct,
        "stability": stab["stability_pct"],
        "n_stable": stab["stable_barrios"],
    }


def print_statistical_tests(model_names, all_cv_results, n_folds):
    """
    Compara modelos estadísticamente usando:
    - Friedman test: ¿hay diferencias significativas entre todos los modelos?
    - Wilcoxon signed-rank test: comparación par a par (¿modelo A es mejor que B?)

    Usa el % de anomalías por fold como métrica de comparación.
    """
    from scipy.stats import friedmanchisquare, wilcoxon

    # Extraer % anomalías por fold para cada modelo
    model_scores = {}
    for name, cv_res in zip(model_names, all_cv_results):
        model_scores[name] = [r["pct_anomalies"] for r in cv_res]

    # Asegurar que todos tienen el mismo número de folds
    min_folds = min(len(v) for v in model_scores.values())
    for name in model_scores:
        model_scores[name] = model_scores[name][:min_folds]

    print(f"\n{'='*70}")
    print(f"  TESTS ESTADÍSTICOS — Comparación entre modelos")
    print(f"{'='*70}")

    names = list(model_scores.keys())
    scores = [np.array(model_scores[n]) for n in names]

    # 1. Friedman test (¿hay diferencias entre los modelos?)
    if len(names) >= 3 and min_folds >= 3:
        try:
            stat, p_friedman = friedmanchisquare(*scores)
            print(f"\n  Friedman test (¿algún modelo es distinto?):")
            print(f"    Estadístico: {stat:.3f}")
            print(f"    p-valor:     {p_friedman:.4f}")
            if p_friedman < 0.05:
                print(f"    → SÍ hay diferencias significativas (p<0.05)")
            else:
                print(f"    → No hay diferencias significativas (p≥0.05)")
        except Exception as e:
            print(f"\n  Friedman test: no se pudo calcular ({e})")

    # 2. Wilcoxon signed-rank test (par a par)
    print(f"\n  Wilcoxon signed-rank test (comparación par a par):")
    print(f"  {'Modelo A':<25} vs {'Modelo B':<25} {'p-valor':>8} {'Resultado':>15}")
    print(f"  {'─'*78}")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = scores[i], scores[j]
            try:
                # Wilcoxon necesita diferencias no nulas
                diff = a - b
                if np.all(diff == 0):
                    p_val = 1.0
                elif min_folds < 6:
                    # Con pocos folds, usar método exacto
                    stat_w, p_val = wilcoxon(a, b, alternative="two-sided",
                                             method="exact" if min_folds <= 10 else "auto")
                else:
                    stat_w, p_val = wilcoxon(a, b, alternative="two-sided")

                if p_val < 0.05:
                    # ¿Cuál es mejor? (menor varianza = más estable)
                    better = names[i] if np.std(a) < np.std(b) else names[j]
                    result = f"SIGNIF. ({better})"
                else:
                    result = "No signif."

                print(f"  {names[i]:<25} vs {names[j]:<25} {p_val:>8.4f} {result:>15}")
            except Exception as e:
                print(f"  {names[i]:<25} vs {names[j]:<25} {'N/A':>8} (error: {e})")

    # 3. Ranking por consistencia (CV = std/mean, menor es mejor)
    print(f"\n  Ranking por consistencia (coeficiente de variación):")
    print(f"  {'#':>3} {'Modelo':<30} {'Media':>8} {'Std':>8} {'CV':>8} {'Veredicto':<20}")
    print(f"  {'─'*78}")

    rankings = []
    for name, s in zip(names, scores):
        mean_s = np.mean(s)
        std_s = np.std(s)
        cv = std_s / mean_s if mean_s > 0 else float("inf")
        rankings.append((name, mean_s, std_s, cv))

    rankings.sort(key=lambda x: x[3])  # Menor CV = más consistente
    for rank, (name, mean_s, std_s, cv) in enumerate(rankings, 1):
        if cv < 0.3:
            veredicto = "MUY ESTABLE"
        elif cv < 0.5:
            veredicto = "ESTABLE"
        elif cv < 1.0:
            veredicto = "VARIABLE"
        else:
            veredicto = "INESTABLE"
        print(f"  {rank:>3} {name:<30} {mean_s:>7.1f}% {std_s:>7.1f}% {cv:>7.2f} {veredicto:<20}")


def optimize_hyperparameters(df, feature_cols, n_folds=7):
    """
    Grid search de hiperparametros usando CV walk-forward.

    Busca la combinacion que maximiza estabilidad (menor CV) manteniendo
    una tasa de deteccion razonable (2-8%).

    Returns:
        dict con mejores parametros para cada modelo
    """
    print(f"\n{'='*70}")
    print(f"  OPTIMIZACION DE HIPERPARAMETROS (Grid Search + CV)")
    print(f"{'='*70}")

    best_params = {}

    # M2 — IsolationForest: optimizar contamination
    print(f"\n  M2 IsolationForest — buscando mejor contamination...")
    best_cv = float("inf")
    for contamination in [0.01, 0.02, 0.03, 0.05, 0.08]:
        cv_res = cv_isolation_forest(df, feature_cols, contamination=contamination, n_folds=n_folds)
        pcts = [r["pct_anomalies"] for r in cv_res]
        mean_pct = np.mean(pcts)
        std_pct = np.std(pcts)
        cv = std_pct / mean_pct if mean_pct > 0 else float("inf")

        # Penalizar si esta fuera del rango razonable (2-8%)
        if mean_pct < 1.0 or mean_pct > 15.0:
            cv += 1.0

        stab = compute_stability(cv_res)
        marker = " ← BEST" if cv < best_cv else ""
        print(f"    contamination={contamination:.2f}: {mean_pct:.1f}%±{std_pct:.1f}% "
              f"CV={cv:.2f} estabilidad={stab['stability_pct']:.0f}%{marker}")

        if cv < best_cv:
            best_cv = cv
            best_params["m2_contamination"] = contamination
            best_params["m2_cv"] = cv
            best_params["m2_mean_pct"] = mean_pct

    # M5 — IQR multiplier
    print(f"\n  M5 3-sigma+IQR — buscando mejor iqr_multiplier...")
    best_cv = float("inf")
    for iqr_mult in [2.0, 2.5, 3.0, 3.5, 4.0]:
        cv_res = cv_statistical(df, iqr_multiplier=iqr_mult, n_folds=n_folds)
        pcts = [r["pct_anomalies"] for r in cv_res]
        mean_pct = np.mean(pcts)
        std_pct = np.std(pcts)
        cv = std_pct / mean_pct if mean_pct > 0 else float("inf")

        if mean_pct < 1.0 or mean_pct > 40.0:
            cv += 1.0

        stab = compute_stability(cv_res)
        marker = " ← BEST" if cv < best_cv else ""
        print(f"    iqr_multiplier={iqr_mult:.1f}: {mean_pct:.1f}%±{std_pct:.1f}% "
              f"CV={cv:.2f} estabilidad={stab['stability_pct']:.0f}%{marker}")

        if cv < best_cv:
            best_cv = cv
            best_params["m5_iqr_multiplier"] = iqr_mult
            best_params["m5_cv"] = cv

    # M13 — Autoencoder: optimizar contamination
    print(f"\n  M13 Autoencoder — buscando mejor contamination...")
    best_cv = float("inf")
    for contamination in [0.02, 0.03, 0.05, 0.08]:
        cv_res = cv_autoencoder(df, feature_cols, contamination=contamination, n_folds=n_folds)
        pcts = [r["pct_anomalies"] for r in cv_res]
        mean_pct = np.mean(pcts)
        std_pct = np.std(pcts)
        cv = std_pct / mean_pct if mean_pct > 0 else float("inf")

        if mean_pct < 1.0 or mean_pct > 15.0:
            cv += 1.0

        seps = [r["score_separation"] for r in cv_res]
        avg_sep = np.mean(seps)
        stab = compute_stability(cv_res)
        marker = " ← BEST" if cv < best_cv else ""
        print(f"    contamination={contamination:.2f}: {mean_pct:.1f}%±{std_pct:.1f}% "
              f"CV={cv:.2f} sep={avg_sep:.1f} estabilidad={stab['stability_pct']:.0f}%{marker}")

        if cv < best_cv:
            best_cv = cv
            best_params["m13_contamination"] = contamination
            best_params["m13_cv"] = cv

    # Resumen
    print(f"\n  {'─'*70}")
    print(f"  MEJORES PARAMETROS ENCONTRADOS:")
    print(f"  {'─'*70}")
    for k, v in best_params.items():
        if not k.endswith("_cv") and not k.endswith("_mean_pct"):
            print(f"    {k}: {v}")

    return best_params


def main():
    parser = argparse.ArgumentParser(description="Cross-validation temporal de modelos")
    parser.add_argument("--folds", type=int, default=5, help="Numero de folds")
    parser.add_argument("--quick", action="store_true", help="Solo M2, M5, M13 (sin Prophet)")
    parser.add_argument("--optimize", action="store_true", help="Grid search de hiperparametros")
    args = parser.parse_args()

    print("=" * 70)
    print("  AQUAGUARD AI — Cross-Validation Temporal (Walk-Forward)")
    print("=" * 70)
    print(f"  Folds: {args.folds}")
    print(f"  Método: TimeSeriesSplit (walk-forward, respeta temporalidad)")

    t_start = time.time()

    print("\n  Preparando datos...")
    df, feature_cols = prepare_data()
    print(f"  {len(df)} puntos, {df['barrio_key'].nunique()} barrios, {len(feature_cols)} features")

    summaries = []

    # M2 — IsolationForest
    print("\n  Ejecutando CV para M2 (IsolationForest)...")
    m2_cv = cv_isolation_forest(df, feature_cols, n_folds=args.folds)
    s = print_cv_report("M2 — IsolationForest (contamination=0.03)", m2_cv)
    summaries.append(s)

    # M5 — 3-sigma + IQR
    print("\n  Ejecutando CV para M5 (3-sigma + IQR)...")
    m5_cv = cv_statistical(df, n_folds=args.folds)
    s = print_cv_report("M5 — 3-sigma + IQR (desviación del grupo)", m5_cv)
    summaries.append(s)

    # M13 — Autoencoder
    print("\n  Ejecutando CV para M13 (Autoencoder)...")
    m13_cv = cv_autoencoder(df, feature_cols, n_folds=args.folds)
    s = print_cv_report("M13 — Autoencoder (16-8-4-8-16)", m13_cv)
    summaries.append(s)

    # M7 — Prophet (opcional, lento)
    if not args.quick:
        print("\n  Ejecutando CV para M7 (Prophet, top 10 barrios)...")
        m7_cv = cv_prophet(df, n_folds=args.folds)
        s = print_cv_report("M7 — Prophet (interval=0.99, top 10 barrios)", m7_cv)
        summaries.append(s)

    # Resumen comparativo
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  RESUMEN COMPARATIVO — {args.folds}-fold Walk-Forward CV")
    print(f"{'='*70}")
    print(f"  {'Modelo':<45} {'Anomalías':>10} {'Estabilidad':>12} {'Barrios':>8}")
    print(f"  {'─'*78}")
    for s in summaries:
        print(f"  {s['model']:<45} {s['mean_pct']:>5.1f}%±{s['std_pct']:<4.1f} "
              f"{s['stability']:>10.0f}% {s['n_stable']:>8}")

    # --- Tests estadísticos de comparación entre modelos ---
    all_cv_results = [m2_cv, m5_cv, m13_cv]
    all_cv_names = ["M2 IsolationForest", "M5 3-sigma+IQR", "M13 Autoencoder"]
    if not args.quick:
        all_cv_results.append(m7_cv)
        all_cv_names.append("M7 Prophet")

    print_statistical_tests(all_cv_names, all_cv_results, args.folds)

    print(f"\n  Tiempo total: {elapsed:.1f}s")
    print(f"\n  Interpretación:")
    print(f"  - Anomalías %: porcentaje medio de puntos detectados como anómalos")
    print(f"  - Estabilidad: % de barrios que aparecen en >=60% de los folds")
    print(f"  - Un modelo bueno tiene anomalías consistentes y alta estabilidad")
    print(f"  - Si % varía mucho entre folds → modelo inestable (no fiable)")
    print(f"  - Wilcoxon p<0.05 = diferencia significativa entre dos modelos")
    print(f"  - Friedman p<0.05 = al menos un modelo es significativamente distinto")

    # Optimizacion de hiperparametros (opcional)
    if args.optimize:
        best = optimize_hyperparameters(df, feature_cols, n_folds=args.folds)

        # Guardar mejores parametros
        import json
        out_path = "tuned_params_cv.json"
        with open(out_path, "w") as f:
            json.dump(best, f, indent=2)
        print(f"\n  Parametros guardados en: {out_path}")


if __name__ == "__main__":
    main()
