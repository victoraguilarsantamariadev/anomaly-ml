"""
M15 — Optimal Transport (Wasserstein Distance): compara DISTRIBUCIONES, no medias.

La distancia de Wasserstein (Earth Mover's Distance) mide cuanto "trabajo" cuesta
transformar una distribucion en otra. Es la metrica gold-standard en papers de
Nature/Science para detectar distribution shift.

Ventaja sobre PSI/KL-divergence: funciona con distribuciones sin soporte comun,
es una metrica real (satisface desigualdad triangular), y tiene interpretacion fisica.

Papers: Villani (2008), Peyre & Cuturi (2019) "Computational Optimal Transport"
"""

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Wasserstein distance entre periodos (train vs test) por barrio
# ---------------------------------------------------------------------------

def _train_test_wasserstein(
    df_barrio: pd.DataFrame,
    consumo_col: str,
    train_frac: float = 0.6,
) -> float:
    """
    Calcula la Wasserstein distance entre la distribucion de consumo
    del periodo de referencia (train) y el periodo de test.

    Cuanto mayor la distancia, mas ha cambiado la distribucion.
    """
    n = len(df_barrio)
    n_train = max(3, int(n * train_frac))

    train_vals = df_barrio[consumo_col].values[:n_train].astype(float)
    test_vals = df_barrio[consumo_col].values[n_train:].astype(float)

    if len(test_vals) < 2 or len(train_vals) < 2:
        return 0.0

    return wasserstein_distance(train_vals, test_vals)


# ---------------------------------------------------------------------------
# 2. Distribution shift detection — flag por barrio
# ---------------------------------------------------------------------------

def _compute_distribution_shift(
    df_monthly: pd.DataFrame,
    consumo_col: str,
    train_frac: float = 0.6,
) -> pd.DataFrame:
    """
    Para cada barrio, calcula la Wasserstein distance entre train y test.
    Devuelve DataFrame con barrio_key, wd_train_test, shift_zscore.
    Los barrios con shift_zscore alto tienen un distribution shift significativo.
    """
    barrios = df_monthly["barrio_key"].unique()
    records = []

    for barrio in barrios:
        df_b = df_monthly[df_monthly["barrio_key"] == barrio].sort_values("fecha")
        if len(df_b) < 6:
            continue

        wd = _train_test_wasserstein(df_b, consumo_col, train_frac)
        records.append({"barrio_key": barrio, "wd_train_test": wd})

    shift_df = pd.DataFrame(records)
    if len(shift_df) == 0:
        return shift_df

    # Z-score de la distancia entre barrios: los que mas han cambiado
    mean_wd = shift_df["wd_train_test"].mean()
    std_wd = shift_df["wd_train_test"].std() + 1e-10
    shift_df["shift_zscore"] = (shift_df["wd_train_test"] - mean_wd) / std_wd

    return shift_df


# ---------------------------------------------------------------------------
# 3. Cross-barrio Wasserstein — cada barrio vs la distribucion tipica
# ---------------------------------------------------------------------------

def _cross_barrio_wasserstein(
    df_monthly: pd.DataFrame,
    consumo_col: str,
) -> pd.DataFrame:
    """
    Compara la distribucion de cada barrio contra la distribucion "tipica"
    (mediana de barrios). Un barrio cuya distribucion esta muy lejos de la
    tipica es un outlier estructural.

    Usa la serie completa de cada barrio.
    """
    barrios = df_monthly["barrio_key"].unique()

    # Construir distribucion de referencia: barrio mediano
    # Tomamos todos los valores de todos los barrios y calculamos la mediana
    # por posicion temporal (mes relativo)
    all_series = {}
    for barrio in barrios:
        df_b = df_monthly[df_monthly["barrio_key"] == barrio].sort_values("fecha")
        vals = df_b[consumo_col].values.astype(float)
        if len(vals) >= 6:
            # Normalizar por la media del barrio para comparar formas
            mean_val = np.mean(vals) + 1e-10
            all_series[barrio] = vals / mean_val

    if len(all_series) < 3:
        return pd.DataFrame(columns=["barrio_key", "wd_cross_barrio"])

    # La distribucion de referencia: mediana de las distribuciones normalizadas
    max_len = max(len(s) for s in all_series.values())
    # Usar la distribucion agregada de todos los barrios como referencia
    ref_distribution = np.concatenate(list(all_series.values()))

    records = []
    for barrio, norm_vals in all_series.items():
        wd = wasserstein_distance(norm_vals, ref_distribution)
        records.append({"barrio_key": barrio, "wd_cross_barrio": wd})

    cross_df = pd.DataFrame(records)

    # Z-score cross-barrio
    mean_wd = cross_df["wd_cross_barrio"].mean()
    std_wd = cross_df["wd_cross_barrio"].std() + 1e-10
    cross_df["cross_barrio_zscore"] = (cross_df["wd_cross_barrio"] - mean_wd) / std_wd

    return cross_df


# ---------------------------------------------------------------------------
# 4. Temporal Wasserstein — rolling window, mes a mes
# ---------------------------------------------------------------------------

def _temporal_wasserstein(
    df_monthly: pd.DataFrame,
    consumo_col: str,
    window: int = 6,
) -> pd.DataFrame:
    """
    Para cada barrio y cada punto temporal, calcula la Wasserstein distance
    entre la ventana [t-window, t] y [t-2*window, t-window].

    Picos en esta distancia = cambio abrupto de distribucion.
    """
    barrios = df_monthly["barrio_key"].unique()
    records = []

    for barrio in barrios:
        df_b = df_monthly[df_monthly["barrio_key"] == barrio].sort_values("fecha")
        if len(df_b) < 2 * window + 1:
            # No hay suficientes datos para dos ventanas
            # Asignar 0 a todos los puntos
            for _, row in df_b.iterrows():
                records.append({
                    "barrio_key": barrio,
                    "fecha": row["fecha"],
                    "wd_temporal": 0.0,
                })
            continue

        vals = df_b[consumo_col].values.astype(float)
        fechas = df_b["fecha"].values

        for i in range(len(vals)):
            if i < 2 * window:
                # No hay suficiente historia
                records.append({
                    "barrio_key": barrio,
                    "fecha": fechas[i],
                    "wd_temporal": 0.0,
                })
                continue

            past_window = vals[i - 2 * window: i - window]
            curr_window = vals[i - window: i + 1]

            wd = wasserstein_distance(past_window, curr_window)
            records.append({
                "barrio_key": barrio,
                "fecha": fechas[i],
                "wd_temporal": wd,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Multivariate Wasserstein (feature-by-feature, averaged)
# ---------------------------------------------------------------------------

def _multivariate_wasserstein(
    df_monthly: pd.DataFrame,
    feature_cols: list[str],
    window: int = 6,
) -> pd.DataFrame:
    """
    Wasserstein multidimensional aproximado: calcula la distancia de Wasserstein
    por cada feature y promedia (sliced Wasserstein approximation).

    scipy no tiene OT multidimensional nativo, asi que usamos la
    Sliced Wasserstein distance: promedio de W1 sobre proyecciones 1D.
    Con features individuales, esto es equivalente a promediar W1 por feature.

    Features usados: consumo_litros, consumption_per_contract, yoy_ratio
    (los que existan en el DataFrame).
    """
    available_cols = [c for c in feature_cols if c in df_monthly.columns]
    if len(available_cols) < 2:
        return pd.DataFrame(columns=["barrio_key", "fecha", "wd_multivariate"])

    barrios = df_monthly["barrio_key"].unique()
    records = []

    for barrio in barrios:
        df_b = df_monthly[df_monthly["barrio_key"] == barrio].sort_values("fecha")
        if len(df_b) < 2 * window + 1:
            for _, row in df_b.iterrows():
                records.append({
                    "barrio_key": barrio,
                    "fecha": row["fecha"],
                    "wd_multivariate": 0.0,
                })
            continue

        fechas = df_b["fecha"].values

        for i in range(len(df_b)):
            if i < 2 * window:
                records.append({
                    "barrio_key": barrio,
                    "fecha": fechas[i],
                    "wd_multivariate": 0.0,
                })
                continue

            wd_per_feature = []
            for col in available_cols:
                vals = df_b[col].values.astype(float)
                past_window = vals[i - 2 * window: i - window]
                curr_window = vals[i - window: i + 1]

                # Filtrar NaN
                past_clean = past_window[~np.isnan(past_window)]
                curr_clean = curr_window[~np.isnan(curr_window)]

                if len(past_clean) >= 2 and len(curr_clean) >= 2:
                    # Normalizar por la escala de cada feature
                    scale = np.std(past_clean) + 1e-10
                    wd = wasserstein_distance(past_clean / scale, curr_clean / scale)
                    wd_per_feature.append(wd)

            wd_avg = float(np.mean(wd_per_feature)) if wd_per_feature else 0.0
            records.append({
                "barrio_key": barrio,
                "fecha": fechas[i],
                "wd_multivariate": wd_avg,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Composite score y deteccion de anomalias
# ---------------------------------------------------------------------------

def _compute_wasserstein_score(row: pd.Series) -> float:
    """
    Combina las diferentes metricas de Wasserstein en un score unico [0, 1].

    Pesos:
      - wd_temporal_zscore:      40%  (cambio reciente, mas actionable)
      - shift_zscore:            25%  (cambio estructural train vs test)
      - cross_barrio_zscore:     20%  (outlier respecto a otros barrios)
      - wd_multivariate_zscore:  15%  (cambio multivariante)
    """
    temporal_z = abs(row.get("wd_temporal_zscore", 0.0))
    shift_z = abs(row.get("shift_zscore", 0.0))
    cross_z = abs(row.get("cross_barrio_zscore", 0.0))
    multi_z = abs(row.get("wd_multivariate_zscore", 0.0))

    raw_score = (
        0.40 * temporal_z +
        0.25 * shift_z +
        0.20 * cross_z +
        0.15 * multi_z
    )

    # Sigmoid para mapear a [0, 1]
    score = 1.0 / (1.0 + np.exp(-0.8 * (raw_score - 2.0)))
    return float(score)


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def run_wasserstein_detection(
    df_monthly: pd.DataFrame,
    consumo_col: str = "consumo_litros",
    train_frac: float = 0.6,
    window: int = 6,
    anomaly_threshold: float = 0.55,
    multivariate_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Ejecuta la deteccion de anomalias basada en Wasserstein distance.

    Combina 4 perspectivas:
      1. Distribution shift (train vs test) por barrio
      2. Cross-barrio Wasserstein (vs distribucion tipica)
      3. Temporal Wasserstein (rolling, mes a mes)
      4. Multivariate Wasserstein (multiples features)

    Args:
        df_monthly:      DataFrame con barrio_key, fecha, consumo_col
        consumo_col:     columna de consumo (default: consumo_litros)
        train_frac:      fraccion de datos como periodo de referencia
        window:          tamanio de ventana para Wasserstein temporal
        anomaly_threshold: umbral de wasserstein_score para flag anomalia
        multivariate_cols: columnas para Wasserstein multivariante
                           (default: consumo_litros, consumption_per_contract, yoy_ratio)

    Returns:
        DataFrame con columnas:
          barrio_key, fecha, wasserstein_score, is_anomaly_wasserstein,
          distribution_shift_pct, wd_temporal, wd_cross_barrio,
          wd_multivariate, wd_train_test
    """
    print(f"\n  [M15] Optimal Transport (Wasserstein Distance)...")

    if multivariate_cols is None:
        multivariate_cols = [consumo_col, "consumption_per_contract", "yoy_ratio"]

    # --- 1. Distribution shift train vs test ---
    shift_df = _compute_distribution_shift(df_monthly, consumo_col, train_frac)
    n_shift = (shift_df["shift_zscore"].abs() > 2.0).sum() if len(shift_df) > 0 else 0
    print(f"    Distribution shift: {n_shift} barrios con shift significativo (|z|>2)")

    # --- 2. Cross-barrio ---
    cross_df = _cross_barrio_wasserstein(df_monthly, consumo_col)
    n_cross = (cross_df["cross_barrio_zscore"].abs() > 2.0).sum() if len(cross_df) > 0 else 0
    print(f"    Cross-barrio outliers: {n_cross} barrios con distribucion atipica")

    # --- 3. Temporal Wasserstein (mes a mes) ---
    temporal_df = _temporal_wasserstein(df_monthly, consumo_col, window)
    print(f"    Temporal Wasserstein: {len(temporal_df)} puntos calculados")

    # --- 4. Multivariate ---
    multi_df = _multivariate_wasserstein(df_monthly, multivariate_cols, window)
    n_multi_cols = len([c for c in multivariate_cols if c in df_monthly.columns])
    print(f"    Multivariate Wasserstein: {n_multi_cols} features usados")

    # --- Merge todo sobre la base temporal ---
    base = df_monthly[["barrio_key", "fecha"]].copy()

    # Merge shift (barrio-level → broadcast a todos los meses)
    if len(shift_df) > 0:
        base = base.merge(
            shift_df[["barrio_key", "wd_train_test", "shift_zscore"]],
            on="barrio_key", how="left",
        )
    else:
        base["wd_train_test"] = 0.0
        base["shift_zscore"] = 0.0

    # Merge cross-barrio (barrio-level → broadcast)
    if len(cross_df) > 0:
        base = base.merge(
            cross_df[["barrio_key", "wd_cross_barrio", "cross_barrio_zscore"]],
            on="barrio_key", how="left",
        )
    else:
        base["wd_cross_barrio"] = 0.0
        base["cross_barrio_zscore"] = 0.0

    # Merge temporal (barrio_key + fecha level)
    if len(temporal_df) > 0:
        base = base.merge(
            temporal_df[["barrio_key", "fecha", "wd_temporal"]],
            on=["barrio_key", "fecha"], how="left",
        )
    else:
        base["wd_temporal"] = 0.0

    # Merge multivariate
    if len(multi_df) > 0:
        base = base.merge(
            multi_df[["barrio_key", "fecha", "wd_multivariate"]],
            on=["barrio_key", "fecha"], how="left",
        )
    else:
        base["wd_multivariate"] = 0.0

    # Rellenar NaN
    for col in ["wd_train_test", "shift_zscore", "wd_cross_barrio",
                 "cross_barrio_zscore", "wd_temporal", "wd_multivariate"]:
        base[col] = base[col].fillna(0.0)

    # --- Z-scores de las metricas temporales (por barrio) ---
    # Temporal z-score: cuanto se desvía el wd_temporal de cada punto
    # respecto a la media de su barrio
    temporal_stats = base.groupby("barrio_key")["wd_temporal"].agg(["mean", "std"]).reset_index()
    temporal_stats.columns = ["barrio_key", "wd_temporal_mean", "wd_temporal_std"]
    base = base.merge(temporal_stats, on="barrio_key", how="left")
    base["wd_temporal_zscore"] = np.where(
        base["wd_temporal_std"] > 1e-10,
        (base["wd_temporal"] - base["wd_temporal_mean"]) / base["wd_temporal_std"],
        0.0,
    )

    # Multivariate z-score
    multi_stats = base.groupby("barrio_key")["wd_multivariate"].agg(["mean", "std"]).reset_index()
    multi_stats.columns = ["barrio_key", "wd_multi_mean", "wd_multi_std"]
    base = base.merge(multi_stats, on="barrio_key", how="left")
    base["wd_multivariate_zscore"] = np.where(
        base["wd_multi_std"] > 1e-10,
        (base["wd_multivariate"] - base["wd_multi_mean"]) / base["wd_multi_std"],
        0.0,
    )

    # --- Composite score ---
    base["wasserstein_score"] = base.apply(_compute_wasserstein_score, axis=1)

    # --- Anomaly flag ---
    base["is_anomaly_wasserstein"] = base["wasserstein_score"] > anomaly_threshold

    # --- Distribution shift percentage ---
    # Expresar el shift como % de cambio respecto a la distribucion de referencia
    # Usamos wd_train_test normalizado por la media del consumo del barrio
    barrio_means = df_monthly.groupby("barrio_key")[consumo_col].mean().reset_index()
    barrio_means.columns = ["barrio_key", "_barrio_mean_consumo"]
    base = base.merge(barrio_means, on="barrio_key", how="left")
    base["distribution_shift_pct"] = np.where(
        base["_barrio_mean_consumo"].abs() > 1e-10,
        (base["wd_train_test"] / base["_barrio_mean_consumo"]) * 100.0,
        0.0,
    )

    # --- Limpiar columnas auxiliares y seleccionar output ---
    output_cols = [
        "barrio_key", "fecha",
        "wasserstein_score", "is_anomaly_wasserstein",
        "distribution_shift_pct",
        "wd_temporal", "wd_cross_barrio", "wd_multivariate",
        "wd_train_test",
    ]
    result = base[output_cols].copy()

    n_anomalies = result["is_anomaly_wasserstein"].sum()
    n_barrios_anom = result[result["is_anomaly_wasserstein"]]["barrio_key"].nunique()
    print(f"    Resultado: {n_anomalies} anomalias en {n_barrios_anom} barrios "
          f"(threshold={anomaly_threshold})")

    return result


def wasserstein_summary(results: pd.DataFrame):
    """
    Imprime un resumen de la deteccion por Wasserstein distance.

    Args:
        results: DataFrame devuelto por run_wasserstein_detection()
    """
    if "wasserstein_score" not in results.columns:
        print("  [M15] No hay resultados de Wasserstein disponibles.")
        return

    print(f"\n{'='*80}")
    print(f"  M15 — OPTIMAL TRANSPORT (WASSERSTEIN DISTANCE)")
    print(f"{'='*80}")

    total = len(results)
    anomalies = results[results["is_anomaly_wasserstein"]]
    n_anomalies = len(anomalies)
    n_barrios = results["barrio_key"].nunique()
    n_barrios_anom = anomalies["barrio_key"].nunique() if n_anomalies > 0 else 0

    print(f"\n  Puntos analizados:  {total}")
    print(f"  Barrios analizados: {n_barrios}")
    print(f"  Anomalias:          {n_anomalies} ({n_anomalies/total*100:.1f}%)")
    print(f"  Barrios con anom:   {n_barrios_anom}")

    # Estadisticas del score
    print(f"\n  Wasserstein score:")
    print(f"    Media:   {results['wasserstein_score'].mean():.3f}")
    print(f"    Mediana: {results['wasserstein_score'].median():.3f}")
    print(f"    P95:     {results['wasserstein_score'].quantile(0.95):.3f}")
    print(f"    Max:     {results['wasserstein_score'].max():.3f}")

    # Distribution shift
    if "distribution_shift_pct" in results.columns:
        shift = results.groupby("barrio_key")["distribution_shift_pct"].first()
        print(f"\n  Distribution shift (train vs test):")
        print(f"    Media:   {shift.mean():.1f}%")
        print(f"    Mediana: {shift.median():.1f}%")
        print(f"    Max:     {shift.max():.1f}%")

    # Top barrios mas anomalos
    if n_anomalies > 0:
        barrio_scores = (
            anomalies.groupby("barrio_key")
            .agg(
                n_meses_anom=("is_anomaly_wasserstein", "sum"),
                max_score=("wasserstein_score", "max"),
                avg_shift=("distribution_shift_pct", "mean"),
            )
            .sort_values("max_score", ascending=False)
        )

        print(f"\n  TOP BARRIOS CON MAYOR ANOMALIA WASSERSTEIN:")
        print(f"  {'─'*75}")
        print(f"  {'Barrio':<35} {'Meses':>6} {'Max score':>10} {'Shift %':>10}")
        print(f"  {'─'*75}")

        for barrio_key, row in barrio_scores.head(15).iterrows():
            barrio_name = barrio_key.split("__")[0][:33]
            print(f"  {barrio_name:<35} {int(row['n_meses_anom']):>6} "
                  f"{row['max_score']:>10.3f} {row['avg_shift']:>9.1f}%")

    # Temporal: meses con mas anomalias
    if n_anomalies > 0 and "fecha" in anomalies.columns:
        anomalies_copy = anomalies.copy()
        anomalies_copy["year_month"] = pd.to_datetime(anomalies_copy["fecha"]).dt.to_period("M")
        by_month = anomalies_copy.groupby("year_month").size().sort_values(ascending=False)
        if len(by_month) > 0:
            print(f"\n  Meses con mas anomalias Wasserstein:")
            for period, count in by_month.head(5).items():
                print(f"    {period}: {count} anomalias")

    # Componentes del score
    print(f"\n  Componentes (medias sobre anomalias):")
    if n_anomalies > 0:
        print(f"    wd_temporal:      {anomalies['wd_temporal'].mean():.4f}")
        print(f"    wd_cross_barrio:  {anomalies['wd_cross_barrio'].mean():.4f}")
        print(f"    wd_multivariate:  {anomalies['wd_multivariate'].mean():.4f}")
        print(f"    wd_train_test:    {anomalies['wd_train_test'].mean():.4f}")
    else:
        print(f"    (sin anomalias detectadas)")


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from train_local import load_hackathon_amaem
    from monthly_features import compute_monthly_features

    DATA_FILE = "data/datos-hackathon-amaem.xlsx-set-de-datos-.csv"
    if not Path(DATA_FILE).exists():
        print(f"ERROR: No se encuentra {DATA_FILE}")
        exit(1)

    print("Cargando datos del hackathon...")
    df_all = load_hackathon_amaem(DATA_FILE)

    print("Calculando features mensuales...")
    df_monthly = compute_monthly_features(df_all, uso="DOMESTICO")

    print("Ejecutando deteccion Wasserstein...")
    results = run_wasserstein_detection(df_monthly, consumo_col="consumo_litros")

    wasserstein_summary(results)

    # Guardar CSV
    output_path = "results_wasserstein.csv"
    results.to_csv(output_path, index=False)
    print(f"\nResultados guardados en {output_path}")
