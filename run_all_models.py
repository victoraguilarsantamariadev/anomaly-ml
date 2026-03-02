"""
Pipeline unificado: ejecuta 4 modelos sobre el dataset del hackathon y muestra resultados.

Modelos:
  M2  — IsolationForest cross-sectional (compara barrios entre si)
  M5  — 3-sigma + IQR (outliers estadisticos puros)
  M6  — Amazon Chronos (transformer pre-entrenado, forecasting)
  M7  — Facebook Prophet (descomposicion estacional)

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
)
from statistical_baseline import score_3sigma, score_iqr

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

    # Filtrar por tipo de uso
    df_uso = df_features[df_features["uso"].str.strip() == uso_filter].copy()
    df_uso = df_uso.sort_values(["barrio_key", "fecha"]).reset_index(drop=True)

    # Usar features relativos (sin valores absolutos que llevan tendencia global)
    use_extended = external_df is not None
    feature_cols = RELATIVE_EXTENDED_FEATURE_COLUMNS if use_extended else RELATIVE_FEATURE_COLUMNS
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
           iqr_multiplier: float = 2.0):
    """
    M5 — 3-sigma + IQR sobre deviation_from_group_trend.
    Para cada barrio, calcula estadisticas de desviacion sobre los primeros 24 meses
    y detecta outliers en los ultimos 12.

    Detecta: barrios cuya desviacion del grupo es extrema.
    """
    print(f"\n  [M5] 3-sigma + IQR sobre desviacion del grupo "
          f"(iqr_multiplier={iqr_multiplier})...")

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

        for i, (_, row) in enumerate(test.iterrows()):
            results.append({
                "barrio_key": barrio_key,
                "fecha": row["fecha"],
                "is_anomaly_3sigma": bool(sigma_flags[i]),
                "is_anomaly_iqr": bool(iqr_flags[i]),
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


def collect_results(m2_results: pd.DataFrame, m5_results: pd.DataFrame,
                    m6_results: pd.DataFrame, m7_results: pd.DataFrame) -> pd.DataFrame:
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

    # Rellenar NaN en columnas booleanas
    for col in ["is_anomaly_3sigma", "is_anomaly_iqr"]:
        result[col] = result[col].fillna(False).astype(bool)

    # Contar cuantos modelos detectan anomalia (ignorar NaN = modelo no ejecutado)
    model_cols = ["is_anomaly_m2", "is_anomaly_3sigma", "is_anomaly_iqr",
                  "is_anomaly_chronos", "is_anomaly_prophet"]
    available = [c for c in model_cols if c in result.columns]

    def _count_detecting(row):
        detecting = []
        model_names = {
            "is_anomaly_m2": "M2",
            "is_anomaly_3sigma": "M5_3sigma",
            "is_anomaly_iqr": "M5_IQR",
            "is_anomaly_chronos": "M6_Chronos",
            "is_anomaly_prophet": "M7_Prophet",
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
    parser.add_argument("--output", type=str, default=None,
                        help="Guardar resultados en CSV")
    # Tuning parameters
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="M2 contamination rate (default: 0.05)")
    parser.add_argument("--prophet-interval", type=float, default=0.97,
                        help="Prophet interval width (default: 0.97)")
    parser.add_argument("--prophet-changepoint", type=float, default=0.15,
                        help="Prophet changepoint_prior_scale (default: 0.15)")
    parser.add_argument("--chronos-sigma", type=float, default=2.5,
                        help="Chronos threshold sigma (default: 2.5)")
    parser.add_argument("--iqr-multiplier", type=float, default=2.0,
                        help="IQR multiplier for fences (default: 2.0)")
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
          ("" if args.skip_prophet else ", M7"))
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
                        iqr_multiplier=args.iqr_multiplier)

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

    # Combinar
    print(f"\n  Combinando resultados...")
    results = collect_results(m2_results, m5_results, m6_results, m7_results)

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
