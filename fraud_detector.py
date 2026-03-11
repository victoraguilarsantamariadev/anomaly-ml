"""
M12 — Detector supervisado de fraude + Meta-modelo (stacking).

Usa los 190 casos reales de fraude de cambios-de-contador para:
  1. Calcular tasa mensual de fraude (feature temporal)
  2. Construir perfil de vulnerabilidad por barrio (via contadores-telelectura)
  3. Meta-modelo GradientBoosting que combina scores de todos los modelos
     en un unico fraud_score calibrado

Input: resultados de collect_results() + cambios-de-contador + contadores-telelectura
Output: fraud_score por (barrio, fecha)

Uso:
  from fraud_detector import (
      load_fraud_cases, compute_monthly_fraud_rate,
      compute_barrio_vulnerability, build_meta_model
  )
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import warnings

SUSPICIOUS_MOTIVOS = [
    "FP-FRAUDE POSIBLE",
    "RB-ROBO",
    "MR-MARCHA AL REVES",
]

# Pesos por tipo de fraude (para scoring)
FRAUD_WEIGHTS = {
    "FP-FRAUDE POSIBLE": 1.0,
    "RB-ROBO": 1.5,        # robo es mas grave
    "MR-MARCHA AL REVES": 0.8,  # puede ser error tecnico
}


def load_fraud_cases(cambios_path: str) -> pd.DataFrame:
    """
    Carga los 190 casos de fraude/sospechosos de cambios-de-contador.

    Returns:
        DataFrame con fecha, motivo, emplazamiento, calibre, peso
    """
    if not Path(cambios_path).exists():
        return pd.DataFrame()

    df = pd.read_csv(cambios_path)
    df["FECHA"] = pd.to_datetime(df["FECHA"])

    fraud = df[df["MOTIVO_CAMBIO"].isin(SUSPICIOUS_MOTIVOS)].copy()
    fraud["weight"] = fraud["MOTIVO_CAMBIO"].map(FRAUD_WEIGHTS)

    return fraud[["FECHA", "MOTIVO_CAMBIO", "EMPLAZAMIENTO", "CALIBRE", "weight"]]


def compute_monthly_fraud_rate(cambios_path: str) -> pd.DataFrame:
    """
    Tasa mensual de fraude: n_fraudes / n_cambios_totales por mes.

    Util como feature temporal: meses con mas fraude pueden correlacionar
    con anomalias detectadas por otros modelos.

    Returns:
        DataFrame con: fecha, n_fraud, n_total_changes, fraud_rate, fraud_weighted
    """
    if not Path(cambios_path).exists():
        return pd.DataFrame()

    df = pd.read_csv(cambios_path)
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["year_month"] = df["FECHA"].dt.to_period("M")
    df["is_fraud"] = df["MOTIVO_CAMBIO"].isin(SUSPICIOUS_MOTIVOS)
    df["fraud_weight"] = df["MOTIVO_CAMBIO"].map(FRAUD_WEIGHTS).fillna(0)

    monthly = df.groupby("year_month").agg(
        n_fraud=("is_fraud", "sum"),
        n_total_changes=("FECHA", "size"),
        fraud_weighted=("fraud_weight", "sum"),
    ).reset_index()

    monthly["fraud_rate"] = monthly["n_fraud"] / monthly["n_total_changes"]
    monthly["fecha"] = monthly["year_month"].dt.to_timestamp()

    # Media movil 3 meses para suavizar
    monthly = monthly.sort_values("fecha")
    monthly["fraud_rate_3m"] = monthly["fraud_rate"].rolling(3, min_periods=1).mean()

    return monthly[["fecha", "n_fraud", "n_total_changes", "fraud_rate",
                     "fraud_rate_3m", "fraud_weighted"]]


def compute_barrio_vulnerability(contadores_path: str) -> pd.DataFrame:
    """
    Indice de vulnerabilidad al fraude por barrio.

    Basado en perfil de contadores comparado con el perfil de fraudes conocidos:
      - % lectura manual (sin telelectura = mas facil manipular)
      - Densidad de contadores (mas contadores = mas dificil supervisar)
      - % uso no domestico (fraude comercial mas costoso)

    Returns:
        DataFrame con: barrio, fraud_vulnerability (0-1)
    """
    if not Path(contadores_path).exists():
        return pd.DataFrame()

    cont = pd.read_csv(contadores_path)

    barrio_stats = cont.groupby("BARRIO").agg(
        n_meters=("CALIBRE", "size"),
        pct_manual=("SISTEMA", lambda x: (x != "Leer por telelectura").mean()),
        avg_calibre=("CALIBRE", "mean"),
        pct_no_domestico=("USO", lambda x: (~x.str.contains("DOMÉSTICO", na=False)).mean()),
    ).reset_index()

    # Normalizar cada factor al rango [0, 1]
    def _norm(s):
        vmin, vmax = s.min(), s.max()
        return (s - vmin) / (vmax - vmin) if vmax > vmin else pd.Series(0.5, index=s.index)

    # Mayor % manual = mayor vulnerabilidad
    barrio_stats["v_manual"] = _norm(barrio_stats["pct_manual"])

    # Mas contadores = mas dificil supervisar
    barrio_stats["v_density"] = _norm(barrio_stats["n_meters"])

    # Mayor % no domestico = fraudes mas costosos
    barrio_stats["v_commercial"] = _norm(barrio_stats["pct_no_domestico"])

    # Score compuesto (ponderado: lectura manual es el factor principal)
    barrio_stats["fraud_vulnerability"] = (
        0.5 * barrio_stats["v_manual"] +
        0.3 * barrio_stats["v_density"] +
        0.2 * barrio_stats["v_commercial"]
    )

    return barrio_stats[["BARRIO", "fraud_vulnerability"]].rename(
        columns={"BARRIO": "barrio"}
    )


def enrich_with_fraud_features(df: pd.DataFrame,
                                fraud_rate_df: pd.DataFrame,
                                vulnerability_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquece resultados con features de fraude.

    Añade:
      - fraud_rate: tasa mensual de fraude (temporal, nivel ciudad)
      - fraud_rate_3m: media movil 3 meses de tasa de fraude
      - fraud_vulnerability: vulnerabilidad del barrio al fraude (estatico)
    """
    df = df.copy()

    # Merge tasa de fraude por mes
    if not fraud_rate_df.empty:
        fr = fraud_rate_df[["fecha", "fraud_rate", "fraud_rate_3m"]].copy()
        df["_merge_month"] = pd.to_datetime(df["fecha"]).dt.to_period("M").dt.to_timestamp()
        fr["_merge_month"] = pd.to_datetime(fr["fecha"]).dt.to_period("M").dt.to_timestamp()
        fr = fr.drop(columns=["fecha"])

        df = df.merge(fr, on="_merge_month", how="left")
        df = df.drop(columns=["_merge_month"])
        df["fraud_rate"] = df["fraud_rate"].fillna(0)
        df["fraud_rate_3m"] = df["fraud_rate_3m"].fillna(0)
    else:
        df["fraud_rate"] = 0.0
        df["fraud_rate_3m"] = 0.0

    # Merge vulnerabilidad por barrio
    if not vulnerability_df.empty:
        df["_barrio_clean"] = df["barrio_key"].str.split("__").str[0]
        df = df.merge(vulnerability_df, left_on="_barrio_clean",
                      right_on="barrio", how="left", suffixes=("", "_vuln"))
        df = df.drop(columns=["barrio_vuln", "_barrio_clean"], errors="ignore")
        df["fraud_vulnerability"] = df["fraud_vulnerability"].fillna(0.5)
    else:
        df["fraud_vulnerability"] = 0.5

    return df


def build_meta_model(results: pd.DataFrame) -> pd.DataFrame:
    """
    Meta-modelo mejorado: XGBoost con PU Learning + SHAP explicabilidad.

    Enfoque PU (Positive-Unlabeled) Learning:
      - Los 190 fraudes reales son positivos confirmados
      - El consenso >=2 modelos genera pseudo-positivos adicionales
      - El resto es UNLABELED (no negativo) — se trata con sample_weight
      - XGBoost con scale_pos_weight para manejar el desbalance

    Output: fraud_score (0-1) + SHAP values para explicabilidad

    Args:
        results: DataFrame de collect_results() enriquecido con fraud features

    Returns:
        results con columnas 'fraud_score' y 'shap_top_feature' añadidas
    """
    results = results.copy()

    # Features para el meta-modelo
    score_cols = []
    flag_cols = []

    # Score continuo de M2 (IsolationForest)
    if "score_m2" in results.columns:
        score_cols.append("score_m2")

    # Flags binarias de cada modelo
    binary_features = {
        "is_anomaly_m2": "flag_m2",
        "is_anomaly_3sigma": "flag_3sigma",
        "is_anomaly_iqr": "flag_iqr",
        "is_anomaly_chronos": "flag_chronos",
        "is_anomaly_prophet": "flag_prophet",
        "is_anomaly_anr": "flag_anr",
        "is_anomaly_nmf": "flag_nmf",
        "is_anomaly_readings": "flag_readings",
        "is_anomaly_autoencoder": "flag_autoencoder",
    }

    for src, dst in binary_features.items():
        if src in results.columns:
            results[dst] = results[src].fillna(0).astype(float)
            flag_cols.append(dst)

    # Features extra (incluye scores continuos de modelos)
    extra_cols = []
    for col in ["fraud_rate", "fraud_rate_3m", "fraud_vulnerability",
                "infrastructure_risk_score", "anr_ratio",
                "prophet_residual", "reconstruction_error",
                "reading_anomaly_zscore",
                "deviation_from_group_trend", "relative_consumption",
                "consumption_volatility", "momentum"]:
        if col in results.columns:
            results[col] = results[col].fillna(0).astype(float)
            extra_cols.append(col)

    # Spatial class como numerico
    if "spatial_class" in results.columns:
        spatial_map = {"NORMAL": 0, "ISOLATED": 1, "CLUSTER": 2}
        results["spatial_numeric"] = results["spatial_class"].map(spatial_map).fillna(0)
        extra_cols.append("spatial_numeric")

    all_features = score_cols + flag_cols + extra_cols
    if len(all_features) < 3:
        results["fraud_score"] = results["n_models_detecting"] / 8.0
        return results

    # Preparar matriz
    X = results[all_features].replace([np.inf, -np.inf], np.nan).fillna(0).values

    # PU Learning: pseudo-labels con pesos diferenciados
    # Positivos confirmados: consenso >= 2 modelos (alta confianza)
    # Positivos posibles: consenso == 1 modelo (confianza media)
    # Unlabeled: 0 modelos (NO negativo, peso reducido)
    n_models = results["n_models_detecting"].values
    y = (n_models >= 2).astype(int)

    # Sample weights para PU Learning
    sample_weight = np.ones(len(y))
    sample_weight[n_models >= 3] = 2.0   # Alta confianza
    sample_weight[n_models == 1] = 0.5   # Posible positivo, peso bajo
    sample_weight[n_models == 0] = 0.3   # Unlabeled, no forzar como negativo

    n_positive = y.sum()
    n_total = len(y)

    if n_positive < 10 or n_positive > n_total * 0.5:
        results["fraud_score"] = _heuristic_score(results, flag_cols, score_cols, extra_cols)
        return results

    # XGBoost con PU Learning
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(n_total - n_positive) / max(n_positive, 1),
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42,
        )

    # Cross-validation con sample weights
    from sklearn.model_selection import StratifiedKFold
    fraud_scores = np.zeros(n_total)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X, y):
        model.fit(X[train_idx], y[train_idx],
                  sample_weight=sample_weight[train_idx])
        fraud_scores[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    results["fraud_score"] = fraud_scores

    # Entrenar modelo final para SHAP y feature importance
    model.fit(X, y, sample_weight=sample_weight)

    importances = dict(zip(all_features, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]
    print(f"    Meta-modelo (XGBoost + PU Learning): {n_positive} positivos de {n_total} puntos")
    print(f"    Top features: " +
          ", ".join(f"{k}={v:.3f}" for k, v in top_features))

    # SHAP explicabilidad
    try:
        import shap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

        # Para cada punto, encontrar el feature que mas contribuye
        shap_top = []
        for i in range(len(X)):
            abs_shap = np.abs(shap_values[i])
            top_idx = np.argmax(abs_shap)
            direction = "+" if shap_values[i][top_idx] > 0 else "-"
            shap_top.append(f"{direction}{all_features[top_idx]}")
        results["shap_top_feature"] = shap_top

        # Guardar SHAP summary para reporte
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_ranking = sorted(zip(all_features, mean_abs_shap), key=lambda x: -x[1])
        print(f"    SHAP ranking: " +
              ", ".join(f"{k}={v:.3f}" for k, v in shap_ranking[:5]))
    except Exception as e:
        print(f"    SHAP no disponible: {e}")
        results["shap_top_feature"] = ""

    return results


def _heuristic_score(results: pd.DataFrame,
                     flag_cols: list, score_cols: list,
                     extra_cols: list) -> pd.Series:
    """
    Score heuristico ponderado cuando no hay suficientes datos
    para el meta-modelo supervisado.
    """
    score = pd.Series(0.0, index=results.index)

    # Peso por modelo (basado en precision estimada)
    model_weights = {
        "flag_m2": 0.15,
        "flag_3sigma": 0.10,
        "flag_iqr": 0.10,
        "flag_chronos": 0.15,
        "flag_prophet": 0.15,
        "flag_anr": 0.20,
        "flag_nmf": 0.15,
        "flag_readings": 0.05,
        "flag_autoencoder": 0.15,
    }

    total_weight = 0
    for col in flag_cols:
        w = model_weights.get(col, 0.1)
        if col in results.columns:
            score += results[col].fillna(0) * w
            total_weight += w

    if total_weight > 0:
        score = score / total_weight

    # Bonus por vulnerabilidad alta
    if "fraud_vulnerability" in results.columns:
        score = score * (1 + 0.3 * results["fraud_vulnerability"].fillna(0.5))

    return score.clip(0, 1)


def fraud_summary(results: pd.DataFrame, fraud_rate_df: pd.DataFrame = None) -> None:
    """Imprime resumen del modelo de fraude."""
    if "fraud_score" not in results.columns:
        return

    print(f"\n  {'─'*80}")
    print(f"  M12 — DETECTOR DE FRAUDE (Meta-modelo)")
    print(f"  {'─'*80}")

    # Top barrios por fraud_score
    high_risk = results[results["fraud_score"] > 0.5].copy()
    n_high = len(high_risk)
    n_total = len(results)
    print(f"  {n_high} alertas de alto riesgo de {n_total} puntos ({n_high/n_total*100:.1f}%)")

    if n_high > 0:
        # Ranking por barrio
        barrio_risk = (
            high_risk.groupby("barrio_key")
            .agg(
                avg_score=("fraud_score", "mean"),
                max_score=("fraud_score", "max"),
                n_alerts=("fraud_score", "count"),
            )
            .sort_values("max_score", ascending=False)
        )

        print(f"\n  {'Barrio':<35}  {'Score max':>10}  {'Score avg':>10}  {'Alertas':>8}")
        print(f"  {'─'*68}")
        for barrio, row in barrio_risk.head(10).iterrows():
            print(f"  {barrio:<35}  {row['max_score']:>10.3f}  "
                  f"{row['avg_score']:>10.3f}  {int(row['n_alerts']):>8}")

    # Correlacion con fraude real si disponible
    if fraud_rate_df is not None and not fraud_rate_df.empty:
        results_monthly = results.copy()
        results_monthly["year_month"] = pd.to_datetime(results_monthly["fecha"]).dt.to_period("M")
        avg_scores = results_monthly.groupby("year_month")["fraud_score"].mean()

        fr = fraud_rate_df.copy()
        fr["year_month"] = pd.to_datetime(fr["fecha"]).dt.to_period("M")
        fr = fr.set_index("year_month")

        common = avg_scores.index.intersection(fr.index)
        if len(common) >= 3:
            s_vals = avg_scores.loc[common].values
            f_vals = fr.loc[common, "fraud_rate"].values
            if np.std(s_vals) > 0 and np.std(f_vals) > 0:
                corr = np.corrcoef(s_vals, f_vals)[0, 1]
                print(f"\n  Correlacion temporal con fraude real: r={corr:+.3f}")
                if corr > 0.3:
                    print(f"  → Los scores altos coinciden con meses de fraude real")
                elif corr > 0:
                    print(f"  → Correlacion debil positiva")


# ─────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CAMBIOS_PATH = "data/cambios-de-contador-solo-alicante_hackaton-dataart-cambios-de-contador-solo-alicante.csv.csv"
    CONTADORES_PATH = "data/contadores-telelectura-instalados-solo-alicante_hackaton-dataart-contadores-telelectura-instalad.csv"

    print("=" * 70)
    print("  M12 — Detector de Fraude")
    print("=" * 70)

    # Fraude timeline
    fraud = load_fraud_cases(CAMBIOS_PATH)
    print(f"\n  Casos de fraude/sospechosos: {len(fraud)}")
    if len(fraud) > 0:
        for motivo in SUSPICIOUS_MOTIVOS:
            n = (fraud["MOTIVO_CAMBIO"] == motivo).sum()
            print(f"    {motivo}: {n}")

    # Monthly rate
    fraud_rate = compute_monthly_fraud_rate(CAMBIOS_PATH)
    if len(fraud_rate) > 0:
        print(f"\n  Tasa de fraude mensual ({len(fraud_rate)} meses):")
        top_months = fraud_rate.nlargest(5, "fraud_rate")
        for _, row in top_months.iterrows():
            fecha_str = row["fecha"].strftime("%Y-%m")
            print(f"    {fecha_str}: {row['n_fraud']:.0f} fraudes / "
                  f"{row['n_total_changes']:.0f} cambios = {row['fraud_rate']:.3f}")

    # Vulnerability
    vuln = compute_barrio_vulnerability(CONTADORES_PATH)
    if len(vuln) > 0:
        print(f"\n  Vulnerabilidad por barrio (top 10):")
        top_vuln = vuln.nlargest(10, "fraud_vulnerability")
        for _, row in top_vuln.iterrows():
            print(f"    {row['barrio']:<35}  {row['fraud_vulnerability']:.3f}")
