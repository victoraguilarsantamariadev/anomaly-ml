"""
Counterfactual Explanations — No solo "es anomalo", sino "que tendria que cambiar".

Cumple con EU AI Act (Art. 13-14): derecho a explicacion de decisiones algoritmicas.

Para cada anomalia, calcula el MINIMO cambio necesario para que el punto
sea clasificado como normal. Esto permite:
  1. Entender POR QUE es anomalo (que feature lo causa)
  2. Cuantificar la magnitud del problema
  3. Dar recomendaciones ACCIONABLES a AMAEM

Basado en Wachter et al. (2017) "Counterfactual Explanations without Opening the Black Box"
y Mothilal et al. (2020) "DiCE: Diverse Counterfactual Explanations"

Uso:
  from counterfactual_explainer import generate_counterfactuals, counterfactual_summary
  cf_df = generate_counterfactuals(results, df_features, top_n=10)
  counterfactual_summary(cf_df)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ─────────────────────────────────────────────────────────────────
# Features descriptivos: nombres legibles para recomendaciones
# ─────────────────────────────────────────────────────────────────

FEATURE_LABELS = {
    "consumption_per_contract": "Consumo por contrato (litros)",
    "yoy_ratio": "Ratio interanual",
    "deviation_from_group_trend": "Desviacion de la tendencia del grupo",
    "relative_consumption": "Consumo relativo al grupo",
    "seasonal_zscore": "Z-score estacional",
    "cross_sectional_zscore": "Z-score transversal",
    "type_percentile": "Percentil dentro del tipo",
    "trend_3m": "Tendencia 3 meses",
    "months_above_mean": "Meses consecutivos sobre la media",
    "consumo_litros": "Consumo total (litros)",
    "anr_ratio": "Ratio ANR (agua no registrada)",
    "prophet_residual": "Residual Prophet",
    "reconstruction_error": "Error de reconstruccion (autoencoder)",
    "vae_score_norm": "Score VAE normalizado",
    "reading_anomaly_rate": "Tasa de lecturas anomalas",
    "reading_anomaly_zscore": "Z-score de lecturas anomalas",
    "log_consumption": "Log-consumo",
    "consumption_volatility": "Volatilidad consumo",
    "momentum": "Momentum",
    "yoy_acceleration": "Aceleracion interanual",
    "smart_meter_ratio": "Ratio contadores inteligentes",
    "avg_calibre": "Calibre medio",
    "regenerada_ratio": "Ratio agua regenerada",
    "fraud_rate": "Tasa de fraude",
    "fraud_rate_3m": "Tasa de fraude (3 meses)",
    "pipe_density_km_per_km2": "Densidad de tuberias (km/km2)",
    "avg_pipe_diameter_mm": "Diametro medio tuberia (mm)",
    "elevation_range": "Rango de elevacion",
    "infrastructure_risk_score": "Score riesgo infraestructura",
}

# Recomendaciones accionables por tipo de feature
ACTION_TEMPLATES = {
    "consumption_per_contract": (
        "Reducir consumo un {pct:.0f}% igualaria al grupo de referencia "
        "(de {current:.0f} a {target:.0f} litros/contrato)"
    ),
    "consumo_litros": (
        "Reducir consumo total un {pct:.0f}% "
        "(de {current:,.0f} a {target:,.0f} litros)"
    ),
    "yoy_ratio": (
        "El ratio interanual deberia bajar de {current:.2f} a {target:.2f} "
        "para estar en rango normal"
    ),
    "anr_ratio": (
        "El ratio ANR deberia bajar de {current:.2f} a {target:.2f} — "
        "investigar posibles fugas o subregistro"
    ),
    "deviation_from_group_trend": (
        "La desviacion respecto al grupo deberia bajar de {current:.3f} a {target:.3f}"
    ),
    "relative_consumption": (
        "El consumo relativo al grupo deberia ajustarse de {current:.3f} a {target:.3f}"
    ),
    "seasonal_zscore": (
        "El z-score estacional deberia bajar de {current:.2f} a {target:.2f} "
        "(dentro de +-2 es normal)"
    ),
    "cross_sectional_zscore": (
        "El z-score transversal deberia bajar de {current:.2f} a {target:.2f}"
    ),
    "trend_3m": (
        "La tendencia de 3 meses deberia cambiar de {current:.4f} a {target:.4f}"
    ),
    "months_above_mean": (
        "Lleva {current:.0f} meses sobre la media; deberia bajar a {target:.0f} "
        "para salir de la zona de alerta"
    ),
    "fraud_rate": (
        "La tasa de fraude ({current:.4f}) deberia bajar a {target:.4f} — "
        "reforzar inspecciones en la zona"
    ),
    "reading_anomaly_rate": (
        "La tasa de lecturas anomalas ({current:.4f}) deberia bajar a {target:.4f} — "
        "revisar contadores sospechosos"
    ),
    "reconstruction_error": (
        "El error de reconstruccion ({current:.0f}) deberia bajar a {target:.0f} — "
        "el patron de consumo es inusual en multiples dimensiones"
    ),
    "infrastructure_risk_score": (
        "El riesgo de infraestructura ({current:.2f}) deberia bajar a {target:.2f} — "
        "considerar renovacion de red"
    ),
}

DEFAULT_ACTION = (
    "{feature_label} deberia cambiar de {current:.4f} a {target:.4f} "
    "({direction} un {pct:.1f}%)"
)


# ─────────────────────────────────────────────────────────────────
# 1. PERCENTILE-BASED COUNTERFACTUALS
# ─────────────────────────────────────────────────────────────────

def _detect_anomaly_columns(results: pd.DataFrame) -> list:
    """Detecta columnas is_anomaly_* presentes en results."""
    return [c for c in results.columns if c.startswith("is_anomaly_")]


def _get_feature_columns(results: pd.DataFrame, df_features: pd.DataFrame = None) -> list:
    """Determina las columnas de features disponibles para counterfactuals."""
    candidates = [
        "consumption_per_contract", "consumo_litros",
        "yoy_ratio", "deviation_from_group_trend", "relative_consumption",
        "seasonal_zscore", "cross_sectional_zscore", "type_percentile",
        "trend_3m", "months_above_mean",
        "anr_ratio", "prophet_residual",
        "reconstruction_error", "vae_score_norm",
        "reading_anomaly_rate", "reading_anomaly_zscore",
        "log_consumption", "consumption_volatility", "momentum",
        "yoy_acceleration", "smart_meter_ratio", "avg_calibre",
        "regenerada_ratio", "fraud_rate", "fraud_rate_3m",
        "pipe_density_km_per_km2", "avg_pipe_diameter_mm",
        "elevation_range", "infrastructure_risk_score",
    ]
    # Buscar en ambos DataFrames
    available = set()
    for col in candidates:
        if col in results.columns:
            available.add(col)
        if df_features is not None and col in df_features.columns:
            available.add(col)
    return sorted(available)


def _merge_features(results: pd.DataFrame,
                    df_features: pd.DataFrame = None) -> pd.DataFrame:
    """Combina results con df_features si se proporcionan."""
    if df_features is None:
        return results.copy()

    merge_cols = ["barrio_key", "fecha"]
    available_merge = [c for c in merge_cols if c in results.columns and c in df_features.columns]
    if len(available_merge) < 2:
        return results.copy()

    # Features que no estan ya en results
    extra_cols = [c for c in df_features.columns
                  if c not in results.columns and c not in merge_cols]
    if not extra_cols:
        return results.copy()

    merged = results.merge(
        df_features[available_merge + extra_cols],
        on=available_merge, how="left", suffixes=("", "_feat")
    )
    return merged


def _compute_normal_bounds(df: pd.DataFrame,
                           feature_cols: list,
                           lower_pct: float = 5.0,
                           upper_pct: float = 95.0) -> pd.DataFrame:
    """
    Calcula los limites normales para cada feature usando percentiles
    de los puntos NO anomalos.
    """
    # Identificar puntos normales
    anom_cols = _detect_anomaly_columns(df)
    if "n_models_detecting" in df.columns:
        normal_mask = df["n_models_detecting"] == 0
    elif anom_cols:
        normal_mask = ~df[anom_cols].any(axis=1)
    else:
        # Fallback: usar todo el dataset
        normal_mask = pd.Series(True, index=df.index)

    normal_data = df.loc[normal_mask]

    bounds = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        vals = normal_data[col].dropna()
        if len(vals) < 10:
            vals = df[col].dropna()
        if len(vals) < 5:
            continue
        bounds[col] = {
            "lower": np.percentile(vals, lower_pct),
            "upper": np.percentile(vals, upper_pct),
            "median": np.median(vals),
            "mean": np.mean(vals),
            "std": np.std(vals),
            "p25": np.percentile(vals, 25),
            "p75": np.percentile(vals, 75),
        }
    return bounds


def _compute_feature_counterfactual(current_val: float,
                                    bounds: dict) -> dict:
    """
    Para un valor actual, calcula el cambio minimo para entrar
    en el rango normal [lower, upper].

    Retorna None si ya esta dentro del rango.
    """
    if pd.isna(current_val):
        return None

    lower = bounds["lower"]
    upper = bounds["upper"]

    # Si ya esta en rango normal, no necesita cambio
    if lower <= current_val <= upper:
        return None

    # Determinar la frontera mas cercana
    if current_val > upper:
        target = upper
        direction = "bajar"
    else:
        target = lower
        direction = "subir"

    diff = target - current_val
    if abs(current_val) > 1e-9:
        change_pct = (diff / abs(current_val)) * 100
    else:
        change_pct = 0.0

    return {
        "required_value": target,
        "change_absolute": diff,
        "change_pct": change_pct,
        "direction": direction,
        "distance_to_median": abs(current_val - bounds["median"]),
        "distance_in_std": (abs(current_val - bounds["mean"]) / bounds["std"]
                            if bounds["std"] > 1e-9 else 0.0),
    }


def _build_action_description(feature: str,
                               current_val: float,
                               target_val: float,
                               change_pct: float,
                               direction: str) -> str:
    """Genera una recomendacion accionable en lenguaje natural."""
    feature_label = FEATURE_LABELS.get(feature, feature)

    if feature in ACTION_TEMPLATES:
        return ACTION_TEMPLATES[feature].format(
            current=current_val,
            target=target_val,
            pct=abs(change_pct),
            feature_label=feature_label,
            direction=direction,
        )
    else:
        return DEFAULT_ACTION.format(
            feature_label=feature_label,
            current=current_val,
            target=target_val,
            pct=abs(change_pct),
            direction=direction,
        )


# ─────────────────────────────────────────────────────────────────
# 2. NEAREST NORMAL NEIGHBOR
# ─────────────────────────────────────────────────────────────────

def _find_nearest_normal(df: pd.DataFrame,
                         anomalous_idx: list,
                         normal_idx: list,
                         feature_cols: list,
                         n_neighbors: int = 1) -> dict:
    """
    Para cada punto anomalo, encuentra el vecino normal mas cercano
    en el espacio de features estandarizado.

    Retorna dict: anomalous_idx -> {neighbor_idx, distance, feature_diffs}
    """
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 2 or len(normal_idx) < 2:
        return {}

    # Preparar matrices
    X_all = df[available].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_all),
        index=X_all.index,
        columns=available,
    )

    X_normal = X_scaled.loc[normal_idx].values
    if len(X_normal) < 2:
        return {}

    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(X_normal)), metric="euclidean")
    nn.fit(X_normal)

    results = {}
    for idx in anomalous_idx:
        if idx not in X_scaled.index:
            continue
        x = X_scaled.loc[[idx]].values
        distances, indices = nn.kneighbors(x)

        neighbor_pos = indices[0][0]
        neighbor_original_idx = normal_idx[neighbor_pos]
        dist = distances[0][0]

        # Diferencias por feature (en escala original)
        feature_diffs = {}
        for col in available:
            anom_val = df.loc[idx, col]
            norm_val = df.loc[neighbor_original_idx, col]
            if pd.notna(anom_val) and pd.notna(norm_val) and abs(anom_val) > 1e-9:
                feature_diffs[col] = {
                    "anomalous_value": anom_val,
                    "normal_value": norm_val,
                    "diff": norm_val - anom_val,
                    "diff_pct": ((norm_val - anom_val) / abs(anom_val)) * 100,
                }

        results[idx] = {
            "neighbor_idx": neighbor_original_idx,
            "distance": dist,
            "feature_diffs": feature_diffs,
        }

    return results


# ─────────────────────────────────────────────────────────────────
# 3. CONFIDENCE ASSESSMENT
# ─────────────────────────────────────────────────────────────────

def _assess_counterfactual_confidence(cf_records: list) -> str:
    """
    Evalua la confianza del contrafactual segun cuantos features
    necesitan cambiar para que el punto sea normal.

    1 feature  -> ALTA   (causa clara, facil de actuar)
    2-3        -> MEDIA  (multiples factores, investigar)
    4+         -> BAJA   (posible problema estructural)
    """
    n_features = len(cf_records)
    if n_features <= 1:
        return "ALTA"
    elif n_features <= 3:
        return "MEDIA"
    else:
        return "BAJA"


def _confidence_explanation(confidence: str, n_features: int) -> str:
    """Explica en lenguaje natural que significa la confianza."""
    if confidence == "ALTA":
        return (f"Confianza ALTA: solo {n_features} feature fuera de rango. "
                "La causa es clara y la accion es directa.")
    elif confidence == "MEDIA":
        return (f"Confianza MEDIA: {n_features} features fuera de rango. "
                "Multiples factores contribuyen; investigar interdependencias.")
    else:
        return (f"Confianza BAJA: {n_features} features fuera de rango. "
                "Posible problema estructural o cambio de regimen en el barrio.")


# ─────────────────────────────────────────────────────────────────
# 4. MAIN API: generate_counterfactuals
# ─────────────────────────────────────────────────────────────────

def generate_counterfactuals(results: pd.DataFrame,
                             df_features: pd.DataFrame = None,
                             top_n: int = 10) -> pd.DataFrame:
    """
    Genera explicaciones contrafactuales para las top_n anomalias
    mas severas.

    Para cada anomalia, calcula:
      - El MINIMO cambio por feature para entrar en rango normal
      - El vecino normal mas cercano
      - Recomendaciones accionables
      - Confianza del contrafactual

    Parametros
    ----------
    results : pd.DataFrame
        Resultados del pipeline con columnas is_anomaly_*, n_models_detecting,
        anomaly_score, barrio_key, fecha, y features.
    df_features : pd.DataFrame, optional
        DataFrame con features mensuales adicionales. Se mergea por
        (barrio_key, fecha).
    top_n : int
        Numero maximo de anomalias a analizar.

    Retorna
    -------
    pd.DataFrame con columnas:
        barrio_key, fecha, counterfactual_feature, current_value,
        required_value, change_pct, action_description,
        nearest_normal_barrio, nearest_normal_distance,
        cf_confidence, n_features_outside_range
    """
    print(f"\n{'='*70}")
    print(f"  COUNTERFACTUAL EXPLANATIONS")
    print(f"  'Que tendria que cambiar para que NO sea anomalo?'")
    print(f"{'='*70}")

    # Merge features si se proporcionan
    df = _merge_features(results, df_features)

    # Detectar features disponibles
    feature_cols = _get_feature_columns(df)
    if not feature_cols:
        print("  [WARN] No se encontraron features para counterfactuals")
        return pd.DataFrame()
    print(f"\n  Features disponibles: {len(feature_cols)}")

    # Identificar anomalias
    anom_cols = _detect_anomaly_columns(df)
    if "n_models_detecting" in df.columns:
        anomalous_mask = df["n_models_detecting"] >= 1
        normal_mask = df["n_models_detecting"] == 0
    elif anom_cols:
        anomalous_mask = df[anom_cols].any(axis=1)
        normal_mask = ~anomalous_mask
    else:
        print("  [WARN] No se encontraron columnas de anomalia")
        return pd.DataFrame()

    n_anomalous = anomalous_mask.sum()
    print(f"  Anomalias detectadas: {n_anomalous}")

    if n_anomalous == 0:
        print("  No hay anomalias para explicar")
        return pd.DataFrame()

    # Ordenar por severidad y tomar top_n
    anomalous_df = df[anomalous_mask].copy()
    if "anomaly_score" in anomalous_df.columns:
        anomalous_df = anomalous_df.sort_values("anomaly_score", ascending=False)
    elif "n_models_detecting" in anomalous_df.columns:
        anomalous_df = anomalous_df.sort_values("n_models_detecting", ascending=False)
    anomalous_df = anomalous_df.head(top_n)
    print(f"  Analizando top {len(anomalous_df)} anomalias...")

    # Calcular limites normales
    bounds = _compute_normal_bounds(df, feature_cols)
    if not bounds:
        print("  [WARN] No se pudieron calcular limites normales")
        return pd.DataFrame()

    # Encontrar vecinos normales
    anomalous_idx = anomalous_df.index.tolist()
    normal_idx = df[normal_mask].index.tolist()
    nn_results = _find_nearest_normal(df, anomalous_idx, normal_idx, feature_cols)

    # Generar counterfactuals por cada anomalia
    all_records = []

    for idx in anomalous_idx:
        row = df.loc[idx]
        barrio = row.get("barrio_key", "?")
        fecha = row.get("fecha", "?")

        # Calcular contrafactuales por feature
        cf_for_row = []
        for col in feature_cols:
            if col not in bounds or col not in df.columns:
                continue
            current_val = row.get(col, np.nan)
            cf = _compute_feature_counterfactual(current_val, bounds[col])
            if cf is not None:
                cf_for_row.append((col, current_val, cf))

        # Ordenar por distancia en desviaciones estandar (mas anomalo primero)
        cf_for_row.sort(key=lambda x: x[2]["distance_in_std"], reverse=True)

        # Confianza
        n_outside = len(cf_for_row)
        confidence = _assess_counterfactual_confidence(cf_for_row)

        # Vecino normal mas cercano
        nn_info = nn_results.get(idx, {})
        nn_barrio = "?"
        nn_distance = np.nan
        if nn_info:
            nn_idx = nn_info["neighbor_idx"]
            nn_barrio = df.loc[nn_idx].get("barrio_key", "?")
            nn_distance = nn_info["distance"]

        # Generar registros
        if cf_for_row:
            for col, current_val, cf in cf_for_row:
                action = _build_action_description(
                    col, current_val, cf["required_value"],
                    cf["change_pct"], cf["direction"],
                )
                all_records.append({
                    "barrio_key": barrio,
                    "fecha": fecha,
                    "counterfactual_feature": col,
                    "current_value": current_val,
                    "required_value": cf["required_value"],
                    "change_pct": cf["change_pct"],
                    "change_absolute": cf["change_absolute"],
                    "direction": cf["direction"],
                    "distance_in_std": cf["distance_in_std"],
                    "action_description": action,
                    "nearest_normal_barrio": nn_barrio,
                    "nearest_normal_distance": nn_distance,
                    "cf_confidence": confidence,
                    "n_features_outside_range": n_outside,
                })
        else:
            # Anomalia sin features fuera de rango (detectada por modelos
            # que no usan features simples, e.g. Prophet, Chronos)
            all_records.append({
                "barrio_key": barrio,
                "fecha": fecha,
                "counterfactual_feature": "(modelo temporal)",
                "current_value": np.nan,
                "required_value": np.nan,
                "change_pct": np.nan,
                "change_absolute": np.nan,
                "direction": "n/a",
                "distance_in_std": 0.0,
                "action_description": (
                    "Anomalia detectada por modelos temporales (Prophet/Chronos). "
                    "Revisar patron temporal del barrio."
                ),
                "nearest_normal_barrio": nn_barrio,
                "nearest_normal_distance": nn_distance,
                "cf_confidence": confidence,
                "n_features_outside_range": 0,
            })

    cf_df = pd.DataFrame(all_records)
    print(f"  Contrafactuales generados: {len(cf_df)} registros "
          f"para {cf_df['barrio_key'].nunique()} barrios")

    return cf_df


# ─────────────────────────────────────────────────────────────────
# 5. HUMAN-READABLE SUMMARY
# ─────────────────────────────────────────────────────────────────

def counterfactual_summary(cf_df: pd.DataFrame) -> None:
    """
    Imprime un resumen legible de los contrafactuales con
    recomendaciones accionables.

    Parametros
    ----------
    cf_df : pd.DataFrame
        Output de generate_counterfactuals().
    """
    if cf_df.empty:
        print("\n  No hay contrafactuales para mostrar.")
        return

    print(f"\n{'='*70}")
    print(f"  RESUMEN CONTRAFACTUAL — EXPLICACIONES ACCIONABLES")
    print(f"{'='*70}")

    # Agrupar por barrio-fecha
    grouped = cf_df.groupby(["barrio_key", "fecha"])

    for (barrio, fecha), group in grouped:
        fecha_str = pd.Timestamp(fecha).strftime("%Y-%m") if pd.notna(fecha) else str(fecha)
        confidence = group["cf_confidence"].iloc[0]
        n_outside = group["n_features_outside_range"].iloc[0]
        nn_barrio = group["nearest_normal_barrio"].iloc[0]
        nn_dist = group["nearest_normal_distance"].iloc[0]

        print(f"\n  {'─'*66}")
        print(f"  BARRIO: {barrio} | MES: {fecha_str}")
        print(f"  {'─'*66}")

        # Confianza
        conf_label = {"ALTA": "ALTA", "MEDIA": "MEDIA", "BAJA": "BAJA"}.get(confidence, confidence)
        conf_explanation = _confidence_explanation(confidence, n_outside)
        print(f"  Confianza contrafactual: {conf_label}")
        print(f"    {conf_explanation}")

        # Top features fuera de rango (max 5 para legibilidad)
        feature_rows = group[group["counterfactual_feature"] != "(modelo temporal)"]
        if not feature_rows.empty:
            print(f"\n  Que tendria que cambiar:")
            for i, (_, row) in enumerate(feature_rows.iterrows()):
                if i >= 5:
                    remaining = len(feature_rows) - 5
                    print(f"    ... y {remaining} features mas")
                    break
                feat = row["counterfactual_feature"]
                label = FEATURE_LABELS.get(feat, feat)
                current = row["current_value"]
                target = row["required_value"]
                pct = row["change_pct"]
                std_dist = row["distance_in_std"]
                print(f"    {i+1}. {label}")
                print(f"       Actual: {current:.4f} -> Necesario: {target:.4f} "
                      f"({pct:+.1f}%, {std_dist:.1f} sigmas)")

        # Recomendaciones accionables (top 3)
        print(f"\n  Recomendaciones accionables:")
        action_rows = group.head(3)
        for i, (_, row) in enumerate(action_rows.iterrows()):
            print(f"    {i+1}. {row['action_description']}")

        # Vecino normal
        if nn_barrio != "?" and pd.notna(nn_dist):
            print(f"\n  Barrio normal mas similar: {nn_barrio} (distancia={nn_dist:.3f})")

    # Resumen global
    print(f"\n  {'='*66}")
    print(f"  RESUMEN GLOBAL")
    print(f"  {'='*66}")

    n_barrios = cf_df["barrio_key"].nunique()
    conf_counts = cf_df.groupby("barrio_key")["cf_confidence"].first().value_counts()

    print(f"  Barrios analizados: {n_barrios}")
    for conf, count in conf_counts.items():
        print(f"    Confianza {conf}: {count} barrios")

    # Features mas frecuentemente fuera de rango
    feature_freq = (cf_df[cf_df["counterfactual_feature"] != "(modelo temporal)"]
                    ["counterfactual_feature"].value_counts().head(5))
    if not feature_freq.empty:
        print(f"\n  Features mas frecuentemente fuera de rango:")
        for feat, count in feature_freq.items():
            label = FEATURE_LABELS.get(feat, feat)
            print(f"    - {label}: {count} veces")

    # Cambios mas grandes necesarios
    if "change_pct" in cf_df.columns:
        biggest = (cf_df.dropna(subset=["change_pct"])
                   .sort_values("change_pct", key=abs, ascending=False)
                   .head(3))
        if not biggest.empty:
            print(f"\n  Mayores cambios necesarios:")
            for _, row in biggest.iterrows():
                label = FEATURE_LABELS.get(row["counterfactual_feature"],
                                           row["counterfactual_feature"])
                print(f"    - {row['barrio_key']} ({row['fecha']}): "
                      f"{label} requiere {row['change_pct']:+.1f}%")

    print()


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Counterfactual Explanations para anomalias")
    parser.add_argument("--results", default="results_full.csv",
                        help="CSV con resultados del pipeline")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Numero de anomalias a analizar")
    parser.add_argument("--output", default=None,
                        help="Guardar CSV de counterfactuals")
    args = parser.parse_args()

    print(f"Cargando resultados de {args.results}...")
    results = pd.read_csv(args.results)

    cf_df = generate_counterfactuals(results, top_n=args.top_n)
    counterfactual_summary(cf_df)

    if args.output:
        cf_df.to_csv(args.output, index=False)
        print(f"  Counterfactuals guardados en {args.output}")
