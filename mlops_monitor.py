"""
MLOps Monitoring — Sistema de monitorizacion en produccion.

1. Data Drift Detection (PSI) — detecta si la distribucion de features cambia
2. Model Decay Tracking — detecta si el modelo pierde calibracion
3. A/B Testing Framework — mide si intervenciones AMAEM funcionan

Un quant no entrega un script, entrega un SISTEMA.

Uso:
  from mlops_monitor import run_monitoring_report
  run_monitoring_report(df_monthly, results, feature_cols)
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────
# 1. PSI — Population Stability Index (Data Drift)
# ─────────────────────────────────────────────────────────────────

def compute_psi(reference: np.ndarray, current: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Population Stability Index.

    PSI < 0.1  → Sin drift (distribucion estable)
    PSI 0.1-0.2 → Drift moderado (investigar)
    PSI > 0.2  → Drift significativo (reentrenar)

    Usado en banca/seguros para monitorizar modelos en produccion.
    """
    # Crear bins basados en la referencia
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    if len(ref_clean) < 10 or len(cur_clean) < 10:
        return 0.0

    breakpoints = np.percentile(ref_clean, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    # Ensure unique breakpoints
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    ref_counts = np.histogram(ref_clean, bins=breakpoints)[0]
    cur_counts = np.histogram(cur_clean, bins=breakpoints)[0]

    # Proporciones (con suavizado para evitar log(0))
    ref_pct = (ref_counts + 1) / (len(ref_clean) + len(breakpoints) - 1)
    cur_pct = (cur_counts + 1) / (len(cur_clean) + len(breakpoints) - 1)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi


def detect_data_drift(df_monthly: pd.DataFrame,
                      feature_cols: list,
                      split_date: str = "2024-01-01") -> pd.DataFrame:
    """
    Detecta drift en features entre periodo de referencia y actual.
    """
    print(f"\n  [MLOPS] Data Drift Detection (PSI)...")

    df = df_monthly.copy()
    df["fecha_dt"] = pd.to_datetime(df["fecha"])

    reference = df[df["fecha_dt"] < split_date]
    current = df[df["fecha_dt"] >= split_date]

    if len(reference) < 20 or len(current) < 20:
        print(f"    Insuficientes datos para drift detection")
        return pd.DataFrame()

    results = []
    available = [c for c in feature_cols if c in df.columns]

    for col in available:
        ref_vals = reference[col].dropna().values.astype(float)
        cur_vals = current[col].dropna().values.astype(float)
        psi = compute_psi(ref_vals, cur_vals)

        if psi < 0.1:
            status = "ESTABLE"
        elif psi < 0.2:
            status = "DRIFT MODERADO"
        else:
            status = "DRIFT ALTO"

        results.append({
            "feature": col,
            "psi": psi,
            "status": status,
            "ref_mean": ref_vals.mean() if len(ref_vals) > 0 else 0,
            "cur_mean": cur_vals.mean() if len(cur_vals) > 0 else 0,
            "pct_change": ((cur_vals.mean() - ref_vals.mean()) /
                          (ref_vals.mean() + 1e-10) * 100) if len(ref_vals) > 0 else 0,
        })

    drift_df = pd.DataFrame(results).sort_values("psi", ascending=False)

    n_stable = (drift_df["status"] == "ESTABLE").sum()
    n_moderate = (drift_df["status"] == "DRIFT MODERADO").sum()
    n_high = (drift_df["status"] == "DRIFT ALTO").sum()

    print(f"    {len(available)} features analizadas:")
    print(f"      Estables: {n_stable}, Drift moderado: {n_moderate}, "
          f"Drift alto: {n_high}")

    # Top features con drift
    if n_moderate + n_high > 0:
        drifted = drift_df[drift_df["status"] != "ESTABLE"].head(5)
        print(f"    Features con drift:")
        for _, row in drifted.iterrows():
            print(f"      {row['feature']}: PSI={row['psi']:.3f} "
                  f"({row['status']}) media {row['ref_mean']:.1f}→{row['cur_mean']:.1f}")

    return drift_df


# ─────────────────────────────────────────────────────────────────
# 2. Model Decay Tracking
# ─────────────────────────────────────────────────────────────────

def track_model_decay(results: pd.DataFrame) -> pd.DataFrame:
    """
    Trackea si los modelos mantienen su calibracion a lo largo del tiempo.

    Si un modelo detecta 5% en enero y 25% en diciembre → posible decay.
    Si la tasa se mantiene estable → modelo bien calibrado.
    """
    print(f"\n  [MLOPS] Model Decay Tracking...")

    anomaly_cols = [c for c in results.columns if c.startswith("is_anomaly_")]
    if not anomaly_cols:
        print(f"    Sin columnas de anomalia")
        return pd.DataFrame()

    results_copy = results.copy()
    results_copy["fecha_dt"] = pd.to_datetime(results_copy["fecha"])
    results_copy["year_quarter"] = results_copy["fecha_dt"].dt.to_period("Q")

    decay_data = []

    for col in anomaly_cols:
        quarterly = results_copy.groupby("year_quarter")[col].agg(
            ["mean", "count"]
        ).reset_index()
        quarterly.columns = ["quarter", "detection_rate", "n_points"]

        rates = quarterly["detection_rate"].values
        if len(rates) < 2:
            continue

        # Trend: correlacion con el tiempo
        x = np.arange(len(rates))
        if rates.std() > 0:
            correlation = np.corrcoef(x, rates)[0, 1]
        else:
            correlation = 0

        # Stability: CV de las tasas
        cv = rates.std() / (rates.mean() + 1e-10)

        if abs(correlation) > 0.7:
            status = "DECAY" if correlation > 0 else "MEJORA"
        elif cv > 0.5:
            status = "INESTABLE"
        else:
            status = "ESTABLE"

        model_name = col.replace("is_anomaly_", "")
        decay_data.append({
            "model": model_name,
            "mean_rate": rates.mean() * 100,
            "cv": cv,
            "trend_corr": correlation,
            "status": status,
            "quarters": len(rates),
            "rates": [f"{r*100:.1f}%" for r in rates],
        })

    decay_df = pd.DataFrame(decay_data)

    if len(decay_df) > 0:
        print(f"    {'Modelo':<20} {'Tasa media':>10} {'CV':>8} "
              f"{'Trend':>8} {'Estado':>12}")
        print(f"    {'─'*60}")
        for _, row in decay_df.iterrows():
            print(f"    {row['model']:<20} {row['mean_rate']:>9.1f}% "
                  f"{row['cv']:>7.2f} {row['trend_corr']:>+7.2f} "
                  f"{row['status']:>12}")

        # Alertas
        decaying = decay_df[decay_df["status"] == "DECAY"]
        if len(decaying) > 0:
            print(f"\n    ALERTA: {len(decaying)} modelo(s) con posible decay:")
            for _, row in decaying.iterrows():
                print(f"      {row['model']}: tasa {' → '.join(row['rates'])}")
                print(f"      → Considerar reentrenar con datos recientes")

    return decay_df


# ─────────────────────────────────────────────────────────────────
# 3. A/B Testing Framework
# ─────────────────────────────────────────────────────────────────

def ab_test_intervention(results: pd.DataFrame,
                         intervention_barrios: list = None,
                         intervention_date: str = "2024-01-01") -> dict:
    """
    A/B Test: mide si una intervencion de AMAEM redujo anomalias.

    Tratamiento: barrios donde AMAEM intervino (reemplazo, reparacion)
    Control: barrios similares sin intervencion
    Metrica: tasa de anomalia antes vs despues
    """
    print(f"\n  [MLOPS] A/B Testing Framework...")

    if intervention_barrios is None:
        # Inferir: barrios con cambios de contador recientes
        # como proxy de intervencion
        if "pct_telelectura" not in results.columns:
            print(f"    Sin datos de intervencion — usando top anomalos como proxy")
            # Usar barrios con >3 modelos como "intervenidos"
            if "n_models_detecting" in results.columns:
                top_barrios = results[
                    results["n_models_detecting"] >= 3
                ]["barrio_key"].unique().tolist()
                intervention_barrios = top_barrios[:5] if len(top_barrios) > 5 else top_barrios
            else:
                return {}
        else:
            high_tele = results.groupby("barrio_key")["pct_telelectura"].mean()
            intervention_barrios = high_tele.nlargest(5).index.tolist()

    if not intervention_barrios:
        print(f"    Sin barrios de intervencion identificados")
        return {}

    results_copy = results.copy()
    results_copy["fecha_dt"] = pd.to_datetime(results_copy["fecha"])

    treated = results_copy[results_copy["barrio_key"].isin(intervention_barrios)]
    control = results_copy[~results_copy["barrio_key"].isin(intervention_barrios)]

    pre_date = pd.to_datetime(intervention_date)

    ab_results = {}

    for group_name, group_df in [("Tratados", treated), ("Control", control)]:
        pre = group_df[group_df["fecha_dt"] < pre_date]
        post = group_df[group_df["fecha_dt"] >= pre_date]

        if len(pre) > 0 and len(post) > 0:
            pre_rate = pre["n_models_detecting"].mean()
            post_rate = post["n_models_detecting"].mean()
            change = post_rate - pre_rate

            ab_results[group_name] = {
                "pre_rate": pre_rate,
                "post_rate": post_rate,
                "change": change,
                "n_pre": len(pre),
                "n_post": len(post),
            }

            print(f"    {group_name}: {pre_rate:.2f} → {post_rate:.2f} "
                  f"(cambio: {change:+.2f} modelos/punto)")

    if "Tratados" in ab_results and "Control" in ab_results:
        # DiD del A/B
        treated_change = ab_results["Tratados"]["change"]
        control_change = ab_results["Control"]["change"]
        net_effect = treated_change - control_change

        print(f"    Efecto neto de intervencion: {net_effect:+.2f} modelos/punto")
        if net_effect < 0:
            print(f"    → La intervencion REDUCE anomalias ({net_effect:.2f} modelos menos)")
        else:
            print(f"    → Sin efecto claro de la intervencion")

        ab_results["net_effect"] = net_effect

    return ab_results


# ─────────────────────────────────────────────────────────────────
# Main: reporte completo de monitoring
# ─────────────────────────────────────────────────────────────────

def run_monitoring_report(df_monthly: pd.DataFrame,
                          results: pd.DataFrame,
                          feature_cols: list) -> dict:
    """Ejecuta todo el monitoring de MLOps."""
    print(f"\n{'='*80}")
    print(f"  MLOPS MONITORING — Sistema de produccion")
    print(f"{'='*80}")

    monitoring = {}

    # 1. Data Drift
    drift_df = detect_data_drift(df_monthly, feature_cols)
    monitoring["drift"] = drift_df

    # 2. Model Decay
    decay_df = track_model_decay(results)
    monitoring["decay"] = decay_df

    # 3. A/B Testing
    ab = ab_test_intervention(results)
    monitoring["ab_test"] = ab

    # Resumen ejecutivo
    print(f"\n  ESTADO DEL SISTEMA:")
    print(f"  {'─'*75}")

    if len(drift_df) > 0:
        n_drift = (drift_df["status"] != "ESTABLE").sum()
        if n_drift == 0:
            print(f"    Data Drift:    OK — Todas las features estables")
        else:
            print(f"    Data Drift:    ATENCION — {n_drift} features con drift")

    if len(decay_df) > 0:
        n_decay = (decay_df["status"] == "DECAY").sum()
        if n_decay == 0:
            print(f"    Model Decay:   OK — Todos los modelos estables")
        else:
            print(f"    Model Decay:   ALERTA — {n_decay} modelos con decay")

    if "net_effect" in ab:
        if ab["net_effect"] < -0.5:
            print(f"    Intervenciones: EFICACES — Reducen anomalias")
        else:
            print(f"    Intervenciones: REVISAR — Sin efecto claro")

    return monitoring
