"""
M14 — Change Point Detection: detecta CUANDO cambia el comportamiento de un barrio.

Algoritmos:
  1. PELT (Pruned Exact Linear Time) — rapido, detecta cambios en media/varianza
  2. BOCPD simplificado — Bayesian Online, da probabilidad de cambio en cada punto
  3. CUSUM — Control chart clasico, detecta drift acumulado

No solo dice "es anomalo", sino "cambio en marzo 2023" → AMAEM sabe cuando buscar.

Uso:
  from changepoint_detector import run_changepoint_detection
  results = run_changepoint_detection(df_monthly, results)
"""

import numpy as np
import pandas as pd


def _detect_pelt(series: np.ndarray, pen: float = 3.0) -> list:
    """
    PELT changepoint detection.
    Usa ruptures si disponible, fallback a implementacion manual.
    """
    try:
        import ruptures as rpt
        algo = rpt.Pelt(model="rbf", min_size=3).fit(series.reshape(-1, 1))
        bkps = algo.predict(pen=pen)
        # ruptures incluye el ultimo indice, lo quitamos
        return [b for b in bkps if b < len(series)]
    except ImportError:
        return _cusum_simple(series)


def _cusum_simple(series: np.ndarray, threshold: float = 2.0) -> list:
    """
    CUSUM simplificado: detecta cambios acumulados en la media.
    Fallback cuando ruptures no esta disponible.
    """
    mean = series.mean()
    std = series.std() + 1e-10
    normalized = (series - mean) / std

    cusum_pos = np.zeros(len(series))
    cusum_neg = np.zeros(len(series))
    changepoints = []

    for i in range(1, len(series)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + normalized[i] - 0.5)
        cusum_neg[i] = max(0, cusum_neg[i-1] - normalized[i] - 0.5)

        if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
            changepoints.append(i)
            cusum_pos[i] = 0
            cusum_neg[i] = 0

    return changepoints


def _bayesian_changepoint_prob(series: np.ndarray,
                                hazard_rate: float = 1/12) -> np.ndarray:
    """
    Bayesian Online Changepoint Detection (simplificado).

    Calcula P(changepoint en t) para cada punto temporal.
    hazard_rate = 1/12 → esperamos ~1 cambio por año.

    Basado en Adams & MacKay (2007).
    """
    n = len(series)
    if n < 3:
        return np.zeros(n)

    # Prior: probabilidad de cambio en cada paso
    H = hazard_rate

    # Run length probabilities
    max_run = n
    R = np.zeros((max_run + 1, n + 1))
    R[0, 0] = 1.0

    cp_prob = np.zeros(n)

    # Predictive sufficient statistics (Gaussian conjugate)
    mu0 = series.mean()
    kappa0 = 1.0
    alpha0 = 1.0
    beta0 = series.var() + 1e-10

    for t in range(n):
        # Predictive probabilities for each run length
        pred_probs = np.zeros(t + 1)
        for r in range(t + 1):
            # Datos en el run actual
            if r == 0:
                # No data in current run → prior predictive
                pred_probs[r] = _gaussian_pred(series[t], mu0, beta0 / alpha0)
            else:
                start = t - r
                segment = series[start:t+1]
                seg_mean = segment.mean()
                seg_var = segment.var() + 1e-10
                pred_probs[r] = _gaussian_pred(series[t], seg_mean, seg_var)

        # Growth probability
        for r in range(t, 0, -1):
            R[r, t+1] = R[r-1, t] * pred_probs[min(r-1, t)] * (1 - H)

        # Changepoint probability
        cp_mass = 0
        for r in range(t + 1):
            cp_mass += R[r, t] * pred_probs[min(r, t)] * H

        R[0, t+1] = cp_mass
        cp_prob[t] = cp_mass

        # Normalize
        total = R[:t+2, t+1].sum()
        if total > 0:
            R[:t+2, t+1] /= total
            cp_prob[t] = R[0, t+1]

    return cp_prob


def _gaussian_pred(x, mu, var):
    """Predictive probability bajo Gaussiana."""
    return np.exp(-0.5 * (x - mu)**2 / (var + 1e-10)) / np.sqrt(2 * np.pi * (var + 1e-10))


def detect_changepoints_per_barrio(df_monthly: pd.DataFrame,
                                    consumo_col: str = "consumo_litros") -> pd.DataFrame:
    """
    Detecta changepoints en la serie temporal de cada barrio.

    Returns:
        DataFrame con columnas:
          - barrio_key, fecha, is_changepoint, cp_method, cp_probability,
            direction (UP/DOWN), magnitude
    """
    print(f"\n  [M14] Change Point Detection...")

    results = []
    barrios = df_monthly["barrio_key"].unique()

    for barrio in barrios:
        df_b = df_monthly[df_monthly["barrio_key"] == barrio].sort_values("fecha")
        if len(df_b) < 6:
            continue

        series = df_b[consumo_col].values.astype(float)
        fechas = df_b["fecha"].values

        # 1. PELT changepoints
        pelt_cps = _detect_pelt(series)

        # 2. Bayesian changepoint probability
        cp_probs = _bayesian_changepoint_prob(series)

        # 3. Para cada punto, determinar si es changepoint
        for i in range(len(series)):
            is_pelt = i in pelt_cps
            bayes_prob = cp_probs[i]
            is_cp = is_pelt or bayes_prob > 0.3

            if is_cp and i > 0:
                # Direccion y magnitud del cambio
                before = series[max(0, i-3):i]
                after = series[i:min(len(series), i+3)]
                if len(before) > 0 and len(after) > 0:
                    mean_before = before.mean()
                    mean_after = after.mean()
                    magnitude = (mean_after - mean_before) / (mean_before + 1e-10) * 100
                    direction = "UP" if magnitude > 0 else "DOWN"
                else:
                    magnitude = 0
                    direction = "UNKNOWN"

                method = []
                if is_pelt:
                    method.append("PELT")
                if bayes_prob > 0.3:
                    method.append(f"BOCPD(p={bayes_prob:.2f})")

                results.append({
                    "barrio_key": barrio,
                    "fecha": fechas[i],
                    "is_changepoint": True,
                    "cp_method": "+".join(method),
                    "cp_probability": bayes_prob,
                    "cp_direction": direction,
                    "cp_magnitude": magnitude,
                })

    cp_df = pd.DataFrame(results)

    if len(cp_df) > 0:
        n_barrios = cp_df["barrio_key"].nunique()
        n_up = (cp_df["cp_direction"] == "UP").sum()
        n_down = (cp_df["cp_direction"] == "DOWN").sum()
        print(f"    {len(cp_df)} changepoints en {n_barrios} barrios")
        print(f"    Subidas: {n_up}, Bajadas: {n_down}")

        # Top changepoints por magnitud
        top = cp_df.nlargest(min(5, len(cp_df)), "cp_magnitude", keep="first")
        print(f"    Top cambios mas grandes:")
        for _, row in top.iterrows():
            barrio = row["barrio_key"].split("__")[0]
            fecha = pd.to_datetime(row["fecha"]).strftime("%Y-%m")
            print(f"      {barrio} {fecha}: {row['cp_direction']} "
                  f"{row['cp_magnitude']:+.1f}% ({row['cp_method']})")
    else:
        print(f"    Sin changepoints detectados")

    return cp_df


def enrich_results_with_changepoints(results: pd.DataFrame,
                                      cp_df: pd.DataFrame) -> pd.DataFrame:
    """Merge changepoint info al results principal."""
    if len(cp_df) == 0:
        results["is_changepoint"] = False
        results["cp_method"] = ""
        results["cp_magnitude"] = 0.0
        return results

    cp_merge = cp_df[["barrio_key", "fecha", "is_changepoint",
                       "cp_method", "cp_magnitude", "cp_direction"]].copy()
    results = results.merge(cp_merge, on=["barrio_key", "fecha"],
                           how="left", suffixes=("", "_cp"))

    results["is_changepoint"] = results["is_changepoint"].fillna(False).astype(bool)
    results["cp_method"] = results["cp_method"].fillna("")
    results["cp_magnitude"] = results["cp_magnitude"].fillna(0.0)
    results["cp_direction"] = results["cp_direction"].fillna("")

    # Anomalia + changepoint = muy interesante
    if "n_models_detecting" in results.columns:
        both = results["is_changepoint"] & (results["n_models_detecting"] >= 2)
        n_both = both.sum()
        if n_both > 0:
            print(f"    Anomalias CON changepoint: {n_both} "
                  f"(cambio de regimen confirmado)")

    return results


def changepoint_summary(results: pd.DataFrame):
    """Resumen de changepoints para el output final."""
    if "is_changepoint" not in results.columns:
        return

    cps = results[results["is_changepoint"]].copy()
    if len(cps) == 0:
        return

    print(f"\n{'='*80}")
    print(f"  CHANGE POINT DETECTION — Cuando empezo cada problema")
    print(f"{'='*80}")

    # Agrupar por barrio: primer changepoint significativo
    barrio_first_cp = cps.sort_values("fecha").groupby("barrio_key").first()

    print(f"\n  {'Barrio':<30} {'Fecha cambio':>12} {'Direccion':>10} "
          f"{'Magnitud':>10} {'Metodo':>20}")
    print(f"  {'─'*85}")

    for barrio_key, row in barrio_first_cp.iterrows():
        barrio = barrio_key.split("__")[0][:28]
        fecha = pd.to_datetime(row["fecha"]).strftime("%Y-%m")
        direction = row.get("cp_direction", "")
        magnitude = row.get("cp_magnitude", 0)
        method = row.get("cp_method", "")
        print(f"  {barrio:<30} {fecha:>12} {direction:>10} "
              f"{magnitude:>+9.1f}% {method:>20}")

    # Analisis temporal: cuando hubo mas cambios
    cps["fecha_dt"] = pd.to_datetime(cps["fecha"])
    cps["year_month"] = cps["fecha_dt"].dt.to_period("M")
    cp_by_month = cps.groupby("year_month").size()
    if len(cp_by_month) > 0:
        peak_month = cp_by_month.idxmax()
        print(f"\n  Mes con mas cambios: {peak_month} ({cp_by_month.max()} cambios)")
        print(f"  → Investigar que ocurrio en la red ese mes")
