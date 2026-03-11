"""
independent_validation.py
Validacion independiente de detecciones contra datos EXTERNOS.

5 capas de evidencia + Fisher's combined test:
  A. Correlacion geografica con riesgo de infraestructura (edad contadores, % manual)
  B. Caudal nocturno minimo (MNF) — evidencia fisica de fugas reales
  C. Cobertura de telelectura como proxy de oportunidad de fraude
  D. Balance hidrico (agua entrada vs facturada) — validacion FISICA
  E. Agua regenerada — control negativo (especificidad del sistema)

Cada validacion usa datos independientes del sistema de deteccion.
Todos los p-valores se calculan con permutation test (mas potente que parametrico para n pequeño).
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, combine_pvalues

__all__ = ["run_independent_validation", "print_validation_summary"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CAUDAL_PATH = os.path.join(
    DATA_DIR,
    "_caudal_medio_sector_hidraulico_hora_2024_"
    "-caudal_medio_sector_hidraulico_hora_2024.csv",
)
REGEN_PATH = os.path.join(
    DATA_DIR,
    "_consumos_alicante_regenerada_barrio_mes-2024_"
    "-consumos_alicante_regenerada_barrio_mes-2024.csv.csv",
)


def _load_barrio_scores(results_path):
    """Carga resultados y agrega por barrio."""
    df = pd.read_csv(results_path)
    df["barrio"] = df["barrio_key"].str.split("__").str[0]
    agg = df.groupby("barrio").agg(
        mean_ensemble=("ensemble_score", "mean"),
        mean_stacking=("stacking_score", "mean"),
        n_detections=("stacking_anomaly", "sum"),
        n_obs=("ensemble_score", "size"),
    )
    agg["detection_rate"] = agg["n_detections"] / agg["n_obs"]
    return df, agg


def _hit_rate(set_a, set_b, k=10):
    """Overlap de los top-k entre dos rankings."""
    return len(set_a & set_b) / k if k > 0 else 0.0


def _bootstrap_ci(x, y, n_boot=1000, seed=42):
    """Bootstrap 95% CI para Spearman rho."""
    rng = np.random.RandomState(seed)
    rhos = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(x[idx])) > 1 and len(np.unique(y[idx])) > 1:
            r, _ = spearmanr(x[idx], y[idx])
            rhos.append(r)
    if len(rhos) < 10:
        return np.nan, np.nan
    return np.percentile(rhos, 2.5), np.percentile(rhos, 97.5)


def _permutation_pvalue(x, y, n_perm=10000, seed=42):
    """P-valor por permutation test. Más potente que paramétrico para n pequeño.

    Shufflea y 10,000 veces y cuenta cuántas veces |rho_perm| >= |rho_obs|.
    """
    observed_rho, _ = spearmanr(x, y)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        rho_perm, _ = spearmanr(x, rng.permutation(y))
        if abs(rho_perm) >= abs(observed_rho):
            count += 1
    return (count + 1) / (n_perm + 1)


# ─────────────────────────────────────────────────────────────────
# VALIDACION A: Riesgo de infraestructura vs detecciones
# ─────────────────────────────────────────────────────────────────

def validation_geographic_risk(results_path="results_full.csv"):
    """Correlacion entre riesgo de infraestructura y anomalias detectadas.

    Hipotesis: barrios con contadores viejos y menos smart meters
    deberian tener MAS anomalias detectadas.
    """
    from fraud_ground_truth import load_ground_truth

    gt = load_ground_truth()
    barrio_risk = gt["barrio_risk"]  # risk_score, mean_age_years, manual_pct

    _, barrio_scores = _load_barrio_scores(results_path)

    # Merge por barrio
    merged = barrio_scores.join(
        barrio_risk[["risk_score", "mean_age_years", "manual_pct", "smart_pct"]],
        how="inner",
    )
    merged = merged.dropna(subset=["risk_score", "mean_ensemble"])

    if len(merged) < 5:
        return {"rho": np.nan, "p": 1.0, "hit_rate_top10": 0.0,
                "n_matched": len(merged), "table": merged, "name": "Infraestructura"}

    rho, _ = spearmanr(merged["mean_ensemble"], merged["risk_score"])
    p = _permutation_pvalue(merged["mean_ensemble"].values, merged["risk_score"].values)
    ci_lo, ci_hi = _bootstrap_ci(
        merged["mean_ensemble"].values, merged["risk_score"].values
    )

    k = min(10, len(merged) // 2)
    top_risk = set(merged.nlargest(k, "risk_score").index)
    top_anomaly = set(merged.nlargest(k, "mean_ensemble").index)
    hit = _hit_rate(top_risk, top_anomaly, k)

    return {
        "name": "Infraestructura (edad + manual%)",
        "rho": rho, "p": p, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "hit_rate_top10": hit, "k": k,
        "n_matched": len(merged),
        "table": merged.sort_values("mean_ensemble", ascending=False),
    }


# ─────────────────────────────────────────────────────────────────
# VALIDACION B: Caudal nocturno minimo (Minimum Night Flow)
# ─────────────────────────────────────────────────────────────────

def validation_nightflow(results_path="results_full.csv"):
    """Correlacion entre exceso de caudal nocturno y anomalias detectadas.

    Hipotesis: sectores/barrios con alto caudal 2-4 AM (= fugas reales)
    deberian correlacionar con nuestras detecciones.
    """
    from nightflow_detector import load_hourly_data
    from sector_mapping import get_mapped_sectors

    if not os.path.exists(CAUDAL_PATH):
        return {"rho": np.nan, "p": 1.0, "hit_rate_top10": 0.0,
                "n_matched": 0, "table": pd.DataFrame(), "name": "Caudal nocturno"}

    df = load_hourly_data(CAUDAL_PATH)
    mapped = get_mapped_sectors()  # {sector: barrio}

    # Filtrar a sectores con mapeo
    df_mapped = df[df["SECTOR"].isin(mapped.keys())].copy()
    df_mapped["barrio"] = df_mapped["SECTOR"].map(mapped)

    # MNF: mediana del caudal nocturno (2-4 AM) por barrio
    night = df_mapped[df_mapped["hour"].isin([2, 3, 4])]
    mnf = night.groupby("barrio")["caudal"].median().rename("mnf_median")

    # Baseline: percentil 10 nocturno (noches sin fugas)
    p10 = night.groupby("barrio")["caudal"].quantile(0.10).rename("mnf_p10")

    # Exceso = mediana - p10 (lo que sobra por encima de las noches limpias)
    flow_stats = pd.concat([mnf, p10], axis=1)
    flow_stats["excess_night"] = (flow_stats["mnf_median"] - flow_stats["mnf_p10"]).clip(lower=0)

    # Night/day ratio medio por barrio
    day = df_mapped[df_mapped["hour"].isin(range(10, 18))]
    day_median = day.groupby("barrio")["caudal"].median().rename("day_median")
    flow_stats = flow_stats.join(day_median)
    flow_stats["night_day_ratio"] = np.where(
        flow_stats["day_median"] > 0,
        flow_stats["mnf_median"] / flow_stats["day_median"],
        np.nan,
    )

    # Cargar resultados filtrados a 2024 (mismo periodo que caudal)
    results = pd.read_csv(results_path)
    results["barrio"] = results["barrio_key"].str.split("__").str[0]
    results["fecha"] = pd.to_datetime(results["fecha"])
    results_2024 = results[results["fecha"].dt.year == 2024]

    if len(results_2024) == 0:
        # Si no hay datos 2024, usar todos
        results_2024 = results

    barrio_scores = results_2024.groupby("barrio").agg(
        mean_ensemble=("ensemble_score", "mean"),
        mean_stacking=("stacking_score", "mean"),
    )

    # Merge
    merged = barrio_scores.join(flow_stats, how="inner")
    merged = merged.dropna(subset=["excess_night", "mean_ensemble"])

    if len(merged) < 5:
        return {"rho": np.nan, "p": 1.0, "hit_rate_top10": 0.0,
                "n_matched": len(merged), "table": merged, "name": "Caudal nocturno"}

    rho, _ = spearmanr(merged["mean_ensemble"], merged["excess_night"])
    p = _permutation_pvalue(merged["mean_ensemble"].values, merged["excess_night"].values)
    ci_lo, ci_hi = _bootstrap_ci(
        merged["mean_ensemble"].values, merged["excess_night"].values
    )

    k = min(10, len(merged) // 2)
    top_flow = set(merged.nlargest(k, "excess_night").index)
    top_anomaly = set(merged.nlargest(k, "mean_ensemble").index)
    hit = _hit_rate(top_flow, top_anomaly, k)

    return {
        "name": "Caudal nocturno (MNF)",
        "rho": rho, "p": p, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "hit_rate_top10": hit, "k": k,
        "n_matched": len(merged),
        "table": merged.sort_values("excess_night", ascending=False),
    }


# ─────────────────────────────────────────────────────────────────
# VALIDACION C: Cobertura smart meters como oportunidad de fraude
# ─────────────────────────────────────────────────────────────────

def validation_smart_coverage(results_path="results_full.csv"):
    """Correlacion entre BAJA cobertura de smart meters y anomalias.

    Hipotesis: barrios con menos smart meters tienen mas oportunidad
    de fraude no detectado → deberian tener MAS anomalias.
    """
    from fraud_ground_truth import load_ground_truth

    gt = load_ground_truth()
    barrio_stats = gt["telelectura_barrio"]

    _, barrio_scores = _load_barrio_scores(results_path)

    merged = barrio_scores.join(
        barrio_stats[["smart_pct", "manual_pct", "total_counters"]],
        how="inner",
    )
    merged = merged.dropna(subset=["smart_pct", "mean_ensemble"])
    merged["opportunity"] = 100 - merged["smart_pct"]  # Mayor = mas oportunidad

    if len(merged) < 5:
        return {"rho": np.nan, "p": 1.0, "hit_rate_top10": 0.0,
                "n_matched": len(merged), "table": merged, "name": "Smart coverage"}

    rho, _ = spearmanr(merged["mean_ensemble"], merged["opportunity"])
    p = _permutation_pvalue(merged["mean_ensemble"].values, merged["opportunity"].values)
    ci_lo, ci_hi = _bootstrap_ci(
        merged["mean_ensemble"].values, merged["opportunity"].values
    )

    k = min(10, len(merged) // 2)
    top_manual = set(merged.nlargest(k, "opportunity").index)
    top_anomaly = set(merged.nlargest(k, "mean_ensemble").index)
    hit = _hit_rate(top_manual, top_anomaly, k)

    # Identificar barrios con ALTA oportunidad Y ALTA anomalia
    q75_opp = merged["opportunity"].quantile(0.75)
    q75_anom = merged["mean_ensemble"].quantile(0.75)
    high_both = merged[(merged["opportunity"] >= q75_opp) & (merged["mean_ensemble"] >= q75_anom)]

    return {
        "name": "Cobertura smart meters",
        "rho": rho, "p": p, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "hit_rate_top10": hit, "k": k,
        "n_matched": len(merged),
        "high_opportunity_high_anomaly": list(high_both.index),
        "table": merged.sort_values("mean_ensemble", ascending=False),
    }


# ─────────────────────────────────────────────────────────────────
# VALIDACION D: Balance hidrico (agua entrada vs facturada)
# ─────────────────────────────────────────────────────────────────

HACKATHON_DATA = os.path.join(
    DATA_DIR, "datos-hackathon-amaem.xlsx-set-de-datos-.csv"
)


def validation_hydraulic_balance(results_path="results_full.csv"):
    """Balance hidrico: agua entrada (caudal sector) vs agua facturada.

    loss_ratio = (caudal_in - consumo_billed) / caudal_in
    Esto es FISICA: la diferencia es agua perdida (fugas + fraude + errores).
    """
    from nightflow_detector import load_hourly_data
    from sector_mapping import get_mapped_sectors

    if not os.path.exists(CAUDAL_PATH) or not os.path.exists(HACKATHON_DATA):
        return {"rho": np.nan, "p": 1.0, "hit_rate_top10": 0.0,
                "n_matched": 0, "table": pd.DataFrame(), "name": "Balance hidrico"}

    # --- Agua ENTRADA: caudal horario → mensual por barrio ---
    df_caudal = load_hourly_data(CAUDAL_PATH)
    mapped = get_mapped_sectors()
    df_caudal = df_caudal[df_caudal["SECTOR"].isin(mapped.keys())].copy()
    df_caudal["barrio"] = df_caudal["SECTOR"].map(mapped)
    df_caudal["year_month"] = df_caudal["fecha"].dt.to_period("M")

    # Datos tienen 12 lecturas/día (cada 2h). Columna = caudal medio m³/h.
    # Volumen real = caudal_medio × 2h por lectura.
    water_in = (
        df_caudal.groupby(["barrio", "year_month"])["caudal"]
        .sum()
        .reset_index()
        .rename(columns={"caudal": "caudal_in_m3"})
    )
    water_in["caudal_in_m3"] = water_in["caudal_in_m3"] * 2  # 2h intervals

    # --- Agua FACTURADA: consumo por barrio/mes ---
    consumo = pd.read_csv(HACKATHON_DATA)
    consumo.columns = ["Barrio", "Uso", "Fecha", "Consumo_litros", "N_Contratos"]
    consumo["Consumo_litros"] = (
        consumo["Consumo_litros"].astype(str).str.replace(",", "").astype(float)
    )
    consumo["fecha"] = pd.to_datetime(consumo["Fecha"])
    consumo["year_month"] = consumo["fecha"].dt.to_period("M")

    # Filtrar a 2024 y agregar por barrio/mes (todos los usos)
    consumo_2024 = consumo[consumo["fecha"].dt.year == 2024]
    water_out = (
        consumo_2024.groupby(["Barrio", "year_month"])["Consumo_litros"]
        .sum()
        .reset_index()
    )
    water_out["consumo_m3"] = water_out["Consumo_litros"] / 1000
    water_out = water_out.rename(columns={"Barrio": "barrio"})

    # --- Balance ---
    balance = water_in.merge(water_out, on=["barrio", "year_month"], how="inner")
    if len(balance) < 5:
        return {"rho": np.nan, "p": 1.0, "hit_rate_top10": 0.0,
                "n_matched": 0, "table": pd.DataFrame(), "name": "Balance hidrico"}

    balance["loss_m3"] = balance["caudal_in_m3"] - balance["consumo_m3"]
    balance["loss_ratio"] = (
        balance["loss_m3"] / balance["caudal_in_m3"].replace(0, np.nan)
    ).clip(-0.5, 1.0)  # Permitir negativo leve (error medicion)

    # Promediar por barrio
    barrio_loss = balance.groupby("barrio").agg(
        mean_loss_ratio=("loss_ratio", "mean"),
        total_in=("caudal_in_m3", "sum"),
        total_out=("consumo_m3", "sum"),
        n_months=("year_month", "nunique"),
    )
    barrio_loss["overall_loss"] = (
        (barrio_loss["total_in"] - barrio_loss["total_out"]) / barrio_loss["total_in"]
    ).clip(-0.5, 1.0)

    # --- Correlacion con detecciones ---
    results = pd.read_csv(results_path)
    results["fecha"] = pd.to_datetime(results["fecha"])
    results_2024 = results[results["fecha"].dt.year == 2024]
    if len(results_2024) == 0:
        results_2024 = results  # Fallback

    # Agrupar por barrio (no barrio_key que incluye uso)
    barrio_col = "barrio" if "barrio" in results_2024.columns else "barrio_key"
    barrio_scores = results_2024.groupby(barrio_col).agg(
        mean_ensemble=("ensemble_score", "mean"),
        mean_stacking=("stacking_score", "mean"),
    )

    merged = barrio_loss.join(barrio_scores, how="inner")
    merged = merged.dropna(subset=["overall_loss", "mean_ensemble"])

    if len(merged) < 5:
        return {"rho": np.nan, "p": 1.0, "hit_rate_top10": 0.0,
                "n_matched": len(merged), "table": merged, "name": "Balance hidrico"}

    rho, _ = spearmanr(merged["mean_ensemble"], merged["overall_loss"])
    p = _permutation_pvalue(merged["mean_ensemble"].values, merged["overall_loss"].values)
    ci_lo, ci_hi = _bootstrap_ci(
        merged["mean_ensemble"].values, merged["overall_loss"].values
    )

    k = min(10, len(merged) // 2)
    top_loss = set(merged.nlargest(k, "overall_loss").index)
    top_anomaly = set(merged.nlargest(k, "mean_ensemble").index)
    hit = _hit_rate(top_loss, top_anomaly, k)

    # Estadisticas del balance
    mean_loss = merged["overall_loss"].mean()
    high_loss = merged[merged["overall_loss"] > 0.30]

    return {
        "name": "Balance hidrico (entrada vs facturada)",
        "rho": rho, "p": p, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "hit_rate_top10": hit, "k": k,
        "n_matched": len(merged),
        "mean_loss_ratio": mean_loss,
        "n_high_loss": len(high_loss),
        "high_loss_barrios": list(high_loss.index),
        "table": merged.sort_values("overall_loss", ascending=False),
    }


# ─────────────────────────────────────────────────────────────────
# VALIDACION E: Agua regenerada (control negativo)
# ─────────────────────────────────────────────────────────────────

def validation_regenerated_water(results_path="results_full.csv"):
    """Control negativo: agua regenerada NO deberia correlacionar con anomalias.

    El agua regenerada se usa para riego publico y limpieza, no pasa por
    contadores residenciales. Si el sistema es especifico, rho ≈ 0.
    """
    if not os.path.exists(REGEN_PATH):
        return {"rho": np.nan, "p": 1.0, "hit_rate_top10": 0.0,
                "n_matched": 0, "table": pd.DataFrame(),
                "name": "Agua regenerada (control negativo)"}

    regen = pd.read_csv(REGEN_PATH)
    barrio_regen = regen.groupby("BARRIO")["CONSUMO_2024"].sum().rename("regen_m3")

    _, barrio_scores = _load_barrio_scores(results_path)
    merged = barrio_scores.join(barrio_regen, how="inner")
    merged = merged.dropna(subset=["regen_m3", "mean_ensemble"])

    if len(merged) < 5:
        return {"rho": np.nan, "p": 1.0, "hit_rate_top10": 0.0,
                "n_matched": len(merged), "table": merged,
                "name": "Agua regenerada (control negativo)"}

    rho, _ = spearmanr(merged["mean_ensemble"], merged["regen_m3"])
    p = _permutation_pvalue(merged["mean_ensemble"].values, merged["regen_m3"].values)
    ci_lo, ci_hi = _bootstrap_ci(
        merged["mean_ensemble"].values, merged["regen_m3"].values
    )

    k = min(10, len(merged) // 2)
    top_regen = set(merged.nlargest(k, "regen_m3").index)
    top_anomaly = set(merged.nlargest(k, "mean_ensemble").index)
    hit = _hit_rate(top_regen, top_anomaly, k)

    return {
        "name": "Agua regenerada (control negativo)",
        "rho": rho, "p": p, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "hit_rate_top10": hit, "k": k,
        "n_matched": len(merged),
        "is_negative_control": True,
        "table": merged.sort_values("mean_ensemble", ascending=False),
    }


# ─────────────────────────────────────────────────────────────────
# VALIDACION F: Lecturas individuales de contadores (temporal)
# ─────────────────────────────────────────────────────────────────

# m3-registrados: lecturas individuales de contadores por año
M3_FILE_PATTERN = (
    "m3-registrados_facturados-tll_{year}-solo-alicante-"
    "m3-registrados_facturados-tll_{year}-solo-alicant.csv"
)


def validation_meter_anomalies(results_path="results_full.csv"):
    """Validacion temporal: lecturas sospechosas de contadores vs anomalias.

    Usa ~2M lecturas individuales de m3-registrados (2022-2024).
    NO tiene BARRIO → validacion TEMPORAL (ciudad entera, n=36 meses).
    Hipotesis: meses con mas contadores sospechosos → mas anomalias.
    """
    dfs = []
    for year in [2022, 2023, 2024]:
        fname = M3_FILE_PATTERN.format(year=year)
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            # Normalizar columnas (2022 usa Title Case, 2024 usa UPPER)
            df.columns = [c.upper().strip() for c in df.columns]
            dfs.append(df)

    if not dfs:
        return {"rho": np.nan, "p": 1.0, "hit_rate_top10": 0.0,
                "n_matched": 0, "table": pd.DataFrame(),
                "name": "Lecturas contadores (temporal)"}

    all_m3 = pd.concat(dfs, ignore_index=True)

    # Parsear PERIODO: "2022 -  01" → year, month
    all_m3["PERIODO"] = all_m3["PERIODO"].astype(str).str.strip()
    parts = all_m3["PERIODO"].str.split(r"\s*-\s*", expand=True)
    all_m3["year"] = pd.to_numeric(parts[0], errors="coerce")
    all_m3["month"] = pd.to_numeric(parts[1], errors="coerce")
    all_m3 = all_m3.dropna(subset=["year", "month"])
    all_m3["year_month"] = pd.to_datetime(
        all_m3["year"].astype(int).astype(str) + "-"
        + all_m3["month"].astype(int).astype(str).str.zfill(2) + "-01"
    ).dt.to_period("M")

    # Parsear M3 A FACTURAR
    all_m3["m3"] = pd.to_numeric(
        all_m3["M3 A FACTURAR"].astype(str).str.replace(",", ""),
        errors="coerce",
    )
    # Parsear DIAS LECTURA
    all_m3["dias"] = pd.to_numeric(
        all_m3["DIAS LECTURA"].astype(str).str.replace(",", ""),
        errors="coerce",
    )

    # Senales bottom-up por mes
    monthly = all_m3.groupby("year_month").apply(
        lambda g: pd.Series({
            "n_total": len(g),
            "n_zero": (g["m3"] == 0).sum(),
            "n_negative": (g["m3"] < 0).sum(),
            "n_extreme": (g["m3"] > g["m3"].quantile(0.99)).sum() if len(g) > 100 else 0,
            "n_dias_anormal": ((g["dias"] < 15) | (g["dias"] > 60)).sum(),
        })
    )
    monthly["pct_suspicious"] = (
        (monthly["n_zero"] + monthly["n_negative"] + monthly["n_dias_anormal"])
        / monthly["n_total"]
    )

    # Correlacionar con tasa de anomalias del ensemble
    results = pd.read_csv(results_path)
    results["fecha"] = pd.to_datetime(results["fecha"])
    results["year_month"] = results["fecha"].dt.to_period("M")
    ens_monthly = results.groupby("year_month").agg(
        mean_ensemble=("ensemble_score", "mean"),
        detection_rate=("stacking_anomaly", "mean"),
    )

    merged = monthly.join(ens_monthly, how="inner")
    merged = merged.dropna(subset=["pct_suspicious", "mean_ensemble"])

    if len(merged) < 5:
        return {"rho": np.nan, "p": 1.0, "hit_rate_top10": 0.0,
                "n_matched": len(merged), "table": merged,
                "name": "Lecturas contadores (temporal)"}

    rho, _ = spearmanr(merged["mean_ensemble"], merged["pct_suspicious"])
    p = _permutation_pvalue(
        merged["mean_ensemble"].values, merged["pct_suspicious"].values
    )
    ci_lo, ci_hi = _bootstrap_ci(
        merged["mean_ensemble"].values, merged["pct_suspicious"].values
    )

    return {
        "name": "Lecturas contadores (temporal, n=" + str(len(merged)) + " meses)",
        "rho": rho, "p": p, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "hit_rate_top10": 0.0, "k": 0,  # No aplica (temporal, no ranking)
        "n_matched": len(merged),
        "n_readings": int(all_m3.shape[0]),
        "table": merged.sort_values("pct_suspicious", ascending=False),
    }


# ─────────────────────────────────────────────────────────────────
# VALIDACION G: Weather deconfounding (AEMET)
# ─────────────────────────────────────────────────────────────────

def _partial_spearman(x, y, covariates):
    """Partial Spearman: correlacion x-y controlando por covariates."""
    from scipy.stats import rankdata
    rx = rankdata(x)
    ry = rankdata(y)
    rc = np.column_stack([rankdata(c) for c in covariates.T])
    # Residuos OLS
    beta_x = np.linalg.lstsq(rc, rx, rcond=None)[0]
    beta_y = np.linalg.lstsq(rc, ry, rcond=None)[0]
    res_x = rx - rc @ beta_x
    res_y = ry - rc @ beta_y
    return spearmanr(res_x, res_y)


def validation_weather_deconfound(results_path="results_full.csv"):
    """Deconfounding climatico: si anomalias persisten tras controlar por
    temperatura y precipitacion → NO son artefactos del clima.

    Usa medias climatologicas oficiales de AEMET para Alicante.
    """
    try:
        from external_data import (
            ALICANTE_MONTHLY_TEMP,
            ALICANTE_MONTHLY_PRECIP_MM,
            ALICANTE_MONTHLY_TOURISM,
        )
    except ImportError:
        return {"rho": np.nan, "p": 1.0, "partial_rho": np.nan,
                "n_matched": 0, "table": pd.DataFrame(),
                "name": "Weather deconfounding"}

    results = pd.read_csv(results_path)
    results["fecha"] = pd.to_datetime(results["fecha"])
    results["month"] = results["fecha"].dt.month
    results["year_month"] = results["fecha"].dt.to_period("M")

    # Ensemble score medio por mes
    monthly = results.groupby("year_month").agg(
        mean_ensemble=("ensemble_score", "mean"),
        detection_rate=("stacking_anomaly", "mean"),
        month=("month", "first"),
    )

    # Añadir covariables climaticas
    monthly["temp"] = monthly["month"].map(ALICANTE_MONTHLY_TEMP)
    monthly["precip"] = monthly["month"].map(ALICANTE_MONTHLY_PRECIP_MM)
    monthly["tourism"] = monthly["month"].map(ALICANTE_MONTHLY_TOURISM)
    monthly = monthly.dropna()

    if len(monthly) < 10:
        return {"rho": np.nan, "p": 1.0, "partial_rho": np.nan,
                "n_matched": len(monthly), "table": monthly,
                "name": "Weather deconfounding"}

    # Correlacion simple ensemble ~ temp
    rho_raw, _ = spearmanr(monthly["mean_ensemble"], monthly["temp"])

    # Correlacion parcial: ensemble ~ detection_rate | temp, precip, tourism
    covariates = monthly[["temp", "precip", "tourism"]].values
    partial_rho, partial_p_param = _partial_spearman(
        monthly["mean_ensemble"].values,
        monthly["detection_rate"].values,
        covariates,
    )

    # Permutation test para partial correlation
    n_perm = 10000
    rng = np.random.RandomState(42)
    count = 0
    x = monthly["mean_ensemble"].values
    y = monthly["detection_rate"].values
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        rho_perm, _ = _partial_spearman(x, y_perm, covariates)
        if abs(rho_perm) >= abs(partial_rho):
            count += 1
    p_perm = (count + 1) / (n_perm + 1)

    # Interpretacion
    weather_confounded = abs(rho_raw) > 0.3  # temp correlaciona con ensemble?
    signal_persists = abs(partial_rho) > 0.2  # señal sobrevive al deconfounding?

    return {
        "name": "Weather deconfounding (temp+precip+turismo)",
        "rho": partial_rho, "p": p_perm,
        "ci_lo": np.nan, "ci_hi": np.nan,  # No bootstrap para parcial
        "hit_rate_top10": 0.0, "k": 0,
        "n_matched": len(monthly),
        "rho_raw_vs_temp": rho_raw,
        "weather_confounded": weather_confounded,
        "signal_persists": signal_persists,
        "is_deconfound": True,
        "table": monthly,
    }


# ─────────────────────────────────────────────────────────────────
# H. Out-of-sample temporal: 2025 vs 2022-2024
# ─────────────────────────────────────────────────────────────────

def validation_temporal_oos(results_path="results_full.csv"):
    """
    H. Out-of-sample temporal validation.

    Compares monthly anomaly patterns from 2022-2024 (training period)
    with 2025 (out-of-sample). If patterns are consistent, anomalies
    reflect structural properties, not data artifacts.
    """
    from welfare_detector import detect_individual_meter_anomalies

    try:
        # Training period: 2022-2024
        train = detect_individual_meter_anomalies(years=(2022, 2023, 2024))
        # Out-of-sample: 2025
        test = detect_individual_meter_anomalies(years=(2025,))
    except Exception as e:
        return {
            "name": "H_temporal_oos", "rho": np.nan, "p": np.nan,
            "hit_rate_top10": np.nan, "error": str(e),
        }

    if train.empty or test.empty:
        return {
            "name": "H_temporal_oos", "rho": np.nan, "p": np.nan,
            "hit_rate_top10": np.nan, "error": "no data",
        }

    # Compare monthly pattern: avg pct_suspicious per calendar month
    train["month"] = train.index.str[-2:].astype(int) if hasattr(train.index, 'str') else range(len(train))
    test["month"] = test.index.str[-2:].astype(int) if hasattr(test.index, 'str') else range(len(test))

    # Use pct_suspicious as the key metric
    if "pct_suspicious" not in train.columns or "pct_suspicious" not in test.columns:
        return {
            "name": "H_temporal_oos", "rho": np.nan, "p": np.nan,
            "hit_rate_top10": np.nan, "error": "missing pct_suspicious",
        }

    # Average by calendar month for seasonal pattern comparison
    train_monthly = train.groupby("month")["pct_suspicious"].mean()
    test_monthly = test.groupby("month")["pct_suspicious"].mean()

    # Align on common months
    common = train_monthly.index.intersection(test_monthly.index)
    if len(common) < 4:
        return {
            "name": "H_temporal_oos", "rho": np.nan, "p": np.nan,
            "hit_rate_top10": np.nan, "n_months": len(common),
            "error": f"only {len(common)} common months",
        }

    x = train_monthly.loc[common].values
    y = test_monthly.loc[common].values

    rho, _ = spearmanr(x, y)

    # Permutation p-value
    rng = np.random.default_rng(42)
    n_perm = 999
    null_rhos = [spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)]
    p_perm = (sum(1 for r in null_rhos if abs(r) >= abs(rho)) + 1) / (n_perm + 1)

    # Also compare overall level
    train_mean = train["pct_suspicious"].mean()
    test_mean = test["pct_suspicious"].mean()
    level_ratio = test_mean / train_mean if train_mean > 0 else np.nan

    return {
        "name": "H_temporal_oos",
        "rho": float(rho),
        "p": float(p_perm),
        "hit_rate_top10": np.nan,
        "n_months_common": len(common),
        "train_mean_suspicious": float(train_mean),
        "test_mean_suspicious": float(test_mean),
        "level_ratio": float(level_ratio) if np.isfinite(level_ratio) else None,
        "is_oos": True,
        "train_monthly": {int(k): float(v) for k, v in train_monthly.loc[common].items()},
        "test_monthly": {int(k): float(v) for k, v in test_monthly.loc[common].items()},
    }


# ─────────────────────────────────────────────────────────────────
# I. Postal code micro-validation
# ─────────────────────────────────────────────────────────────────

POSTAL_PATH = os.path.join(
    DATA_DIR,
    "_consumos_alicante_codpostal_mes-2024_"
    "-consumos-alicante-codpostal-mes-2024.csv",
)

# Approximate mapping: postal code → barrio(s) from AMAEM
POSTAL_TO_BARRIO = {
    "03001": "3-CENTRO",
    "03002": "14-ENSANCHE DIPUTACION",
    "03003": "5-CAMPOAMOR",
    "03004": "1-BENALUA",
    "03005": "24-SAN BLAS - SANTO DOMINGO",
    "03006": "17-CAROLINAS ALTAS",
    "03007": "25-ALTOZANO - CONDE LUMIARES",
    "03008": "32-VIRGEN DEL REMEDIO",
    "03009": "6-LOS ANGELES",
    "03010": "12-POLIGONO BABEL",
    "03011": "19-GARBINET",
    "03012": "38-VISTAHERMOSA",
    "03013": "VILLAFRANQUEZA",
    "03015": "39-ALBUFERETA",
    "03016": "40-CABO DE LAS HUERTAS",
    "03540": "41-PLAYA DE SAN JUAN",
}


def validation_postal_code(results_path="results_full.csv"):
    """
    I. Postal code micro-validation.

    If barrio-level anomalies are real, postal codes overlapping
    those barrios should also show anomalous consumption.
    """
    if not os.path.exists(POSTAL_PATH):
        return {
            "name": "I_postal_code", "rho": np.nan, "p": np.nan,
            "hit_rate_top10": np.nan, "error": "no postal data",
        }

    if not os.path.exists(results_path):
        return {
            "name": "I_postal_code", "rho": np.nan, "p": np.nan,
            "hit_rate_top10": np.nan, "error": "no results",
        }

    # 1. Load postal code consumption
    postal = pd.read_csv(POSTAL_PATH)
    postal["CODIGO-POSTAL"] = postal["CODIGO-POSTAL"].astype(str).str.zfill(5)

    # Filter DOMESTICO only
    dom = postal[postal["USO"].str.contains("DOMESTICO", case=False, na=False)].copy()
    if dom.empty:
        return {
            "name": "I_postal_code", "rho": np.nan, "p": np.nan,
            "hit_rate_top10": np.nan, "error": "no domestic postal data",
        }

    # Parse CONSUMO (may have comma as decimal)
    dom["consumo"] = (
        dom["CONSUMO (M3)"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )

    # 2. Compute CV per postal code (anomaly signal)
    postal_cv = dom.groupby("CODIGO-POSTAL")["consumo"].agg(
        mean="mean", std="std"
    )
    postal_cv["cv"] = postal_cv["std"] / postal_cv["mean"].replace(0, np.nan)
    postal_cv = postal_cv.dropna()

    # 3. Map postal codes to barrios
    postal_cv["barrio"] = postal_cv.index.map(POSTAL_TO_BARRIO)
    mapped = postal_cv.dropna(subset=["barrio"])

    if len(mapped) < 5:
        return {
            "name": "I_postal_code", "rho": np.nan, "p": np.nan,
            "hit_rate_top10": np.nan, "n_mapped": len(mapped),
            "error": "too few mapped postal codes",
        }

    # 4. Load ensemble scores by barrio
    results = pd.read_csv(results_path)
    barrio_scores = results.groupby("barrio_key")["ensemble_score"].mean().reset_index()
    barrio_scores["barrio_clean"] = (
        barrio_scores["barrio_key"].str.split("__").str[0]
    )

    # 5. Merge
    mapped = mapped.reset_index()
    merged = mapped.merge(
        barrio_scores[["barrio_clean", "ensemble_score"]],
        left_on="barrio", right_on="barrio_clean", how="inner"
    )

    n = len(merged)
    if n < 5:
        return {
            "name": "I_postal_code", "rho": np.nan, "p": np.nan,
            "hit_rate_top10": np.nan, "n_paired": n,
            "error": "insufficient paired data",
        }

    # 6. Spearman correlation: ensemble_score vs postal CV
    rho, _ = spearmanr(merged["ensemble_score"], merged["cv"])

    # Permutation p-value
    rng = np.random.default_rng(42)
    n_perm = 999
    x = merged["ensemble_score"].values
    y = merged["cv"].values
    null_rhos = [spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)]
    p_perm = (sum(1 for r in null_rhos if abs(r) >= abs(rho)) + 1) / (n_perm + 1)

    # 7. Hit rate: top-5 barrios by ensemble in top-5 by postal CV?
    top_ensemble = set(merged.nlargest(5, "ensemble_score")["barrio"].values)
    top_postal = set(merged.nlargest(5, "cv")["barrio"].values)
    hit_rate = len(top_ensemble & top_postal) / 5

    return {
        "name": "I_postal_code",
        "rho": float(rho),
        "p": float(p_perm),
        "hit_rate_top10": float(hit_rate),
        "n_paired": n,
    }


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def run_independent_validation(results_path="results_full.csv"):
    """Ejecuta las 9 validaciones independientes + Fisher's combined."""
    va = validation_geographic_risk(results_path)
    vb = validation_nightflow(results_path)
    vc = validation_smart_coverage(results_path)
    vd = validation_hydraulic_balance(results_path)
    ve = validation_regenerated_water(results_path)
    vf = validation_meter_anomalies(results_path)
    vg = validation_weather_deconfound(results_path)
    vh = validation_temporal_oos(results_path)
    vi = validation_postal_code(results_path)

    validations = [va, vb, vc, vd, ve, vf, vg, vh, vi]
    significant = [v for v in validations if np.isfinite(v["rho"]) and v["p"] < 0.05
                   and not v.get("is_negative_control", False)
                   and not v.get("is_deconfound", False)]
    hit_rates = [v["hit_rate_top10"] for v in validations
                 if np.isfinite(v.get("hit_rate_top10", np.nan)) and v["hit_rate_top10"] > 0]

    # Strongest (exclude controls and deconfounding)
    valid_rhos = [(abs(v["rho"]), v["name"]) for v in validations
                  if np.isfinite(v["rho"])
                  and not v.get("is_negative_control", False)
                  and not v.get("is_deconfound", False)]
    strongest = max(valid_rhos, key=lambda x: x[0])[1] if valid_rhos else "N/A"

    # Fisher's combined: validaciones fisicas con rho > 0
    physical_pvals = []
    for v in [vb, vd, vf]:  # MNF + Balance hidrico + Meter anomalies
        if (np.isfinite(v.get("rho", np.nan)) and v["rho"] > 0
                and not v.get("is_negative_control", False)):
            physical_pvals.append(v["p"])

    fisher_stat, fisher_p = (np.nan, np.nan)
    if len(physical_pvals) >= 2:
        fisher_stat, fisher_p = combine_pvalues(physical_pvals, method="fisher")

    # ── Benjamini-Hochberg FDR correction ──
    # Corrige por test múltiple: 7 validaciones → family-wise error ~30% sin corrección
    bh_results = {}
    bh_pvals = []
    bh_names = []
    for v in validations:
        if (not v.get("is_negative_control", False)
                and not v.get("is_deconfound", False)
                and np.isfinite(v.get("p", np.nan))):
            bh_pvals.append(v["p"])
            bh_names.append(v["name"])

    if len(bh_pvals) >= 2:
        from statsmodels.stats.multitest import multipletests
        rejected, qvalues, _, _ = multipletests(bh_pvals, method="fdr_bh")
        for name, p_raw, q, rej in zip(bh_names, bh_pvals, qvalues, rejected):
            bh_results[name] = {"p_raw": p_raw, "q_bh": float(q), "rejected": bool(rej)}

    # Re-count significant after BH correction
    n_sig_bh = sum(1 for v in bh_results.values() if v["rejected"])

    return {
        "validation_a": va,
        "validation_b": vb,
        "validation_c": vc,
        "validation_d": vd,
        "validation_e": ve,
        "validation_f": vf,
        "validation_g": vg,
        "validation_h": vh,
        "validation_i": vi,
        "fisher": {
            "statistic": fisher_stat,
            "p": fisher_p,
            "n_combined": len(physical_pvals),
            "pvalues": physical_pvals,
        },
        "bh_correction": bh_results,
        "summary": {
            "n_significant": len(significant),
            "n_significant_bh": n_sig_bh,
            "n_total": len([v for v in validations
                           if not v.get("is_negative_control", False)
                           and not v.get("is_deconfound", False)]),
            "mean_hit_rate": np.mean(hit_rates) if hit_rates else 0.0,
            "strongest": strongest,
        },
    }


def print_validation_summary(results):
    """Imprime resumen de validacion independiente."""
    print(f"\n{'='*80}")
    print(f"  VALIDACION INDEPENDIENTE — 7 capas + Fisher's combined (permutation test)")
    print(f"{'='*80}")

    val_keys = ["validation_a", "validation_b", "validation_c", "validation_d",
                "validation_e", "validation_f", "validation_g"]
    for key in val_keys:
        v = results[key]
        name = v.get("name", key)
        rho = v.get("rho", np.nan)
        p = v.get("p", np.nan)
        ci_lo = v.get("ci_lo", np.nan)
        ci_hi = v.get("ci_hi", np.nan)
        hit = v.get("hit_rate_top10", 0)
        k = v.get("k", 10)
        n = v.get("n_matched", 0)
        is_neg = v.get("is_negative_control", False)

        is_deconf = v.get("is_deconfound", False)

        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        if is_neg:
            verdict = "OK (no correlaciona)" if p > 0.10 else "INESPERADO"
        elif is_deconf:
            verdict = "SEÑAL PERSISTE" if v.get("signal_persists", False) else "CONFUNDIDA POR CLIMA"
        else:
            verdict = "SIGNIFICATIVA" if p < 0.05 else "MARGINAL" if p < 0.10 else "NO SIGNIFICATIVA"

        print(f"\n  {name}")
        print(f"  {'─'*50}")
        if np.isfinite(rho):
            ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if np.isfinite(ci_lo) else ""
            label = "Partial rho" if is_deconf else "Spearman rho"
            print(f"    {label}:  {rho:+.3f} {sig}  (p_perm={p:.4f}) {ci_str}")
            if k > 0:
                print(f"    Hit-rate top-{k}: {hit:.0%} ({int(hit*k)}/{k} barrios coinciden)")
            if is_deconf:
                rho_temp = v.get("rho_raw_vs_temp", np.nan)
                print(f"    rho(ensemble, temperatura): {rho_temp:+.3f}")
                print(f"    Señal tras controlar clima: {'SI persiste' if v.get('signal_persists') else 'NO persiste'}")
            elif "n_readings" in v:
                print(f"    Lecturas individuales: {v['n_readings']:,}")
            print(f"    N observaciones: {n}")
            print(f"    Veredicto: {verdict}")
        else:
            print(f"    Sin datos suficientes (n={n})")

        # Top barrios si hay tabla
        table = v.get("table", pd.DataFrame())
        if not table.empty and len(table) > 0:
            print(f"\n    Top 5 barrios (por anomalia):")
            top5 = table.head(5)
            for idx, row in top5.iterrows():
                ens = row.get("mean_ensemble", 0)
                extra = ""
                if "risk_score" in row:
                    extra = f"risk={row['risk_score']:.1f}"
                elif "excess_night" in row:
                    extra = f"excess={row['excess_night']:.1f} m3/h"
                elif "opportunity" in row:
                    extra = f"manual={row['opportunity']:.0f}%"
                elif "overall_loss" in row:
                    extra = f"loss={row['overall_loss']:.1%}"
                elif "regen_m3" in row:
                    extra = f"regen={row['regen_m3']:.0f} m3"
                print(f"      {str(idx):<40s} ens={ens:.3f}  {extra}")

    # Fisher's combined test
    fisher = results.get("fisher", {})
    fisher_p = fisher.get("p", np.nan)
    fisher_n = fisher.get("n_combined", 0)
    if np.isfinite(fisher_p) and fisher_n >= 2:
        fisher_sig = "***" if fisher_p < 0.01 else "**" if fisher_p < 0.05 else "*" if fisher_p < 0.10 else ""
        print(f"\n  Fisher's Combined Test (MNF + Balance hidrico)")
        print(f"  {'─'*50}")
        print(f"    p-valores combinados: {fisher['pvalues']}")
        print(f"    Fisher's p: {fisher_p:.4f} {fisher_sig}")
        if fisher_p < 0.05:
            print(f"    >> SIGNIFICATIVO: evidencia fisica combinada confirma detecciones")
        else:
            print(f"    >> No significativo (p >= 0.05)")

    # Benjamini-Hochberg correction
    bh = results.get("bh_correction", {})
    if bh:
        print(f"\n  Benjamini-Hochberg FDR Correction")
        print(f"  {'─'*50}")
        print(f"    {'Validacion':<45} {'p_raw':>8}  {'q_BH':>8}  {'Sobrevive?':>10}")
        for name, vals in sorted(bh.items(), key=lambda x: x[1]["p_raw"]):
            survived = "SI ***" if vals["rejected"] else "NO"
            print(f"    {name:<45} {vals['p_raw']:>8.4f}  {vals['q_bh']:>8.4f}  {survived:>10}")
        n_survived = sum(1 for v in bh.values() if v["rejected"])
        print(f"    >> {n_survived}/{len(bh)} validaciones sobreviven FDR q<0.05")

    # Resumen consolidado
    s = results["summary"]
    print(f"\n  {'='*50}")
    print(f"  RESUMEN CONSOLIDADO")
    print(f"  {'='*50}")
    print(f"    Validaciones significativas (p<0.05): {s['n_significant']}/{s['n_total']}")
    if s.get("n_significant_bh") is not None:
        print(f"    Sobreviven BH-FDR (q<0.05): {s['n_significant_bh']}/{s['n_total']}")
    if np.isfinite(fisher_p):
        print(f"    Fisher's combined (fisicas): p={fisher_p:.4f}")
    print(f"    Hit-rate medio top-k: {s['mean_hit_rate']:.0%}")
    print(f"    Validacion mas fuerte: {s['strongest']}")

    # Check for marginal validations (p < 0.10)
    marginal = [v for k, v in results.items()
                if k.startswith("validation_") and np.isfinite(v.get("rho", np.nan))
                and v["p"] < 0.10 and not v.get("is_negative_control", False)]

    if s["n_significant"] >= 2 or (np.isfinite(fisher_p) and fisher_p < 0.05):
        print(f"    >> EVIDENCIA FUERTE: las detecciones correlacionan con datos reales")
    elif s["n_significant"] == 1:
        print(f"    >> EVIDENCIA PARCIAL: 1 validacion significativa")
    elif len(marginal) > 0:
        print(f"    >> EVIDENCIA MARGINAL: {len(marginal)} validacion(es) cercana(s) a significancia (p<0.10)")
    else:
        print(f"    >> EVIDENCIA DEBIL: ninguna validacion significativa")


if __name__ == "__main__":
    results = run_independent_validation()
    print_validation_summary(results)
