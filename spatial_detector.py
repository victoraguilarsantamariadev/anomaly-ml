"""
M11 — Detector de anomalias espaciales.

Usa la adyacencia geografica de barrios para distinguir:
  - Problema de RED: un barrio y sus vecinos tienen anomalia → probablemente
    fuga en tuberia compartida o problema de sector hidraulico.
  - Problema PUNTUAL: un barrio tiene anomalia pero sus vecinos no →
    probablemente fraude, contador roto, o problema individual.

Tambien usa las features de infraestructura (elevacion, densidad de tuberias,
red de saneamiento) como indicadores de riesgo de fuga.

Input: resultados de los otros modelos (collect_results) + adjacency + infra features.
Output: columnas adicionales con clasificacion espacial.

Uso:
  from spatial_detector import classify_spatial_anomalies, compute_infrastructure_risk
  results = classify_spatial_anomalies(results_df, adjacency)
  risk = compute_infrastructure_risk(infra_df)
"""

import numpy as np
import pandas as pd
from typing import Optional


def classify_spatial_anomalies(
    results: pd.DataFrame,
    adjacency: dict[str, list[str]],
    min_models: int = 1,
) -> pd.DataFrame:
    """
    Clasifica anomalias como CLUSTER (red) o ISOLATED (puntual).

    Para cada (barrio, fecha) con anomalia:
      - Mira cuantos vecinos tambien tienen anomalia en ese mes
      - Si >= 1 vecino anomalo → CLUSTER (posible problema de red)
      - Si 0 vecinos anomalos → ISOLATED (posible fraude/individual)

    Args:
        results: DataFrame de collect_results() con barrio_key, fecha,
                 n_models_detecting
        adjacency: dict barrio → lista de barrios vecinos
        min_models: minimo de modelos detectando para considerar anomalia

    Returns:
        DataFrame con columnas extra:
          - spatial_class: "CLUSTER", "ISOLATED", o "NORMAL"
          - n_anomalous_neighbors: cuantos vecinos tienen anomalia
          - cluster_size: tamaño del cluster (barrio + vecinos anomalos)
    """
    results = results.copy()

    # Extraer barrio limpio del barrio_key (quitar __USO)
    results["_barrio_clean"] = results["barrio_key"].str.split("__").str[0]

    # Para cada fecha, saber que barrios son anomalos
    anomalous_by_date: dict[str, set[str]] = {}
    for fecha, group in results.groupby("fecha"):
        fecha_key = str(fecha)
        anomalous = set(
            group[group["n_models_detecting"] >= min_models]["_barrio_clean"].values
        )
        anomalous_by_date[fecha_key] = anomalous

    # Clasificar cada fila
    spatial_classes = []
    n_neighbors = []
    cluster_sizes = []

    for _, row in results.iterrows():
        barrio = row["_barrio_clean"]
        fecha_key = str(row["fecha"])
        n_detecting = row.get("n_models_detecting", 0)

        if n_detecting < min_models:
            spatial_classes.append("NORMAL")
            n_neighbors.append(0)
            cluster_sizes.append(0)
            continue

        # Buscar vecinos anomalos
        neighbors = adjacency.get(barrio, [])
        anomalous_set = anomalous_by_date.get(fecha_key, set())

        anomalous_neighbors = [n for n in neighbors if n in anomalous_set]
        n_anom = len(anomalous_neighbors)

        if n_anom >= 1:
            spatial_classes.append("CLUSTER")
            n_neighbors.append(n_anom)
            cluster_sizes.append(1 + n_anom)  # barrio + vecinos
        else:
            spatial_classes.append("ISOLATED")
            n_neighbors.append(0)
            cluster_sizes.append(1)

    results["spatial_class"] = spatial_classes
    results["n_anomalous_neighbors"] = n_neighbors
    results["cluster_size"] = cluster_sizes

    results = results.drop(columns=["_barrio_clean"])
    return results


def compute_infrastructure_risk(infra_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula un indice de riesgo de fuga basado en infraestructura.

    Factores de riesgo (normalizados 0-1, luego sumados):
      - Baja densidad de tuberias: menos red = peor mantenimiento
      - Alta elevacion: mas presion = mas riesgo de rotura
      - Bajo diametro medio: tuberias estrechas se atascan/rompen mas
      - Baja densidad de hidrantes: menos puntos de control
      - Alta % red unitaria: red mixta (sanitario+pluvial) tiene mas problemas

    Output: DataFrame con barrio + infrastructure_risk_score (0-7)
    """
    if infra_df.empty:
        return pd.DataFrame(columns=["barrio", "infrastructure_risk_score"])

    df = infra_df.copy()

    # Normalizar cada factor al rango [0, 1]
    def _normalize_asc(series):
        """Mayor valor = mayor riesgo"""
        vmin, vmax = series.min(), series.max()
        if vmax == vmin:
            return pd.Series(0.5, index=series.index)
        return (series - vmin) / (vmax - vmin)

    def _normalize_desc(series):
        """Menor valor = mayor riesgo"""
        return 1.0 - _normalize_asc(series)

    risk = pd.DataFrame({"barrio": df["barrio"]})

    # Baja densidad de tuberias → mayor riesgo (menos infraestructura)
    risk["r_pipe_density"] = _normalize_desc(
        df["pipe_density_km_per_km2"].fillna(0)
    )

    # Alta elevacion → mayor riesgo (mas presion en la red)
    risk["r_elevation"] = _normalize_asc(
        df["elevation_mean"].fillna(0)
    )

    # Bajo diametro → mayor riesgo
    risk["r_diameter"] = _normalize_desc(
        df["avg_pipe_diameter_mm"].replace(0, np.nan).fillna(
            df["avg_pipe_diameter_mm"].median()
        )
    )

    # Baja densidad de hidrantes → mayor riesgo
    risk["r_hydrant"] = _normalize_desc(
        df["hydrant_density_per_km2"].fillna(0)
    )

    # Alta % unitaria → mayor riesgo (red mixta mas vieja/problematica)
    risk["r_sewer_type"] = _normalize_asc(
        df["pct_sewer_unitaria"].fillna(0.5)
    )

    # Baja densidad de imbornales → mayor riesgo (menos drenaje)
    if "imbornal_density_per_km2" in df.columns:
        risk["r_imbornal"] = _normalize_desc(
            df["imbornal_density_per_km2"].fillna(0)
        )
    else:
        risk["r_imbornal"] = 0.5

    # Baja cobertura de colectores → mayor riesgo
    if "colector_coverage_pct" in df.columns:
        risk["r_colector"] = _normalize_desc(
            df["colector_coverage_pct"].fillna(0)
        )
    else:
        risk["r_colector"] = 0.5

    # Score total (suma de 7 factores, rango 0-7)
    risk_cols = ["r_pipe_density", "r_elevation", "r_diameter",
                 "r_hydrant", "r_sewer_type", "r_imbornal", "r_colector"]
    risk["infrastructure_risk_score"] = risk[risk_cols].sum(axis=1)

    return risk[["barrio", "infrastructure_risk_score"]]


def spatial_summary(results: pd.DataFrame) -> None:
    """Imprime resumen de la clasificacion espacial."""
    if "spatial_class" not in results.columns:
        print("  No hay clasificacion espacial disponible.")
        return

    flagged = results[results["spatial_class"] != "NORMAL"]
    if len(flagged) == 0:
        print("  Sin anomalias espaciales.")
        return

    n_cluster = (flagged["spatial_class"] == "CLUSTER").sum()
    n_isolated = (flagged["spatial_class"] == "ISOLATED").sum()

    print(f"\n  {'─'*80}")
    print(f"  M11 — ANALISIS ESPACIAL")
    print(f"  {'─'*80}")
    print(f"  CLUSTER (posible problema de red): {n_cluster} alertas")
    print(f"  ISOLATED (posible fraude/puntual): {n_isolated} alertas")

    if n_cluster > 0:
        clusters = flagged[flagged["spatial_class"] == "CLUSTER"]
        print(f"\n  Clusters detectados:")
        for fecha, group in clusters.groupby("fecha"):
            fecha_str = fecha.strftime("%Y-%m") if hasattr(fecha, "strftime") else str(fecha)[:7]
            barrios = sorted(group["barrio_key"].unique())
            max_cluster = group["cluster_size"].max()
            print(f"    {fecha_str}: {len(barrios)} barrios en cluster "
                  f"(max tamaño={max_cluster})")
            for b in barrios[:5]:
                row = group[group["barrio_key"] == b].iloc[0]
                print(f"      {b}: {row['n_anomalous_neighbors']} vecinos anomalos")

    if n_isolated > 0:
        isolated = flagged[flagged["spatial_class"] == "ISOLATED"]
        print(f"\n  Anomalias aisladas (posible fraude):")
        barrio_counts = isolated.groupby("barrio_key").size().sort_values(ascending=False)
        for barrio_key, count in barrio_counts.head(10).items():
            print(f"    {barrio_key}: {count} meses aislado")


def infrastructure_risk_summary(risk_df: pd.DataFrame) -> None:
    """Imprime ranking de barrios por riesgo de infraestructura."""
    if risk_df.empty:
        return

    print(f"\n  {'─'*80}")
    print(f"  RIESGO DE INFRAESTRUCTURA POR BARRIO")
    print(f"  {'─'*80}")

    risk_sorted = risk_df.sort_values("infrastructure_risk_score", ascending=False)
    print(f"\n  {'Barrio':<35}  {'Score':>7}  {'Nivel':>8}")
    print(f"  {'─'*55}")
    for _, row in risk_sorted.head(15).iterrows():
        score = row["infrastructure_risk_score"]
        level = "ALTO" if score > 4.5 else "MEDIO" if score > 3.0 else "BAJO"
        print(f"  {row['barrio']:<35}  {score:>7.2f}  {level:>8}")


# ─────────────────────────────────────────────────────────────────
# Moran's I — Autocorrelación Espacial
# ─────────────────────────────────────────────────────────────────

def compute_morans_i(barrio_scores: "pd.Series", adjacency_dict: dict,
                     n_perm: int = 999, seed: int = 42) -> dict:
    """
    Moran's I: test de autocorrelación espacial de anomaly scores.

    I > 0 → clustering (barrios anómalos agrupados) → evidencia de causa física
    I < 0 → dispersión
    I ≈ 0 → aleatorio

    Args:
        barrio_scores: Series indexada por barrio con ensemble_score medio
        adjacency_dict: dict[barrio] → [barrios vecinos] (de gis_features.py)
        n_perm: permutaciones para p-value

    Returns:
        dict con I_observed, E_I, p_value, verdict
    """
    import numpy as np

    # Filtrar a barrios que existen en ambos datasets
    common = [b for b in barrio_scores.index if b in adjacency_dict]
    if len(common) < 5:
        return {"error": f"Solo {len(common)} barrios en comun"}

    scores = barrio_scores.loc[common]
    n = len(scores)
    x = scores.values.astype(float)
    x_mean = x.mean()
    barrio_list = list(scores.index)

    # Weight matrix
    W = np.zeros((n, n))
    for i, b in enumerate(barrio_list):
        for neighbor in adjacency_dict.get(b, []):
            if neighbor in barrio_list:
                j = barrio_list.index(neighbor)
                W[i, j] = 1.0

    W_sum = W.sum()
    if W_sum == 0:
        return {"error": "Sin conexiones espaciales entre barrios"}

    # Moran's I
    z = x - x_mean
    numerator = n * float(z @ W @ z)
    denominator = W_sum * float(z @ z)
    I_obs = numerator / denominator if denominator > 0 else 0.0

    # Expected I under H0
    E_I = -1.0 / (n - 1)

    # Permutation test
    rng = np.random.RandomState(seed)
    null_Is = np.zeros(n_perm)
    for p in range(n_perm):
        z_perm = rng.permutation(z)
        num_p = n * float(z_perm @ W @ z_perm)
        null_Is[p] = num_p / denominator if denominator > 0 else 0.0

    p_value = (np.sum(null_Is >= I_obs) + 1) / (n_perm + 1)

    verdict = "CLUSTERING" if I_obs > 0 and p_value < 0.05 else \
              "DISPERSION" if I_obs < 0 and p_value < 0.05 else "ALEATORIO"

    return {
        "I_observed": float(I_obs),
        "E_I": float(E_I),
        "p_value": float(p_value),
        "n_barrios": n,
        "n_connections": int(W_sum),
        "verdict": verdict,
    }


def print_morans_i(result: dict):
    """Print Moran's I results."""
    print(f"\n  Moran's I — Autocorrelación Espacial")
    print(f"  {'─'*50}")
    if "error" in result:
        print(f"    Error: {result['error']}")
        return

    I = result["I_observed"]
    E = result["E_I"]
    p = result["p_value"]
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

    print(f"    I observado: {I:+.4f} (esperado bajo H0: {E:+.4f})")
    print(f"    p-value (permutation): {p:.4f} {sig}")
    print(f"    Barrios: {result['n_barrios']}, Conexiones: {result['n_connections']}")
    print(f"    Veredicto: {result['verdict']}")
    if result["verdict"] == "CLUSTERING":
        print(f"    >> Las anomalías se AGRUPAN geográficamente → causa física compartida")
    elif result["verdict"] == "ALEATORIO":
        print(f"    >> Distribución espacial aleatoria (no invalida detecciones, solo no clustering)")


# ─────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from gis_features import load_infrastructure_features, compute_barrio_adjacency

    print("=" * 70)
    print("  M11 — Detector de Anomalias Espaciales")
    print("=" * 70)

    # Infrastructure risk
    infra = load_infrastructure_features("data/")
    if len(infra) > 0:
        risk = compute_infrastructure_risk(infra)
        infrastructure_risk_summary(risk)

    # Adjacency
    print("\n  Calculando adyacencia...")
    adj = compute_barrio_adjacency("data/")
    print(f"  {len(adj)} barrios con vecinos calculados")
    for b in sorted(adj.keys())[:5]:
        print(f"    {b}: {len(adj[b])} vecinos")
