"""
M16 — Topological Data Analysis (TDA): la forma del consumo importa.

Persistent Homology encuentra estructura topologica (agujeros, loops, clusters)
en los datos de consumo. Un barrio "normal" tiene una topologia suave y predecible.
Un barrio anomalo tiene agujeros, bifurcaciones, o estructura irregular.

Basado en Takens' theorem (1981) para delay embedding y
Edelsbrunner et al. (2002) para persistent homology.

Publicado en Nature, Science, PNAS para deteccion de anomalias en series temporales.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage


# ---------------------------------------------------------------------------
# 1. Time-delay embedding (Takens' theorem)
# ---------------------------------------------------------------------------

def _takens_embedding(series: np.ndarray, dim: int = 3, tau: int = 1) -> np.ndarray:
    """
    Convierte una serie temporal en una nube de puntos via delay embedding.

    Dado x(t), construye vectores:
        [x(t), x(t+tau), x(t+2*tau), ..., x(t+(dim-1)*tau)]

    Takens (1981): el embedding reconstruye la topologia del atractor subyacente
    si dim >= 2*d + 1 donde d es la dimension del atractor.

    Args:
        series: serie temporal 1D
        dim: dimension del embedding (2 o 3 tipicamente)
        tau: delay entre coordenadas

    Returns:
        Matriz (N - (dim-1)*tau, dim) — la nube de puntos
    """
    n = len(series)
    n_points = n - (dim - 1) * tau
    if n_points < 4:
        return np.empty((0, dim))

    embedding = np.zeros((n_points, dim))
    for d in range(dim):
        embedding[:, d] = series[d * tau: d * tau + n_points]

    return embedding


# ---------------------------------------------------------------------------
# 2. Simplified persistent homology (H0) via single-linkage clustering
# ---------------------------------------------------------------------------

def _compute_persistence_diagram_h0(point_cloud: np.ndarray) -> np.ndarray:
    """
    Calcula el diagrama de persistencia H0 (componentes conectados)
    usando single-linkage clustering, que es equivalente al Vietoris-Rips
    filtration para H0.

    Single linkage: al fusionar dos clusters a distancia d,
      - El cluster mas joven "muere" a epsilon = d
      - El cluster mas viejo sobrevive
    Cada punto "nace" a epsilon = 0.

    Returns:
        Array (n_features, 2) con columnas [birth, death].
        El componente que nunca muere (infinito) se excluye.
    """
    n = len(point_cloud)
    if n < 2:
        return np.empty((0, 2))

    # Pairwise distances
    dists = pdist(point_cloud)

    # Single linkage = minimal spanning tree = Rips for H0
    Z = linkage(dists, method="single")

    # Z tiene filas: [cluster_i, cluster_j, distance, size]
    # Cada merge mata una componente. Todas nacen a 0.
    # La ultima componente (la que sobrevive) tiene vida infinita — la excluimos.
    births = np.zeros(n - 1)
    deaths = Z[:, 2]  # merge distances

    # Excluir la ultima (la que viviria para siempre)
    diagram = np.column_stack([births[:-1], deaths[:-1]])

    return diagram


# ---------------------------------------------------------------------------
# 3. Persistence statistics
# ---------------------------------------------------------------------------

def _persistence_stats(diagram: np.ndarray) -> dict:
    """
    Extrae estadisticas topologicas del diagrama de persistencia.

    Args:
        diagram: array (n, 2) con [birth, death]

    Returns:
        dict con metricas topologicas
    """
    if len(diagram) == 0:
        return {
            "total_persistence": 0.0,
            "max_persistence": 0.0,
            "mean_persistence": 0.0,
            "std_persistence": 0.0,
            "persistence_entropy": 0.0,
            "n_significant": 0,
            "n_features": 0,
        }

    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes > 0]  # filtrar ceros

    if len(lifetimes) == 0:
        return {
            "total_persistence": 0.0,
            "max_persistence": 0.0,
            "mean_persistence": 0.0,
            "std_persistence": 0.0,
            "persistence_entropy": 0.0,
            "n_significant": 0,
            "n_features": 0,
        }

    total = lifetimes.sum()
    max_pers = lifetimes.max()
    mean_pers = lifetimes.mean()
    std_pers = lifetimes.std() if len(lifetimes) > 1 else 0.0

    # Persistence entropy: entropia de Shannon de las vidas normalizadas
    probs = lifetimes / total if total > 0 else np.ones(len(lifetimes)) / len(lifetimes)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))

    # Features significativos: lifetime > mediana
    median_life = np.median(lifetimes)
    n_significant = int(np.sum(lifetimes > median_life))

    return {
        "total_persistence": float(total),
        "max_persistence": float(max_pers),
        "mean_persistence": float(mean_pers),
        "std_persistence": float(std_pers),
        "persistence_entropy": float(entropy),
        "n_significant": n_significant,
        "n_features": len(lifetimes),
    }


# ---------------------------------------------------------------------------
# 4. Bottleneck / Wasserstein distance between persistence diagrams
# ---------------------------------------------------------------------------

def _wasserstein_distance_diagrams(diag_a: np.ndarray, diag_b: np.ndarray) -> float:
    """
    Distancia de Wasserstein simplificada (p=2) entre dos diagramas de persistencia.

    Aproximacion: comparamos los vectores de lifetimes ordenados.
    Si tienen distinto tamaño, paddeamos con ceros (equivalente a emparejar
    con la diagonal, que es la convencion estandar en TDA).

    La distancia Wasserstein real requiere optimal matching (Hungarian algorithm),
    pero esta aproximacion funciona bien para deteccion de anomalias.
    """
    if len(diag_a) == 0 and len(diag_b) == 0:
        return 0.0

    lifetimes_a = np.sort(diag_a[:, 1] - diag_a[:, 0])[::-1] if len(diag_a) > 0 else np.array([])
    lifetimes_b = np.sort(diag_b[:, 1] - diag_b[:, 0])[::-1] if len(diag_b) > 0 else np.array([])

    # Pad to same length with zeros (matching to diagonal)
    max_len = max(len(lifetimes_a), len(lifetimes_b))
    if max_len == 0:
        return 0.0

    padded_a = np.zeros(max_len)
    padded_b = np.zeros(max_len)
    padded_a[:len(lifetimes_a)] = lifetimes_a
    padded_b[:len(lifetimes_b)] = lifetimes_b

    # L2 Wasserstein
    return float(np.sqrt(np.sum((padded_a - padded_b) ** 2)))


# ---------------------------------------------------------------------------
# 5. Main detection pipeline
# ---------------------------------------------------------------------------

def run_tda_detection(
    df_monthly: pd.DataFrame,
    consumo_col: str = "consumption_per_contract",
    embedding_dim: int = 3,
    embedding_tau: int = 1,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Deteccion de anomalias topologicas por barrio.

    Pipeline:
      1. Para cada barrio, time-delay embedding de la serie de consumo
      2. Calcular diagrama de persistencia H0 (componentes conectados)
      3. Extraer estadisticas topologicas (total persistence, entropy, etc.)
      4. Comparar topologia de cada barrio con la topologia "mediana" (referencia)
      5. Barrios con topologia inusual → anomalos

    Args:
        df_monthly: DataFrame con barrio_key, fecha, consumo_col
        consumo_col: columna de consumo a analizar
        embedding_dim: dimension del delay embedding (2 o 3)
        embedding_tau: delay temporal
        contamination: fraccion esperada de anomalias

    Returns:
        DataFrame con columnas:
          barrio_key, tda_persistence, tda_entropy, tda_n_features,
          tda_score, is_anomaly_tda
    """
    print(f"\n  [M16] Topological Data Analysis (dim={embedding_dim}, "
          f"tau={embedding_tau}, contamination={contamination})...")

    barrios = df_monthly["barrio_key"].unique()
    barrio_stats = []
    barrio_diagrams = {}

    # --- Paso 1-3: calcular topologia por barrio ---
    n_skipped = 0
    for barrio in barrios:
        df_b = df_monthly[df_monthly["barrio_key"] == barrio].sort_values("fecha")

        if consumo_col not in df_b.columns or len(df_b) < 6:
            n_skipped += 1
            continue

        series = df_b[consumo_col].values.astype(float)

        # Normalizar la serie (para que la topologia sea comparable entre barrios)
        s_mean = np.mean(series)
        s_std = np.std(series)
        if s_std < 1e-10:
            # Serie constante — topologia trivial
            n_skipped += 1
            continue
        series_norm = (series - s_mean) / s_std

        # Time-delay embedding
        point_cloud = _takens_embedding(series_norm, dim=embedding_dim, tau=embedding_tau)
        if len(point_cloud) < 4:
            n_skipped += 1
            continue

        # Persistent homology H0
        diagram = _compute_persistence_diagram_h0(point_cloud)
        barrio_diagrams[barrio] = diagram

        # Estadisticas
        stats = _persistence_stats(diagram)
        stats["barrio_key"] = barrio
        barrio_stats.append(stats)

    if not barrio_stats:
        print("    Sin barrios con datos suficientes para TDA")
        return pd.DataFrame(columns=[
            "barrio_key", "tda_persistence", "tda_entropy",
            "tda_n_features", "tda_score", "is_anomaly_tda",
        ])

    stats_df = pd.DataFrame(barrio_stats)
    print(f"    {len(stats_df)} barrios analizados, {n_skipped} omitidos (datos insuficientes)")

    # --- Paso 4: construir diagrama "referencia" (mediana) ---
    # Usamos el barrio cuya total_persistence esta mas cerca de la mediana
    median_persistence = stats_df["total_persistence"].median()
    ref_idx = (stats_df["total_persistence"] - median_persistence).abs().idxmin()
    ref_barrio = stats_df.loc[ref_idx, "barrio_key"]
    ref_diagram = barrio_diagrams[ref_barrio]

    # Calcular distancia Wasserstein de cada barrio a la referencia
    wasserstein_dists = []
    for barrio in stats_df["barrio_key"]:
        d = _wasserstein_distance_diagrams(barrio_diagrams[barrio], ref_diagram)
        wasserstein_dists.append(d)

    stats_df["wasserstein_dist"] = wasserstein_dists

    # --- Paso 5: score compuesto y deteccion ---
    # El score combina varias senales topologicas:
    #   - Persistence entropy alta o baja (vs mediana) → topologia inusual
    #   - Wasserstein distance alta → topologia diferente de lo normal
    #   - Numero de features significativos inusual

    # Z-scores robustos para cada metrica
    def _robust_zscore(x: np.ndarray) -> np.ndarray:
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        mad = mad if mad > 1e-10 else 1e-10
        return np.abs(x - med) / (1.4826 * mad)

    z_entropy = _robust_zscore(stats_df["persistence_entropy"].values)
    z_wasserstein = _robust_zscore(stats_df["wasserstein_dist"].values)
    z_total = _robust_zscore(stats_df["total_persistence"].values)
    z_nfeatures = _robust_zscore(stats_df["n_significant"].values.astype(float))

    # Score compuesto: media ponderada de z-scores
    tda_score = (
        0.30 * z_entropy
        + 0.35 * z_wasserstein
        + 0.20 * z_total
        + 0.15 * z_nfeatures
    )

    stats_df["tda_score"] = tda_score

    # Threshold: percentil basado en contamination
    threshold = np.percentile(tda_score, (1 - contamination) * 100)
    stats_df["is_anomaly_tda"] = stats_df["tda_score"] >= threshold

    # Preparar resultado final
    result = stats_df[["barrio_key"]].copy()
    result["tda_persistence"] = stats_df["total_persistence"]
    result["tda_entropy"] = stats_df["persistence_entropy"]
    result["tda_n_features"] = stats_df["n_significant"]
    result["tda_wasserstein"] = stats_df["wasserstein_dist"]
    result["tda_score"] = stats_df["tda_score"]
    result["is_anomaly_tda"] = stats_df["is_anomaly_tda"]

    n_anomalies = result["is_anomaly_tda"].sum()
    print(f"    {n_anomalies} barrios con topologia anomala "
          f"(threshold={threshold:.3f})")

    # Top anomalias topologicas
    top = result.nlargest(min(5, len(result)), "tda_score")
    print(f"    Top barrios por anomalia topologica:")
    for _, row in top.iterrows():
        barrio_short = row["barrio_key"].split("__")[0][:30]
        flag = " *** ANOMALO" if row["is_anomaly_tda"] else ""
        print(f"      {barrio_short:<30} score={row['tda_score']:.3f} "
              f"entropy={row['tda_entropy']:.3f} "
              f"wasserstein={row['tda_wasserstein']:.3f}{flag}")

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------

def tda_summary(results: pd.DataFrame):
    """
    Resumen de la deteccion topologica para el output final.

    Args:
        results: DataFrame devuelto por run_tda_detection()
    """
    if results.empty or "is_anomaly_tda" not in results.columns:
        print("\n  TDA: sin resultados")
        return

    n_total = len(results)
    n_anomalies = results["is_anomaly_tda"].sum()

    print(f"\n{'='*80}")
    print(f"  TOPOLOGICAL DATA ANALYSIS (TDA) — La forma del consumo")
    print(f"{'='*80}")
    print(f"\n  Barrios analizados: {n_total}")
    print(f"  Barrios con topologia anomala: {n_anomalies} "
          f"({n_anomalies/n_total*100:.1f}%)")

    if n_anomalies == 0:
        print(f"  Todos los barrios tienen topologia de consumo normal.")
        return

    # Estadisticas globales
    print(f"\n  Estadisticas de persistencia:")
    print(f"    Entropia media (normal):  "
          f"{results[~results['is_anomaly_tda']]['tda_entropy'].mean():.3f}")
    print(f"    Entropia media (anomalo): "
          f"{results[results['is_anomaly_tda']]['tda_entropy'].mean():.3f}")

    if "tda_wasserstein" in results.columns:
        print(f"    Wasserstein media (normal):  "
              f"{results[~results['is_anomaly_tda']]['tda_wasserstein'].mean():.3f}")
        print(f"    Wasserstein media (anomalo): "
              f"{results[results['is_anomaly_tda']]['tda_wasserstein'].mean():.3f}")

    # Tabla de anomalos
    anomalos = results[results["is_anomaly_tda"]].sort_values("tda_score", ascending=False)
    print(f"\n  {'Barrio':<35} {'Score':>8} {'Entropy':>9} {'N_feat':>7} "
          f"{'Wasserstein':>12}")
    print(f"  {'─'*75}")

    for _, row in anomalos.iterrows():
        barrio = row["barrio_key"].split("__")[0][:33]
        wass = row.get("tda_wasserstein", 0.0)
        print(f"  {barrio:<35} {row['tda_score']:>8.3f} "
              f"{row['tda_entropy']:>9.3f} {row['tda_n_features']:>7.0f} "
              f"{wass:>12.3f}")

    # Interpretacion
    print(f"\n  Interpretacion:")
    print(f"    - Entropia ALTA: la serie tiene muchas escalas temporales activas")
    print(f"      (consumo caotico, posible fraude intermitente)")
    print(f"    - Wasserstein ALTA: la topologia es muy diferente de lo normal")
    print(f"      (patron de consumo con estructura inusual)")
    print(f"    - N_features ALTO: la serie tiene muchos clusters en el espacio de fases")
    print(f"      (multiples regimenes de consumo, posible manipulacion)")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generar datos sinteticos para probar
    np.random.seed(42)

    n_barrios = 20
    n_meses = 36
    fechas = pd.date_range("2021-01-01", periods=n_meses, freq="MS")

    rows = []
    for i in range(n_barrios):
        barrio = f"BARRIO-{i:02d}__DOMESTICO"
        base = 100 + np.random.randn() * 20

        if i < 17:
            # Barrios normales: patron estacional + ruido
            seasonal = 15 * np.sin(2 * np.pi * np.arange(n_meses) / 12)
            noise = np.random.randn(n_meses) * 5
            consumo = base + seasonal + noise
        elif i == 17:
            # Anomalo: salto brusco
            consumo = np.concatenate([
                base + np.random.randn(18) * 5,
                base * 2.5 + np.random.randn(18) * 5,
            ])
        elif i == 18:
            # Anomalo: patron caotico
            consumo = base + np.random.randn(n_meses) * 40
        else:
            # Anomalo: picos puntuales
            consumo = base + np.random.randn(n_meses) * 5
            consumo[np.array([5, 12, 25])] *= 4

        for j, fecha in enumerate(fechas):
            rows.append({
                "barrio_key": barrio,
                "fecha": fecha,
                "consumption_per_contract": max(0, consumo[j]),
            })

    df_test = pd.DataFrame(rows)

    print("=" * 80)
    print("  TDA DETECTOR — Demo con datos sinteticos")
    print("=" * 80)

    results = run_tda_detection(df_test, contamination=0.10)
    tda_summary(results)
