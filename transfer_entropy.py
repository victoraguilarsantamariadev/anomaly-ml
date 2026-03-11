"""
Transfer Entropy — Flujo de informacion direccional entre barrios.

Transfer Entropy (Schreiber, 2000) mide cuanta informacion el pasado de X
proporciona sobre el futuro de Y, mas alla de lo que el pasado de Y ya dice.

Ventajas sobre Granger Causality:
  - No asume linealidad
  - Captura dependencias no lineales
  - Base information-theoretic (model-free)

Detecta:
  - Barrios "hub" que propagan anomalias a vecinos (posible fuga en red compartida)
  - Cadenas de cascada: A→B→C (propagacion por la red de distribucion)
  - Leading indicators: barrios que predicen problemas en otros

Paper: Schreiber (2000) "Measuring Information Transfer", Physical Review Letters
"""

import numpy as np
import pandas as pd
from itertools import permutations


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENTROPY ESTIMATION (histogram-based)
# ─────────────────────────────────────────────────────────────────────────────

def _entropy(x, bins=10):
    """Estimate Shannon entropy of a 1D array via histogram."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or np.std(x) < 1e-12:
        return 0.0
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    dx = (x.max() - x.min()) / bins
    return -np.sum(hist * np.log(hist + 1e-10)) * dx


def _joint_entropy(x, y, bins=10):
    """Estimate joint Shannon entropy of two 1D arrays via 2D histogram."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0
    hist2d, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    hist2d = hist2d[hist2d > 0]
    dx = (x.max() - x.min()) / bins if np.std(x) > 1e-12 else 1.0
    dy = (y.max() - y.min()) / bins if np.std(y) > 1e-12 else 1.0
    return -np.sum(hist2d * np.log(hist2d + 1e-10)) * dx * dy


def _conditional_entropy(y, *conds, bins=10):
    """
    Estimate H(Y | cond_1, cond_2, ...) = H(Y, cond_1, ...) - H(cond_1, ...).

    Uses multidimensional histograms for up to 3 conditioning variables.
    Falls back to pairwise approximation for higher dimensions.
    """
    y = np.asarray(y, dtype=float)
    conds = [np.asarray(c, dtype=float) for c in conds]
    n_conds = len(conds)

    if n_conds == 0:
        return _entropy(y, bins=bins)

    if n_conds == 1:
        # H(Y | X) = H(Y, X) - H(X)
        return _joint_entropy(y, conds[0], bins=bins) - _entropy(conds[0], bins=bins)

    # For multiple conditions, stack them into a combined representation.
    # Use a flattened index approach: discretize each condition into bins
    # and create a composite index.
    all_vars = conds + [y]
    n = len(y)

    # Digitize each variable into bins
    digitized = []
    for v in all_vars:
        if np.std(v) < 1e-12:
            digitized.append(np.zeros(n, dtype=int))
        else:
            edges = np.linspace(v.min() - 1e-10, v.max() + 1e-10, bins + 1)
            digitized.append(np.digitize(v, edges) - 1)

    # Composite index for conditions only
    cond_digits = digitized[:-1]
    multipliers = [1]
    for i in range(1, len(cond_digits)):
        multipliers.append(multipliers[-1] * bins)
    cond_composite = np.zeros(n, dtype=int)
    for i, d in enumerate(cond_digits):
        cond_composite += d * multipliers[i]

    # Composite index for all variables (conditions + Y)
    all_digits = digitized
    multipliers_all = [1]
    for i in range(1, len(all_digits)):
        multipliers_all.append(multipliers_all[-1] * bins)
    all_composite = np.zeros(n, dtype=int)
    for i, d in enumerate(all_digits):
        all_composite += d * multipliers_all[i]

    # H(Y, conds) via histogram of all_composite
    _, counts_all = np.unique(all_composite, return_counts=True)
    p_all = counts_all / counts_all.sum()
    h_joint = -np.sum(p_all * np.log(p_all + 1e-10))

    # H(conds) via histogram of cond_composite
    _, counts_cond = np.unique(cond_composite, return_counts=True)
    p_cond = counts_cond / counts_cond.sum()
    h_cond = -np.sum(p_cond * np.log(p_cond + 1e-10))

    return h_joint - h_cond


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSFER ENTROPY
# ─────────────────────────────────────────────────────────────────────────────

def transfer_entropy(x, y, k=1, bins=10):
    """
    Compute Transfer Entropy from X to Y: TE(X -> Y).

    TE(X->Y) = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-k})

    This measures how much knowing the past of X reduces uncertainty about
    the future of Y, beyond what the past of Y already provides.

    Parameters
    ----------
    x : array-like
        Source time series.
    y : array-like
        Target time series.
    k : int
        Number of lags (history length).
    bins : int
        Number of bins for histogram estimation.

    Returns
    -------
    float
        Transfer entropy value (>= 0, higher = more information flow).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))

    if n <= k + 1:
        return 0.0

    x = x[:n]
    y = y[:n]

    # Build lagged arrays
    y_future = y[k:]  # Y_t
    y_lags = [y[k - i - 1: n - i - 1] for i in range(k)]  # Y_{t-1}, ..., Y_{t-k}
    x_lags = [x[k - i - 1: n - i - 1] for i in range(k)]  # X_{t-1}, ..., X_{t-k}

    # H(Y_t | Y_past)
    h_y_given_ypast = _conditional_entropy(y_future, *y_lags, bins=bins)

    # H(Y_t | Y_past, X_past)
    h_y_given_ypast_xpast = _conditional_entropy(y_future, *y_lags, *x_lags, bins=bins)

    te = h_y_given_ypast - h_y_given_ypast_xpast
    return max(te, 0.0)  # TE is non-negative by definition


def transfer_entropy_multilag(x, y, lags=(1, 2, 3), bins=10):
    """
    Compute TE for multiple lags and return the maximum.

    Using multiple lags captures different timescales of information flow.
    """
    te_values = []
    for k in lags:
        te = transfer_entropy(x, y, k=k, bins=bins)
        te_values.append(te)
    return max(te_values), lags[np.argmax(te_values)]


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIGNIFICANCE TESTING (shuffle-based null distribution)
# ─────────────────────────────────────────────────────────────────────────────

def te_significance(x, y, k=1, bins=10, n_shuffles=100, seed=42):
    """
    Test significance of TE(X->Y) via permutation test.

    Procedure:
      1. Compute observed TE(X->Y)
      2. Shuffle X n_shuffles times, compute TE each time
      3. p-value = fraction of shuffled TE >= observed TE

    Parameters
    ----------
    x, y : array-like
        Source and target time series.
    k : int
        Lag order.
    bins : int
        Histogram bins.
    n_shuffles : int
        Number of permutations for the null distribution.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    observed_te : float
    p_value : float
    null_mean : float
        Mean TE under null distribution.
    """
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    observed_te = transfer_entropy(x, y, k=k, bins=bins)

    null_tes = np.zeros(n_shuffles)
    for i in range(n_shuffles):
        x_shuffled = rng.permutation(x)
        null_tes[i] = transfer_entropy(x_shuffled, y, k=k, bins=bins)

    p_value = np.mean(null_tes >= observed_te)
    return observed_te, p_value, null_tes.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 4. PAIRWISE TE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def compute_te_matrix(series_dict, lags=(1, 2, 3), bins=10,
                      n_shuffles=100, alpha=0.05, seed=42):
    """
    Compute pairwise Transfer Entropy matrix for all pairs of time series.

    Parameters
    ----------
    series_dict : dict
        {barrio_name: np.array of time series values}
    lags : tuple of int
        Lag orders to test.
    bins : int
        Histogram bins for entropy estimation.
    n_shuffles : int
        Permutations for significance testing.
    alpha : float
        Significance threshold.
    seed : int
        Random seed.

    Returns
    -------
    te_matrix : pd.DataFrame
        TE(row -> col) values.
    pval_matrix : pd.DataFrame
        p-values for each pair.
    best_lag_matrix : pd.DataFrame
        Best lag for each pair.
    """
    names = sorted(series_dict.keys())
    n = len(names)
    te_mat = np.zeros((n, n))
    pval_mat = np.ones((n, n))
    lag_mat = np.zeros((n, n), dtype=int)

    total_pairs = n * (n - 1)
    computed = 0

    for i, src in enumerate(names):
        for j, tgt in enumerate(names):
            if i == j:
                continue
            computed += 1
            if computed % 50 == 0:
                print(f"    Calculando TE: {computed}/{total_pairs} pares...")

            x = series_dict[src]
            y = series_dict[tgt]

            # Find best lag
            best_te = 0.0
            best_k = 1
            best_p = 1.0
            for k in lags:
                te_obs, p_val, _ = te_significance(
                    x, y, k=k, bins=bins,
                    n_shuffles=n_shuffles, seed=seed + i * n + j
                )
                if te_obs > best_te:
                    best_te = te_obs
                    best_k = k
                    best_p = p_val

            te_mat[i, j] = best_te
            pval_mat[i, j] = best_p
            lag_mat[i, j] = best_k

    te_matrix = pd.DataFrame(te_mat, index=names, columns=names)
    pval_matrix = pd.DataFrame(pval_mat, index=names, columns=names)
    best_lag_matrix = pd.DataFrame(lag_mat, index=names, columns=names)

    return te_matrix, pval_matrix, best_lag_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 5. NETWORK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def build_te_network(te_matrix, pval_matrix, alpha=0.05):
    """
    Build directed network of significant TE connections.

    Returns
    -------
    edges : list of dict
        Each dict: {source, target, te, p_value, lag}
    hub_scores : pd.DataFrame
        For each barrio: outgoing_te, incoming_te, net_flow, n_outgoing, n_incoming
    """
    names = te_matrix.index.tolist()
    edges = []

    for src in names:
        for tgt in names:
            if src == tgt:
                continue
            if pval_matrix.loc[src, tgt] < alpha:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "te": te_matrix.loc[src, tgt],
                    "p_value": pval_matrix.loc[src, tgt],
                })

    # Hub scores
    hub_data = []
    for name in names:
        outgoing = [e for e in edges if e["source"] == name]
        incoming = [e for e in edges if e["target"] == name]

        out_te = sum(e["te"] for e in outgoing)
        in_te = sum(e["te"] for e in incoming)

        hub_data.append({
            "barrio": name,
            "outgoing_te": out_te,
            "incoming_te": in_te,
            "net_flow": out_te - in_te,
            "n_outgoing": len(outgoing),
            "n_incoming": len(incoming),
            "role": "HUB (propagador)" if out_te > in_te and len(outgoing) > 0
                    else "SINK (receptor)" if in_te > out_te and len(incoming) > 0
                    else "NEUTRAL",
        })

    hub_scores = pd.DataFrame(hub_data).set_index("barrio")
    hub_scores = hub_scores.sort_values("net_flow", ascending=False)

    return edges, hub_scores


def find_cascade_chains(edges, max_depth=4):
    """
    Find propagation chains A -> B -> C -> ... in the TE network.

    Uses DFS to find all simple paths of length >= 2.

    Parameters
    ----------
    edges : list of dict
        Significant edges from build_te_network.
    max_depth : int
        Maximum chain length to search.

    Returns
    -------
    chains : list of dict
        Each dict: {path: [A, B, C, ...], total_te: float, length: int}
    """
    # Build adjacency list
    adj = {}
    te_lookup = {}
    for e in edges:
        src, tgt = e["source"], e["target"]
        if src not in adj:
            adj[src] = []
        adj[src].append(tgt)
        te_lookup[(src, tgt)] = e["te"]

    chains = []

    def _dfs(node, path, total_te):
        if len(path) >= 3:  # At least A -> B -> C
            chains.append({
                "path": list(path),
                "total_te": total_te,
                "length": len(path) - 1,
                "path_str": " -> ".join(str(p) for p in path),
            })
        if len(path) >= max_depth + 1:
            return
        for neighbor in adj.get(node, []):
            if neighbor not in path:  # Avoid cycles
                edge_te = te_lookup.get((node, neighbor), 0)
                _dfs(neighbor, path + [neighbor], total_te + edge_te)

    all_nodes = set()
    for e in edges:
        all_nodes.add(e["source"])
        all_nodes.add(e["target"])

    for start in sorted(all_nodes):
        _dfs(start, [start], 0.0)

    # Sort by total TE descending, deduplicate
    chains.sort(key=lambda c: c["total_te"], reverse=True)

    # Keep only top chains (avoid combinatorial explosion)
    seen = set()
    unique_chains = []
    for c in chains:
        key = c["path_str"]
        if key not in seen:
            seen.add(key)
            unique_chains.append(c)
        if len(unique_chains) >= 50:
            break

    return unique_chains


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANOMALY PROPAGATION SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_propagation_scores(hub_scores, te_matrix, pval_matrix, alpha=0.05):
    """
    For each barrio, compute a composite anomaly propagation score.

    Score components:
      - influence: normalized outgoing TE (how much it affects others)
      - susceptibility: normalized incoming TE (how much others affect it)
      - centrality: total significant connections (in + out)

    Returns
    -------
    pd.DataFrame with columns: influence, susceptibility, centrality,
                                propagation_score, rank
    """
    scores = hub_scores.copy()

    # Normalize
    max_out = scores["outgoing_te"].max()
    max_in = scores["incoming_te"].max()
    max_conn = (scores["n_outgoing"] + scores["n_incoming"]).max()

    if max_out > 0:
        scores["influence"] = scores["outgoing_te"] / max_out
    else:
        scores["influence"] = 0.0

    if max_in > 0:
        scores["susceptibility"] = scores["incoming_te"] / max_in
    else:
        scores["susceptibility"] = 0.0

    if max_conn > 0:
        scores["centrality"] = (scores["n_outgoing"] + scores["n_incoming"]) / max_conn
    else:
        scores["centrality"] = 0.0

    # Composite score: weighted combination
    # Higher weight on influence (propagation hubs are more actionable)
    scores["propagation_score"] = (
        0.5 * scores["influence"] +
        0.3 * scores["susceptibility"] +
        0.2 * scores["centrality"]
    )

    scores["rank"] = scores["propagation_score"].rank(ascending=False).astype(int)
    scores = scores.sort_values("propagation_score", ascending=False)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN API
# ─────────────────────────────────────────────────────────────────────────────

def run_transfer_entropy_analysis(df_monthly, results,
                                  consumo_col="consumption_per_contract",
                                  top_n=15, lags=(1, 2, 3), bins=10,
                                  n_shuffles=100, alpha=0.05) -> dict:
    """
    Run complete Transfer Entropy analysis on barrio consumption data.

    Parameters
    ----------
    df_monthly : pd.DataFrame
        Monthly consumption data with columns: barrio_key, fecha, consumo_col.
    results : pd.DataFrame
        Model results with anomaly scores (used to select top anomalous barrios).
    consumo_col : str
        Column name for consumption metric.
    top_n : int
        Number of top anomalous barrios to analyze (reduces computation).
    lags : tuple of int
        Lag orders to test for TE.
    bins : int
        Histogram bins for entropy estimation.
    n_shuffles : int
        Permutations for significance testing.
    alpha : float
        Significance level.

    Returns
    -------
    dict with keys:
        te_matrix : pd.DataFrame — TE(row -> col)
        pval_matrix : pd.DataFrame — p-values
        significant_edges : list of dict — significant directional connections
        hub_scores : pd.DataFrame — hub/sink classification
        cascade_chains : list of dict — A->B->C propagation paths
        propagation_scores : pd.DataFrame — composite scores per barrio
    """
    print("\n" + "=" * 70)
    print("  TRANSFER ENTROPY — Flujo de informacion direccional")
    print("=" * 70)

    # --- Select top anomalous barrios ---
    if results is not None and "anomaly_score" in results.columns:
        barrio_col = "barrio_key" if "barrio_key" in results.columns else "barrio"
        if barrio_col in results.columns:
            top_barrios = (
                results.groupby(barrio_col)["anomaly_score"]
                .mean()
                .nlargest(top_n)
                .index.tolist()
            )
            print(f"\n  Analizando top {len(top_barrios)} barrios anomalos")
        else:
            top_barrios = df_monthly["barrio_key"].unique()[:top_n].tolist()
    else:
        top_barrios = df_monthly["barrio_key"].unique()[:top_n].tolist()
        print(f"\n  Sin resultados de anomalia, usando primeros {top_n} barrios")

    # --- Build time series dict ---
    barrio_col_monthly = "barrio_key" if "barrio_key" in df_monthly.columns else "barrio"
    fecha_col = "fecha" if "fecha" in df_monthly.columns else df_monthly.columns[
        df_monthly.columns.str.contains("fecha|date", case=False)
    ][0]

    series_dict = {}
    for barrio in top_barrios:
        mask = df_monthly[barrio_col_monthly] == barrio
        ts = df_monthly.loc[mask].sort_values(fecha_col)[consumo_col].values
        if len(ts) >= 6:  # Need minimum length for meaningful TE
            series_dict[barrio] = ts.astype(float)

    if len(series_dict) < 2:
        print("  Insuficientes barrios con datos temporales para TE.")
        return {
            "te_matrix": pd.DataFrame(),
            "pval_matrix": pd.DataFrame(),
            "significant_edges": [],
            "hub_scores": pd.DataFrame(),
            "cascade_chains": [],
            "propagation_scores": pd.DataFrame(),
        }

    print(f"  {len(series_dict)} barrios con series temporales validas")
    print(f"  Lags: {lags}, bins: {bins}, shuffles: {n_shuffles}, alpha: {alpha}")

    # --- Compute pairwise TE ---
    n_pairs = len(series_dict) * (len(series_dict) - 1)
    print(f"\n  Calculando {n_pairs} pares de Transfer Entropy...")
    te_matrix, pval_matrix, lag_matrix = compute_te_matrix(
        series_dict, lags=lags, bins=bins,
        n_shuffles=n_shuffles, alpha=alpha, seed=42
    )

    # --- Build network ---
    print("\n  Construyendo red de flujo de informacion...")
    edges, hub_scores = build_te_network(te_matrix, pval_matrix, alpha=alpha)
    print(f"  {len(edges)} conexiones significativas (p < {alpha})")

    # --- Find cascade chains ---
    print("  Buscando cadenas de cascada...")
    chains = find_cascade_chains(edges, max_depth=4)
    print(f"  {len(chains)} cadenas de propagacion encontradas")

    # --- Propagation scores ---
    propagation_scores = compute_propagation_scores(
        hub_scores, te_matrix, pval_matrix, alpha=alpha
    )

    result = {
        "te_matrix": te_matrix,
        "pval_matrix": pval_matrix,
        "best_lag_matrix": lag_matrix,
        "significant_edges": edges,
        "hub_scores": hub_scores,
        "cascade_chains": chains,
        "propagation_scores": propagation_scores,
    }

    # Print summary
    transfer_entropy_summary(result)

    return result


def transfer_entropy_summary(te_results):
    """
    Print human-readable summary of Transfer Entropy network analysis.

    Parameters
    ----------
    te_results : dict
        Output from run_transfer_entropy_analysis.
    """
    te_matrix = te_results.get("te_matrix", pd.DataFrame())
    edges = te_results.get("significant_edges", [])
    hub_scores = te_results.get("hub_scores", pd.DataFrame())
    chains = te_results.get("cascade_chains", [])
    prop_scores = te_results.get("propagation_scores", pd.DataFrame())

    if te_matrix.empty:
        print("\n  No hay resultados de Transfer Entropy para mostrar.")
        return

    print("\n" + "-" * 70)
    print("  RESUMEN — Transfer Entropy Network")
    print("-" * 70)

    # --- Matrix stats ---
    n_barrios = len(te_matrix)
    n_edges = len(edges)
    density = n_edges / (n_barrios * (n_barrios - 1)) if n_barrios > 1 else 0
    print(f"\n  Barrios analizados: {n_barrios}")
    print(f"  Conexiones significativas: {n_edges}")
    print(f"  Densidad de red: {density:.1%}")

    # --- Top propagation hubs ---
    if not hub_scores.empty:
        hubs = hub_scores[hub_scores["role"] == "HUB (propagador)"]
        sinks = hub_scores[hub_scores["role"] == "SINK (receptor)"]

        print(f"\n  HUBS (propagan anomalias): {len(hubs)}")
        for barrio in hubs.head(5).index:
            row = hubs.loc[barrio]
            print(f"    {barrio}: {row['n_outgoing']} conexiones salientes, "
                  f"TE total = {row['outgoing_te']:.4f}")

        print(f"\n  SINKS (reciben anomalias): {len(sinks)}")
        for barrio in sinks.head(5).index:
            row = sinks.loc[barrio]
            print(f"    {barrio}: {row['n_incoming']} conexiones entrantes, "
                  f"TE total = {row['incoming_te']:.4f}")

    # --- Top edges ---
    if edges:
        print("\n  TOP 10 conexiones mas fuertes:")
        sorted_edges = sorted(edges, key=lambda e: e["te"], reverse=True)
        for i, e in enumerate(sorted_edges[:10]):
            print(f"    {i+1}. {e['source']} -> {e['target']}: "
                  f"TE = {e['te']:.4f} (p = {e['p_value']:.3f})")

    # --- Top cascade chains ---
    if chains:
        print(f"\n  TOP 5 cadenas de cascada:")
        for i, c in enumerate(chains[:5]):
            print(f"    {i+1}. {c['path_str']}  "
                  f"(TE acumulado = {c['total_te']:.4f}, "
                  f"longitud = {c['length']})")

    # --- Propagation score ranking ---
    if not prop_scores.empty:
        print("\n  RANKING de propagacion (composite score):")
        for barrio in prop_scores.head(10).index:
            row = prop_scores.loc[barrio]
            print(f"    #{int(row['rank']):2d} {barrio}: "
                  f"score = {row['propagation_score']:.3f} "
                  f"(influence={row['influence']:.2f}, "
                  f"suscept={row['susceptibility']:.2f}, "
                  f"central={row['centrality']:.2f}) "
                  f"[{row['role']}]")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Transfer Entropy — analisis de flujo de informacion entre barrios")
    print("Uso: importar run_transfer_entropy_analysis() desde otro script.\n")

    # Quick demo with synthetic data
    np.random.seed(42)
    n_t = 36  # 36 months
    n_barrios = 5

    # Create synthetic series where barrio_0 drives barrio_1 with lag 1
    demo_series = {}
    base = np.random.randn(n_t).cumsum()
    demo_series["barrio_A"] = base + np.random.randn(n_t) * 0.3
    demo_series["barrio_B"] = np.roll(base, 1) + np.random.randn(n_t) * 0.5
    demo_series["barrio_C"] = np.roll(base, 2) + np.random.randn(n_t) * 0.8
    demo_series["barrio_D"] = np.random.randn(n_t).cumsum()  # independent
    demo_series["barrio_E"] = np.roll(demo_series["barrio_B"], 1) + np.random.randn(n_t) * 0.4

    print("Series sinteticas: A -> B -> E, A -> C (con lag), D independiente\n")
    print("Calculando TE matrix...")

    te_mat, pval_mat, lag_mat = compute_te_matrix(
        demo_series, lags=(1, 2, 3), bins=8, n_shuffles=50
    )

    edges, hub_scores = build_te_network(te_mat, pval_mat, alpha=0.10)
    chains = find_cascade_chains(edges)
    prop_scores = compute_propagation_scores(hub_scores, te_mat, pval_mat, alpha=0.10)

    demo_results = {
        "te_matrix": te_mat,
        "pval_matrix": pval_mat,
        "significant_edges": edges,
        "hub_scores": hub_scores,
        "cascade_chains": chains,
        "propagation_scores": prop_scores,
    }

    transfer_entropy_summary(demo_results)
