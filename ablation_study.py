"""
ablation_study.py
Leave-One-Model-Out ablation + pairwise redundancy analysis.

Demuestra la contribucion marginal de cada modelo al ensemble.
"""

import numpy as np
import pandas as pd

__all__ = [
    "run_ablation_study",
    "compute_pairwise_redundancy",
    "ablation_summary",
]

ALL_FLAG_COLS = [
    "is_anomaly_m2", "is_anomaly_autoencoder", "is_anomaly_vae",
    "is_anomaly_3sigma", "is_anomaly_iqr", "is_anomaly_prophet",
    "is_anomaly_chronos", "is_anomaly_anr", "is_anomaly_nmf",
]

SCORE_COLS = [
    "score_m2", "vae_score_norm", "reconstruction_error",
    "anr_ratio",
    # ensemble_score EXCLUIDO: depende de voting weights → causa oscilación
]

MODEL_NAMES = {
    "is_anomaly_m2": "M2 IsoForest",
    "is_anomaly_autoencoder": "M13 Autoencoder",
    "is_anomaly_vae": "M14 VAE",
    "is_anomaly_3sigma": "M5a 3-sigma",
    "is_anomaly_iqr": "M5b IQR",
    "is_anomaly_prophet": "M7 Prophet",
    "is_anomaly_chronos": "M8 Chronos",
    "is_anomaly_anr": "M8 ANR",
    "is_anomaly_nmf": "M9 NMF",
}


def _train_stacking_auc(results, flag_cols, score_cols, y_true, min_months=6):
    """Train walk-forward stacking and return AUC-PR."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score

    meta_features = flag_cols + score_cols
    X = results[meta_features].fillna(0).replace([np.inf, -np.inf], 0).values
    all_dates = sorted(results["fecha"].unique())

    if len(all_dates) < min_months + 1:
        return np.nan

    predictions = np.full(len(results), np.nan)

    for t_idx in range(min_months, len(all_dates)):
        test_date = all_dates[t_idx]
        train_dates = set(all_dates[:t_idx])

        train_mask = results["fecha"].isin(train_dates).values
        test_mask = (results["fecha"] == test_date).values

        X_train, y_train = X[train_mask], y_true[train_mask]
        X_test = X[test_mask]

        if len(np.unique(y_train)) < 2 or X_test.shape[0] == 0:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        lr = LogisticRegression(
            penalty="l2", C=1.0, max_iter=1000,
            random_state=42, class_weight="balanced"
        )
        lr.fit(X_train_s, y_train)
        predictions[test_mask] = lr.predict_proba(X_test_s)[:, 1]

    evaluated = ~np.isnan(predictions)
    if evaluated.sum() < 20:
        return np.nan

    return average_precision_score(y_true[evaluated], predictions[evaluated])


def run_ablation_study(results, pseudo_labels=None):
    """Leave-one-model-out ablation study.

    Parameters
    ----------
    results : pd.DataFrame
        Full pipeline results.
    pseudo_labels : array-like, optional
        Ground truth labels. Uses results["pseudo_label"] if not provided.

    Returns
    -------
    pd.DataFrame with columns: model, auc_full, auc_without, delta, verdict
    """
    if pseudo_labels is None:
        pseudo_labels = results["pseudo_label"].values
    y_true = np.array(pseudo_labels)

    available_flags = [c for c in ALL_FLAG_COLS if c in results.columns]
    available_scores = [c for c in SCORE_COLS if c in results.columns]

    if len(available_flags) < 3:
        return pd.DataFrame()

    # Full model AUC
    auc_full = _train_stacking_auc(results, available_flags, available_scores, y_true)

    # Leave-one-out
    rows = []
    for drop_col in available_flags:
        reduced_flags = [c for c in available_flags if c != drop_col]
        # Also remove associated score column if it matches
        reduced_scores = available_scores.copy()
        if drop_col == "is_anomaly_m2" and "score_m2" in reduced_scores:
            reduced_scores.remove("score_m2")
        elif drop_col == "is_anomaly_vae" and "vae_score_norm" in reduced_scores:
            reduced_scores.remove("vae_score_norm")
        elif drop_col == "is_anomaly_autoencoder" and "reconstruction_error" in reduced_scores:
            reduced_scores.remove("reconstruction_error")
        elif drop_col == "is_anomaly_anr" and "anr_ratio" in reduced_scores:
            reduced_scores.remove("anr_ratio")

        auc_without = _train_stacking_auc(results, reduced_flags, reduced_scores, y_true)
        delta = auc_full - auc_without if not np.isnan(auc_without) else np.nan

        if np.isnan(delta):
            verdict = "N/A"
        elif delta >= 0.03:
            verdict = "ESSENTIAL"
        elif delta >= 0.01:
            verdict = "USEFUL"
        elif delta >= 0.005:
            verdict = "MARGINAL"
        else:
            verdict = "REDUNDANT"

        rows.append({
            "model": MODEL_NAMES.get(drop_col, drop_col),
            "flag_col": drop_col,
            "auc_full": auc_full,
            "auc_without": auc_without,
            "delta": delta,
            "verdict": verdict,
        })

    return pd.DataFrame(rows).sort_values("delta", ascending=False)


def compute_pairwise_redundancy(results):
    """Jaccard similarity between all model pairs.

    Returns
    -------
    pd.DataFrame with columns: model_a, model_b, jaccard, verdict
    """
    available = [c for c in ALL_FLAG_COLS if c in results.columns]
    rows = []

    for i, col_a in enumerate(available):
        for col_b in available[i+1:]:
            a = results[col_a].fillna(0).astype(bool).values
            b = results[col_b].fillna(0).astype(bool).values

            intersection = (a & b).sum()
            union = (a | b).sum()
            jaccard = intersection / union if union > 0 else 0

            verdict = "REDUNDANT" if jaccard > 0.7 else "OVERLAP" if jaccard > 0.4 else "COMPLEMENTARY"

            rows.append({
                "model_a": MODEL_NAMES.get(col_a, col_a),
                "model_b": MODEL_NAMES.get(col_b, col_b),
                "jaccard": jaccard,
                "verdict": verdict,
            })

    return pd.DataFrame(rows).sort_values("jaccard", ascending=False)


def ablation_summary(ablation_df, redundancy_df):
    """Print formatted ablation and redundancy report."""
    print(f"\n{'='*80}")
    print(f"  ABLATION STUDY — Contribucion marginal de cada modelo")
    print(f"{'='*80}")

    if ablation_df.empty:
        print("  No hay suficientes modelos para ablation.")
        return

    auc_full = ablation_df.iloc[0]["auc_full"]
    print(f"\n  AUC-PR completo (todos los modelos): {auc_full:.4f}")

    print(f"\n  {'Modelo':<20} {'AUC-PR sin':>10} {'Delta':>8} {'Verdict':<12}")
    print(f"  {'─'*55}")
    for _, row in ablation_df.iterrows():
        delta_str = f"+{row['delta']:.4f}" if not np.isnan(row['delta']) else "N/A"
        print(f"  {row['model']:<20} {row['auc_without']:>10.4f} "
              f"{delta_str:>8} {row['verdict']:<12}")

    # Count essential/useful
    n_essential = (ablation_df["verdict"] == "ESSENTIAL").sum()
    n_useful = (ablation_df["verdict"] == "USEFUL").sum()
    n_redundant = (ablation_df["verdict"] == "REDUNDANT").sum()
    print(f"\n  Resumen: {n_essential} esenciales, {n_useful} utiles, "
          f"{n_redundant} redundantes")

    # Redundancy
    if not redundancy_df.empty:
        high_overlap = redundancy_df[redundancy_df["jaccard"] > 0.4]
        if len(high_overlap) > 0:
            print(f"\n  REDUNDANCIA (Jaccard > 0.4):")
            for _, row in high_overlap.head(10).iterrows():
                print(f"    {row['model_a']} <-> {row['model_b']}: "
                      f"Jaccard={row['jaccard']:.3f} ({row['verdict']})")
