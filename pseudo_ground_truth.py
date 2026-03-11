"""
pseudo_ground_truth.py
Construye pseudo-labels por triangulacion de 3 senales independientes
y calcula metricas clasicas (Precision, Recall, F1, AUC-ROC, AUC-PR).

Senales (v2 — sin circularidad con M8/ANR):
  A. Riesgo infraestructura (edad contador + % lectura manual) — estatica
  B. Desviacion del grupo (consumo vs peers del mismo tipo) — temporal
  C. Tasa de reemplazo reciente (contadores instalados >= 2023) — operacional

Nota: ANR eliminado de pseudo-labels para romper circularidad con M8 detector.
"""

import os
import numpy as np
import pandas as pd

__all__ = [
    "build_pseudo_labels",
    "evaluate_against_pseudo",
    "pseudo_ground_truth_summary",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TELELECTURA_FILE = (
    "contadores-telelectura-instalados-solo-alicante_hackaton-dataart-"
    "contadores-telelectura-instalad.csv"
)


def _rank_normalize(series):
    """Rank-percentile normalization to [0, 1]."""
    ranked = series.rank(method="average", na_option="bottom")
    return (ranked - 1) / max(ranked.max() - 1, 1)


def _compute_replacement_rate(telelectura_path=None):
    """Compute per-barrio recent counter replacement rate."""
    if telelectura_path is None:
        telelectura_path = os.path.join(DATA_DIR, TELELECTURA_FILE)
    if not os.path.exists(telelectura_path):
        return {}

    tele = pd.read_csv(telelectura_path)
    tele = tele[tele["BARRIO"] != "BARRIO"].copy()
    tele["FECHA_INSTALACION"] = pd.to_datetime(
        tele["FECHA INSTALACION"], format="%d/%m/%Y", errors="coerce"
    )
    tele["is_recent"] = tele["FECHA_INSTALACION"] >= pd.Timestamp("2023-01-01")

    barrio_rates = tele.groupby("BARRIO").agg(
        total=("BARRIO", "size"),
        recent=("is_recent", "sum"),
    )
    barrio_rates["replacement_rate"] = barrio_rates["recent"] / barrio_rates["total"]
    return barrio_rates["replacement_rate"].to_dict()


def build_pseudo_labels(results, ground_truth,
                        w_infra=0.35, w_deviation=0.35, w_replace=0.30,
                        threshold_percentile=85):
    """Build pseudo-labels from 3 independent signals (sin circularidad).

    Parameters
    ----------
    results : pd.DataFrame
        Pipeline results with barrio_key, deviation_from_group_trend, etc.
    ground_truth : dict
        Output of load_ground_truth() with 'barrio_risk' key.
    w_infra, w_deviation, w_replace : float
        Weights for each signal (must sum to 1.0).
    threshold_percentile : int
        Percentile cutoff for positive pseudo-label.

    Returns
    -------
    pd.DataFrame — results with 'pseudo_score' and 'pseudo_label' added.
    """
    df = results.copy()

    # Extract clean barrio name
    df["_barrio_clean"] = df["barrio_key"].apply(
        lambda x: x.split("__")[0] if "__" in x else x
    )

    # Signal A: Infrastructure risk (from ground_truth)
    barrio_risk = ground_truth.get("barrio_risk", pd.DataFrame())
    if "risk_score" in barrio_risk.columns:
        # Lower risk_score = higher risk (it's a rank sum, lower = worse)
        risk_map = barrio_risk["risk_score"].to_dict()
        df["_infra_raw"] = df["_barrio_clean"].map(risk_map)
        # Invert: lower rank sum → higher risk → higher signal
        max_risk = df["_infra_raw"].max()
        df["_infra_signal"] = 1 - (df["_infra_raw"] / max_risk) if max_risk > 0 else 0
        df["_infra_signal"] = df["_infra_signal"].fillna(0.5)
    else:
        df["_infra_signal"] = 0.5

    # Signal B: Deviation from group trend (independent of ANR — no circularity)
    # Uses cross-sectional deviation: how different is this barrio from its peers?
    if "deviation_from_group_trend" in df.columns:
        abs_dev = df["deviation_from_group_trend"].abs()
        df["_deviation_signal"] = _rank_normalize(abs_dev.fillna(0))
    elif "cross_sectional_zscore" in df.columns:
        abs_z = df["cross_sectional_zscore"].abs()
        df["_deviation_signal"] = _rank_normalize(abs_z.fillna(0))
    elif "seasonal_zscore" in df.columns:
        abs_z = df["seasonal_zscore"].abs()
        df["_deviation_signal"] = _rank_normalize(abs_z.fillna(0))
    else:
        df["_deviation_signal"] = 0.0

    # Signal C: Replacement rate
    replacement_rates = _compute_replacement_rate()
    if replacement_rates:
        df["_replace_raw"] = df["_barrio_clean"].map(replacement_rates)
        df["_replace_signal"] = _rank_normalize(df["_replace_raw"].fillna(0))
    else:
        df["_replace_signal"] = 0.0

    # Composite pseudo-score (NO ANR — breaks circularity with M8)
    df["pseudo_score"] = (
        w_infra * df["_infra_signal"] +
        w_deviation * df["_deviation_signal"] +
        w_replace * df["_replace_signal"]
    )

    # Pseudo-label: top quartile
    threshold = np.percentile(df["pseudo_score"].dropna(), threshold_percentile)
    df["pseudo_label"] = (df["pseudo_score"] >= threshold).astype(int)

    # Cleanup temp columns
    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)

    return df


def evaluate_against_pseudo(results):
    """Compute classification metrics against pseudo-labels.

    Evaluates multiple detection methods:
    - stacking_anomaly (if exists)
    - ensemble_score >= 0.25
    - n_models_detecting >= 2
    - conformal_anomaly (if exists)

    Returns
    -------
    dict with metrics per method + AUC-ROC and AUC-PR for continuous scores.
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, cohen_kappa_score,
    )

    y_true = results["pseudo_label"].values

    metrics = {"n_positive": int(y_true.sum()), "n_total": len(y_true)}

    # Define detection methods to evaluate
    methods = {}
    if "stacking_anomaly" in results.columns:
        methods["Stacking (score>=0.5)"] = results["stacking_anomaly"].fillna(False).astype(int).values
    if "ensemble_score" in results.columns:
        methods["Ensemble (score>=0.25)"] = (results["ensemble_score"] >= 0.25).astype(int).values
    if "n_models_detecting" in results.columns:
        methods["Consenso (>=2 modelos)"] = (results["n_models_detecting"] >= 2).astype(int).values
        methods["Consenso (>=3 modelos)"] = (results["n_models_detecting"] >= 3).astype(int).values
    if "conformal_anomaly" in results.columns:
        methods["Conformal (p<0.05)"] = results["conformal_anomaly"].fillna(False).astype(int).values

    method_metrics = {}
    for name, y_pred in methods.items():
        try:
            method_metrics[name] = {
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "kappa": cohen_kappa_score(y_true, y_pred),
                "n_detected": int(y_pred.sum()),
            }
        except Exception:
            pass

    metrics["methods"] = method_metrics

    # Continuous score AUC
    auc_metrics = {}
    score_cols = {
        "stacking_score": "Stacking Score",
        "ensemble_score": "Ensemble Score",
    }
    for col, label in score_cols.items():
        if col in results.columns:
            scores = results[col].fillna(0).values
            try:
                auc_metrics[label] = {
                    "auc_roc": roc_auc_score(y_true, scores),
                    "auc_pr": average_precision_score(y_true, scores),
                }
            except Exception:
                pass

    # Inverted conformal p-value (lower p = more anomalous)
    if "conformal_pvalue" in results.columns:
        inv_pval = 1 - results["conformal_pvalue"].fillna(1.0).values
        try:
            auc_metrics["Conformal (1-pvalue)"] = {
                "auc_roc": roc_auc_score(y_true, inv_pval),
                "auc_pr": average_precision_score(y_true, inv_pval),
            }
        except Exception:
            pass

    metrics["auc"] = auc_metrics

    return metrics


def pseudo_ground_truth_summary(metrics):
    """Print formatted report of pseudo-ground-truth evaluation."""
    print(f"\n{'='*80}")
    print(f"  PSEUDO-GROUND-TRUTH — Metricas contra triangulacion multi-senal")
    print(f"{'='*80}")

    n_pos = metrics["n_positive"]
    n_total = metrics["n_total"]
    prevalence = n_pos / n_total * 100 if n_total > 0 else 0

    print(f"\n  Pseudo-labels: {n_pos} positivos / {n_total} total "
          f"(prevalencia {prevalence:.1f}%)")
    print(f"  Senales: infraestructura (35%) + desviacion grupo (35%) + "
          f"tasa reemplazo (30%) [sin ANR — no circularidad]")

    # Method comparison table
    methods = metrics.get("methods", {})
    if methods:
        print(f"\n  {'Metodo':<28} {'Prec':>6} {'Recall':>6} {'F1':>6} "
              f"{'Kappa':>6} {'Det':>5}")
        print(f"  {'─'*62}")
        for name, m in sorted(methods.items(), key=lambda x: -x[1]["f1"]):
            print(f"  {name:<28} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{m['f1']:>6.3f} {m['kappa']:>+6.3f} {m['n_detected']:>5}")

    # AUC table
    auc = metrics.get("auc", {})
    if auc:
        print(f"\n  {'Score continuo':<28} {'AUC-ROC':>8} {'AUC-PR':>8}")
        print(f"  {'─'*46}")
        for name, a in sorted(auc.items(), key=lambda x: -x[1].get("auc_pr", 0)):
            print(f"  {name:<28} {a['auc_roc']:>8.3f} {a['auc_pr']:>8.3f}")

        # Baseline AUC-PR = prevalence
        print(f"\n  Baseline AUC-PR (random): {prevalence/100:.3f}")
        best_pr = max(a["auc_pr"] for a in auc.values())
        lift = best_pr / (prevalence / 100) if prevalence > 0 else 0
        print(f"  Mejor AUC-PR: {best_pr:.3f} (lift {lift:.1f}x vs random)")
