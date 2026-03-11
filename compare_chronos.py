"""
Comparacion rapida de Chronos small vs base vs large.

Corre solo 5 barrios para ver la diferencia en deteccion.

Uso:
  python compare_chronos.py
  python compare_chronos.py --models small base
  python compare_chronos.py --models small base large --barrios 3
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

from train_local import load_hackathon_amaem
from monthly_features import compute_monthly_features
from validate_model import (
    _load_barrio, inject_anomalies,
    SYNTHETIC_ANOMALIES_EXTREME, SYNTHETIC_ANOMALIES_SUBTLE,
)

DATA_FILE = "data/datos-hackathon-amaem.xlsx-set-de-datos-.csv"


def _compute_prf(detected, true_labels):
    tp = int(np.sum(detected & true_labels))
    fp = int(np.sum(detected & ~true_labels))
    fn = int(np.sum(~detected & true_labels))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def compare_models(df_all, barrios, uso, model_sizes, threshold_sigma=3.5):
    from chronos_detector import score_chronos

    results = []

    for barrio in barrios:
        df = _load_barrio(df_all, barrio, uso)
        if df is None:
            continue

        n_train = min(25, int(len(df) * 0.7))
        n_test = len(df) - n_train

        for anom_name, anom_list in [("extreme", SYNTHETIC_ANOMALIES_EXTREME),
                                      ("subtle", SYNTHETIC_ANOMALIES_SUBTLE)]:
            if n_test < max(a["offset"] for a in anom_list) + 1:
                continue

            df_inj, anom_idx = inject_anomalies(df, n_train, anomalies=anom_list, verbose=False)
            train_vals = df_inj.iloc[:n_train]["consumption"].values.astype(float)
            test_vals = df_inj.iloc[n_train:]["consumption"].values.astype(float)

            true_labels = np.zeros(len(test_vals), dtype=bool)
            for idx in anom_idx:
                rel = idx - n_train
                if 0 <= rel < len(true_labels):
                    true_labels[rel] = True

            for model_size in model_sizes:
                t0 = time.time()
                flags = score_chronos(train_vals, test_vals,
                                      threshold_sigma=threshold_sigma,
                                      num_samples=30, model_size=model_size)
                elapsed = time.time() - t0

                metrics = _compute_prf(flags, true_labels)
                metrics.update({
                    "barrio": barrio,
                    "anomaly_set": anom_name,
                    "model_size": model_size,
                    "time_s": elapsed,
                })
                results.append(metrics)

                print(f"  {model_size:>5} | {barrio:<25} | {anom_name:<8} | "
                      f"P={metrics['precision']:.0%} R={metrics['recall']:.0%} "
                      f"F1={metrics['f1']:.0%} | {elapsed:.1f}s")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["small", "base"],
                        choices=["small", "base", "large"])
    parser.add_argument("--barrios", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=3.5)
    parser.add_argument("--uso", default="DOMESTICO")
    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"  Comparacion Chronos: {' vs '.join(args.models)}")
    print(f"  Barrios: {args.barrios}, sigma: {args.sigma}")
    print(f"{'='*80}")

    df_all = load_hackathon_amaem(DATA_FILE)
    barrios = sorted(df_all[df_all["uso"] == args.uso]["barrio"].unique())[:args.barrios]

    print(f"\n  {'Model':>5} | {'Barrio':<25} | {'Anomaly':<8} | "
          f"{'P':>3} {'R':>3} {'F1':>3} | {'Time':>5}")
    print(f"  {'─'*75}")

    results = compare_models(df_all, barrios, args.uso, args.models, args.sigma)

    if len(results) > 0:
        print(f"\n{'='*80}")
        print(f"  RESUMEN (media sobre {len(barrios)} barrios)")
        print(f"{'='*80}")

        summary = results.groupby(["model_size", "anomaly_set"]).agg(
            mean_p=("precision", "mean"),
            mean_r=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_time=("time_s", "mean"),
        ).reset_index()

        print(f"\n  {'Model':>5} | {'Anomaly':<8} | {'Precision':>9} | "
              f"{'Recall':>6} | {'F1':>6} | {'Avg time':>8}")
        print(f"  {'─'*55}")
        for _, row in summary.iterrows():
            print(f"  {row['model_size']:>5} | {row['anomaly_set']:<8} | "
                  f"{row['mean_p']:>8.0%} | {row['mean_r']:>5.0%} | "
                  f"{row['mean_f1']:>5.0%} | {row['mean_time']:>7.1f}s")

        # Total por modelo
        total = results.groupby("model_size").agg(
            mean_f1=("f1", "mean"),
            total_time=("time_s", "sum"),
        ).reset_index()

        print(f"\n  TOTAL:")
        for _, row in total.iterrows():
            print(f"    {row['model_size']:>5}: F1={row['mean_f1']:.0%}, "
                  f"tiempo total={row['total_time']:.0f}s")


if __name__ == "__main__":
    main()
