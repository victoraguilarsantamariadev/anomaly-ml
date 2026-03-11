"""
Meter Readings Distribution Detector -- deep analysis of ~4.5M individual counter readings.

Exploits distribution-level features from individual meter readings that the existing
M10 detector barely uses. Instead of just computing anomaly rates, this module computes
per-month distributional statistics (Gini, entropy, KS test, skewness, kurtosis, etc.)
that reveal systemic anomalies invisible to per-reading outlier detection.

Data: data/m3-registrados_facturados-tll_{year}-solo-alicante-*.csv (~750K rows/year)
Columns: Explotacion, Fecha Factura, Periodo, Periodicidad, Dias Lectura,
         Fecha Lectura, Fecha Prevista Lectura, M3 A Facturar

Usage:
  from meter_readings_detector import run_meter_analysis, meter_readings_summary
  results = run_meter_analysis()
  meter_readings_summary(results)
"""

import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp, entropy as shannon_entropy, skew, kurtosis

__all__ = ["run_meter_analysis", "meter_readings_summary"]

DATA_DIR = Path("data")
FILE_PATTERN = "m3-registrados_facturados-tll_{year}-solo-alicante-m3-registrados_facturados-tll_{year}-solo-alicant.csv"


# ---------------------------------------------------------------------------
# Function 1: Load readings
# ---------------------------------------------------------------------------

def load_readings_sample(years=(2020, 2021, 2022, 2023, 2024, 2025), sample_frac=0.3) -> pd.DataFrame:
    """
    Load individual meter readings for the specified years.

    Processes year-by-year to limit peak memory usage.  Optionally samples
    rows (default 30 %) so that the full 4.5 M-row dataset stays manageable.
    Computes derived columns: m3_per_day, year_month, delay_days.

    Parameters
    ----------
    years : iterable of int
        Which yearly CSV files to load (default 2022-2024).
    sample_frac : float or None
        Fraction of rows to keep per file (0 < frac <= 1).
        Pass None or 1.0 to keep everything.

    Returns
    -------
    pd.DataFrame  with columns:
        Explotacion, Fecha Factura, Periodo, Periodicidad, Dias Lectura,
        Fecha Lectura, Fecha Prevista Lectura, M3 A Facturar,
        fecha_factura, fecha_lectura, fecha_prevista,
        m3, dias, m3_per_day, year_month, delay_days
    """
    frames = []
    for year in sorted(years):
        fname = FILE_PATTERN.format(year=year)
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  [meter_readings] WARNING: {path} not found, skipping.")
            continue

        df = pd.read_csv(path)
        print(f"  [meter_readings] {path.name}: {len(df):,} rows", end="")

        # Sample if requested
        if sample_frac is not None and 0 < sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
            print(f" -> sampled {len(df):,}", end="")
        print()

        frames.append(df)

    if not frames:
        print("  [meter_readings] ERROR: No data files found.")
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    del frames  # free memory early

    # ---- Parse dates ----
    all_df["fecha_factura"] = pd.to_datetime(
        all_df["Fecha Factura"], format="%d/%m/%Y", dayfirst=True, errors="coerce"
    )
    all_df["fecha_lectura"] = pd.to_datetime(
        all_df["Fecha Lectura"], format="%d/%m/%Y", dayfirst=True, errors="coerce"
    )
    all_df["fecha_prevista"] = pd.to_datetime(
        all_df["Fecha Prevista Lectura"], format="%d/%m/%Y", dayfirst=True, errors="coerce"
    )

    # ---- Numeric columns (float32 for memory) ----
    all_df["m3"] = pd.to_numeric(all_df["M3 A Facturar"], errors="coerce").fillna(0).astype(np.float32)
    all_df["dias"] = pd.to_numeric(all_df["Dias Lectura"], errors="coerce").fillna(0).astype(np.float32)

    # ---- Derived columns ----
    all_df["m3_per_day"] = np.where(
        all_df["dias"] > 0,
        all_df["m3"] / all_df["dias"],
        np.float32(0),
    ).astype(np.float32)

    all_df["year_month"] = all_df["fecha_factura"].dt.to_period("M")

    all_df["delay_days"] = (
        (all_df["fecha_lectura"] - all_df["fecha_prevista"]).dt.days
    ).astype("Int32")  # nullable int for NaT cases

    return all_df


# ---------------------------------------------------------------------------
# Function 2: Distribution features per month
# ---------------------------------------------------------------------------

def _gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient of a 1-D array (values >= 0)."""
    v = np.sort(np.abs(values))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float(((2 * index - n - 1) * v).sum() / (n * v.sum()))


def _binned_entropy(values: np.ndarray, n_bins: int = 50) -> float:
    """Shannon entropy of the histogram of *values*."""
    counts, _ = np.histogram(values, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(shannon_entropy(probs, base=2))


def compute_distribution_features(readings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distributional features per year_month that capture systemic
    anomalies invisible at the individual-reading level.

    Features
    --------
    pct_zero        % of readings with M3 == 0
    pct_negative    % of readings with M3 < 0
    pct_extreme     % of readings with m3_per_day > global 99th percentile
    gini_coefficient  Gini of consumption distribution (inequality)
    entropy         Shannon entropy of binned consumption
    mean_delay      Mean delay_days
    pct_late        % of readings taken > 5 days after scheduled
    ks_statistic    KS stat vs rolling 6-month baseline distribution
    skewness        Skewness of m3_per_day
    kurtosis        Kurtosis of m3_per_day

    Returns
    -------
    pd.DataFrame indexed by year_month with the features above plus
    n_readings (count).
    """
    df = readings_df.copy()

    # Global 99th percentile for extreme detection
    positive_m3pd = df.loc[df["m3_per_day"] > 0, "m3_per_day"]
    p99 = float(np.percentile(positive_m3pd.dropna(), 99)) if len(positive_m3pd) > 0 else 1.0

    # Sort months chronologically
    months_sorted = sorted(df["year_month"].dropna().unique())

    # Pre-compute per-month m3_per_day arrays for the KS rolling baseline
    month_arrays = {}
    for ym in months_sorted:
        mask = df["year_month"] == ym
        month_arrays[ym] = df.loc[mask, "m3_per_day"].dropna().values.astype(np.float64)

    records = []
    for i, ym in enumerate(months_sorted):
        mdf = df[df["year_month"] == ym]
        n = len(mdf)
        if n == 0:
            continue

        m3_vals = mdf["m3"].values
        m3pd_vals = mdf["m3_per_day"].values.astype(np.float64)
        delay_vals = mdf["delay_days"].dropna().values.astype(np.float64)

        # Basic percentage features
        pct_zero = float((m3_vals == 0).sum()) / n * 100
        pct_negative = float((m3_vals < 0).sum()) / n * 100
        pct_extreme = float((mdf["m3_per_day"].values > p99).sum()) / n * 100

        # Distributional shape
        gini = _gini(m3pd_vals[m3pd_vals >= 0])
        ent = _binned_entropy(m3pd_vals, n_bins=50)

        # Delay features
        mean_delay = float(np.nanmean(delay_vals)) if len(delay_vals) > 0 else 0.0
        pct_late = float((delay_vals > 5).sum()) / max(len(delay_vals), 1) * 100

        # KS statistic: compare this month vs rolling 6-month baseline
        baseline_months = months_sorted[max(0, i - 6):i]
        if len(baseline_months) >= 1:
            baseline_arr = np.concatenate([month_arrays[m] for m in baseline_months])
            # Subsample if baseline is very large to keep KS fast
            if len(baseline_arr) > 50000:
                rng = np.random.RandomState(42)
                baseline_arr = rng.choice(baseline_arr, 50000, replace=False)
            current_arr = m3pd_vals
            if len(current_arr) > 50000:
                rng = np.random.RandomState(42)
                current_arr = rng.choice(current_arr, 50000, replace=False)
            ks_stat, _ = ks_2samp(current_arr, baseline_arr)
        else:
            ks_stat = 0.0

        # Skewness and kurtosis
        sk = float(skew(m3pd_vals, nan_policy="omit")) if n > 2 else 0.0
        ku = float(kurtosis(m3pd_vals, nan_policy="omit")) if n > 3 else 0.0

        records.append({
            "year_month": ym,
            "n_readings": n,
            "pct_zero": pct_zero,
            "pct_negative": pct_negative,
            "pct_extreme": pct_extreme,
            "gini_coefficient": gini,
            "entropy": ent,
            "mean_delay": mean_delay,
            "pct_late": pct_late,
            "ks_statistic": ks_stat,
            "skewness": sk,
            "kurtosis": ku,
        })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("year_month").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Function 3: Detect distribution-level anomalies
# ---------------------------------------------------------------------------

DISTRIBUTION_FEATURES = [
    "pct_zero", "pct_negative", "pct_extreme", "gini_coefficient",
    "entropy", "mean_delay", "pct_late", "ks_statistic",
    "skewness", "kurtosis",
]


def detect_reading_anomalies(dist_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag months where the consumption distribution is anomalous.

    Method:
      - Z-score each distributional feature against its own history.
      - A month is anomalous if >= 3 features have |z| > 2.
      - Composite score = mean of |z-scores| of anomalous features.

    Returns
    -------
    pd.DataFrame  with original columns plus:
        z_{feature}  per-feature z-scores
        n_anomalous_features
        is_anomaly_readings_dist   (bool)
        reading_distribution_score (float)
    """
    df = dist_features_df.copy()

    z_cols = []
    for feat in DISTRIBUTION_FEATURES:
        if feat not in df.columns:
            continue
        col = df[feat].astype(np.float64)
        mu = col.mean()
        sigma = col.std()
        z_name = f"z_{feat}"
        df[z_name] = (col - mu) / sigma if sigma > 0 else 0.0
        z_cols.append(z_name)

    # Count how many features are anomalous per row (|z| > 2)
    z_matrix = df[z_cols].abs()
    df["n_anomalous_features"] = (z_matrix > 2).sum(axis=1)
    df["is_anomaly_readings_dist"] = df["n_anomalous_features"] >= 3

    # Composite score: mean |z| of the features that are anomalous (|z|>2)
    def _composite_score(row):
        vals = []
        for zc in z_cols:
            if abs(row[zc]) > 2:
                vals.append(abs(row[zc]))
        return np.mean(vals) if vals else 0.0

    df["reading_distribution_score"] = df.apply(_composite_score, axis=1)

    return df


# ---------------------------------------------------------------------------
# Function 4: Main entry point
# ---------------------------------------------------------------------------

def run_meter_analysis(results_df=None,
                       years=(2020, 2021, 2022, 2023, 2024, 2025),
                       sample_frac=0.3) -> dict:
    """
    Run the full meter-readings distribution analysis.

    Parameters
    ----------
    results_df : pd.DataFrame or None
        If provided, must contain 'year_month' (Period[M]) and numeric model
        detection columns so we can correlate distributional anomalies with
        model detections.
    years : tuple of int
        Years to load.
    sample_frac : float
        Sampling fraction per file (0-1).  Use None / 1.0 for full data.

    Returns
    -------
    dict with keys:
        dist_features      : pd.DataFrame  (monthly distribution features)
        anomalous_months   : pd.DataFrame  (subset flagged as anomalous)
        correlation_with_models : dict or None
    """
    print("=" * 70)
    print("  Meter Readings Distribution Analysis")
    print("=" * 70)

    # Step 1 -- load
    print("\n  Step 1: Loading readings...")
    readings = load_readings_sample(years=years, sample_frac=sample_frac)
    if readings.empty:
        return {"dist_features": pd.DataFrame(), "anomalous_months": pd.DataFrame(),
                "correlation_with_models": None}

    n_total = len(readings)
    n_months = readings["year_month"].nunique()
    print(f"  Loaded {n_total:,} readings across {n_months} months.")

    # Step 2 -- distribution features
    print("\n  Step 2: Computing distribution features...")
    dist_features = compute_distribution_features(readings)
    del readings  # free memory

    # Step 3 -- detect anomalies
    print("\n  Step 3: Detecting distribution-level anomalies...")
    dist_features = detect_reading_anomalies(dist_features)
    anomalous = dist_features[dist_features["is_anomaly_readings_dist"]].copy()
    print(f"  Found {len(anomalous)} anomalous months out of {len(dist_features)}.")

    # Step 4 -- correlate with model results if provided
    correlation = None
    if results_df is not None and not results_df.empty:
        correlation = _correlate_with_models(dist_features, results_df)

    return {
        "dist_features": dist_features,
        "anomalous_months": anomalous,
        "correlation_with_models": correlation,
    }


def _correlate_with_models(dist_features: pd.DataFrame,
                           results_df: pd.DataFrame) -> dict:
    """
    Correlate the reading_distribution_score with model detection columns.

    Returns dict mapping model_column -> pearson_r.
    """
    # Ensure year_month in results_df is Period[M]
    rdf = results_df.copy()
    if "year_month" not in rdf.columns:
        print("  [correlation] WARNING: results_df has no 'year_month'. Skipping.")
        return {}

    if not hasattr(rdf["year_month"].dtype, "freq"):
        try:
            rdf["year_month"] = rdf["year_month"].apply(
                lambda x: pd.Period(str(x), freq="M")
            )
        except Exception:
            print("  [correlation] WARNING: Cannot convert year_month to Period. Skipping.")
            return {}

    # Aggregate model detections per year_month (mean detection rate)
    model_cols = [c for c in rdf.columns if c.startswith("is_anomaly") or c.startswith("M")]
    numeric_model_cols = []
    for c in model_cols:
        if pd.api.types.is_numeric_dtype(rdf[c]):
            numeric_model_cols.append(c)

    if not numeric_model_cols:
        # Try to detect boolean columns
        for c in rdf.columns:
            if rdf[c].dtype == bool:
                rdf[c] = rdf[c].astype(int)
                numeric_model_cols.append(c)

    if not numeric_model_cols:
        print("  [correlation] WARNING: No numeric model columns found.")
        return {}

    agg_dict = {c: "mean" for c in numeric_model_cols}
    monthly_models = rdf.groupby("year_month").agg(agg_dict).reset_index()

    merged = pd.merge(
        dist_features[["year_month", "reading_distribution_score"]],
        monthly_models,
        on="year_month",
        how="inner",
    )

    correlations = {}
    for c in numeric_model_cols:
        if merged[c].std() > 0 and merged["reading_distribution_score"].std() > 0:
            r = merged["reading_distribution_score"].corr(merged[c])
            correlations[c] = round(r, 4)

    if correlations:
        print(f"  [correlation] Computed correlations for {len(correlations)} model columns.")
    return correlations


# ---------------------------------------------------------------------------
# Function 5: Summary printer
# ---------------------------------------------------------------------------

def meter_readings_summary(analysis_results: dict):
    """
    Print a compelling human-readable summary of the meter readings analysis.

    Parameters
    ----------
    analysis_results : dict
        Output of run_meter_analysis().
    """
    dist = analysis_results.get("dist_features", pd.DataFrame())
    anomalous = analysis_results.get("anomalous_months", pd.DataFrame())
    correlations = analysis_results.get("correlation_with_models")

    if dist.empty:
        print("  No distribution features computed. Nothing to summarise.")
        return

    total_readings = int(dist["n_readings"].sum())
    n_months = len(dist)

    print()
    print("=" * 70)
    print("  METER READINGS DISTRIBUTION ANALYSIS -- SUMMARY")
    print("=" * 70)

    # ---- Scale ----
    millions = total_readings / 1_000_000
    print(f"\n  Analyzed {millions:.2f} million individual readings across {n_months} months.")

    # ---- Zero-reading rates ----
    print(f"\n  ZERO-READING RATES (M3 = 0):")
    print(f"  {'─' * 55}")
    mean_zero = dist["pct_zero"].mean()
    min_zero = dist["pct_zero"].min()
    max_zero = dist["pct_zero"].max()
    print(f"    Average: {mean_zero:.1f}%   Min: {min_zero:.1f}%   Max: {max_zero:.1f}%")
    # Trend (first half vs second half)
    half = n_months // 2
    if half > 0:
        first_half = dist.iloc[:half]["pct_zero"].mean()
        second_half = dist.iloc[half:]["pct_zero"].mean()
        direction = "INCREASING" if second_half > first_half else "DECREASING"
        print(f"    Trend: {direction} ({first_half:.1f}% -> {second_half:.1f}%)")

    # ---- Negative readings ----
    mean_neg = dist["pct_negative"].mean()
    if mean_neg > 0:
        print(f"\n  NEGATIVE READINGS (M3 < 0, impossible -- counter manipulation?):")
        print(f"  {'─' * 55}")
        print(f"    Average: {mean_neg:.3f}%")
        worst_neg = dist.loc[dist["pct_negative"].idxmax()]
        print(f"    Worst month: {worst_neg['year_month']} ({worst_neg['pct_negative']:.3f}%)")

    # ---- Distribution shifts (high KS) ----
    print(f"\n  DISTRIBUTION SHIFT MONTHS (high KS statistic):")
    print(f"  {'─' * 55}")
    ks_threshold = dist["ks_statistic"].mean() + 2 * dist["ks_statistic"].std()
    high_ks = dist[dist["ks_statistic"] > ks_threshold].sort_values("ks_statistic", ascending=False)
    if len(high_ks) > 0:
        for _, row in high_ks.head(5).iterrows():
            print(f"    {row['year_month']}:  KS = {row['ks_statistic']:.4f}")
    else:
        print(f"    No months with KS > {ks_threshold:.4f} (mean + 2*std)")

    # ---- Gini coefficient ----
    print(f"\n  CONSUMPTION INEQUALITY (Gini coefficient):")
    print(f"  {'─' * 55}")
    mean_gini = dist["gini_coefficient"].mean()
    print(f"    Average Gini: {mean_gini:.4f} (1 = perfect inequality, 0 = perfect equality)")
    if half > 0:
        gini_first = dist.iloc[:half]["gini_coefficient"].mean()
        gini_second = dist.iloc[half:]["gini_coefficient"].mean()
        gini_dir = "INCREASING" if gini_second > gini_first else "DECREASING"
        print(f"    Trend: {gini_dir} ({gini_first:.4f} -> {gini_second:.4f})")

    # ---- Entropy ----
    print(f"\n  CONSUMPTION ENTROPY (low = suspicious uniformity):")
    print(f"  {'─' * 55}")
    mean_ent = dist["entropy"].mean()
    min_ent_row = dist.loc[dist["entropy"].idxmin()]
    print(f"    Average: {mean_ent:.2f} bits")
    print(f"    Lowest:  {min_ent_row['year_month']} ({min_ent_row['entropy']:.2f} bits)")

    # ---- Delay patterns ----
    print(f"\n  READING DELAYS:")
    print(f"  {'─' * 55}")
    mean_late = dist["pct_late"].mean()
    max_late_row = dist.loc[dist["pct_late"].idxmax()]
    print(f"    Average pct late (>5 days): {mean_late:.1f}%")
    print(f"    Worst month: {max_late_row['year_month']} ({max_late_row['pct_late']:.1f}%)")

    # ---- Anomalous months ----
    print(f"\n  ANOMALOUS MONTHS (>= 3 distributional features with |z| > 2):")
    print(f"  {'─' * 70}")
    if len(anomalous) > 0:
        print(f"  {'Month':>10} {'Score':>8} {'#Anom':>6}  Key features")
        print(f"  {'─' * 70}")
        for _, row in anomalous.sort_values("reading_distribution_score", ascending=False).iterrows():
            # Identify which features drove the anomaly
            drivers = []
            for feat in DISTRIBUTION_FEATURES:
                zc = f"z_{feat}"
                if zc in row.index and abs(row[zc]) > 2:
                    drivers.append(f"{feat}(z={row[zc]:+.1f})")
            drivers_str = ", ".join(drivers[:4])
            print(f"  {str(row['year_month']):>10} {row['reading_distribution_score']:>8.2f} "
                  f"{int(row['n_anomalous_features']):>6}  {drivers_str}")
    else:
        print("    No months flagged as anomalous.")

    # ---- Correlation with models ----
    if correlations:
        print(f"\n  CORRELATION WITH MODEL DETECTIONS:")
        print(f"  {'─' * 55}")
        for col, r in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"    {col:>35s}:  R = {r:+.4f}")
        top_r = max(abs(v) for v in correlations.values())
        print(f"\n  Months with highest distribution anomalies correlate "
              f"R={top_r:.3f} with multi-model detections.")

    print(f"\n{'=' * 70}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Meter Readings Distribution Detector"
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025],
        help="Years to analyse (default: 2020-2025)",
    )
    parser.add_argument(
        "--sample-frac", type=float, default=0.3,
        help="Fraction of rows to sample per file (default: 0.3). Use 1.0 for full data.",
    )
    parser.add_argument(
        "--results-csv", type=str, default=None,
        help="Path to results CSV with model detection columns for correlation.",
    )
    args = parser.parse_args()

    results_df = None
    if args.results_csv:
        results_df = pd.read_csv(args.results_csv)

    analysis = run_meter_analysis(
        results_df=results_df,
        years=tuple(args.years),
        sample_frac=args.sample_frac if args.sample_frac < 1.0 else None,
    )
    meter_readings_summary(analysis)
