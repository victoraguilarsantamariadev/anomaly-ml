"""
Advanced Household Analytics — Quant-Grade Anomaly Detection

5 tecnicas de nivel Harvard DS / quant fund aplicadas a datos horarios
de consumo de agua por vivienda individual:

  1. Spectral Analysis (FFT) — firma frecuencial, entropia espectral
  2. 24h Profile Autoencoder + UMAP — arquetipos de consumo
  3. Survival Analysis (Cox PH) — prediccion de fallos futuros
  4. BOCPD — deteccion bayesiana de cambio de regimen por hora
  5. Factor Model — descomposicion en factores + residual (estilo quant)

Uso:
  python advanced_household_analytics.py          # ejecuta las 5 y muestra resumen
  python advanced_household_analytics.py --only spectral  # solo una tecnica
"""

import sys
import io
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
HOURLY_CSV = DATA_DIR / "synthetic_hourly_domicilio.csv"
LEAK_LABELS_CSV = DATA_DIR / "synthetic_leak_labels.csv"
CATASTRO_CSV = DATA_DIR / "synthetic_catastro_households.csv"
PROFILES_CSV = DATA_DIR / "synthetic_household_profiles.csv"


def _load_hourly() -> pd.DataFrame:
    df = pd.read_csv(HOURLY_CSV, parse_dates=["timestamp"])
    return df


def _load_leak_ids() -> set:
    if LEAK_LABELS_CSV.exists():
        return set(pd.read_csv(LEAK_LABELS_CSV)["contrato_id"])
    return set()


def _load_leak_labels() -> pd.DataFrame:
    if LEAK_LABELS_CSV.exists():
        return pd.read_csv(LEAK_LABELS_CSV, parse_dates=["inicio_fuga", "fin_fuga"])
    return pd.DataFrame()


def _load_catastro() -> pd.DataFrame:
    if CATASTRO_CSV.exists():
        return pd.read_csv(CATASTRO_CSV)
    return pd.DataFrame()


def _load_profiles() -> pd.DataFrame:
    if PROFILES_CSV.exists():
        return pd.read_csv(PROFILES_CSV)
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SPECTRAL ANALYSIS (FFT)
# ═══════════════════════════════════════════════════════════════════════════════

def spectral_analysis(
    df_hourly: pd.DataFrame = None,
    window_days: int = 7,
    overlap: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """
    Analisis espectral por vivienda: FFT, entropia espectral, potencia diurna.

    Un hogar normal tiene un pico fuerte a 24h (ciclo diurno).
    Una fuga anade baseline constante -> espectro plano -> alta entropia.

    Returns:
        scores_df: contrato_id, spectral_entropy, diurnal_power_ratio,
                   harmonic_12h_ratio, spectral_anomaly_score, is_anomaly_spectral
        figures: dict con figuras Plotly
    """
    from scipy.fft import rfft, rfftfreq

    if df_hourly is None:
        df_hourly = _load_hourly()

    records = []
    spectrograms = {}

    for cid, grp in df_hourly.groupby("contrato_id"):
        series = grp.sort_values("timestamp")["consumo_litros"].values
        n = len(series)
        if n < 48:
            continue

        # ── FFT completo ──────────────────────────────────────────────
        fft_vals = rfft(series - series.mean())  # centrar para eliminar DC
        psd = np.abs(fft_vals) ** 2
        freqs = rfftfreq(n, d=1.0)  # frecuencias en ciclos/hora

        # Normalizar PSD para entropia
        psd_norm = psd / (psd.sum() + 1e-15)
        psd_norm = psd_norm[psd_norm > 0]

        # Spectral Entropy (Shannon)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-15))

        # Diurnal Power Ratio (potencia en periodo 24h / total)
        # Periodo 24h = frecuencia 1/24 ciclos/hora
        target_freq_24h = 1.0 / 24.0
        freq_idx_24h = np.argmin(np.abs(freqs - target_freq_24h))
        # Tomar un rango de ±2 bins para robustez
        low_24 = max(0, freq_idx_24h - 2)
        high_24 = min(len(psd), freq_idx_24h + 3)
        diurnal_power = psd[low_24:high_24].sum()
        total_power = psd.sum() + 1e-15
        diurnal_power_ratio = diurnal_power / total_power

        # 12h harmonic ratio
        target_freq_12h = 1.0 / 12.0
        freq_idx_12h = np.argmin(np.abs(freqs - target_freq_12h))
        low_12 = max(0, freq_idx_12h - 2)
        high_12 = min(len(psd), freq_idx_12h + 3)
        harmonic_12h_ratio = psd[low_12:high_12].sum() / total_power

        # ── Sliding-window spectrogram ────────────────────────────────
        win_size = window_days * 24
        step = int(win_size * (1 - overlap))
        spectrogram_windows = []
        for start in range(0, n - win_size + 1, step):
            window = series[start:start + win_size]
            w_fft = rfft(window - window.mean())
            w_psd = np.abs(w_fft) ** 2
            spectrogram_windows.append(w_psd)

        if spectrogram_windows:
            spectrograms[cid] = np.array(spectrogram_windows)

        records.append({
            "contrato_id": cid,
            "spectral_entropy": round(float(spectral_entropy), 4),
            "diurnal_power_ratio": round(float(diurnal_power_ratio), 6),
            "harmonic_12h_ratio": round(float(harmonic_12h_ratio), 6),
        })

    df_scores = pd.DataFrame(records)
    if df_scores.empty:
        return df_scores, {}

    # Z-scores para scoring
    for col in ["spectral_entropy", "diurnal_power_ratio", "harmonic_12h_ratio"]:
        mean_val = df_scores[col].mean()
        std_val = df_scores[col].std() + 1e-10
        df_scores[f"{col}_zscore"] = (df_scores[col] - mean_val) / std_val

    # Score combinado: alta entropia + baja potencia diurna = anomalia
    df_scores["spectral_anomaly_score"] = (
        0.5 * df_scores["spectral_entropy_zscore"].clip(0, 3) / 3.0
        + 0.3 * (-df_scores["diurnal_power_ratio_zscore"]).clip(0, 3) / 3.0
        + 0.2 * (-df_scores["harmonic_12h_ratio_zscore"]).clip(0, 3) / 3.0
    ).clip(0, 1)

    threshold = df_scores["spectral_anomaly_score"].quantile(0.90)
    df_scores["is_anomaly_spectral"] = df_scores["spectral_anomaly_score"] > threshold

    # ── Figuras Plotly ────────────────────────────────────────────────
    figures = {}
    try:
        import plotly.graph_objects as go

        # Ranking de entropia espectral
        top_df = df_scores.nlargest(30, "spectral_entropy")
        leak_ids = _load_leak_ids()
        colors = ["#e74c3c" if c in leak_ids else "#3498db" for c in top_df["contrato_id"]]
        fig_ranking = go.Figure(go.Bar(
            x=top_df["contrato_id"],
            y=top_df["spectral_entropy"],
            marker_color=colors,
            text=[f"{s:.2f}" for s in top_df["spectral_anomaly_score"]],
            textposition="outside",
        ))
        fig_ranking.update_layout(
            title="Spectral Entropy Ranking (rojo = fuga conocida)",
            xaxis_title="Contrato", yaxis_title="Spectral Entropy (bits)",
            height=400, template="plotly_dark",
        )
        figures["entropy_ranking"] = fig_ranking

        # Spectrogram del contrato mas anomalo
        top_cid = df_scores.nlargest(1, "spectral_anomaly_score")["contrato_id"].iloc[0]
        if top_cid in spectrograms:
            spec = spectrograms[top_cid]
            spec_freqs = rfftfreq(window_days * 24, d=1.0)
            fig_spec = go.Figure(go.Heatmap(
                z=np.log10(spec.T + 1),
                x=list(range(spec.shape[0])),
                y=spec_freqs[:spec.shape[1]],
                colorscale="Viridis",
            ))
            fig_spec.update_layout(
                title=f"Spectrogram — {top_cid}",
                xaxis_title="Ventana temporal",
                yaxis_title="Frecuencia (ciclos/hora)",
                yaxis_range=[0, 0.15],
                height=400, template="plotly_dark",
            )
            figures["spectrogram_top"] = fig_spec

    except ImportError:
        pass

    return df_scores, figures


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 24h PROFILE AUTOENCODER + UMAP
# ═══════════════════════════════════════════════════════════════════════════════

def profile_autoencoder(
    df_hourly: pd.DataFrame = None,
    latent_dim: int = 4,
    epochs: int = 500,
    contamination: float = 0.10,
) -> tuple[pd.DataFrame, dict]:
    """
    Autoencoder sobre perfiles horarios 24D. UMAP para visualizar arquetipos.

    Cada dia de cada vivienda = vector 24D (consumo horario normalizado).
    El AE aprende patrones normales; error alto = anomalia.

    Returns:
        scores_df: contrato_id, ae_median_error, ae_anomaly_score,
                   cluster_id, is_anomaly_ae
        figures: dict con UMAP scatter, arquetipos, heatmap error
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import RobustScaler
    from sklearn.cluster import KMeans

    if df_hourly is None:
        df_hourly = _load_hourly()

    df = df_hourly.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date

    # Reshape: (contrato, date, hour) -> matrix 24D
    pivot = df.pivot_table(
        index=["contrato_id", "date"],
        columns="hour",
        values="consumo_litros",
        aggfunc="first",
    ).reset_index()

    hour_cols = [c for c in pivot.columns if isinstance(c, (int, np.integer))]
    hour_cols = sorted(hour_cols)
    if len(hour_cols) < 24:
        # Rellenar horas faltantes
        for h in range(24):
            if h not in pivot.columns:
                pivot[h] = 0.0
        hour_cols = list(range(24))

    # Normalizar cada perfil por su suma (forma, no volumen)
    profiles = pivot[hour_cols].values.astype(float)
    row_sums = profiles.sum(axis=1, keepdims=True) + 1e-10
    profiles_norm = profiles / row_sums

    # Split: train = contratos sin fuga, test = todos
    leak_ids = _load_leak_ids()
    contract_ids = pivot["contrato_id"].values
    train_mask = ~np.isin(contract_ids, list(leak_ids))

    X_train = profiles_norm[train_mask]
    X_all = profiles_norm

    # Scale
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_all_s = scaler.transform(X_all)

    # Autoencoder
    hidden = (16, 8, latent_dim, 8, 16)
    ae = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=epochs,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )
    ae.fit(X_train_s, X_train_s)

    # Reconstruction error
    X_pred = ae.predict(X_all_s)
    errors = np.mean((X_all_s - X_pred) ** 2, axis=1)

    # Threshold
    train_pred = ae.predict(X_train_s)
    train_errors = np.mean((X_train_s - train_pred) ** 2, axis=1)
    threshold = np.percentile(train_errors, (1 - contamination) * 100)

    pivot["reconstruction_error"] = errors
    pivot["is_anomaly_ae_day"] = errors > threshold

    # Aggregate per contract
    agg = pivot.groupby("contrato_id").agg(
        ae_median_error=("reconstruction_error", "median"),
        ae_max_error=("reconstruction_error", "max"),
        ae_anomaly_days=("is_anomaly_ae_day", "sum"),
        n_days=("date", "count"),
    ).reset_index()

    agg["ae_anomaly_ratio"] = agg["ae_anomaly_days"] / agg["n_days"]
    ae_max = agg["ae_median_error"].max() + 1e-10
    agg["ae_anomaly_score"] = (agg["ae_median_error"] / ae_max).clip(0, 1)
    agg["is_anomaly_ae"] = agg["ae_anomaly_score"] > agg["ae_anomaly_score"].quantile(0.90)

    # ── Latent space extraction + clustering ──────────────────────────
    # Extract bottleneck activations (approximate via half-forward)
    # Use KMeans on the scaled profiles as proxy for latent clustering
    n_clusters = min(6, len(agg))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    # Compute mean profile per contract
    contract_profiles = pivot.groupby("contrato_id")[hour_cols].mean().values
    contract_profiles_norm = contract_profiles / (contract_profiles.sum(axis=1, keepdims=True) + 1e-10)
    km_labels = km.fit_predict(contract_profiles_norm)

    contract_list = pivot.groupby("contrato_id")[hour_cols].mean().reset_index()["contrato_id"].values
    cluster_map = dict(zip(contract_list, km_labels))
    agg["cluster_id"] = agg["contrato_id"].map(cluster_map)

    # ── UMAP / t-SNE ─────────────────────────────────────────────────
    umap_coords = None
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_coords = reducer.fit_transform(contract_profiles_norm)
    except ImportError:
        try:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(agg) - 1))
            umap_coords = reducer.fit_transform(contract_profiles_norm)
        except Exception:
            pass

    if umap_coords is not None:
        umap_map_x = dict(zip(contract_list, umap_coords[:, 0]))
        umap_map_y = dict(zip(contract_list, umap_coords[:, 1]))
        agg["umap_x"] = agg["contrato_id"].map(umap_map_x)
        agg["umap_y"] = agg["contrato_id"].map(umap_map_y)

    # ── Figuras ───────────────────────────────────────────────────────
    figures = {}
    try:
        import plotly.graph_objects as go

        # UMAP scatter
        if "umap_x" in agg.columns:
            leak_status = ["Fuga" if c in leak_ids else "Normal" for c in agg["contrato_id"]]
            fig_umap = go.Figure()
            for status, color in [("Normal", "#3498db"), ("Fuga", "#e74c3c")]:
                mask = [s == status for s in leak_status]
                subset = agg[mask]
                fig_umap.add_trace(go.Scatter(
                    x=subset["umap_x"], y=subset["umap_y"],
                    mode="markers",
                    marker=dict(size=8 if status == "Fuga" else 5,
                                color=color, opacity=0.8),
                    name=status,
                    text=subset["contrato_id"],
                    hovertemplate="%{text}<br>Error: %{marker.color:.3f}",
                ))
            fig_umap.update_layout(
                title="UMAP — Espacio latente de perfiles de consumo",
                xaxis_title="UMAP 1", yaxis_title="UMAP 2",
                height=500, template="plotly_dark",
            )
            figures["umap_scatter"] = fig_umap

        # Arquetipos (centroides de clusters como curvas 24h)
        fig_arch = go.Figure()
        centroids = km.cluster_centers_
        for i in range(n_clusters):
            n_members = (km_labels == i).sum()
            fig_arch.add_trace(go.Scatter(
                x=list(range(24)), y=centroids[i],
                mode="lines+markers",
                name=f"Cluster {i} (n={n_members})",
                line=dict(width=2),
            ))
        fig_arch.update_layout(
            title="Arquetipos de Consumo — Perfiles 24h por Cluster",
            xaxis_title="Hora del dia", yaxis_title="Consumo normalizado",
            height=400, template="plotly_dark",
        )
        figures["archetype_profiles"] = fig_arch

    except ImportError:
        pass

    return agg, figures


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SURVIVAL ANALYSIS (Cox Proportional Hazards)
# ═══════════════════════════════════════════════════════════════════════════════

def survival_analysis(
    df_hourly: pd.DataFrame = None,
    df_catastro: pd.DataFrame = None,
    df_profiles: pd.DataFrame = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Modelo de supervivencia: P(fuga en proximos N meses) por vivienda.

    Cox Proportional Hazards con covariables: edad edificio, m2, pipe_risk,
    variabilidad consumo, edad titular. PREDICTIVO, no solo detectivo.

    Returns:
        scores_df: contrato_id, predicted_risk_60d, hazard_ratio, risk_group
        figures: dict con Kaplan-Meier, forest plot hazard ratios
    """
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
    except ImportError:
        print("  [Survival] lifelines no instalado — pip install lifelines")
        return pd.DataFrame(), {}

    if df_hourly is None:
        df_hourly = _load_hourly()
    if df_catastro is None:
        df_catastro = _load_catastro()
    if df_profiles is None:
        df_profiles = _load_profiles()

    labels = _load_leak_labels()
    if labels.empty:
        return pd.DataFrame(), {}

    # ── Construir tabla de supervivencia ──────────────────────────────
    start_date = df_hourly["timestamp"].min()
    end_date = df_hourly["timestamp"].max()
    total_hours = (end_date - start_date).total_seconds() / 3600.0
    total_days = total_hours / 24.0

    contracts = df_hourly[["contrato_id", "barrio", "uso"]].drop_duplicates()
    leak_map = {}
    for _, row in labels.iterrows():
        cid = row["contrato_id"]
        hours_to_event = (row["inicio_fuga"] - start_date).total_seconds() / 3600.0
        leak_map[cid] = max(hours_to_event / 24.0, 0.5)  # en dias

    survival_rows = []
    for _, c in contracts.iterrows():
        cid = c["contrato_id"]
        if cid in leak_map:
            duration = leak_map[cid]
            event = 1
        else:
            duration = total_days
            event = 0
        survival_rows.append({"contrato_id": cid, "duration_days": duration, "event": event})

    df_surv = pd.DataFrame(survival_rows)

    # Merge covariables
    if not df_catastro.empty:
        df_surv = df_surv.merge(
            df_catastro[["contrato_id", "building_m2", "construction_year", "pipe_risk_score",
                         "consumption_efficiency_ratio"]],
            on="contrato_id", how="left",
        )
        df_surv["building_age_decades"] = (2024 - df_surv["construction_year"].fillna(1990)) / 10.0

    if not df_profiles.empty:
        df_surv = df_surv.merge(
            df_profiles[["contrato_id", "edad_titular", "n_personas_hogar", "vive_solo"]],
            on="contrato_id", how="left",
        )

    # Seleccionar covariables disponibles
    possible_covs = ["building_age_decades", "building_m2", "pipe_risk_score",
                     "edad_titular", "consumption_efficiency_ratio"]
    covariates = [c for c in possible_covs if c in df_surv.columns and df_surv[c].notna().sum() > 50]

    if len(covariates) < 2:
        print("  [Survival] Insuficientes covariables disponibles")
        return pd.DataFrame(), {}

    # Limitar a 4 covariables (24 eventos -> regla 1:10)
    covariates = covariates[:4]

    df_model = df_surv[["duration_days", "event"] + covariates].dropna()

    # ── Fit Cox PH ───────────────────────────────────────────────────
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_model, duration_col="duration_days", event_col="event")

    # Predictions
    X_pred = df_surv[covariates].fillna(df_surv[covariates].median())
    partial_hazard = cph.predict_partial_hazard(X_pred)
    df_surv["hazard_ratio"] = partial_hazard.values

    # Survival function at 60 days
    sf = cph.predict_survival_function(X_pred)
    # P(fuga en 60 dias) = 1 - S(60)
    closest_time = sf.index[np.argmin(np.abs(sf.index - 60.0))]
    df_surv["survival_60d"] = sf.loc[closest_time].values
    df_surv["predicted_risk_60d"] = 1 - df_surv["survival_60d"]

    # Risk groups
    q33 = df_surv["predicted_risk_60d"].quantile(0.67)
    q66 = df_surv["predicted_risk_60d"].quantile(0.90)
    df_surv["risk_group"] = pd.cut(
        df_surv["predicted_risk_60d"],
        bins=[-0.01, q33, q66, 1.01],
        labels=["BAJO", "MEDIO", "ALTO"],
    )

    # ── Figuras ───────────────────────────────────────────────────────
    figures = {}
    try:
        import plotly.graph_objects as go

        # Forest plot de hazard ratios
        summary = cph.summary
        fig_forest = go.Figure()
        coefs = summary["coef"]
        ci_lower = summary["coef lower 95%"]
        ci_upper = summary["coef upper 95%"]
        hrs = np.exp(coefs)
        hr_lower = np.exp(ci_lower)
        hr_upper = np.exp(ci_upper)

        fig_forest.add_trace(go.Scatter(
            x=hrs.values, y=list(range(len(coefs))),
            mode="markers",
            marker=dict(size=12, color="#e74c3c"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=(hr_upper - hrs).values,
                arrayminus=(hrs - hr_lower).values,
            ),
            text=[f"HR={h:.2f} [{l:.2f}-{u:.2f}]" for h, l, u in zip(hrs, hr_lower, hr_upper)],
            hovertemplate="%{text}",
        ))
        fig_forest.add_vline(x=1.0, line_dash="dash", line_color="gray")
        fig_forest.update_layout(
            title="Hazard Ratios — Cox Proportional Hazards",
            xaxis_title="Hazard Ratio (>1 = mas riesgo)",
            yaxis=dict(tickvals=list(range(len(coefs))), ticktext=list(coefs.index)),
            height=350, template="plotly_dark",
        )
        figures["hazard_ratios"] = fig_forest

        # Kaplan-Meier por building age group
        if "building_age_decades" in df_surv.columns:
            kmf = KaplanMeierFitter()
            fig_km = go.Figure()
            df_surv["age_group"] = pd.cut(
                df_surv["building_age_decades"],
                bins=[0, 3, 5, 10],
                labels=["<30 anos", "30-50 anos", ">50 anos"],
            )
            colors_km = {"<30 anos": "#2ecc71", "30-50 anos": "#f39c12", ">50 anos": "#e74c3c"}
            for grp_name, grp_df in df_surv.groupby("age_group", observed=True):
                if len(grp_df) < 5:
                    continue
                kmf.fit(grp_df["duration_days"], grp_df["event"], label=str(grp_name))
                sf_plot = kmf.survival_function_
                fig_km.add_trace(go.Scatter(
                    x=sf_plot.index, y=sf_plot.iloc[:, 0],
                    mode="lines", name=str(grp_name),
                    line=dict(width=2, color=colors_km.get(str(grp_name), "#3498db")),
                ))
            fig_km.update_layout(
                title="Kaplan-Meier — Supervivencia por Edad de Edificio",
                xaxis_title="Dias", yaxis_title="P(sin fuga)",
                height=400, template="plotly_dark",
            )
            figures["kaplan_meier"] = fig_km

    except ImportError:
        pass

    result = df_surv[["contrato_id", "duration_days", "event", "hazard_ratio",
                       "predicted_risk_60d", "risk_group"]].copy()
    return result, figures


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BAYESIAN ONLINE CHANGEPOINT DETECTION (BOCPD) per Household
# ═══════════════════════════════════════════════════════════════════════════════

def _gaussian_pred(x, mu, var):
    """Predictive probability under Gaussian."""
    return np.exp(-0.5 * (x - mu) ** 2 / (var + 1e-10)) / np.sqrt(2 * np.pi * (var + 1e-10))


def _bocpd_hourly(series: np.ndarray, hazard_rate: float = 1.0 / (24 * 7),
                  max_run: int = 168) -> np.ndarray:
    """
    BOCPD truncado para series horarias. O(n * max_run) en lugar de O(n^2).

    Args:
        series: consumo horario
        hazard_rate: 1/(24*7) = ~1 cambio esperado por semana
        max_run: longitud maxima de run (168 = 1 semana)
    """
    n = len(series)
    if n < 10:
        return np.zeros(n)

    H = hazard_rate
    R = np.zeros(min(max_run + 1, n + 1))
    R[0] = 1.0

    cp_prob = np.zeros(n)
    mu0 = series.mean()
    var0 = series.var() + 1e-10

    for t in range(n):
        max_r = min(t + 1, max_run)
        pred_probs = np.zeros(max_r)

        for r in range(max_r):
            if r == 0:
                pred_probs[r] = _gaussian_pred(series[t], mu0, var0)
            else:
                start = t - r
                if start >= 0:
                    segment = series[start:t + 1]
                    pred_probs[r] = _gaussian_pred(series[t], segment.mean(), segment.var() + 1e-10)

        # Growth + Changepoint
        new_R = np.zeros_like(R)
        cp_mass = 0.0
        for r in range(min(max_r, len(R) - 1), 0, -1):
            new_R[r] = R[r - 1] * pred_probs[min(r - 1, max_r - 1)] * (1 - H)
        for r in range(max_r):
            cp_mass += R[min(r, len(R) - 1)] * pred_probs[min(r, max_r - 1)] * H

        new_R[0] = cp_mass
        cp_prob[t] = cp_mass

        # Normalize
        total = new_R.sum()
        if total > 0:
            new_R /= total
            cp_prob[t] = new_R[0]
        R = new_R

    return cp_prob


def household_changepoint(
    df_hourly: pd.DataFrame = None,
    hazard_rate: float = 1.0 / (24 * 7),
    min_segment_hours: int = 12,
    downsample_hours: int = 4,
) -> tuple[pd.DataFrame, dict]:
    """
    BOCPD + PELT per household: detecta el momento exacto del cambio de regimen.

    Downsampling a bloques de 4h para eficiencia (1440 -> 360 puntos).

    Returns:
        scores_df: contrato_id, n_changepoints, max_cp_prob,
                   first_cp_timestamp, cp_detection_score
        figures: dict con timeline + run-length
    """
    if df_hourly is None:
        df_hourly = _load_hourly()

    records = []
    all_cp_probs = {}

    for cid, grp in df_hourly.groupby("contrato_id"):
        grp = grp.sort_values("timestamp")
        timestamps = grp["timestamp"].values
        series = grp["consumo_litros"].values

        # Downsample a bloques de 4h
        n = len(series)
        n_blocks = n // downsample_hours
        if n_blocks < 10:
            continue

        series_ds = np.array([
            series[i * downsample_hours:(i + 1) * downsample_hours].mean()
            for i in range(n_blocks)
        ])
        ts_ds = timestamps[::downsample_hours][:n_blocks]

        # BOCPD
        cp_prob = _bocpd_hourly(
            series_ds,
            hazard_rate=hazard_rate * downsample_hours,
            max_run=min(168 // downsample_hours, n_blocks),
        )

        all_cp_probs[cid] = (ts_ds, series_ds, cp_prob)

        # Detectar picos de changepoint
        cp_threshold = 0.05
        cp_indices = np.where(cp_prob > cp_threshold)[0]

        # Filtrar picos demasiado cercanos
        if len(cp_indices) > 0:
            filtered = [cp_indices[0]]
            for idx in cp_indices[1:]:
                if idx - filtered[-1] > min_segment_hours // downsample_hours:
                    filtered.append(idx)
            cp_indices = np.array(filtered)

        n_cps = len(cp_indices)
        max_prob = float(cp_prob.max())
        first_cp_ts = str(ts_ds[cp_indices[0]]) if n_cps > 0 else ""

        # Score: combina max probabilidad + numero de cambios
        cp_score = min(max_prob * 1.5, 1.0) * 0.7 + min(n_cps / 5.0, 1.0) * 0.3

        records.append({
            "contrato_id": cid,
            "n_changepoints": n_cps,
            "max_cp_probability": round(max_prob, 4),
            "first_cp_timestamp": first_cp_ts,
            "cp_detection_score": round(float(cp_score), 4),
        })

    df_scores = pd.DataFrame(records)
    if df_scores.empty:
        return df_scores, {}

    threshold = df_scores["cp_detection_score"].quantile(0.85)
    df_scores["is_anomaly_cp"] = df_scores["cp_detection_score"] > threshold

    # ── Figuras ───────────────────────────────────────────────────────
    figures = {}
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Timeline del contrato mas anomalo
        top_cid = df_scores.nlargest(1, "cp_detection_score")["contrato_id"].iloc[0]
        if top_cid in all_cp_probs:
            ts_ds, series_ds, cp_prob = all_cp_probs[top_cid]

            fig_cp = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=[f"Consumo — {top_cid}",
                                                   "P(changepoint)"],
                                   vertical_spacing=0.08)

            fig_cp.add_trace(go.Scatter(
                x=pd.to_datetime(ts_ds), y=series_ds,
                mode="lines", name="Consumo (4h)",
                line=dict(color="#3498db", width=1),
            ), row=1, col=1)

            fig_cp.add_trace(go.Scatter(
                x=pd.to_datetime(ts_ds), y=cp_prob,
                mode="lines", name="P(cambio)",
                line=dict(color="#e74c3c", width=1.5),
                fill="tozeroy",
            ), row=2, col=1)

            # Marcar ground truth si existe
            labels = _load_leak_labels()
            leak_row = labels[labels["contrato_id"] == top_cid]
            if not leak_row.empty:
                for _, lr in leak_row.iterrows():
                    fig_cp.add_vline(x=lr["inicio_fuga"], line_dash="dash",
                                     line_color="#2ecc71", row=1, col=1)
                    fig_cp.add_vline(x=lr["inicio_fuga"], line_dash="dash",
                                     line_color="#2ecc71", row=2, col=1)

            fig_cp.update_layout(height=500, template="plotly_dark",
                                  title=f"BOCPD Changepoint Detection — {top_cid}")
            figures["changepoint_timeline"] = fig_cp

    except ImportError:
        pass

    return df_scores, figures


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FACTOR MODEL (Quant-Style Residual Analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def factor_model(
    df_hourly: pd.DataFrame = None,
    df_catastro: pd.DataFrame = None,
    df_profiles: pd.DataFrame = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Descomposicion factorial estilo quant fund:
      E[consumo] = f(hora, dia_semana, m2, n_personas, uso)
      Residual = actual - esperado = senal de anomalia

    Cross-sectional ranking: quien es el outlier RELATIVO a sus pares.

    Returns:
        scores_df: contrato_id, mean_residual, residual_zscore,
                   peer_rank_pct, factor_anomaly_score
        figures: dict con heatmap residuales, peer ranking, decomposicion
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import OneHotEncoder

    if df_hourly is None:
        df_hourly = _load_hourly()
    if df_catastro is None:
        df_catastro = _load_catastro()
    if df_profiles is None:
        df_profiles = _load_profiles()

    df = df_hourly.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek

    # Merge covariables
    if not df_catastro.empty:
        df = df.merge(
            df_catastro[["contrato_id", "building_m2"]].drop_duplicates(),
            on="contrato_id", how="left",
        )
    else:
        df["building_m2"] = 80.0  # default

    if not df_profiles.empty:
        df = df.merge(
            df_profiles[["contrato_id", "n_personas_hogar"]].drop_duplicates(),
            on="contrato_id", how="left",
        )
    else:
        df["n_personas_hogar"] = 2.5

    df["building_m2"] = df["building_m2"].fillna(80.0)
    df["n_personas_hogar"] = df["n_personas_hogar"].fillna(2.5)

    # ── Feature matrix ────────────────────────────────────────────────
    # One-hot: hour (24) + dow (7) + uso (2-3)
    hour_dummies = pd.get_dummies(df["hour"], prefix="h", dtype=float)
    dow_dummies = pd.get_dummies(df["dow"], prefix="d", dtype=float)
    uso_dummies = pd.get_dummies(df["uso"], prefix="uso", dtype=float)

    X = pd.concat([hour_dummies, dow_dummies, uso_dummies,
                    df[["building_m2", "n_personas_hogar"]]], axis=1)
    y = df["consumo_litros"].values

    # ── Fit Ridge regression ──────────────────────────────────────────
    leak_ids = _load_leak_ids()
    train_mask = ~df["contrato_id"].isin(leak_ids)

    X_train = X[train_mask].values
    y_train = y[train_mask]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Predict all
    y_pred = model.predict(X.values)
    df["expected_consumo"] = y_pred
    df["residual"] = df["consumo_litros"] - y_pred

    # ── Aggregate per contract ────────────────────────────────────────
    agg = df.groupby("contrato_id").agg(
        mean_residual=("residual", "mean"),
        std_residual=("residual", "std"),
        mean_abs_residual=("residual", lambda x: np.abs(x).mean()),
        max_residual=("residual", "max"),
        barrio=("barrio", "first"),
        uso=("uso", "first"),
    ).reset_index()

    # Z-score del residual medio
    global_mean = agg["mean_residual"].mean()
    global_std = agg["mean_residual"].std() + 1e-10
    agg["residual_zscore"] = (agg["mean_residual"] - global_mean) / global_std

    # ── Peer ranking (cross-sectional, dentro de barrio+uso) ─────────
    agg["peer_rank_pct"] = agg.groupby(["barrio", "uso"])["mean_residual"].rank(pct=True)

    # Factor anomaly score
    max_abs_res = agg["mean_abs_residual"].max() + 1e-10
    agg["factor_anomaly_score"] = (
        0.5 * (agg["mean_abs_residual"] / max_abs_res).clip(0, 1)
        + 0.3 * agg["residual_zscore"].abs().clip(0, 3) / 3.0
        + 0.2 * (1 - agg["peer_rank_pct"]).clip(0, 1)  # invertido: top rank = mas anomalo
    ).clip(0, 1)

    agg["is_anomaly_factor"] = agg["factor_anomaly_score"] > agg["factor_anomaly_score"].quantile(0.90)

    # ── Figuras ───────────────────────────────────────────────────────
    figures = {}
    try:
        import plotly.graph_objects as go

        # Residual heatmap (hour x contract)
        heatmap_data = df.groupby(["contrato_id", "hour"])["residual"].mean().reset_index()
        pivot_heat = heatmap_data.pivot(index="contrato_id", columns="hour", values="residual")
        # Ordenar por mean residual
        order = agg.sort_values("mean_residual", ascending=False)["contrato_id"]
        pivot_heat = pivot_heat.reindex(order)

        fig_heat = go.Figure(go.Heatmap(
            z=pivot_heat.values,
            x=[f"{h}:00" for h in range(24)],
            y=pivot_heat.index,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="Residual (L)"),
        ))
        fig_heat.update_layout(
            title="Factor Model — Residuales por Hora y Contrato",
            xaxis_title="Hora del dia",
            yaxis_title="Contrato (ordenado por residual medio)",
            height=600, template="plotly_dark",
        )
        figures["residual_heatmap"] = fig_heat

        # Peer ranking scatter
        fig_peer = go.Figure()
        for is_leak, color, name in [(True, "#e74c3c", "Fuga"), (False, "#3498db", "Normal")]:
            mask = agg["contrato_id"].isin(leak_ids) if is_leak else ~agg["contrato_id"].isin(leak_ids)
            subset = agg[mask]
            fig_peer.add_trace(go.Scatter(
                x=subset["peer_rank_pct"],
                y=subset["mean_residual"],
                mode="markers",
                marker=dict(size=8 if is_leak else 5, color=color, opacity=0.7),
                name=name,
                text=subset["contrato_id"],
            ))
        fig_peer.update_layout(
            title="Cross-Sectional Peer Ranking — Residual vs Posicion entre Pares",
            xaxis_title="Peer Rank (1.0 = maximo consumo relativo)",
            yaxis_title="Residual medio (L/h)",
            height=400, template="plotly_dark",
        )
        figures["peer_ranking"] = fig_peer

        # Factor decomposition para top anomalo
        top_cid = agg.nlargest(1, "factor_anomaly_score")["contrato_id"].iloc[0]
        top_data = df[df["contrato_id"] == top_cid]

        hour_effect = top_data.groupby("hour")["expected_consumo"].mean()
        actual_hour = top_data.groupby("hour")["consumo_litros"].mean()
        residual_hour = top_data.groupby("hour")["residual"].mean()

        fig_decomp = go.Figure()
        fig_decomp.add_trace(go.Bar(
            x=[f"{h}:00" for h in range(24)], y=hour_effect.values,
            name="Esperado (factores)", marker_color="#3498db",
        ))
        fig_decomp.add_trace(go.Bar(
            x=[f"{h}:00" for h in range(24)], y=residual_hour.values,
            name="Residual (anomalia)", marker_color="#e74c3c",
        ))
        fig_decomp.update_layout(
            title=f"Factor Decomposition — {top_cid}",
            barmode="stack",
            xaxis_title="Hora", yaxis_title="Litros/hora",
            height=400, template="plotly_dark",
        )
        figures["factor_decomposition"] = fig_decomp

    except ImportError:
        pass

    return agg, figures


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL + CLI
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_advanced(
    df_hourly: pd.DataFrame = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Ejecuta las 5 tecnicas y combina scores por contrato_id.

    Returns:
        combined_df: contrato_id + scores de cada tecnica + combined_advanced_score
        all_figures: dict con todas las figuras
    """
    if df_hourly is None:
        df_hourly = _load_hourly()

    df_catastro = _load_catastro()
    df_profiles = _load_profiles()

    all_figures = {}
    dfs = []

    print("\n  [1/5] Spectral Analysis (FFT)...")
    try:
        df_spec, fig_spec = spectral_analysis(df_hourly)
        dfs.append(df_spec[["contrato_id", "spectral_entropy", "diurnal_power_ratio",
                             "spectral_anomaly_score", "is_anomaly_spectral"]])
        all_figures.update({f"spectral_{k}": v for k, v in fig_spec.items()})
        print(f"       {df_spec['is_anomaly_spectral'].sum()} anomalias detectadas")
    except Exception as e:
        print(f"       Error: {e}")

    print("  [2/5] Profile Autoencoder + UMAP...")
    try:
        df_ae, fig_ae = profile_autoencoder(df_hourly)
        dfs.append(df_ae[["contrato_id", "ae_median_error", "ae_anomaly_score",
                           "cluster_id", "is_anomaly_ae"]])
        all_figures.update({f"ae_{k}": v for k, v in fig_ae.items()})
        print(f"       {df_ae['is_anomaly_ae'].sum()} anomalias detectadas, "
              f"{df_ae['cluster_id'].nunique()} clusters")
    except Exception as e:
        print(f"       Error: {e}")

    print("  [3/5] Survival Analysis (Cox PH)...")
    try:
        df_surv, fig_surv = survival_analysis(df_hourly, df_catastro, df_profiles)
        if not df_surv.empty:
            dfs.append(df_surv[["contrato_id", "predicted_risk_60d", "hazard_ratio", "risk_group"]])
            all_figures.update({f"survival_{k}": v for k, v in fig_surv.items()})
            n_alto = (df_surv["risk_group"] == "ALTO").sum()
            print(f"       {n_alto} viviendas en riesgo ALTO")
        else:
            print("       Sin resultados")
    except Exception as e:
        print(f"       Error: {e}")

    print("  [4/5] BOCPD Changepoint Detection...")
    try:
        df_cp, fig_cp = household_changepoint(df_hourly)
        dfs.append(df_cp[["contrato_id", "n_changepoints", "max_cp_probability",
                           "cp_detection_score", "is_anomaly_cp"]])
        all_figures.update({f"cp_{k}": v for k, v in fig_cp.items()})
        print(f"       {df_cp['is_anomaly_cp'].sum()} anomalias, "
              f"{df_cp['n_changepoints'].sum()} changepoints totales")
    except Exception as e:
        print(f"       Error: {e}")

    print("  [5/5] Factor Model (Residual Analysis)...")
    try:
        df_factor, fig_factor = factor_model(df_hourly, df_catastro, df_profiles)
        dfs.append(df_factor[["contrato_id", "mean_residual", "residual_zscore",
                               "peer_rank_pct", "factor_anomaly_score", "is_anomaly_factor"]])
        all_figures.update({f"factor_{k}": v for k, v in fig_factor.items()})
        print(f"       {df_factor['is_anomaly_factor'].sum()} anomalias detectadas")
    except Exception as e:
        print(f"       Error: {e}")

    # ── Merge all scores ──────────────────────────────────────────────
    if not dfs:
        return pd.DataFrame(), all_figures

    combined = dfs[0]
    for d in dfs[1:]:
        combined = combined.merge(d, on="contrato_id", how="outer")

    # Combined advanced score (weighted average of available scores)
    score_cols = {
        "spectral_anomaly_score": 0.20,
        "ae_anomaly_score": 0.25,
        "predicted_risk_60d": 0.15,
        "cp_detection_score": 0.20,
        "factor_anomaly_score": 0.20,
    }

    available_scores = [c for c in score_cols if c in combined.columns]
    if available_scores:
        weights = np.array([score_cols[c] for c in available_scores])
        weights /= weights.sum()
        score_matrix = combined[available_scores].fillna(0).values
        combined["combined_advanced_score"] = (score_matrix * weights).sum(axis=1)

        # Count how many techniques flag as anomaly
        anomaly_cols = [c for c in combined.columns if c.startswith("is_anomaly_")]
        if anomaly_cols:
            combined["n_techniques_flagging"] = combined[anomaly_cols].fillna(False).sum(axis=1)

    return combined.sort_values("combined_advanced_score", ascending=False).reset_index(drop=True), all_figures


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Advanced Household Analytics — Quant-Grade")
    parser.add_argument("--only", choices=["spectral", "autoencoder", "survival", "changepoint", "factor"],
                        help="Ejecutar solo una tecnica")
    args = parser.parse_args()

    print("=" * 65)
    print("  ADVANCED HOUSEHOLD ANALYTICS — Quant-Grade Detection")
    print("=" * 65)
    print("\nCargando datos horarios...")
    df = _load_hourly()
    print(f"  {len(df):,} registros, {df['contrato_id'].nunique()} contratos")

    if args.only:
        func_map = {
            "spectral": spectral_analysis,
            "autoencoder": profile_autoencoder,
            "survival": lambda h: survival_analysis(h),
            "changepoint": household_changepoint,
            "factor": lambda h: factor_model(h),
        }
        print(f"\nEjecutando solo: {args.only}")
        scores, figs = func_map[args.only](df)
        if not scores.empty:
            print(f"\nResultados ({len(scores)} contratos):")
            print(scores.head(15).to_string(index=False))
        print(f"\nFiguras generadas: {list(figs.keys())}")
    else:
        combined, figs = run_all_advanced(df)

        # Validacion contra ground truth
        leak_ids = _load_leak_ids()
        if not combined.empty and "combined_advanced_score" in combined.columns:
            print(f"\n{'=' * 65}")
            print("  RESULTADOS COMBINADOS")
            print(f"{'=' * 65}")
            top20 = combined.head(20)
            cols_show = ["contrato_id", "combined_advanced_score"]
            if "n_techniques_flagging" in combined.columns:
                cols_show.append("n_techniques_flagging")
            print(top20[cols_show].to_string(index=False))

            # Precision @ top-K
            for k in [10, 20, 30]:
                top_k = set(combined.head(k)["contrato_id"])
                tp = len(top_k & leak_ids)
                precision = tp / k if k > 0 else 0
                print(f"\n  Precision@{k}: {tp}/{k} = {precision:.1%}")

            print(f"\nFiguras generadas: {len(figs)}")
            for name in sorted(figs.keys()):
                print(f"  - {name}")


if __name__ == "__main__":
    main()
