"""
Deteccion de anomalias a nivel de vivienda individual (granularidad horaria).

Detecta:
  - fuga_fisica      : consumo constante 24h, alto ratio nocturno/diurno
  - fuga_silenciosa  : crecimiento gradual sostenido (cisterna, microperforacion)
  - contador_roto    : consumo a cero durante dias consecutivos
  - fraude_contador  : caida diurna brusca, nocturno estable (manipulacion)

Datos de entrada: data/synthetic_hourly_domicilio.csv
  Columnas: timestamp, contrato_id, barrio, uso, consumo_litros

Uso:
  python household_detector.py                  # top-10 global
  python household_detector.py --barrio 17      # top-10 del barrio 17
  python household_detector.py --top 20         # top-20 global
"""

import sys
import io
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

HOURLY_DATA = Path(__file__).parent / "data" / "synthetic_hourly_domicilio.csv"
LEAK_LABELS = Path(__file__).parent / "data" / "synthetic_leak_labels.csv"

# ─── Umbrales de clasificacion ──────────────────────────────────────────────

NIGHT_RATIO_THRESHOLD   = 0.45   # mean(0-5h) / mean(8-22h) — sospechoso si >0.45
NOCTURNAL_BASELINE_L    = 1.0    # litros/hora minimo 2-4AM — sospechoso si >1 L/h
CV_LOW_THRESHOLD        = 0.35   # std/mean — fuga constante si CV < 0.35
TREND_SLOPE_THRESHOLD   = 0.8    # litros/dia de crecimiento — fuga silenciosa si >0.8
ZERO_DAYS_THRESHOLD     = 3      # dias con <5L total — contador roto si >3


# ─── Feature engineering ────────────────────────────────────────────────────

def compute_household_features(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features de anomalia por contrato_id sobre todos los datos disponibles.

    Features calculadas:
      night_flow_ratio    -- consumo nocturno vs diurno
      nocturnal_baseline  -- litros/hora minimos en franja 2-4AM
      consumption_cv      -- coeficiente de variacion (bajo = constante = fuga)
      daily_trend_slope   -- pendiente de consumo diario (creciente = fuga silenciosa)
      zero_consumption_days -- dias con menos de 5L (contador roto)
      total_mean_daily_L  -- consumo medio diario en litros
    """
    df = df_hourly.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date

    records = []

    for cid, grp in df.groupby("contrato_id"):
        barrio = grp["barrio"].iloc[0]
        uso    = grp["uso"].iloc[0]

        # ── Ratio nocturno / diurno ──────────────────────────────────────────
        night = grp[grp["hour"].between(0, 5)]["consumo_litros"].mean()
        day   = grp[grp["hour"].between(8, 22)]["consumo_litros"].mean()
        night_flow_ratio = night / (day + 1e-9)

        # ── Baseline nocturno (minimo 2-4 AM) ───────────────────────────────
        deep_night = grp[grp["hour"].between(2, 4)]["consumo_litros"]
        # Usa percentil 10 para ignorar picos, capturar el minimo real
        nocturnal_baseline = float(deep_night.quantile(0.10)) if len(deep_night) > 0 else 0.0

        # ── Coeficiente de variacion ─────────────────────────────────────────
        mean_c = grp["consumo_litros"].mean()
        std_c  = grp["consumo_litros"].std()
        consumption_cv = float(std_c / (mean_c + 1e-9))

        # ── Tendencia de consumo diario (regresion lineal sobre totales) ─────
        daily = grp.groupby("date")["consumo_litros"].sum().reset_index()
        daily = daily.rename(columns={"consumo_litros": "total"})
        if len(daily) >= 5:
            x = np.arange(len(daily))
            slope, _ = np.polyfit(x, daily["total"].values, 1)
        else:
            slope = 0.0

        # ── Dias con consumo casi cero ───────────────────────────────────────
        zero_days = int((daily["total"] < 5).sum())

        # ── Consumo medio diario ─────────────────────────────────────────────
        mean_daily = float(daily["total"].mean())

        records.append({
            "contrato_id":           cid,
            "barrio":                barrio,
            "uso":                   uso,
            "night_flow_ratio":      round(float(night_flow_ratio), 4),
            "nocturnal_baseline":    round(nocturnal_baseline, 3),
            "consumption_cv":        round(float(consumption_cv), 4),
            "daily_trend_slope":     round(float(slope), 4),
            "zero_consumption_days": zero_days,
            "mean_daily_L":          round(mean_daily, 1),
        })

    return pd.DataFrame(records)


# ─── Clasificacion de anomalias ──────────────────────────────────────────────

def _classify_anomaly(row: pd.Series) -> tuple[str, float]:
    """Devuelve (tipo_sospecha, score) para una vivienda."""
    scores = {}

    # fuga_fisica: consumo constante alto incluso de madrugada
    if row["night_flow_ratio"] > NIGHT_RATIO_THRESHOLD and row["consumption_cv"] < CV_LOW_THRESHOLD:
        s = (
            min((row["night_flow_ratio"] - NIGHT_RATIO_THRESHOLD) / 0.5, 1.0) * 0.5
            + min((CV_LOW_THRESHOLD - row["consumption_cv"]) / CV_LOW_THRESHOLD, 1.0) * 0.3
            + min(row["nocturnal_baseline"] / 3.0, 1.0) * 0.2
        )
        scores["fuga_fisica"] = round(s, 3)

    # fuga_silenciosa: crecimiento gradual continuo
    if row["daily_trend_slope"] > TREND_SLOPE_THRESHOLD:
        s = min(row["daily_trend_slope"] / (TREND_SLOPE_THRESHOLD * 6), 1.0) * 0.7
        if row["night_flow_ratio"] > 0.3:
            s += 0.2
        scores["fuga_silenciosa"] = round(s, 3)

    # contador_roto: dias seguidos sin consumo
    if row["zero_consumption_days"] > ZERO_DAYS_THRESHOLD:
        s = min(row["zero_consumption_days"] / 10.0, 1.0)
        scores["contador_roto"] = round(s, 3)

    # fraude_contador: consumo total muy bajo pero con baseline nocturno (medidor manipulado)
    if row["mean_daily_L"] < 40 and row["nocturnal_baseline"] > 0.5 and row["consumption_cv"] < 0.5:
        s = 0.4 + min((0.5 - row["mean_daily_L"] / 100.0), 0.5)
        scores["fraude_contador"] = round(max(s, 0.0), 3)

    if not scores:
        return "normal", 0.0

    tipo = max(scores, key=scores.__getitem__)
    return tipo, scores[tipo]


def detect_household_anomalies(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la clasificacion de anomalias a todas las viviendas.

    Retorna DataFrame con columnas adicionales:
      tipo_sospecha, anomaly_score, alerta (bool)
    """
    resultados = features_df.copy()
    clasificaciones = resultados.apply(_classify_anomaly, axis=1)
    resultados["tipo_sospecha"] = clasificaciones.apply(lambda x: x[0])
    resultados["anomaly_score"] = clasificaciones.apply(lambda x: x[1])
    resultados["alerta"] = resultados["anomaly_score"] > 0.3
    return resultados.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def _estimate_leak_start(cid: str, tipo: str, df: pd.DataFrame) -> str:
    """Estima el timestamp de inicio de la anomalia buscando el primer cambio significativo."""
    grp = df[df["contrato_id"] == cid].sort_values("timestamp")
    if grp.empty:
        return "desconocido"

    grp = grp.copy()
    grp["hour"] = grp["timestamp"].dt.hour

    if tipo == "fuga_fisica":
        # Primer dia con consumo nocturno alto (0-5h > 1 L/h)
        night = grp[grp["hour"].between(0, 5)].copy()
        night["date"] = night["timestamp"].dt.date
        daily_night = night.groupby("date")["consumo_litros"].mean()
        suspects = daily_night[daily_night > NOCTURNAL_BASELINE_L]
        if not suspects.empty:
            return str(pd.Timestamp(suspects.index[0]))

    elif tipo == "fuga_silenciosa":
        # Punto de inflexion: primer dia donde el consumo supera la media + 1 std
        grp["date"] = grp["timestamp"].dt.date
        daily = grp.groupby("date")["consumo_litros"].sum()
        threshold = daily.mean() + daily.std() * 0.5
        suspects = daily[daily > threshold]
        if not suspects.empty:
            return str(pd.Timestamp(suspects.index[0]))

    elif tipo == "contador_roto":
        # Primer dia con total < 5L
        grp["date"] = grp["timestamp"].dt.date
        daily = grp.groupby("date")["consumo_litros"].sum()
        suspects = daily[daily < 5]
        if not suspects.empty:
            return str(pd.Timestamp(suspects.index[0]))

    return str(grp["timestamp"].iloc[0])[:10]


# ─── API publica ─────────────────────────────────────────────────────────────

def load_hourly_data() -> pd.DataFrame:
    """Carga data/synthetic_hourly_domicilio.csv."""
    df = pd.read_csv(HOURLY_DATA, parse_dates=["timestamp"])
    return df


def get_suspicious_households(
    df_hourly: pd.DataFrame | None = None,
    barrio: str | None = None,
    top_n: int = 10,
    include_leak_labels: bool = True,
) -> pd.DataFrame:
    """
    Devuelve tabla de viviendas sospechosas, ordenadas por anomaly_score.

    Args:
        df_hourly:          DataFrame horario. Si None, carga desde disco.
        barrio:             Filtrar por barrio (nombre o numero). None = global.
        top_n:              Numero de resultados a devolver.
        include_leak_labels: Si True, añade columna 'fuga_conocida' desde ground truth.

    Returns:
        DataFrame con columnas:
          contrato_id, barrio, tipo_sospecha, anomaly_score,
          inicio_estimado, night_flow_ratio, nocturnal_baseline,
          consumption_cv, daily_trend_slope, fuga_conocida (opcional)
    """
    if df_hourly is None:
        df_hourly = load_hourly_data()

    # Filtrar por barrio si se especifica
    if barrio is not None:
        mask = df_hourly["barrio"].str.contains(str(barrio), case=False, na=False)
        df_hourly = df_hourly[mask]
        if df_hourly.empty:
            return pd.DataFrame()

    features = compute_household_features(df_hourly)
    results  = detect_household_anomalies(features)

    # Estimar inicio de anomalia
    top = results[results["alerta"]].head(top_n).copy()
    top["inicio_estimado"] = top.apply(
        lambda r: _estimate_leak_start(r["contrato_id"], r["tipo_sospecha"], df_hourly),
        axis=1,
    )

    # Añadir ground truth si existe
    if include_leak_labels and LEAK_LABELS.exists():
        labels = pd.read_csv(LEAK_LABELS)
        # Simplificar tipo_fuga a categorias comparables
        fuga_map = {
            "rotura_tuberia":         "fuga_fisica",
            "consumo_nocturno_anomalo": "fuga_fisica",
            "fuga_lenta_continua":    "fuga_fisica",
            "fuga_intermitente":      "fuga_silenciosa",
            "degradacion_gradual":    "fuga_silenciosa",
        }
        labels["tipo_real"] = labels["tipo_fuga"].map(fuga_map).fillna("otro")
        known = set(labels["contrato_id"])
        top["fuga_conocida"] = top["contrato_id"].apply(
            lambda c: labels.loc[labels["contrato_id"] == c, "tipo_real"].values[0]
            if c in known else ""
        )

    cols = ["contrato_id", "barrio", "uso", "tipo_sospecha", "anomaly_score",
            "inicio_estimado", "night_flow_ratio", "nocturnal_baseline",
            "consumption_cv", "daily_trend_slope"]
    if "fuga_conocida" in top.columns:
        cols.append("fuga_conocida")

    return top[cols].reset_index(drop=True)


def get_all_scores(df_hourly: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Devuelve scores para TODAS las viviendas (incluidas normales).
    Util para el dashboard (mapa de calor de contratos).
    """
    if df_hourly is None:
        df_hourly = load_hourly_data()
    features = compute_household_features(df_hourly)
    return detect_household_anomalies(features)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Detector de fugas en viviendas individuales")
    parser.add_argument("--barrio", default=None,
                        help="Numero o nombre del barrio (ej: 17 o CAROLINAS)")
    parser.add_argument("--top", type=int, default=10,
                        help="Numero de viviendas a mostrar (default: 10)")
    parser.add_argument("--all", action="store_true",
                        help="Mostrar todas las viviendas, no solo las alertas")
    args = parser.parse_args()

    print("Cargando datos horarios...")
    df = load_hourly_data()
    print(f"  {len(df):,} registros, {df['contrato_id'].nunique()} contratos, "
          f"{df['barrio'].nunique()} barrios")

    if args.all:
        results = get_all_scores(df)
        if args.barrio:
            results = results[results["barrio"].str.contains(args.barrio, case=False, na=False)]
        print(f"\nTop {args.top} viviendas (todas):\n")
        print(results.head(args.top).to_string(index=False))
    else:
        top = get_suspicious_households(df, barrio=args.barrio, top_n=args.top)
        if top.empty:
            print("No se detectaron alertas con los criterios actuales.")
        else:
            print(f"\nTop {len(top)} viviendas SOSPECHOSAS:\n")
            print(top.to_string(index=False))

            # Resumen por tipo
            print("\n--- Resumen por tipo de anomalia ---")
            for tipo, cnt in top["tipo_sospecha"].value_counts().items():
                print(f"  {tipo}: {cnt} viviendas")

            # Precision vs ground truth
            if "fuga_conocida" in top.columns:
                tp = (top["fuga_conocida"] != "").sum()
                print(f"\nCoincidencias con ground truth: {tp}/{len(top)} "
                      f"({100*tp/len(top):.0f}%)")


if __name__ == "__main__":
    main()
