"""
M9 — Nighttime Minimum Flow (NMF): deteccion de fugas y uso no autorizado.

Tecnica estandar en la industria del agua:
  - Entre 2am-5am el consumo domestico deberia ser minimo
  - Si el caudal nocturno es alto respecto al diurno → fugas o fraude
  - Si hay picos nocturnos en dias especificos → uso no autorizado puntual

Datos: Caudal medio por sector hidraulico, hora a hora, 2024 completo.

Metricas:
  1. Night/Day ratio por sector: caudal medio 2-5am / caudal medio 10-18h
  2. NMF anomalo: sectores cuyo ratio es estadisticamente superior al grupo
  3. Picos nocturnos: dias concretos con caudal nocturno extremo

Uso:
  python nightflow_detector.py
  python nightflow_detector.py --zscore-threshold 2.0
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from sector_mapping import get_mapped_sectors

CAUDAL_PATH = "data/_caudal_medio_sector_hidraulico_hora_2024_-caudal_medio_sector_hidraulico_hora_2024.csv"

NIGHT_HOURS = range(2, 5)   # 2am, 3am, 4am
DAY_HOURS = range(10, 18)   # 10am - 17pm


def load_hourly_data(caudal_path: str) -> pd.DataFrame:
    """Carga y parsea datos horarios de caudal."""
    df = pd.read_csv(caudal_path)
    df["fecha"] = pd.to_datetime(df["FECHA_HORA"], format="%d/%m/%Y %H:%M", dayfirst=True)
    df["date"] = df["fecha"].dt.date
    df["hour"] = df["fecha"].dt.hour
    df["month"] = df["fecha"].dt.month
    df["caudal"] = (
        df["CAUDAL MEDIO(M3)"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    return df


def compute_night_day_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula ratio nocturno/diurno por sector y por dia.

    Un ratio > 1.0 = mas caudal de noche que de dia (muy sospechoso)
    Un ratio > 0.5 en zona residencial = posibles perdidas
    """
    # Caudal medio nocturno por sector por dia
    night = (
        df[df["hour"].isin(NIGHT_HOURS)]
        .groupby(["SECTOR", "date"])["caudal"]
        .mean()
        .rename("night_flow")
    )

    # Caudal medio diurno por sector por dia
    day = (
        df[df["hour"].isin(DAY_HOURS)]
        .groupby(["SECTOR", "date"])["caudal"]
        .mean()
        .rename("day_flow")
    )

    daily = pd.concat([night, day], axis=1).reset_index()
    daily["night_day_ratio"] = np.where(
        daily["day_flow"] > 0,
        daily["night_flow"] / daily["day_flow"],
        np.nan,
    )

    return daily


def detect_nmf_anomalies(daily: pd.DataFrame,
                          zscore_threshold: float = 2.0) -> pd.DataFrame:
    """
    Detecta sectores con consumo nocturno anomalo.

    Metodo: para cada sector, calcula media y std del ratio nocturno/diurno.
    Si el ratio medio del sector supera la media global + zscore_threshold * std → anomalia.

    Tambien detecta dias concretos con picos nocturnos (ratio > media_sector + 2*std_sector).
    """
    # Estadisticas por sector
    sector_stats = daily.groupby("SECTOR").agg(
        avg_ratio=("night_day_ratio", "mean"),
        std_ratio=("night_day_ratio", "std"),
        median_ratio=("night_day_ratio", "median"),
        avg_night=("night_flow", "mean"),
        avg_day=("day_flow", "mean"),
        n_days=("date", "count"),
    ).reset_index()

    # Z-score de cada sector respecto al grupo
    global_mean = sector_stats["avg_ratio"].mean()
    global_std = sector_stats["avg_ratio"].std()

    sector_stats["zscore"] = np.where(
        global_std > 0,
        (sector_stats["avg_ratio"] - global_mean) / global_std,
        0,
    )
    sector_stats["is_anomaly_nmf"] = sector_stats["zscore"] > zscore_threshold

    # Añadir barrio
    mapping = get_mapped_sectors()
    sector_stats["barrio"] = sector_stats["SECTOR"].map(mapping)

    return sector_stats.sort_values("avg_ratio", ascending=False)


def detect_spike_nights(daily: pd.DataFrame,
                         spike_sigma: float = 3.0) -> pd.DataFrame:
    """
    Detecta DIAS concretos donde el caudal nocturno de un sector
    es anormalmente alto respecto a su propio historico.

    Ej: un sector normalmente tiene ratio 0.6, pero el 15 de marzo tiene 2.5 → spike.
    """
    # Media y std por sector
    sector_stats = daily.groupby("SECTOR")["night_day_ratio"].agg(["mean", "std"]).reset_index()

    spikes = daily.merge(sector_stats, on="SECTOR")
    spikes["is_spike"] = (
        (spikes["night_day_ratio"] > spikes["mean"] + spike_sigma * spikes["std"]) &
        (spikes["night_day_ratio"] > 0.8)  # filtrar ratios triviales
    )

    return spikes[spikes["is_spike"]].sort_values("night_day_ratio", ascending=False)


def print_report(sector_stats: pd.DataFrame, spikes: pd.DataFrame, zscore_threshold: float):
    """Imprime reporte de NMF."""
    print("=" * 70)
    print("  M9 — Nighttime Minimum Flow (NMF) — Consumo Nocturno")
    print("=" * 70)

    print(f"\n  Horario nocturno: 2am-5am")
    print(f"  Horario diurno: 10am-18pm")
    print(f"  Z-score threshold: {zscore_threshold}")
    print(f"  Sectores analizados: {len(sector_stats)}")

    global_mean = sector_stats["avg_ratio"].mean()
    print(f"  Ratio nocturno/diurno medio global: {global_mean:.2f}")

    # Sectores anomalos
    anomalies = sector_stats[sector_stats["is_anomaly_nmf"]]
    print(f"\n  SECTORES CON CONSUMO NOCTURNO ANOMALO: {len(anomalies)}")
    print(f"  {'─' * 90}")
    print(f"  {'Sector':<35} {'Barrio':<30} {'Ratio N/D':>10} {'Z-score':>8} {'Noche M3/h':>11}")
    print(f"  {'─' * 90}")

    for _, row in anomalies.iterrows():
        barrio = row["barrio"] if pd.notna(row["barrio"]) else "—"
        print(f"  {row['SECTOR']:<35} {barrio:<30} {row['avg_ratio']:>10.2f} "
              f"{row['zscore']:>8.1f} {row['avg_night']:>11.1f}")

    # Top picos nocturnos
    if len(spikes) > 0:
        print(f"\n  TOP PICOS NOCTURNOS (dias concretos):")
        print(f"  {'─' * 80}")
        top_spikes = spikes.head(15)
        for _, row in top_spikes.iterrows():
            print(f"  {row['SECTOR']:<30} {row['date']}  ratio={row['night_day_ratio']:.2f} "
                  f"(noche={row['night_flow']:.1f} dia={row['day_flow']:.1f} M3/h)")

    # Sectores normales (para referencia)
    normal = sector_stats[~sector_stats["is_anomaly_nmf"]].nsmallest(5, "avg_ratio")
    print(f"\n  SECTORES MAS EFICIENTES (menor ratio nocturno):")
    for _, row in normal.iterrows():
        barrio = row["barrio"] if pd.notna(row["barrio"]) else "—"
        print(f"    {row['SECTOR']:<35} ratio={row['avg_ratio']:.2f} ({barrio})")


def main():
    parser = argparse.ArgumentParser(description="M9 — Detector de consumo nocturno anomalo")
    parser.add_argument("--zscore-threshold", type=float, default=2.0,
                        help="Z-score threshold para anomalia (default: 2.0)")
    parser.add_argument("--spike-sigma", type=float, default=3.0,
                        help="Sigmas para detectar picos nocturnos (default: 3.0)")
    args = parser.parse_args()

    if not Path(CAUDAL_PATH).exists():
        print(f"ERROR: No se encuentra {CAUDAL_PATH}")
        return

    print("Cargando datos horarios 2024...")
    df = load_hourly_data(CAUDAL_PATH)
    print(f"  {len(df)} registros horarios, {df['SECTOR'].nunique()} sectores")

    print("Calculando ratios nocturnos...")
    daily = compute_night_day_ratios(df)

    print("Detectando anomalias NMF...")
    sector_stats = detect_nmf_anomalies(daily, zscore_threshold=args.zscore_threshold)

    print("Detectando picos nocturnos...")
    spikes = detect_spike_nights(daily, spike_sigma=args.spike_sigma)

    print_report(sector_stats, spikes, args.zscore_threshold)

    return sector_stats, spikes


if __name__ == "__main__":
    main()
