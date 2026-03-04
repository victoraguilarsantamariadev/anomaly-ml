"""
M10 — Detector de anomalias en lecturas individuales de contadores.

Datos: ~4M lecturas de contadores (m3 registrados/facturados, 2020-2025).
Cada fila = una lectura de un contador (sin ID de contador ni barrio).

Metodo:
  1. Normalizar: m3_per_day = M3 / Dias Lectura
  2. Para cada periodo mensual: detectar lecturas outlier (IQR + z-score)
  3. Agregar: % lecturas anomalas por mes
  4. Detectar meses con tasa anomala significativamente alta
  5. Detectar patrones sospechosos (zeros, extremos, delays)

Metricas:
  - zero_rate: % de lecturas con M3=0 (contadores parados o manipulados)
  - extreme_rate: % de lecturas con m3/dia extremo (posible fuga o fraude)
  - delay_rate: % de lecturas con retraso significativo vs fecha prevista
  - anomaly_score: combinacion ponderada de las 3 metricas

Uso:
  python meter_reading_detector.py
  python meter_reading_detector.py --year 2024
  python meter_reading_detector.py --iqr-multiplier 3.0
"""

import argparse
import glob
import pandas as pd
import numpy as np
from pathlib import Path


M3_DATA_PATTERN = "data/m3-registrados_facturados-tll_*-solo-alicante-*.csv"


def load_all_readings(pattern: str = M3_DATA_PATTERN,
                      year_filter: int = None) -> pd.DataFrame:
    """Carga todas las lecturas de contadores de los CSVs disponibles."""
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  No se encontraron archivos: {pattern}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        # Extraer año del nombre de archivo
        fname = Path(f).name
        file_year = None
        for part in fname.split("_"):
            if part.isdigit() and len(part) == 4:
                file_year = int(part)
                break

        if year_filter and file_year and file_year != year_filter:
            continue

        df = pd.read_csv(f)
        df["source_year"] = file_year
        dfs.append(df)
        print(f"    {fname}: {len(df):,} lecturas")

    if not dfs:
        return pd.DataFrame()

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


def preprocess_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa lecturas: parsea fechas, calcula m3/dia, limpia datos."""
    df = df.copy()

    # Parsear fechas
    df["fecha_factura"] = pd.to_datetime(df["Fecha Factura"], format="%d/%m/%Y", dayfirst=True)
    df["fecha_lectura"] = pd.to_datetime(df["Fecha Lectura"], format="%d/%m/%Y", dayfirst=True)
    df["fecha_prevista"] = pd.to_datetime(df["Fecha Prevista Lectura"], format="%d/%m/%Y", dayfirst=True)

    # Periodo mensual
    df["year_month"] = df["fecha_factura"].dt.to_period("M")

    # M3 a facturar (ya es numerico)
    df["m3"] = pd.to_numeric(df["M3 A Facturar"], errors="coerce").fillna(0)

    # Dias lectura
    df["dias"] = pd.to_numeric(df["Dias Lectura"], errors="coerce").fillna(0)

    # M3 por dia (consumo diario normalizado)
    df["m3_per_day"] = np.where(df["dias"] > 0, df["m3"] / df["dias"], 0)

    # Retraso en lectura (dias entre prevista y real)
    df["delay_days"] = (df["fecha_lectura"] - df["fecha_prevista"]).dt.days

    # Periodicidad
    df["is_trimestral"] = df["Periodicidad"].str.strip().str.lower() == "trimestral"

    return df


def detect_reading_anomalies(df: pd.DataFrame,
                              iqr_multiplier: float = 3.0,
                              zero_threshold_days: int = 30) -> pd.DataFrame:
    """
    Detecta lecturas individuales anomalas.

    Tipos de anomalia:
    1. ZERO: M3=0 con dias_lectura > threshold (contador parado)
    2. EXTREME_HIGH: m3/dia > Q3 + iqr_mult * IQR (consumo extremo)
    3. EXTREME_LOW: m3/dia negativo (marcha atras, fraude)
    4. DELAY: retraso > 30 dias respecto fecha prevista
    """
    df = df.copy()

    # 1. Zeros sospechosos: M3=0 con periodo de lectura largo
    df["is_zero"] = (df["m3"] == 0) & (df["dias"] > zero_threshold_days)

    # 2. Extremos por IQR (calculado por periodo para ser justo)
    df["is_extreme_high"] = False
    df["is_extreme_low"] = False

    for period, group in df.groupby("year_month"):
        vals = group["m3_per_day"].values
        q1 = np.percentile(vals[vals > 0], 25) if (vals > 0).any() else 0
        q3 = np.percentile(vals[vals > 0], 75) if (vals > 0).any() else 1
        iqr = q3 - q1
        upper = q3 + iqr_multiplier * iqr

        mask_high = (group["m3_per_day"] > upper) & (group["m3_per_day"] > 0)
        mask_low = group["m3"] < 0

        df.loc[group.index[mask_high], "is_extreme_high"] = True
        df.loc[group.index[mask_low], "is_extreme_low"] = True

    # 3. Retrasos significativos
    df["is_delayed"] = df["delay_days"].abs() > 30

    # Anomalia combinada
    df["is_anomaly_reading"] = (
        df["is_zero"] | df["is_extreme_high"] |
        df["is_extreme_low"] | df["is_delayed"]
    )

    return df


def compute_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega estadisticas de anomalias por mes.
    Detecta meses con tasas de anomalia significativamente altas.
    """
    monthly = df.groupby("year_month").agg(
        total_readings=("m3", "count"),
        n_zero=("is_zero", "sum"),
        n_extreme_high=("is_extreme_high", "sum"),
        n_extreme_low=("is_extreme_low", "sum"),
        n_delayed=("is_delayed", "sum"),
        n_anomaly=("is_anomaly_reading", "sum"),
        avg_m3=("m3", "mean"),
        median_m3=("m3", "median"),
        avg_m3_per_day=("m3_per_day", "mean"),
        std_m3_per_day=("m3_per_day", "std"),
        max_m3=("m3", "max"),
    ).reset_index()

    # Tasas
    monthly["zero_rate"] = monthly["n_zero"] / monthly["total_readings"] * 100
    monthly["extreme_rate"] = monthly["n_extreme_high"] / monthly["total_readings"] * 100
    monthly["anomaly_rate"] = monthly["n_anomaly"] / monthly["total_readings"] * 100

    # Detectar meses anomalos: anomaly_rate > media + 2*std
    mean_rate = monthly["anomaly_rate"].mean()
    std_rate = monthly["anomaly_rate"].std()
    monthly["is_anomalous_month"] = monthly["anomaly_rate"] > mean_rate + 2 * std_rate

    # Z-score del mes
    monthly["zscore"] = np.where(
        std_rate > 0,
        (monthly["anomaly_rate"] - mean_rate) / std_rate,
        0,
    )

    return monthly.sort_values("year_month")


def compute_periodicidad_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Estadisticas de anomalia por tipo de periodicidad."""
    return df.groupby("is_trimestral").agg(
        total=("m3", "count"),
        n_anomaly=("is_anomaly_reading", "sum"),
        avg_m3=("m3", "mean"),
        n_zero=("is_zero", "sum"),
    ).reset_index()


def print_report(df: pd.DataFrame, monthly: pd.DataFrame):
    """Imprime reporte de anomalias en lecturas."""
    print("=" * 70)
    print("  M10 — Anomalias en Lecturas Individuales de Contadores")
    print("=" * 70)

    total = len(df)
    print(f"\n  Total lecturas analizadas: {total:,}")
    print(f"  Periodo: {df['year_month'].min()} - {df['year_month'].max()}")

    # Resumen de anomalias
    n_zero = df["is_zero"].sum()
    n_high = df["is_extreme_high"].sum()
    n_low = df["is_extreme_low"].sum()
    n_delay = df["is_delayed"].sum()
    n_total = df["is_anomaly_reading"].sum()

    print(f"\n  TIPOS DE ANOMALIA:")
    print(f"  {'─' * 55}")
    print(f"    Contadores parados (M3=0, >30 dias):  {n_zero:>8,} ({n_zero/total*100:.2f}%)")
    print(f"    Consumo extremo (IQR outlier):         {n_high:>8,} ({n_high/total*100:.2f}%)")
    print(f"    Consumo negativo (marcha atras):       {n_low:>8,} ({n_low/total*100:.2f}%)")
    print(f"    Retraso significativo (>30 dias):      {n_delay:>8,} ({n_delay/total*100:.2f}%)")
    print(f"    TOTAL ANOMALAS:                        {n_total:>8,} ({n_total/total*100:.2f}%)")

    # Top consumos extremos
    extremes = df[df["is_extreme_high"]].nlargest(10, "m3_per_day")
    if len(extremes) > 0:
        print(f"\n  TOP 10 LECTURAS MAS EXTREMAS:")
        print(f"  {'─' * 60}")
        for _, row in extremes.iterrows():
            print(f"    {row['year_month']}  M3={row['m3']:>8.0f}  "
                  f"({row['dias']:.0f} dias, {row['m3_per_day']:.1f} M3/dia)")

    # Meses anomalos
    anom_months = monthly[monthly["is_anomalous_month"]]
    if len(anom_months) > 0:
        print(f"\n  MESES CON TASA ANOMALA ELEVADA:")
        print(f"  {'─' * 70}")
        print(f"  {'Mes':>10} {'Lecturas':>10} {'Anomalas':>10} {'Tasa':>8} {'Z-score':>8}")
        print(f"  {'─' * 70}")
        for _, row in anom_months.iterrows():
            print(f"  {str(row['year_month']):>10} {int(row['total_readings']):>10,} "
                  f"{int(row['n_anomaly']):>10,} {row['anomaly_rate']:>7.2f}% "
                  f"{row['zscore']:>7.1f}")

    # Timeline
    print(f"\n  TIMELINE MENSUAL (tasa de anomalia %):")
    print(f"  {'─' * 70}")
    for _, row in monthly.iterrows():
        bar_len = int(row["anomaly_rate"] * 2)
        bar = "█" * min(bar_len, 40)
        flag = " *** ANOMALO" if row["is_anomalous_month"] else ""
        print(f"  {str(row['year_month']):>10}: {row['anomaly_rate']:>5.1f}% {bar}{flag}")

    # Resumen por periodicidad
    print(f"\n  NOTA: Los contadores parados (M3=0) pueden indicar manipulacion,")
    print(f"  abandono del inmueble, o simplemente contadores defectuosos.")
    print(f"  Los consumos extremos pueden indicar fugas no reparadas o fraude.")


def main():
    parser = argparse.ArgumentParser(description="M10 — Detector de anomalias en lecturas individuales")
    parser.add_argument("--year", type=int, default=None,
                        help="Filtrar por año (default: todos)")
    parser.add_argument("--iqr-multiplier", type=float, default=3.0,
                        help="IQR multiplier para detectar extremos (default: 3.0)")
    args = parser.parse_args()

    print("Cargando lecturas de contadores...")
    df = load_all_readings(year_filter=args.year)
    if df.empty:
        print("ERROR: No se encontraron datos")
        return

    print(f"\n  Total: {len(df):,} lecturas cargadas")

    print("Preprocesando...")
    df = preprocess_readings(df)

    print("Detectando anomalias...")
    df = detect_reading_anomalies(df, iqr_multiplier=args.iqr_multiplier)

    print("Calculando estadisticas mensuales...")
    monthly = compute_monthly_stats(df)

    print_report(df, monthly)

    return df, monthly


if __name__ == "__main__":
    main()
