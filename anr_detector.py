"""
M8 — Agua No Registrada (ANR): deteccion de perdidas por sector.

ANR = Caudal_inyectado_sector - Consumo_facturado_barrio

Un ratio ANR alto indica perdidas tecnicas (fugas) o comerciales (fraude).
Esto es el metodo real que usan las utilities para detectar problemas.

Datos necesarios:
  - Caudal horario por sector (2024)
  - Consumo por barrio/mes del hackathon
  - Mapeo sector → barrio

Limitaciones:
  - Solo 2024 (12 meses de caudal)
  - Mapeo sector→barrio es aproximado (no 1:1)
  - Caudal incluye perdidas tecnicas reales (fugas), no solo fraude
  - Unidades: caudal en M3/hora, consumo en litros/mes

Uso:
  python anr_detector.py
  python anr_detector.py --threshold 1.5  # flag if ANR_ratio > 1.5
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from sector_mapping import get_mapped_sectors

CAUDAL_PATH = "data/_caudal_medio_sector_hidraulico_hora_2024_-caudal_medio_sector_hidraulico_hora_2024.csv"
DATA_PATH = "data/datos-hackathon-amaem.xlsx-set-de-datos-.csv"


def load_caudal_monthly(caudal_path: str) -> pd.DataFrame:
    """
    Carga caudal horario y agrega a nivel mensual por sector.

    El caudal horario es el flujo medio en M3/hora.
    Para obtener M3 totales del mes: sum(caudal_hora) por mes.
    (Cada registro = 1 hora, asi que sum = total M3 del mes)
    """
    df = pd.read_csv(caudal_path)

    # Parsear fecha (formato DD/MM/YYYY H:MM)
    df["fecha"] = pd.to_datetime(df["FECHA_HORA"], format="%d/%m/%Y %H:%M", dayfirst=True)
    df["year_month"] = df["fecha"].dt.to_period("M")

    # Convertir caudal (formato europeo: coma decimal)
    df["caudal_m3"] = (
        df["CAUDAL MEDIO(M3)"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    # Agregar: total M3 por sector por mes
    monthly = (
        df.groupby(["SECTOR", "year_month"])["caudal_m3"]
        .sum()
        .reset_index()
        .rename(columns={"caudal_m3": "caudal_total_m3"})
    )

    return monthly


def load_consumo_monthly(data_path: str, uso_filter: str = None) -> pd.DataFrame:
    """Carga consumo del hackathon y agrega por barrio/mes (en M3)."""
    from train_local import load_hackathon_amaem

    df = load_hackathon_amaem(data_path)
    if uso_filter:
        df = df[df["uso"].str.strip() == uso_filter]

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["year_month"] = df["fecha"].dt.to_period("M")

    # Convertir litros a M3
    monthly = (
        df.groupby(["barrio", "year_month"])
        .agg(consumo_m3=("consumo_litros", lambda x: x.sum() / 1000))
        .reset_index()
    )
    monthly["barrio"] = monthly["barrio"].str.strip()

    return monthly


def compute_anr(caudal_monthly: pd.DataFrame,
                consumo_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula ANR (Agua No Registrada) por barrio/mes.

    ANR_absoluto = caudal_inyectado - consumo_facturado (en M3)
    ANR_ratio = caudal_inyectado / consumo_facturado
      - ratio ~1.0 → todo el agua inyectada se factura (ideal)
      - ratio > 1.5 → 50%+ del agua no se factura (perdidas)
      - ratio < 1.0 → facturamos mas de lo que inyectamos (error de mapeo)
    """
    mapping = get_mapped_sectors()

    # Añadir barrio a caudal
    caudal_monthly = caudal_monthly.copy()
    caudal_monthly["barrio"] = caudal_monthly["SECTOR"].map(mapping)
    caudal_mapped = caudal_monthly[caudal_monthly["barrio"].notna()]

    # Agregar caudal por barrio (puede haber multiples sectores → mismo barrio)
    caudal_by_barrio = (
        caudal_mapped.groupby(["barrio", "year_month"])["caudal_total_m3"]
        .sum()
        .reset_index()
    )

    # Merge con consumo
    anr = caudal_by_barrio.merge(
        consumo_monthly,
        on=["barrio", "year_month"],
        how="inner",
    )

    # Calcular ANR
    anr["anr_m3"] = anr["caudal_total_m3"] - anr["consumo_m3"]
    anr["anr_ratio"] = np.where(
        anr["consumo_m3"] > 0,
        anr["caudal_total_m3"] / anr["consumo_m3"],
        np.inf,
    )

    # ANR como porcentaje del caudal inyectado
    anr["anr_pct"] = np.where(
        anr["caudal_total_m3"] > 0,
        anr["anr_m3"] / anr["caudal_total_m3"] * 100,
        0,
    )

    return anr.sort_values("anr_ratio", ascending=False)


def detect_anr_anomalies(anr: pd.DataFrame,
                          threshold_ratio: float = 2.0) -> pd.DataFrame:
    """
    Detecta barrios con ANR anormalmente alto.

    Un ANR_ratio > threshold significa que el sector inyecta más del doble
    de lo que se factura en el barrio → posibles perdidas significativas.
    """
    # Calcular media y std de ANR_ratio por barrio (a lo largo del año)
    barrio_stats = anr.groupby("barrio").agg(
        avg_anr_ratio=("anr_ratio", "mean"),
        std_anr_ratio=("anr_ratio", "std"),
        avg_anr_pct=("anr_pct", "mean"),
        total_caudal=("caudal_total_m3", "sum"),
        total_consumo=("consumo_m3", "sum"),
        months=("year_month", "count"),
    ).reset_index()

    barrio_stats["total_anr_m3"] = barrio_stats["total_caudal"] - barrio_stats["total_consumo"]
    barrio_stats["total_anr_pct"] = np.where(
        barrio_stats["total_caudal"] > 0,
        barrio_stats["total_anr_m3"] / barrio_stats["total_caudal"] * 100,
        0,
    )

    barrio_stats["is_anomaly_anr"] = barrio_stats["avg_anr_ratio"] > threshold_ratio
    barrio_stats = barrio_stats.sort_values("avg_anr_ratio", ascending=False)

    return barrio_stats


def print_report(anr: pd.DataFrame, barrio_stats: pd.DataFrame, threshold: float):
    """Imprime reporte de ANR."""
    print("=" * 70)
    print("  M8 — Agua No Registrada (ANR) por Sector/Barrio")
    print("=" * 70)

    mapped = get_mapped_sectors()
    print(f"\n  Sectores hidraulicos mapeados: {len(mapped)} / 43")
    print(f"  Barrios con datos de caudal+consumo: {len(barrio_stats)}")
    print(f"  Periodo: 2024 (12 meses)")
    print(f"  Threshold ANR_ratio: {threshold}")

    print(f"\n  {'BARRIO':<40} {'ANR ratio':>10} {'ANR %':>8} {'Caudal M3':>12} {'Consumo M3':>12} {'Anomalia':>10}")
    print("  " + "-" * 94)
    for _, row in barrio_stats.iterrows():
        flag = "*** SI ***" if row["is_anomaly_anr"] else ""
        print(f"  {row['barrio']:<40} {row['avg_anr_ratio']:>10.2f} "
              f"{row['avg_anr_pct']:>7.1f}% "
              f"{row['total_caudal']:>12,.0f} {row['total_consumo']:>12,.0f} "
              f"{flag:>10}")

    anomalies = barrio_stats[barrio_stats["is_anomaly_anr"]]
    print(f"\n  Barrios con ANR anomalo (ratio > {threshold}): {len(anomalies)}")
    if len(anomalies) > 0:
        for _, row in anomalies.iterrows():
            print(f"    {row['barrio']}: ratio={row['avg_anr_ratio']:.2f} "
                  f"({row['avg_anr_pct']:.1f}% agua no registrada)")

    # Estadisticas globales
    total_caudal = barrio_stats["total_caudal"].sum()
    total_consumo = barrio_stats["total_consumo"].sum()
    total_anr = total_caudal - total_consumo
    if total_caudal > 0:
        anr_pct = total_anr / total_caudal * 100
        print(f"\n  ANR global (barrios mapeados):")
        print(f"    Caudal inyectado: {total_caudal:,.0f} M3")
        print(f"    Consumo facturado: {total_consumo:,.0f} M3")
        print(f"    Agua No Registrada: {total_anr:,.0f} M3 ({anr_pct:.1f}%)")

    print("\n  NOTA: ANR incluye perdidas tecnicas (fugas) + comerciales (fraude)")
    print("  Un ANR >30% se considera alto en la industria del agua")


def main():
    parser = argparse.ArgumentParser(description="M8 — Detector de Agua No Registrada")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="ANR ratio threshold (default: 2.0 = doble de lo facturado)")
    parser.add_argument("--uso", type=str, default=None,
                        help="Filtrar por tipo de uso (default: todos)")
    args = parser.parse_args()

    if not Path(CAUDAL_PATH).exists():
        print(f"ERROR: No se encuentra {CAUDAL_PATH}")
        return

    print("Cargando caudal horario 2024...")
    caudal_monthly = load_caudal_monthly(CAUDAL_PATH)
    print(f"  {len(caudal_monthly)} registros mensuales por sector")

    print("Cargando consumo del hackathon...")
    consumo_monthly = load_consumo_monthly(DATA_PATH, uso_filter=args.uso)
    print(f"  {len(consumo_monthly)} registros mensuales por barrio")

    print("Calculando ANR...")
    anr = compute_anr(caudal_monthly, consumo_monthly)
    barrio_stats = detect_anr_anomalies(anr, threshold_ratio=args.threshold)

    print_report(anr, barrio_stats, args.threshold)

    return barrio_stats


if __name__ == "__main__":
    main()
