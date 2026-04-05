"""
Generador de datos sintéticos horarios con fugas inyectadas.

Simula el formato real de Aguas de Alicante: consumo hora a hora por domicilio,
con anomalías deliberadas para demostrar el detector de fugas.

Tipos de anomalía inyectados:
  1. Fuga lenta continua (cisterna/grifo que gotea): +flujo constante 24/7
  2. Rotura de tubería: pico súbito y sostenido durante horas
  3. Degradación gradual: fuga que empeora linealmente con el tiempo
  4. Consumo nocturno anómalo: flujo solo entre 1-5am (fraude/uso no autorizado)
  5. Fuga intermitente: picos periódicos cada N horas

Salida:
  - data/synthetic_hourly_domicilio.csv  (datos horarios)
  - data/synthetic_leak_labels.csv       (ground truth: qué domicilios tienen fuga y cuándo)

Uso:
  python generate_synthetic_leaks.py
  python generate_synthetic_leaks.py --n-domicilios 500 --days 90
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ── Curva diurna típica España (de disaggregate_monthly_to_hourly.py) ──
DIURNAL_CURVE = np.array([
    # 0h    1h    2h    3h    4h    5h    6h    7h    8h    9h   10h   11h
    0.35, 0.25, 0.20, 0.18, 0.18, 0.25, 0.55, 1.45, 1.70, 1.45, 1.20, 1.10,
    # 12h   13h   14h   15h   16h   17h   18h   19h   20h   21h   22h   23h
    1.20, 1.35, 1.10, 0.80, 0.75, 0.80, 0.95, 1.15, 1.25, 1.15, 0.90, 0.55,
])
DIURNAL_CURVE = DIURNAL_CURVE / DIURNAL_CURVE.sum()  # normalizar a distribución

WEEKDAY_FACTORS = np.array([1.0, 1.0, 1.0, 1.0, 1.05, 1.15, 1.15])

# Barrios reales de Alicante (del dataset original)
BARRIOS = [
    "10-FLORIDA BAJA", "11-CIUDAD DE ASIS", "12-POLIGONO BABEL",
    "13-SAN GABRIEL", "14-ENSANCHE DIPUTACION", "15-POLIGONO SAN BLAS",
    "16-PLA DEL BON REPOS", "17-CAROLINAS ALTAS", "18-CAROLINAS BAJAS",
    "19-GARBINET", "20-BENALUA", "21-ALIPARK", "22-ALTOZANO",
    "23-CENTRO TRADICIONAL", "24-CASCO ANTIGUO",
]

USOS = ["DOMESTICO", "COMERCIAL", "NO DOMESTICO"]

# Consumo medio mensual por contrato en litros (aproximado desde datos reales)
CONSUMO_MEDIO_MENSUAL = {
    "DOMESTICO": 8_000,      # ~8 m³/mes por contrato doméstico
    "COMERCIAL": 15_000,     # ~15 m³/mes
    "NO DOMESTICO": 12_000,  # ~12 m³/mes
}

# Factores estacionales mensuales (verano más consumo en Alicante)
SEASONAL_FACTORS = {
    1: 0.80, 2: 0.82, 3: 0.88, 4: 0.95, 5: 1.05, 6: 1.20,
    7: 1.35, 8: 1.30, 9: 1.10, 10: 0.95, 11: 0.85, 12: 0.78,
}


def generate_domicilios(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Genera catálogo de domicilios sintéticos."""
    records = []
    for i in range(n):
        barrio = rng.choice(BARRIOS)
        uso = rng.choice(USOS, p=[0.70, 0.15, 0.15])
        contrato = f"CTR-{barrio[:2]}-{i:05d}"
        factor_individual = rng.uniform(0.6, 1.4)
        records.append({
            "contrato_id": contrato,
            "barrio": barrio,
            "uso": uso,
            "factor_individual": factor_individual,
        })
    return pd.DataFrame(records)


def generate_baseline_hourly(
    domicilios: pd.DataFrame,
    start_date: str,
    n_days: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Genera consumo horario base (sin anomalías) para todos los domicilios."""
    dates = pd.date_range(start=start_date, periods=n_days * 24, freq="h")

    all_rows = []
    for _, dom in domicilios.iterrows():
        base_monthly = CONSUMO_MEDIO_MENSUAL[dom["uso"]] * dom["factor_individual"]
        base_hourly = base_monthly / (30 * 24)  # litros/hora promedio

        for ts in dates:
            hour = ts.hour
            weekday = ts.weekday()
            month = ts.month

            consumption = (
                base_hourly
                * (DIURNAL_CURVE[hour] * 24)
                * WEEKDAY_FACTORS[weekday]
                * SEASONAL_FACTORS.get(month, 1.0)
            )

            noise = rng.normal(1.0, 0.15)
            consumption = max(0, consumption * noise)

            if dom["uso"] == "DOMESTICO" and hour in range(1, 5):
                if rng.random() < 0.3:
                    consumption = 0.0

            all_rows.append({
                "timestamp": ts,
                "contrato_id": dom["contrato_id"],
                "barrio": dom["barrio"],
                "uso": dom["uso"],
                "consumo_litros": round(consumption, 2),
            })

    return pd.DataFrame(all_rows)


def inject_leaks(
    df: pd.DataFrame,
    domicilios: pd.DataFrame,
    n_days: int,
    rng: np.random.Generator,
    leak_fraction: float = 0.12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inyecta anomalías de fuga en un subconjunto de domicilios.

    Returns:
        df con fugas inyectadas, DataFrame de labels (ground truth)
    """
    df = df.copy()
    n_leaks = max(1, int(len(domicilios) * leak_fraction))
    leak_ids = rng.choice(domicilios["contrato_id"].values, size=n_leaks, replace=False)

    leak_types = [
        "fuga_lenta_continua",
        "rotura_tuberia",
        "degradacion_gradual",
        "consumo_nocturno_anomalo",
        "fuga_intermitente",
    ]

    labels = []
    timestamps = sorted(df["timestamp"].unique())
    total_hours = len(timestamps)

    for contrato_id in leak_ids:
        leak_type = rng.choice(leak_types)
        mask = df["contrato_id"] == contrato_id

        start_idx = rng.integers(int(total_hours * 0.2), int(total_hours * 0.7))
        leak_start = timestamps[start_idx]

        if leak_type == "rotura_tuberia":
            duration_hours = rng.integers(6, 48)
        else:
            duration_hours = rng.integers(24 * 7, min(24 * 30, total_hours - start_idx))

        leak_end_idx = min(start_idx + duration_hours, total_hours - 1)
        leak_end = timestamps[leak_end_idx]

        time_mask = (df["timestamp"] >= leak_start) & (df["timestamp"] <= leak_end)
        affected = mask & time_mask

        uso = domicilios.loc[domicilios["contrato_id"] == contrato_id, "uso"].iloc[0]
        base_hourly = CONSUMO_MEDIO_MENSUAL[uso] / (30 * 24)

        if leak_type == "fuga_lenta_continua":
            extra = rng.uniform(2, 8)
            df.loc[affected, "consumo_litros"] += extra

        elif leak_type == "rotura_tuberia":
            multiplier = rng.uniform(5, 15)
            df.loc[affected, "consumo_litros"] += base_hourly * multiplier

        elif leak_type == "degradacion_gradual":
            n_affected = affected.sum()
            if n_affected > 0:
                ramp = np.linspace(0, base_hourly * rng.uniform(2, 6), n_affected)
                df.loc[affected, "consumo_litros"] += ramp

        elif leak_type == "consumo_nocturno_anomalo":
            night_mask = affected & df["timestamp"].dt.hour.isin(range(1, 5))
            extra = rng.uniform(10, 30)
            df.loc[night_mask, "consumo_litros"] += extra

        elif leak_type == "fuga_intermitente":
            period = rng.integers(4, 9)
            affected_indices = df.index[affected]
            for j, idx in enumerate(affected_indices):
                if j % period == 0:
                    df.loc[idx, "consumo_litros"] += base_hourly * rng.uniform(3, 8)

        labels.append({
            "contrato_id": contrato_id,
            "barrio": domicilios.loc[domicilios["contrato_id"] == contrato_id, "barrio"].iloc[0],
            "uso": uso,
            "tipo_fuga": leak_type,
            "inicio_fuga": str(leak_start),
            "fin_fuga": str(leak_end),
            "duracion_horas": int(duration_hours),
        })

    labels_df = pd.DataFrame(labels)
    return df, labels_df


def main():
    parser = argparse.ArgumentParser(description="Genera datos sintéticos horarios con fugas")
    parser.add_argument("--n-domicilios", type=int, default=200,
                        help="Número de domicilios a simular (default: 200)")
    parser.add_argument("--days", type=int, default=60,
                        help="Días de datos a generar (default: 60)")
    parser.add_argument("--start-date", type=str, default="2024-06-01",
                        help="Fecha de inicio (default: 2024-06-01)")
    parser.add_argument("--leak-fraction", type=float, default=0.12,
                        help="Fracción de domicilios con fuga (default: 0.12)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path("data")

    print(f"Generando {args.n_domicilios} domicilios, {args.days} días desde {args.start_date}...")

    domicilios = generate_domicilios(args.n_domicilios, rng)
    print(f"  Domicilios: {len(domicilios)} ({domicilios['uso'].value_counts().to_dict()})")

    print("  Generando consumo horario base...")
    df = generate_baseline_hourly(domicilios, args.start_date, args.days, rng)
    print(f"  Filas generadas: {len(df):,}")

    print(f"  Inyectando fugas (~{args.leak_fraction*100:.0f}% de domicilios)...")
    df, labels = inject_leaks(df, domicilios, args.days, rng, args.leak_fraction)

    print(f"\n  Fugas inyectadas: {len(labels)}")
    print(f"  Tipos de fuga:")
    for tipo, count in labels["tipo_fuga"].value_counts().items():
        print(f"    - {tipo}: {count}")

    hourly_path = out_dir / "synthetic_hourly_domicilio.csv"
    labels_path = out_dir / "synthetic_leak_labels.csv"

    df.to_csv(hourly_path, index=False)
    labels.to_csv(labels_path, index=False)

    print(f"\n  Datos guardados:")
    print(f"    {hourly_path} ({hourly_path.stat().st_size / 1e6:.1f} MB)")
    print(f"    {labels_path}")

    print(f"\n  Rango temporal: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Consumo medio (litros/hora): {df['consumo_litros'].mean():.2f}")
    print(f"  Consumo max  (litros/hora): {df['consumo_litros'].max():.2f}")

    print("\n  Ejemplo de labels (ground truth):")
    print(labels.to_string(index=False))


if __name__ == "__main__":
    main()
