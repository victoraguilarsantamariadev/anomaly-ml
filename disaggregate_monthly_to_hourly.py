"""
Disaggregación temporal: convierte totales mensuales en datos horarios sintéticos.

Estrategia B — usa la curva de demanda diurna típica de consumo de agua en España
para expandir cada total mensual en 720 horas ficticias (30 días × 24h).

Esto permite usar el pipeline completo de feature_engineering.py (nocturnal_min,
day_night_ratio, active_hours) aunque los datos base sean mensuales.

Basado en:
  - Curva de demanda típica publicada por AEAS (Asociación Española de Abastecimientos
    de Agua y Saneamiento) y trabajos académicos sobre patrones de consumo español.
  - DAIAD dataset (Alicante): si está disponible, se usa para calibrar la curva real.
    Descarga: https://data.hellenicdataservice.gr/dataset/78776f38-a58b-4a2a-a8f9-85b964fe5c95

Uso:
  from disaggregate_monthly_to_hourly import disaggregate_to_hourly
  df_hourly = disaggregate_to_hourly(df_monthly)  # columnas: timestamp, consumption
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# Curva de demanda diurna típica (España / Mediterráneo)
# Fuente: AEAS 2022, Ministerio MITECO, literatura académica
#
# Patrón: 24 valores (un factor de multiplicación por hora del día).
# Los factores suman 24 (equivalente a la media = 1.0 por hora).
# Horas pico: 7-9h (ducha mañana), 12-14h (mediodía), 20-22h (noche)
# Horas valle: 1-5h (nocturno profundo)
# ─────────────────────────────────────────────────────────────────
DIURNAL_CURVE_SPAIN = np.array([
    # 0h    1h    2h    3h    4h    5h    6h    7h    8h    9h   10h   11h
    0.35, 0.25, 0.20, 0.18, 0.18, 0.25, 0.55, 1.45, 1.70, 1.45, 1.20, 1.10,
    # 12h   13h   14h   15h   16h   17h   18h   19h   20h   21h   22h   23h
    1.20, 1.35, 1.10, 0.80, 0.75, 0.80, 0.95, 1.15, 1.25, 1.15, 0.90, 0.55,
], dtype=float)

# Factor para cada día de la semana (0=Lun, 6=Dom).
# Fines de semana: +15% (más tiempo en casa), laborables: base 1.0
WEEKDAY_FACTORS = np.array([1.0, 1.0, 1.0, 1.0, 1.05, 1.15, 1.15])


def load_daiad_curve(daiad_file: str) -> Optional[np.ndarray]:
    """
    Carga el DAIAD dataset (Alicante hourly) y calcula la curva diurna real.

    Si tienes el DAIAD descargado, pasa la ruta aquí para una curva más precisa
    que la curva genérica española.

    El DAIAD está disponible en:
    https://data.hellenicdataservice.gr/dataset/78776f38-a58b-4a2a-a8f9-85b964fe5c95

    Args:
        daiad_file: ruta al CSV del DAIAD (columnas: household_id, timestamp, consumption)

    Returns:
        Array de 24 factores diurnos, o None si no se puede cargar.
    """
    path = Path(daiad_file)
    if not path.exists():
        return None

    try:
        df = pd.read_csv(daiad_file)
        # Detectar columnas de timestamp y consumption
        ts_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
        cons_col = next((c for c in df.columns if "cons" in c.lower() or "volume" in c.lower()), None)
        if ts_col is None or cons_col is None:
            return None

        df[ts_col] = pd.to_datetime(df[ts_col])
        df["hour"] = df[ts_col].dt.hour
        df[cons_col] = pd.to_numeric(df[cons_col], errors="coerce").fillna(0)

        # Calcular el patrón diurno promedio normalizado
        hourly_mean = df.groupby("hour")[cons_col].mean()
        if len(hourly_mean) < 24:
            return None

        curve = hourly_mean.values[:24].astype(float)
        # Normalizar para que la suma sea 24 (media=1.0 por hora)
        curve = curve / (curve.sum() / 24)
        print(f"  Curva diurna cargada desde DAIAD Alicante ({len(df)} registros)")
        return curve

    except Exception as e:
        print(f"  Aviso: no se pudo cargar DAIAD ({e}). Usando curva genérica española.")
        return None


def disaggregate_to_hourly(
    df_monthly: pd.DataFrame,
    daiad_file: Optional[str] = None,
    noise_level: float = 0.08,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Convierte totales mensuales en lecturas horarias sintéticas.

    Args:
        df_monthly: DataFrame con columnas ['timestamp', 'consumption']
                    donde timestamp es el primer día del mes y consumption
                    es el total mensual.
        daiad_file: (opcional) ruta al DAIAD para calibrar la curva diurna.
        noise_level: nivel de ruido gaussiano añadido (0.08 = 8% de variabilidad).
        random_seed: semilla para reproducibilidad.

    Returns:
        DataFrame con columnas ['timestamp', 'consumption'] a granularidad horaria.
        Cada mes original se expande a ~720 horas (30-31 días × 24h).
    """
    # Cargar curva diurna (DAIAD si disponible, si no curva genérica española)
    diurnal_curve = None
    if daiad_file:
        diurnal_curve = load_daiad_curve(daiad_file)

    if diurnal_curve is None:
        diurnal_curve = DIURNAL_CURVE_SPAIN.copy()

    # Normalizar curva: suma = 1.0 (factor de distribución)
    diurnal_curve_normalized = diurnal_curve / diurnal_curve.sum()

    rng = np.random.default_rng(random_seed)

    hourly_records = []

    for _, row in df_monthly.iterrows():
        ts = pd.to_datetime(row["timestamp"])
        monthly_total = float(row["consumption"])

        # Si la fecha es el último día del mes (ej: 2022-01-31 del hackathon),
        # retroceder al primer día del mes para generar el mes completo.
        month_end_check = ts + pd.offsets.MonthEnd(0)
        if ts.date() == month_end_check.date():
            month_start = ts.replace(day=1)
        else:
            month_start = ts

        if monthly_total <= 0:
            # Mes con consumo cero — generar lecturas casi nulas (ruido de sensor)
            monthly_total_for_expand = 0.0
        else:
            monthly_total_for_expand = monthly_total

        # Generar todas las horas del mes
        month_end = month_start + pd.offsets.MonthEnd(0)
        n_days = (month_end - month_start).days + 1
        hours = pd.date_range(start=month_start, periods=n_days * 24, freq="h")

        # Distribuir el total mensual por hora usando la curva diurna
        # Factor de cada hora = curva_diurna × factor_día_semana
        weekday_factor = np.array([
            WEEKDAY_FACTORS[h.weekday()] for h in hours
        ])
        hour_factor = np.array([
            diurnal_curve_normalized[h.hour] for h in hours
        ])

        # Combinación de factores diurno × día semana
        combined = hour_factor * weekday_factor
        combined /= combined.sum()  # renormalizar a suma=1

        # Distribución base del total mensual
        hourly_base = combined * monthly_total_for_expand

        # Añadir ruido gaussiano proporcional a cada hora (preserva señal nocturna)
        if monthly_total_for_expand > 0:
            # Ruido relativo: mínimo 10% de la media para evitar ruido cero en horas bajas
            noise_std = noise_level * np.maximum(hourly_base, hourly_base.mean() * 0.1)
            noise = rng.normal(0, noise_std)
            hourly_values = np.clip(hourly_base + noise, 0, None)

            # Reescalar para que el total sea exactamente igual al original
            if hourly_values.sum() > 0:
                hourly_values = hourly_values * (monthly_total_for_expand / hourly_values.sum())
        else:
            # Consumo cero con ruido muy pequeño (detecta contador parado)
            hourly_values = np.abs(rng.normal(0, 1.0, size=len(hours)))

        for h, v in zip(hours, hourly_values):
            hourly_records.append({"timestamp": h, "consumption": float(v)})

    return pd.DataFrame(hourly_records)


def disaggregate_barrio(
    df_all: pd.DataFrame,
    barrio: str,
    uso: str = "DOMESTICO",
    daiad_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Atajo para desagregar un barrio concreto del hackathon.

    Args:
        df_all: DataFrame completo del hackathon (salida de load_hackathon_amaem)
        barrio: nombre del barrio
        uso: tipo de uso ('DOMESTICO', 'INDUSTRIAL', etc.)
        daiad_file: ruta opcional al DAIAD

    Returns:
        DataFrame horario con ['timestamp', 'consumption']
    """
    if "uso" in df_all.columns:
        mask = (df_all["barrio"] == barrio) & (df_all["uso"] == uso)
    else:
        mask = df_all["barrio"] == barrio

    subset = df_all[mask].copy().sort_values("fecha")
    subset["timestamp"] = pd.to_datetime(subset["fecha"])
    subset["consumption"] = pd.to_numeric(
        subset["consumo_litros"].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    ).fillna(0).astype(float)

    return disaggregate_to_hourly(subset[["timestamp", "consumption"]], daiad_file=daiad_file)


# ─────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    from train_local import load_hackathon_amaem
    from feature_engineering import compute_daily_features, add_rolling_features

    DATA_FILE = "data/datos-hackathon-amaem.xlsx-set-de-datos-.csv"
    DAIAD_FILE = "data/daiad_alicante.csv"  # opcional — si está disponible

    if not Path(DATA_FILE).exists():
        print(f"ERROR: No se encuentra {DATA_FILE}")
        sys.exit(1)

    print("Cargando datos del hackathon...")
    df_all = load_hackathon_amaem(DATA_FILE)

    barrio = "10-FLORIDA BAJA"
    uso    = "DOMESTICO"
    print(f"\nDesagregando {barrio} ({uso}) de mensual a horario...")

    # Usar DAIAD si está disponible
    daiad = DAIAD_FILE if Path(DAIAD_FILE).exists() else None
    if daiad:
        print(f"  Calibrando curva con DAIAD Alicante: {daiad}")
    else:
        print(f"  Usando curva genérica española (AEAS/MITECO)")
        print(f"  Descarga DAIAD (Alicante hourly) para mayor precisión:")
        print(f"  https://data.hellenicdataservice.gr/dataset/78776f38-a58b-4a2a-a8f9-85b964fe5c95")

    df_hourly = disaggregate_barrio(df_all, barrio, uso, daiad_file=daiad)
    print(f"\nDatos horarios generados: {len(df_hourly)} registros")
    print(f"Rango: {df_hourly['timestamp'].min()} → {df_hourly['timestamp'].max()}")
    print(f"\nPrimeras 24h (un día típico):")
    print(df_hourly.head(24)[["timestamp", "consumption"]].to_string(index=False))

    # Calcular features sobre los datos horarios sintéticos
    print(f"\nCalculando features con el pipeline completo...")
    daily = compute_daily_features(df_hourly)
    daily = add_rolling_features(daily)
    print(f"Features calculadas para {len(daily)} días")
    print(f"\nEjemplo de features (últimas 7 filas):")
    print(daily[["date", "daily_total", "nocturnal_min", "night_day_ratio", "zscore"]].tail(7).to_string())

    # Gráfica: curva diurna media generada
    Path("output").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Curva diurna usada
    ax1 = axes[0]
    ax1.bar(range(24), DIURNAL_CURVE_SPAIN / DIURNAL_CURVE_SPAIN.mean(), color="steelblue", alpha=0.7)
    ax1.axhspan(0, 1, alpha=0.1, color="blue", label="Media")
    ax1.axvspan(2, 5, alpha=0.15, color="orange", label="Zona nocturna (02-05h)")
    ax1.set_title("Curva diurna de consumo — España (AEAS)")
    ax1.set_xlabel("Hora del día")
    ax1.set_ylabel("Factor relativo (1.0 = media)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Total diario de los datos desagregados
    ax2 = axes[1]
    ax2.plot(daily["date"], daily["daily_total"], "b-", linewidth=1)
    ax2.set_title(f"Consumo diario sintético — {barrio} ({uso})")
    ax2.set_ylabel("Consumo (litros/día)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f"output/{barrio}_{uso}_disaggregated.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGráfica guardada: {output_path}")
    print(f"\nPróximo paso: entrenar el modelo M2 sobre estos datos horarios sintéticos")
    print(f"  python train_local.py --file <csv> --mode hackathon")
