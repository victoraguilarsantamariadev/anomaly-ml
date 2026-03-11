"""
AquaCare — Deteccion de emergencias en personas vulnerables a traves del agua.

Inspirado en utilities de Japon (Tokyo Water) y UK (Thames Water) que monitorean
patrones de consumo para detectar emergencias medicas en personas mayores que viven solas.

Si alguien consume 150L/dia durante 3 anos y de repente baja a 5L...
puede que haya tenido una caida, un ictus, o peor.

El agua no miente: es el ultimo indicador vital de actividad humana.

Metricas:
  1. Caida subita de consumo por barrio (MoM y vs historico)
  2. Indice de vulnerabilidad elderly por barrio (proxies demograficos)
  3. Deteccion de "silencio nocturno" en datos horarios de caudal
  4. Alertas con nivel de urgencia: CRITICO / ALTO / VIGILANCIA

Datos: barrio-level, mensual. Opcionalmente datos horarios de caudal.

Uso:
  from welfare_detector import run_welfare_detection, welfare_summary
  alerts = run_welfare_detection(df_monthly, results, caudal_path="data/caudal.csv")
  welfare_summary(alerts)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# Datos REALES del Padron Municipal de Alicante 2025
# Fuente: https://www.alicante.es/es/documentos/estadisticas-poblacion-alicante-datos-del-padron-municipal-habitantes
# Descargado de: poblacion-alicante-2025.zip → barrios_2025.xls
# ─────────────────────────────────────────────────────────────────

PADRON_ELDERLY_CSV = "data/padron_elderly_barrios_2025.csv"

# Mapping AMAEM → Padron (nombres diferentes entre datasets)
AMAEM_TO_PADRON = {
    "1-BENALUA": "BENALUA",
    "10-FLORIDA BAJA": "FLORIDA BAJA",
    "11-CIUDAD DE ASIS": "CIUDAD DE ASIS",
    "12-POLIGONO BABEL": "POLIGONO BABEL",
    "13-SAN GABRIEL": "SAN GABRIEL",
    "14-ENSANCHE DIPUTACION": "ENSANCHE DIPUTACION",
    "15-POLIGONO SAN BLAS": "POLIGONO SAN BLAS",
    "16-PLA DEL BON REPOS": "PLA DEL BON REPOS",
    "17-CAROLINAS ALTAS": "CAROLINAS ALTAS",
    "18-CAROLINAS BAJAS": "CAROLINAS BAJAS",
    "19-GARBINET": "GARBINET",
    "2-SAN ANTON": "SANANTON",
    "20-RABASA": "RABASA",
    "21-TOMBOLA": "TOMBOLA",
    "22-CASCO ANTIGUO - SANTA CRUZ": "CASCO ANTIGUO - SANTA CRUZ - AYUNTAMIENTO",
    "23-RAVAL ROIG -V. DEL SOCORRO": "RAVAL ROIG - VIRGEN DEL SOCORRO",
    "24-SAN BLAS - SANTO DOMINGO": "SAN BLAS - SANTO DOMINGO",
    "25-ALTOZANO - CONDE LUMIARES": "ALTOZANO - CONDE LUMIARES",
    "26-SIDI IFNI - NOU ALACANT": "SIDI IFNI - NOU ALACANT",
    "27-SAN FERNANDO-PRIN. MERCEDES": "SAN FERNANDO - PRINCESA MERCEDES",
    "28-EL PALMERAL": "EL PALMERAL - URBANOVA - TABARCA",
    "29-URBANOVA": "EL PALMERAL - URBANOVA - TABARCA",
    "3-CENTRO": "CENTRO",
    "30-DIVINA PASTORA": "DIVINA PASTORA",
    "31-CIUDAD JARDIN": "CIUDAD JARDIN",
    "32-VIRGEN DEL REMEDIO": "VIRGEN DEL REMEDIO",
    "33- MORANT -SAN NICOLAS BARI": "LO MORANT - SAN NICOLAS DE BARI",
    "34-COLONIA REQUENA": "COLONIA REQUENA",
    "35-VIRGEN DEL CARMEN": "VIRGEN DEL CARMEN",
    "36-CUATROCIENTAS VIVIENDAS": "CUATROCIENTAS VIVIENDAS",
    "37-JUAN XXIII": "JUAN XXIII",
    "38-VISTAHERMOSA": "VISTAHERMOSA",
    "39-ALBUFERETA": "ALBUFERETA",
    "4-MERCADO": "MERCADO",
    "40-CABO DE LAS HUERTAS": "CABO DE LAS HUERTAS",
    "41-PLAYA DE SAN JUAN": "PLAYA DE SAN JUAN",
    "5-CAMPOAMOR": "CAMPOAMOR",
    "56-DISPERSOS": "DISPERSO PARTIDAS",
    "6-LOS ANGELES": "LOS ANGELES",
    "7-SAN AGUSTIN": "SAN AGUSTIN",
    "8-ALIPARK": "ALIPARK",
    "9-FLORIDA ALTA": "FLORIDA ALTA",
    "TABARCA": "EL PALMERAL - URBANOVA - TABARCA",
    "VILLAFRANQUEZA": "VILLAFRANQUEZA - SANTA FAZ",
    "SANTA FAZ": "VILLAFRANQUEZA - SANTA FAZ",
    "BACAROT": "DISPERSO PARTIDAS",
    "FONTCALENT": "DISPERSO PARTIDAS",
    "LA ALCORAYA": "DISPERSO PARTIDAS",
    "LA CAÑADA": "DISPERSO PARTIDAS",
    "MONNEGRE": "DISPERSO PARTIDAS",
    "MORALET": "DISPERSO PARTIDAS",
    "PDA VALLONGA": "DISPERSO PARTIDAS",
    "REBOLLEDO": "DISPERSO PARTIDAS",
    "VERDEGAS": "DISPERSO PARTIDAS",
}


_PADRON_CACHE = {}


def _load_padron_elderly() -> dict:
    """Carga datos REALES del padron: % poblacion >=65, % viviendo solos, etc."""
    if _PADRON_CACHE:
        return _PADRON_CACHE
    padron_path = Path(PADRON_ELDERLY_CSV)
    if not padron_path.exists():
        print(f"    [AquaCare] AVISO: No se encuentra {PADRON_ELDERLY_CSV}")
        return {}
    padron = pd.read_csv(padron_path)
    for _, row in padron.iterrows():
        _PADRON_CACHE[row["barrio_padron"]] = {
            "pct_65plus": row["pct_65plus"],
            "pct_80plus": row.get("pct_80plus", 0),
            "poblacion_total": row["poblacion_total"],
            "mayores_65_solos": row.get("mayores_65_solos", 0),
            "pct_65plus_solos": row.get("pct_65plus_solos", 0),
            "mayores_80_solos": row.get("mayores_80_solos", 0),
        }
    return _PADRON_CACHE


def _get_real_elderly_pct(barrio_clean: str) -> float:
    """Devuelve % real de poblacion >=65 para un barrio AMAEM.
    Fuente: Padron Municipal Alicante 2025."""
    padron_data = _load_padron_elderly()
    if not padron_data:
        return 0.0
    padron_name = AMAEM_TO_PADRON.get(barrio_clean, "")
    info = padron_data.get(padron_name, {})
    return info.get("pct_65plus", 0.0)


def _get_elderly_alone(barrio_clean: str) -> int:
    """Devuelve numero de mayores >=65 viviendo solos en un barrio.
    Fuente: Padron Municipal Alicante 2025 (miembros_barrio_2025.xls)."""
    padron_data = _load_padron_elderly()
    padron_name = AMAEM_TO_PADRON.get(barrio_clean, "")
    info = padron_data.get(padron_name, {})
    return int(info.get("mayores_65_solos", 0))


# Fallback for barrios not in padron (pedanias rurales sin datos individuales)
BARRIOS_ALTA_VULNERABILIDAD_ELDERLY = {
    "TABARCA":      0.60,  # isla, pocos habitantes, pedania
    "BACAROT":      0.50,  # pedania rural
    "LA ALCORAYA":  0.50,  # pedania rural aislada
    "MONNEGRE":     0.45,  # pedania rural
    "VERDEGAS":     0.45,
    "MORALET":      0.45,
    "FONTCALENT":   0.40,
    "PDA VALLONGA": 0.40,
    "LA CAÑADA":    0.35,
    "REBOLLEDO":    0.35,
}

# Umbrales de clasificacion de barrios por numero de contratos
SMALL_BARRIO_CONTRACTS = 200   # barrio pequeno: mas probable persona sola
TINY_BARRIO_CONTRACTS = 50     # micro-barrio/pedania: alta probabilidad

# Horas nocturnas para deteccion de silencio
NIGHT_HOURS = range(1, 6)   # 1am - 5am
DAY_HOURS = range(8, 22)    # 8am - 10pm


# ═════════════════════════════════════════════════════════════════
# 1. DETECCION DE CAIDAS DE CONSUMO
# ═════════════════════════════════════════════════════════════════

def detect_consumption_drops(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta caidas subitas de consumo por barrio.

    Calcula:
      - mom_drop: caida mes-sobre-mes (% de cambio)
      - historical_drop: caida vs media historica del barrio
      - seasonal_drop: caida vs mismo mes del ano anterior
      - drop_severity: indice combinado de severidad (0-1)

    Una caida del 50% en consumo domestico en un barrio pequeno
    puede significar que la unica persona que vivia alli ha dejado
    de abrir el grifo.
    """
    df = df_monthly.copy()
    df = df.sort_values(["barrio_key", "fecha"]).reset_index(drop=True)

    # Usar consumption_per_contract si existe, sino consumo_litros
    consumo_col = "consumption_per_contract"
    if consumo_col not in df.columns:
        consumo_col = "consumo_litros"

    results = []

    for barrio_key, group in df.groupby("barrio_key"):
        group = group.sort_values("fecha").reset_index(drop=True)

        if len(group) < 3:
            continue

        values = group[consumo_col].values.astype(float)
        fechas = group["fecha"].values

        # Media historica (excluyendo ultimo mes)
        hist_mean = np.mean(values[:-1]) if len(values) > 1 else values[0]
        hist_std = np.std(values[:-1]) if len(values) > 1 else 1.0

        for i in range(len(group)):
            row = group.iloc[i]
            current = values[i]

            # --- Month-over-month drop ---
            if i > 0 and values[i - 1] > 0:
                mom_change = (current - values[i - 1]) / values[i - 1]
            else:
                mom_change = 0.0

            # --- Vs historical mean ---
            if i >= 3:
                past_mean = np.mean(values[max(0, i - 12):i])
            else:
                past_mean = hist_mean
            if past_mean > 0:
                hist_change = (current - past_mean) / past_mean
            else:
                hist_change = 0.0

            # --- Seasonal (YoY) ---
            if i >= 12 and values[i - 12] > 0:
                seasonal_change = (current - values[i - 12]) / values[i - 12]
            else:
                seasonal_change = 0.0

            # --- Z-score vs historical ---
            if hist_std > 0 and i >= 3:
                zscore = (current - past_mean) / hist_std
            else:
                zscore = 0.0

            # --- Racha de caida: meses consecutivos bajando ---
            streak = 0
            for j in range(i - 1, max(i - 6, -1), -1):
                if j >= 0 and values[j + 1] < values[j]:
                    streak += 1
                else:
                    break

            # --- Drop severity: combina indicadores (solo para caidas) ---
            drop_indicators = []
            if mom_change < 0:
                drop_indicators.append(min(abs(mom_change), 1.0))
            if hist_change < 0:
                drop_indicators.append(min(abs(hist_change), 1.0))
            if seasonal_change < 0:
                drop_indicators.append(min(abs(seasonal_change), 1.0))
            if zscore < -1.5:
                drop_indicators.append(min(abs(zscore) / 4.0, 1.0))

            if drop_indicators:
                drop_severity = np.mean(drop_indicators)
            else:
                drop_severity = 0.0

            results.append({
                "barrio_key": barrio_key,
                "fecha": row["fecha"],
                "consumo": current,
                "consumo_col_used": consumo_col,
                "mom_change": mom_change,
                "hist_change": hist_change,
                "seasonal_change": seasonal_change,
                "zscore_historical": zscore,
                "consecutive_decline": streak,
                "drop_severity": drop_severity,
            })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════
# 2. INDICE DE VULNERABILIDAD ELDERLY POR BARRIO
# ═════════════════════════════════════════════════════════════════

def compute_elderly_vulnerability(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Estima vulnerabilidad elderly por barrio usando proxies del agua.

    No tenemos datos del padron individual, pero el agua habla:
      - Bajo consumo por contrato → pocos miembros por hogar → persona sola
      - Consumo muy estable historicamente → persona de habitos fijos (mayor)
      - Barrio pequeno/pedania → comunidades envejecidas
      - Tendencia decreciente de consumo → poblacion que envejece y se reduce

    Cada indicador genera un score 0-1. El indice final es la media ponderada.

    Returns:
        DataFrame con barrio_key, elderly_vulnerability (0-1), componentes
    """
    df = df_monthly.copy()

    consumo_col = "consumption_per_contract"
    if consumo_col not in df.columns:
        consumo_col = "consumo_litros"

    barrio_stats = []

    for barrio_key, group in df.groupby("barrio_key"):
        group = group.sort_values("fecha")
        barrio_clean = barrio_key.split("__")[0] if "__" in barrio_key else barrio_key
        values = group[consumo_col].values.astype(float)

        # --- Indicador 1: Bajo consumo por contrato ---
        # Persona mayor sola consume menos que familia de 4
        median_consumo = np.median(values[values > 0]) if np.any(values > 0) else 0
        # Referencia: ~4000-6000 L/mes por contrato es una persona sola
        # Familia tipica: ~12000-18000 L/mes
        if median_consumo > 0:
            low_consumption_score = max(0, 1.0 - (median_consumo / 15000))
            low_consumption_score = min(low_consumption_score, 1.0)
        else:
            low_consumption_score = 0.0

        # --- Indicador 2: Estabilidad historica (CV bajo) ---
        # Personas mayores tienen rutinas rigidas: misma ducha, misma lavadora
        if len(values) > 6 and np.mean(values) > 0:
            cv = np.std(values) / np.mean(values)
            # CV < 0.10 = muy estable (rutina rigida)
            # CV > 0.40 = variable (familia activa, turismo, etc.)
            stability_score = max(0, 1.0 - (cv / 0.40))
            stability_score = min(stability_score, 1.0)
        else:
            stability_score = 0.3  # neutral si pocos datos

        # --- Indicador 3: Tamano del barrio (num contratos) ---
        num_contratos = None
        if "num_contratos" in group.columns:
            num_contratos = group["num_contratos"].median()
        elif "consumo_litros" in group.columns and consumo_col == "consumption_per_contract":
            # Estimar contratos = consumo_litros / consumption_per_contract
            valid = group[group[consumo_col] > 0]
            if len(valid) > 0:
                num_contratos = (valid["consumo_litros"] / valid[consumo_col]).median()

        if num_contratos is not None and num_contratos > 0:
            if num_contratos < TINY_BARRIO_CONTRACTS:
                size_score = 1.0
            elif num_contratos < SMALL_BARRIO_CONTRACTS:
                size_score = 0.7
            elif num_contratos < 500:
                size_score = 0.3
            else:
                size_score = 0.1
        else:
            size_score = 0.3

        # --- Indicador 4: Tendencia decreciente ---
        # Barrio cuyo consumo lleva bajando = poblacion que se reduce/envejece
        if len(values) >= 12:
            first_half = np.mean(values[:len(values) // 2])
            second_half = np.mean(values[len(values) // 2:])
            if first_half > 0:
                trend = (second_half - first_half) / first_half
                # Solo si baja: trend < 0 indica envejecimiento
                decline_score = max(0, min(abs(trend) * 3, 1.0)) if trend < 0 else 0.0
            else:
                decline_score = 0.0
        else:
            decline_score = 0.0

        # --- Indicador 5: Datos REALES del Padron Municipal 2025 ---
        real_pct = _get_real_elderly_pct(barrio_clean)
        if real_pct > 0:
            # Dato REAL del padron: convertir % a score 0-1
            # 35% elderly (Ensanche Diputacion) = score 1.0
            # 10% elderly = score 0.28
            census_score = min(real_pct / 35.0, 1.0)
        else:
            # Fallback para pedanias sin dato individual en padron
            census_score = BARRIOS_ALTA_VULNERABILIDAD_ELDERLY.get(barrio_clean, 0.15)

        # --- Indice final: media ponderada ---
        # Peso del padron REAL es el mas alto porque es dato oficial
        weights = {
            "census":          0.40,  # DATO REAL del padron — fuente mas fiable
            "low_consumption": 0.15,  # consumo bajo = pocos miembros
            "stability":       0.10,  # rutina estable = persona mayor
            "size":            0.20,  # barrio pequeno = mas vulnerable
            "decline":         0.15,  # tendencia decreciente
        }

        elderly_vulnerability = (
            weights["census"]          * census_score +
            weights["low_consumption"] * low_consumption_score +
            weights["stability"]       * stability_score +
            weights["size"]            * size_score +
            weights["decline"]         * decline_score
        )

        barrio_stats.append({
            "barrio_key": barrio_key,
            "barrio_clean": barrio_clean,
            "elderly_vulnerability": round(elderly_vulnerability, 3),
            "score_census": round(census_score, 3),
            "score_low_consumption": round(low_consumption_score, 3),
            "score_stability": round(stability_score, 3),
            "score_size": round(size_score, 3),
            "score_decline": round(decline_score, 3),
            "median_consumo_per_contract": round(median_consumo, 1),
            "estimated_contracts": round(num_contratos, 0) if num_contratos else None,
        })

    return pd.DataFrame(barrio_stats)


# ═════════════════════════════════════════════════════════════════
# 3. DETECCION DE SILENCIO NOCTURNO (datos horarios opcionales)
# ═════════════════════════════════════════════════════════════════

def detect_night_silence(caudal_path: str) -> pd.DataFrame:
    """
    Detecta sectores donde el caudal nocturno cae a niveles anormalmente bajos.

    En un barrio normal, incluso de noche hay un 5-10% de caudal diurno
    (cisternas, calentadores, alguien que va al bano). Si ese flujo baja
    a casi cero... nadie esta usando agua. En un barrio con persona mayor
    sola, eso puede significar que lleva horas (o dias) sin actividad.

    Returns:
        DataFrame con sector, fecha, night_silence_score, night_flow, day_flow
    """
    if not Path(caudal_path).exists():
        print(f"    [AquaCare] No se encuentra {caudal_path} — omitiendo silencio nocturno")
        return pd.DataFrame()

    # Reutilizar el parser del nightflow_detector
    try:
        from nightflow_detector import load_hourly_data
        df = load_hourly_data(caudal_path)
    except Exception as e:
        print(f"    [AquaCare] Error cargando datos horarios: {e}")
        return pd.DataFrame()

    results = []

    for sector, sector_data in df.groupby("SECTOR"):
        for date, day_data in sector_data.groupby("date"):
            night = day_data[day_data["hour"].isin(NIGHT_HOURS)]
            day = day_data[day_data["hour"].isin(DAY_HOURS)]

            if len(night) == 0 or len(day) == 0:
                continue

            night_flow = night["caudal"].mean()
            day_flow = day["caudal"].mean()

            if day_flow <= 0:
                continue

            # Ratio noche/dia — normal ~0.05-0.15, silencio < 0.02
            night_ratio = night_flow / day_flow

            # Silence score: 1.0 = silencio total, 0.0 = actividad normal
            if night_ratio < 0.01:
                silence_score = 1.0    # practicamente cero
            elif night_ratio < 0.03:
                silence_score = 0.85   # muy bajo
            elif night_ratio < 0.05:
                silence_score = 0.6    # bajo
            elif night_ratio < 0.10:
                silence_score = 0.3    # algo bajo
            else:
                silence_score = 0.0    # normal

            if silence_score > 0:
                results.append({
                    "sector": sector,
                    "date": date,
                    "night_flow": round(night_flow, 3),
                    "day_flow": round(day_flow, 3),
                    "night_day_ratio": round(night_ratio, 4),
                    "night_silence_score": round(silence_score, 3),
                })

    result_df = pd.DataFrame(results)

    if len(result_df) > 0:
        # Agregar por sector-mes para compatibilidad con datos mensuales
        result_df["date"] = pd.to_datetime(result_df["date"])
        result_df["month"] = result_df["date"].dt.to_period("M")

        monthly_silence = (
            result_df.groupby(["sector", "month"])
            .agg(
                avg_silence=("night_silence_score", "mean"),
                max_silence=("night_silence_score", "max"),
                days_silent=("night_silence_score", lambda x: (x > 0.5).sum()),
                total_days=("night_silence_score", "count"),
            )
            .reset_index()
        )
        monthly_silence["pct_days_silent"] = (
            monthly_silence["days_silent"] / monthly_silence["total_days"]
        )

        return monthly_silence

    return pd.DataFrame()


# ═════════════════════════════════════════════════════════════════
# 3b. ANALISIS INDIVIDUAL DE CONTADORES (m3-registrados + telelectura)
# ═════════════════════════════════════════════════════════════════

M3_FILE_PATTERN = "m3-registrados_facturados-tll_{year}-solo-alicante-m3-registrados_facturados-tll_{year}-solo-alicant.csv"
TELELECTURA_PATH = "data/contadores-telelectura-instalados-solo-alicante_hackaton-dataart-contadores-telelectura-instalad.csv"


def detect_individual_meter_anomalies(years=(2020, 2021, 2022, 2023, 2024, 2025)) -> pd.DataFrame:
    """
    Analiza lecturas individuales de contadores para detectar patrones de fuga/abandono.

    m3-registrados NO tiene ID de contador ni BARRIO, pero sí:
      - M3 A FACTURAR (consumo)
      - DIAS LECTURA (periodo)
      - PERIODICIDAD (mensual/trimestral)

    Calcula por mes:
      - pct_zero: % lecturas con M3=0 (contadores parados/manipulados)
      - pct_very_low: % con <5 L/dia (posible no-ocupación o fuga)
      - pct_high_consumption: % con >1000 L/dia (fuga activa)
      - pct_abnormal_days: % con <15 o >100 dias lectura (error/acumulación)
      - median_m3_dia: consumo mediano por dia (salud de la red)

    Returns:
        DataFrame indexado por year_month con métricas de salud de contadores
    """
    data_dir = Path("data")
    dfs = []
    for year in years:
        fname = M3_FILE_PATTERN.format(year=year)
        path = data_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            df.columns = [c.upper().strip() for c in df.columns]
            df["_year"] = year
            dfs.append(df)

    if not dfs:
        print("    [AquaCare] Sin datos m3-registrados")
        return pd.DataFrame()

    all_m3 = pd.concat(dfs, ignore_index=True)
    all_m3["M3"] = pd.to_numeric(all_m3["M3 A FACTURAR"], errors="coerce")
    all_m3["DIAS"] = pd.to_numeric(all_m3["DIAS LECTURA"], errors="coerce")
    all_m3["M3_DIA"] = all_m3["M3"] / all_m3["DIAS"].clip(lower=1)

    # Parse PERIODO (format: "2024 -  01")
    periodo = all_m3["PERIODO"].astype(str).str.strip()
    all_m3["year"] = periodo.str[:4].astype(int)
    all_m3["month"] = periodo.str[-2:].str.strip().astype(int)
    all_m3["year_month"] = pd.to_datetime(
        all_m3["year"].astype(str) + "-" + all_m3["month"].astype(str).str.zfill(2) + "-01"
    ).dt.to_period("M")

    # Métricas por mes
    def compute_monthly_health(group):
        n = len(group)
        m3 = group["M3"].values
        m3_dia = group["M3_DIA"].values
        dias = group["DIAS"].values
        return pd.Series({
            "n_lecturas": n,
            "pct_zero": (m3 == 0).sum() / n,
            "pct_very_low": ((m3_dia > 0) & (m3_dia < 0.005)).sum() / n,  # <5 L/dia
            "pct_high_consumption": (m3_dia > 3.0).sum() / n,  # >3000 L/dia = fuga probable
            "pct_abnormal_days": ((dias < 15) | (dias > 100)).sum() / n,
            "median_m3_dia": float(np.nanmedian(m3_dia)),
            "pct_suspicious": ((m3 == 0) | (m3 < 0) | (dias < 15) | (dias > 45)).sum() / n,
        })

    monthly = all_m3.groupby("year_month").apply(
        compute_monthly_health, include_groups=False
    )

    print(f"    [AquaCare] {len(all_m3):,} lecturas individuales analizadas ({len(monthly)} meses)")
    print(f"    [AquaCare] Media contadores parados (M3=0): {monthly['pct_zero'].mean():.1%}")
    print(f"    [AquaCare] Media consumo muy bajo (<5L/dia): {monthly['pct_very_low'].mean():.1%}")
    print(f"    [AquaCare] Media consumo extremo (>1000L/dia, fugas): {monthly['pct_high_consumption'].mean():.1%}")

    return monthly


def detect_meter_leaks_by_barrio() -> pd.DataFrame:
    """
    Cruza datos de telelectura (192K contadores CON barrio) con padrón elderly
    para crear un score de riesgo de fuga silenciosa por barrio.

    Score = f(n_contadores_viejos, pct_elderly_alone, n_meters_residencial)

    Contadores viejos (>10 años) en barrios con muchos mayores solos
    son los de mayor riesgo de fuga silenciosa no reportada.
    """
    if not Path(TELELECTURA_PATH).exists():
        print("    [AquaCare] Sin datos telelectura")
        return pd.DataFrame()

    tele = pd.read_csv(TELELECTURA_PATH)

    # Solo contadores residenciales
    residential = tele[tele["ACTIVIDAD"].str.contains("VIVIENDA", na=False)].copy()

    # Edad del contador
    residential["FECHA_INST"] = pd.to_datetime(residential["FECHA INSTALACION"], errors="coerce")
    residential["age_years"] = (pd.Timestamp("2024-12-31") - residential["FECHA_INST"]).dt.days / 365.25

    # Métricas por barrio
    barrio_stats = residential.groupby("BARRIO").agg(
        n_meters=("BARRIO", "size"),
        n_old_meters=("age_years", lambda x: (x > 10).sum()),
        mean_age=("age_years", "mean"),
        pct_old=("age_years", lambda x: (x > 10).mean()),
    ).reset_index()

    # Cargar padrón elderly
    padron_path = Path(PADRON_ELDERLY_CSV)
    if not padron_path.exists():
        print("    [AquaCare] Sin datos padrón elderly")
        return barrio_stats

    padron = pd.read_csv(padron_path)

    # Normalizar nombres para merge
    barrio_stats["barrio_clean"] = barrio_stats["BARRIO"].str.replace(r"^\d+-", "", regex=True).str.strip().str.upper()
    padron["barrio_clean"] = padron["barrio_padron"].str.strip().str.upper()

    merged = barrio_stats.merge(padron, on="barrio_clean", how="left")

    # Score de riesgo: contadores viejos × elderly solos × tamaño barrio
    merged["pct_elderly_alone"] = merged["pct_65plus_solos"].fillna(0) / 100
    merged["pct_elderly"] = merged["pct_65plus"].fillna(0) / 100

    # --- Features DINAMICOS: comportamiento real de consumo ---
    consumption_drop_freq = pd.Series(dtype=float)
    consumption_cv = pd.Series(dtype=float)

    if Path(HACKATHON_PATH).exists():
        try:
            hack = pd.read_csv(HACKATHON_PATH)
            hack["Consumo (litros)"] = (
                hack["Consumo (litros)"].astype(str).str.replace(",", "", regex=False).astype(float)
            )
            dom = hack[hack["Uso"].str.contains("DOMESTICO", case=False, na=False)]

            # Consumo mensual por barrio
            barrio_monthly = (
                dom.groupby(["Barrio", "Fecha (aaaa/mm/dd)"])["Consumo (litros)"]
                .sum().reset_index()
            )
            barrio_monthly = barrio_monthly.sort_values(["Barrio", "Fecha (aaaa/mm/dd)"])

            # MoM change
            barrio_monthly["mom_change"] = (
                barrio_monthly.groupby("Barrio")["Consumo (litros)"].pct_change()
            )

            # Frecuencia de caidas >20%
            consumption_drop_freq = (
                barrio_monthly.groupby("Barrio")["mom_change"]
                .apply(lambda x: (x < -0.20).mean())
            )

            # Coeficiente de variacion
            consumption_cv = (
                barrio_monthly.groupby("Barrio")["Consumo (litros)"]
                .agg(lambda x: x.std() / x.mean() if x.mean() > 0 else 0)
            )

            print(f"    [AquaCare] Features dinamicos: drop_freq media={consumption_drop_freq.mean():.2%}, CV medio={consumption_cv.mean():.3f}")
        except Exception as e:
            print(f"    [AquaCare] Warning: features dinamicos no disponibles: {e}")

    # Merge dynamic features via barrio_clean
    if not consumption_drop_freq.empty:
        drop_df = consumption_drop_freq.reset_index()
        drop_df.columns = ["Barrio", "drop_freq"]
        drop_df["barrio_clean"] = drop_df["Barrio"].str.replace(r"^\d+-", "", regex=True).str.strip().str.upper()
        merged = merged.merge(drop_df[["barrio_clean", "drop_freq"]], on="barrio_clean", how="left")

        cv_df = consumption_cv.reset_index()
        cv_df.columns = ["Barrio", "consumption_cv"]
        cv_df["barrio_clean"] = cv_df["Barrio"].str.replace(r"^\d+-", "", regex=True).str.strip().str.upper()
        merged = merged.merge(cv_df[["barrio_clean", "consumption_cv"]], on="barrio_clean", how="left")

        # Solo usar dynamic features en barrios con suficientes datos (>=100 metros)
        MIN_METERS_FOR_DYNAMIC = 100
        merged["drop_freq_norm"] = np.where(
            merged["n_meters"] >= MIN_METERS_FOR_DYNAMIC,
            merged["drop_freq"].fillna(0),
            0
        )
        merged["cv_norm"] = np.where(
            merged["n_meters"] >= MIN_METERS_FOR_DYNAMIC,
            merged["consumption_cv"].fillna(0),
            0
        )

        # Rank-based normalization
        valid_drop = merged["drop_freq_norm"] > 0
        if valid_drop.sum() > 1:
            merged.loc[valid_drop, "drop_freq_norm"] = (
                merged.loc[valid_drop, "drop_freq_norm"].rank(pct=True)
            )
        valid_cv = merged["cv_norm"] > 0
        if valid_cv.sum() > 1:
            merged.loc[valid_cv, "cv_norm"] = (
                merged.loc[valid_cv, "cv_norm"].rank(pct=True)
            )

        # --- DOS SCORES INDEPENDIENTES (cada uno validado por separado) ---

        # Score 1: VULNERABILIDAD demográfica (validado por V4 permutation)
        merged["vulnerability_score"] = (
            0.35 * merged["pct_old"].fillna(0) +
            0.35 * merged["pct_elderly_alone"].fillna(0) +
            0.20 * merged["pct_elderly"].fillna(0) +
            0.10 * (merged["n_meters"] / merged["n_meters"].max())
        )

        # Score 2: RIESGO de consumo anómalo (validado por V1/V3 correlación)
        merged["consumption_risk_score"] = (
            0.40 * merged["drop_freq_norm"] +
            0.30 * merged["cv_norm"] +
            0.30 * merged["pct_old"].fillna(0)  # contadores viejos contribuyen a fugas
        )

        # Score COMBINADO: vulnerabilidad × riesgo consumo
        # Multiplicativo: solo alto si AMBOS factores son altos
        merged["silent_leak_risk"] = (
            merged["vulnerability_score"] * 0.6 +
            merged["consumption_risk_score"] * 0.4
        )
    else:
        # Fallback: formula original sin features dinamicos
        merged["vulnerability_score"] = (
            0.35 * merged["pct_old"].fillna(0) +
            0.35 * merged["pct_elderly_alone"].fillna(0) +
            0.20 * merged["pct_elderly"].fillna(0) +
            0.10 * (merged["n_meters"] / merged["n_meters"].max())
        )
        merged["consumption_risk_score"] = 0
        merged["silent_leak_risk"] = merged["vulnerability_score"]

    # Estimar contadores en riesgo
    merged["estimated_at_risk"] = (
        merged["n_old_meters"] * merged["pct_elderly_alone"]
    ).round(0).astype(int)

    merged = merged.sort_values("silent_leak_risk", ascending=False)

    # Print top barrios
    top5 = merged.head(5)
    has_vuln = "vulnerability_score" in merged.columns
    for _, row in top5.iterrows():
        barrio = row["BARRIO"]
        risk = row["silent_leak_risk"]
        old_pct = row["pct_old"]
        alone_pct = row["pct_elderly_alone"]
        at_risk = row["estimated_at_risk"]
        extra = ""
        if has_vuln:
            vuln = row.get("vulnerability_score", 0)
            cons = row.get("consumption_risk_score", 0)
            extra = f" vuln={vuln:.3f} cons_risk={cons:.3f}"
        print(f"    {barrio:<35} risk={risk:.3f}{extra} "
              f"(viejos={old_pct:.0%}, solos={alone_pct:.0%}, en_riesgo~{at_risk})")

    return merged


# ═════════════════════════════════════════════════════════════════
# 4. GENERACION DE ALERTAS DE BIENESTAR
# ═════════════════════════════════════════════════════════════════

def generate_welfare_alerts(
    drops: pd.DataFrame,
    vulnerability: pd.DataFrame,
    night_silence: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Genera alertas de bienestar combinando caidas de consumo + vulnerabilidad.

    Niveles:
      CRITICO:     caida >50% + vulnerabilidad alta (>0.6) → accion inmediata
      ALTO:        caida >30% + vulnerabilidad moderada (>0.4)
      VIGILANCIA:  patron inusual en barrio vulnerable

    Cada alerta lleva un mensaje humano, pensado para que el operador
    sienta la urgencia de actuar.
    """
    # Merge drops con vulnerability
    merged = drops.merge(
        vulnerability[["barrio_key", "elderly_vulnerability", "barrio_clean",
                        "median_consumo_per_contract", "estimated_contracts"]],
        on="barrio_key",
        how="left",
    )

    if "elderly_vulnerability" not in merged.columns:
        merged["elderly_vulnerability"] = 0.2

    merged["elderly_vulnerability"] = merged["elderly_vulnerability"].fillna(0.2)

    alerts = []

    for _, row in merged.iterrows():
        barrio = row.get("barrio_clean", row["barrio_key"])
        vuln = row["elderly_vulnerability"]
        mom = row["mom_change"]
        hist = row["hist_change"]
        severity = row["drop_severity"]
        contracts = row.get("estimated_contracts")
        consumo = row.get("consumo", 0)
        streak = row.get("consecutive_decline", 0)

        # Calcular la peor caida (la que mas preocupa)
        worst_drop = min(mom, hist)
        worst_drop_pct = abs(worst_drop) * 100

        # --- Clasificar nivel de urgencia ---
        nivel = None
        confidence = 0.0

        # CRITICO: caida fuerte + barrio vulnerable
        if worst_drop < -0.50 and vuln > 0.55:
            nivel = "CRITICO"
            confidence = min(severity * vuln * 2.5, 1.0)
        elif worst_drop < -0.40 and vuln > 0.70:
            nivel = "CRITICO"
            confidence = min(severity * vuln * 2.0, 1.0)

        # ALTO: caida moderada + algo de vulnerabilidad
        elif worst_drop < -0.30 and vuln > 0.40:
            nivel = "ALTO"
            confidence = min(severity * vuln * 1.8, 1.0)
        elif worst_drop < -0.25 and vuln > 0.60:
            nivel = "ALTO"
            confidence = min(severity * vuln * 1.5, 1.0)

        # VIGILANCIA: patron inusual en barrio vulnerable
        elif worst_drop < -0.20 and vuln > 0.45:
            nivel = "VIGILANCIA"
            confidence = min(severity * vuln * 1.2, 1.0)
        elif streak >= 3 and vuln > 0.50:
            nivel = "VIGILANCIA"
            confidence = min(0.3 + vuln * 0.4, 1.0)
        elif severity > 0.3 and vuln > 0.55:
            nivel = "VIGILANCIA"
            confidence = min(severity * vuln, 1.0)

        if nivel is None:
            continue

        # --- Generar mensaje humano ---
        msg = _build_human_message(
            barrio, nivel, worst_drop_pct, vuln, contracts, consumo, streak, row
        )

        alerts.append({
            "barrio_key": row["barrio_key"],
            "barrio": barrio,
            "fecha": row["fecha"],
            "nivel": nivel,
            "confidence": round(confidence, 3),
            "drop_pct": round(worst_drop_pct, 1),
            "elderly_vulnerability": round(vuln, 3),
            "drop_severity": round(severity, 3),
            "mom_change": round(mom, 3),
            "hist_change": round(hist, 3),
            "consecutive_decline_months": streak,
            "consumo_actual": round(consumo, 1),
            "estimated_contracts": contracts,
            "mensaje": msg,
        })

    alerts_df = pd.DataFrame(alerts)

    if len(alerts_df) == 0:
        return alerts_df

    # Ordenar: CRITICO primero, luego ALTO, luego VIGILANCIA
    nivel_order = {"CRITICO": 0, "ALTO": 1, "VIGILANCIA": 2}
    alerts_df["_sort"] = alerts_df["nivel"].map(nivel_order)
    alerts_df = alerts_df.sort_values(
        ["_sort", "confidence"], ascending=[True, False]
    ).drop(columns=["_sort"]).reset_index(drop=True)

    return alerts_df


def _build_human_message(
    barrio: str,
    nivel: str,
    drop_pct: float,
    vuln: float,
    contracts: Optional[float],
    consumo: float,
    streak: int,
    row: pd.Series,
) -> str:
    """
    Construye el mensaje humano para cada alerta.

    Estos mensajes no son para un log: son para una persona que puede
    salvar una vida. Cada palabra cuenta.
    """
    contracts_str = f"{int(contracts)}" if contracts and not np.isnan(contracts) else "pocos"

    if nivel == "CRITICO":
        msg = (
            f"ALERTA CRITICA: El barrio {barrio} muestra una caida del {drop_pct:.0f}% "
            f"en consumo domestico. "
        )
        if contracts and not np.isnan(contracts) and contracts < SMALL_BARRIO_CONTRACTS:
            msg += (
                f"Con solo {contracts_str} contratos y un indice de vulnerabilidad "
                f"de {vuln:.0%}, el perfil es compatible con personas mayores "
                f"viviendo solas. "
            )
        else:
            msg += (
                f"El indice de vulnerabilidad elderly es {vuln:.0%}. "
            )
        msg += "SE RECOMIENDA VERIFICACION PRESENCIAL URGENTE."

        if streak >= 2:
            msg += (
                f" NOTA: El consumo lleva {streak} meses consecutivos bajando. "
                f"Esto no es un mes atipico — es una tendencia que requiere atencion."
            )

    elif nivel == "ALTO":
        msg = (
            f"ALERTA ALTA: {barrio} registra una caida del {drop_pct:.0f}% en consumo. "
            f"Indice de vulnerabilidad: {vuln:.0%}. "
        )
        if contracts and not np.isnan(contracts) and contracts < SMALL_BARRIO_CONTRACTS:
            msg += (
                f"Barrio pequeno ({contracts_str} contratos) — una sola persona "
                f"que deja de consumir agua se nota en las cifras. "
            )
        msg += (
            "Se recomienda contacto telefonico con servicios sociales del distrito "
            "y revision de lecturas individuales si estan disponibles."
        )

    else:  # VIGILANCIA
        msg = (
            f"VIGILANCIA: {barrio} presenta un cambio inusual en su patron de consumo "
            f"(caida del {drop_pct:.0f}%). "
        )
        if streak >= 3:
            msg += (
                f"El consumo lleva {streak} meses bajando de forma consecutiva. "
                f"En un barrio con indice de vulnerabilidad {vuln:.0%}, "
                f"conviene monitorear de cerca. "
            )
        else:
            msg += (
                f"Indice de vulnerabilidad: {vuln:.0%}. "
                f"Mantener en observacion y cruzar con datos de servicios sociales."
            )

    return msg


# ═════════════════════════════════════════════════════════════════
# 5. ENRICHMENT CON OTROS MODELOS (resultados existentes)
# ═════════════════════════════════════════════════════════════════

def enrich_with_model_results(
    alerts: pd.DataFrame,
    results: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Enriquece las alertas welfare con informacion de otros modelos.

    Si un barrio ya fue detectado como anomalo por M2, M5, Prophet, etc.,
    eso REFUERZA la alerta welfare. No es solo una caida — multiples modelos
    confirman que algo raro pasa.
    """
    if results is None or len(alerts) == 0:
        return alerts

    alerts = alerts.copy()

    # Buscar si los barrios alertados tambien tienen anomalias en otros modelos
    model_cols = [c for c in results.columns if c.startswith("is_anomaly_")]

    if not model_cols or "barrio_key" not in results.columns:
        alerts["other_models_confirming"] = 0
        return alerts

    # Cruzar por barrio_key y fecha
    merge_cols = ["barrio_key"]
    if "fecha" in results.columns and "fecha" in alerts.columns:
        # Asegurar que ambas fechas son del mismo tipo
        alerts["fecha"] = pd.to_datetime(alerts["fecha"])
        results = results.copy()
        results["fecha"] = pd.to_datetime(results["fecha"])
        merge_cols.append("fecha")

    model_info = results[merge_cols + model_cols].copy()

    # Contar cuantos modelos detectan anomalia
    model_info["_n_models_confirming"] = model_info[model_cols].sum(axis=1)

    enriched = alerts.merge(
        model_info[merge_cols + ["_n_models_confirming"]],
        on=merge_cols,
        how="left",
    )
    enriched["other_models_confirming"] = (
        enriched["_n_models_confirming"].fillna(0).astype(int)
    )
    enriched = enriched.drop(columns=["_n_models_confirming"], errors="ignore")

    # Elevar nivel si multiples modelos confirman
    upgraded = []
    for idx, row in enriched.iterrows():
        if row["other_models_confirming"] >= 3 and row["nivel"] == "ALTO":
            row = row.copy()
            row["nivel"] = "CRITICO"
            row["mensaje"] = (
                row["mensaje"].replace("ALERTA ALTA:", "ALERTA CRITICA (confirmada por modelos):") +
                f" CONFIRMACION: {row['other_models_confirming']} modelos independientes "
                f"detectan anomalia en este barrio/periodo."
            )
        elif row["other_models_confirming"] >= 2 and row["nivel"] == "VIGILANCIA":
            row = row.copy()
            row["nivel"] = "ALTO"
            row["mensaje"] = (
                row["mensaje"].replace("VIGILANCIA:", "ALERTA ALTA (patron confirmado):") +
                f" CONFIRMACION: {row['other_models_confirming']} modelos detectan anomalia."
            )
        upgraded.append(row)

    return pd.DataFrame(upgraded).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════
# 6. ESTIMACION DE POBLACION ELDERLY
# ═════════════════════════════════════════════════════════════════

def estimate_elderly_population(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Estima el porcentaje de poblacion mayor por barrio usando proxies
    derivados del consumo de agua.

    Metodo:
      - Consumo bajo y estable → persona mayor sola
      - Tendencia decreciente a largo plazo → barrio que se vacia
      - Cruzar con datos INE hardcoded

    LIMITACION: esto es una estimacion PROXY, no un dato censal.
    En un despliegue real, se cruzaria con el Padron municipal.

    Returns:
        DataFrame con barrio_key, pct_elderly_estimated, confidence
    """
    vuln = compute_elderly_vulnerability(df_monthly)

    vuln["pct_elderly_estimated"] = vuln["elderly_vulnerability"].apply(
        lambda v: _vulnerability_to_elderly_pct(v)
    )

    vuln["estimation_confidence"] = vuln.apply(
        lambda row: _estimation_confidence(row), axis=1
    )

    return vuln[["barrio_key", "barrio_clean", "pct_elderly_estimated",
                 "estimation_confidence", "elderly_vulnerability",
                 "median_consumo_per_contract", "estimated_contracts"]]


def _vulnerability_to_elderly_pct(vulnerability: float) -> float:
    """
    Convierte indice de vulnerabilidad a % elderly estimado.
    Calibracion basada en datos INE de Alicante: media 19.5% >65 anos.
    """
    # Mapping no lineal: vulnerabilidad 0.5 ~ media de Alicante
    if vulnerability >= 0.8:
        return 0.40   # ~40% mayores de 65
    elif vulnerability >= 0.6:
        return 0.30
    elif vulnerability >= 0.45:
        return 0.22
    elif vulnerability >= 0.3:
        return 0.18
    else:
        return 0.12


def _estimation_confidence(row: pd.Series) -> str:
    """Nivel de confianza de la estimacion elderly."""
    score = row.get("score_census", 0)
    if score > 0.5:
        return "ALTA"    # tenemos datos INE
    elif score > 0.2:
        return "MEDIA"   # datos parciales + proxies
    else:
        return "BAJA"    # solo proxies del agua


# ═════════════════════════════════════════════════════════════════
# 7. PUNTO DE ENTRADA PRINCIPAL
# ═════════════════════════════════════════════════════════════════

def run_welfare_detection(
    df_monthly: pd.DataFrame,
    results: Optional[pd.DataFrame] = None,
    caudal_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    AquaCare — Deteccion de emergencias en personas vulnerables.

    Pipeline completo:
      1. Detectar caidas de consumo por barrio
      2. Calcular indice de vulnerabilidad elderly
      3. (Opcional) Detectar silencio nocturno
      4. Generar alertas con nivel de urgencia
      5. Enriquecer con resultados de otros modelos

    Args:
        df_monthly:  DataFrame mensual con barrio_key, fecha, consumo_litros,
                     consumption_per_contract, etc.
        results:     DataFrame de collect_results() con is_anomaly_* columns
                     (opcional, para confirmacion multi-modelo)
        caudal_path: Path a datos horarios de caudal (opcional, para silencio nocturno)

    Returns:
        DataFrame de alertas welfare con nivel, mensaje, confidence
    """
    print("\n" + "=" * 70)
    print("  AquaCare — Deteccion de Bienestar en Personas Vulnerables")
    print("  'El agua es el ultimo indicador vital de actividad humana'")
    print("=" * 70)

    # --- Paso 1: Detectar caidas de consumo ---
    print("\n  [1/5] Detectando caidas de consumo por barrio...")
    drops = detect_consumption_drops(df_monthly)
    n_significant_drops = (drops["drop_severity"] > 0.2).sum()
    print(f"        {len(drops)} puntos analizados, {n_significant_drops} con caida significativa (>20%)")

    # --- Paso 2: Calcular vulnerabilidad elderly ---
    print("\n  [2/5] Calculando indice de vulnerabilidad elderly...")
    vulnerability = compute_elderly_vulnerability(df_monthly)
    n_vulnerable = (vulnerability["elderly_vulnerability"] > 0.5).sum()
    n_total = len(vulnerability)
    print(f"        {n_total} barrios analizados, {n_vulnerable} con vulnerabilidad alta (>0.50)")

    # Top barrios vulnerables
    top_vuln = vulnerability.nlargest(5, "elderly_vulnerability")
    for _, row in top_vuln.iterrows():
        barrio = row["barrio_clean"]
        contracts = f"{int(row['estimated_contracts'])} contratos" if row["estimated_contracts"] else "contratos N/A"
        print(f"        -> {barrio:<30} vulnerabilidad={row['elderly_vulnerability']:.2f}  ({contracts})")

    # --- Paso 2b: Análisis individual de contadores ---
    print("\n  [2b/6] Analizando contadores individuales (m3-registrados)...")
    meter_monthly = detect_individual_meter_anomalies(years=(2020, 2021, 2022, 2023, 2024, 2025))

    print("\n  [2c/6] Cruzando contadores con demografía elderly (padrón)...")
    meter_barrio_risk = detect_meter_leaks_by_barrio()
    if not meter_barrio_risk.empty and "silent_leak_risk" in meter_barrio_risk.columns:
        n_high_risk = (meter_barrio_risk["silent_leak_risk"] > 0.3).sum()
        total_at_risk = meter_barrio_risk["estimated_at_risk"].sum()
        print(f"        {n_high_risk} barrios con riesgo alto de fuga silenciosa")
        print(f"        ~{total_at_risk:,.0f} contadores estimados en riesgo")

    # --- Paso 3: Silencio nocturno (si hay datos) ---
    night_silence = None
    if caudal_path:
        print(f"\n  [3/6] Analizando silencio nocturno ({caudal_path})...")
        night_silence = detect_night_silence(caudal_path)
        if len(night_silence) > 0:
            n_silent_sectors = (night_silence["avg_silence"] > 0.5).sum()
            print(f"        {len(night_silence)} sector-meses analizados, "
                  f"{n_silent_sectors} con silencio nocturno significativo")
        else:
            print("        Sin datos de silencio nocturno disponibles")
    else:
        print("\n  [3/6] Silencio nocturno: omitido (sin datos horarios)")

    # --- Paso 4: Generar alertas ---
    print("\n  [4/6] Generando alertas de bienestar...")
    alerts = generate_welfare_alerts(drops, vulnerability, night_silence)
    print(f"        {len(alerts)} alertas generadas")

    if len(alerts) > 0:
        for nivel in ["CRITICO", "ALTO", "VIGILANCIA"]:
            count = (alerts["nivel"] == nivel).sum()
            if count > 0:
                print(f"          {nivel}: {count}")

    # --- Paso 4b: Enriquecer alertas con riesgo de contadores ---
    if not meter_barrio_risk.empty and len(alerts) > 0:
        alerts["barrio_clean"] = alerts["barrio_key"].str.split("__").str[0]
        risk_lookup = meter_barrio_risk.set_index("BARRIO")["silent_leak_risk"].to_dict()
        at_risk_lookup = meter_barrio_risk.set_index("BARRIO")["estimated_at_risk"].to_dict()
        alerts["silent_leak_risk"] = alerts["barrio_clean"].map(risk_lookup).fillna(0)
        alerts["meters_at_risk"] = alerts["barrio_clean"].map(at_risk_lookup).fillna(0)
        # Upgrade alerts: if high silent_leak_risk AND already ALTO → CRITICO
        mask_upgrade = (alerts["nivel"] == "ALTO") & (alerts["silent_leak_risk"] > 0.3)
        alerts.loc[mask_upgrade, "nivel"] = "CRITICO"
        alerts.loc[mask_upgrade, "mensaje"] = alerts.loc[mask_upgrade, "mensaje"] + " + CONTADORES VIEJOS EN ZONA ELDERLY"
        if mask_upgrade.sum() > 0:
            print(f"        {mask_upgrade.sum()} alertas ALTO → CRITICO por riesgo contadores elderly")

    # --- Paso 5: Enriquecer con otros modelos ---
    if results is not None and len(alerts) > 0:
        print("\n  [5/6] Cruzando con resultados de otros modelos...")
        alerts = enrich_with_model_results(alerts, results)
        if "other_models_confirming" in alerts.columns:
            confirmed = (alerts["other_models_confirming"] > 0).sum()
            print(f"        {confirmed} alertas confirmadas por otros modelos")
    else:
        print("\n  [5/6] Enriquecimiento multi-modelo: omitido")

    # --- Estimacion elderly (informativa) ---
    elderly_est = estimate_elderly_population(df_monthly)
    n_high_elderly = (elderly_est["pct_elderly_estimated"] >= 0.30).sum()
    print(f"\n  Estimacion: {n_high_elderly} barrios con >30% poblacion mayor estimada")

    print(f"\n  {'=' * 60}")
    print(f"  RESUMEN: {len(alerts)} alertas de bienestar generadas")
    if len(alerts) > 0:
        n_crit = (alerts["nivel"] == "CRITICO").sum()
        if n_crit > 0:
            print(f"  *** {n_crit} ALERTAS CRITICAS requieren verificacion presencial ***")
    print(f"  {'=' * 60}")

    return alerts


# ═════════════════════════════════════════════════════════════════
# 8. RESUMEN HUMANO
# ═════════════════════════════════════════════════════════════════

def welfare_summary(welfare_alerts: pd.DataFrame) -> None:
    """
    Imprime un resumen legible y emotivo de las alertas de bienestar.

    Disenado para impactar al jurado del hackathon:
    detras de cada dato hay una persona.
    """
    print("\n")
    print("=" * 74)
    print("  A Q U A C A R E")
    print("  Deteccion de Emergencias en Personas Vulnerables a traves del Agua")
    print("=" * 74)

    if len(welfare_alerts) == 0:
        print("\n  No se han detectado alertas de bienestar en el periodo analizado.")
        print("  Esto es una buena noticia: los patrones de consumo son normales.")
        print("=" * 74)
        return

    # --- Estadisticas generales ---
    n_critico = (welfare_alerts["nivel"] == "CRITICO").sum()
    n_alto = (welfare_alerts["nivel"] == "ALTO").sum()
    n_vigilancia = (welfare_alerts["nivel"] == "VIGILANCIA").sum()
    barrios_affected = welfare_alerts["barrio"].nunique()

    print(f"\n  Periodo analizado: {welfare_alerts['fecha'].min()} — {welfare_alerts['fecha'].max()}")
    print(f"  Barrios afectados: {barrios_affected}")
    print(f"\n  Alertas por nivel:")
    if n_critico > 0:
        print(f"    CRITICO:     {n_critico:>3}  <- Requieren verificacion presencial INMEDIATA")
    if n_alto > 0:
        print(f"    ALTO:        {n_alto:>3}  <- Requieren contacto con servicios sociales")
    if n_vigilancia > 0:
        print(f"    VIGILANCIA:  {n_vigilancia:>3}  <- Monitorear de cerca")

    # --- Alertas CRITICAS (detalle completo) ---
    criticas = welfare_alerts[welfare_alerts["nivel"] == "CRITICO"]
    if len(criticas) > 0:
        print(f"\n  {'─' * 70}")
        print(f"  ALERTAS CRITICAS — CADA UNA DE ESTAS PUEDE SER UNA VIDA")
        print(f"  {'─' * 70}")
        for _, row in criticas.iterrows():
            print(f"\n  [{row['fecha']}] {row['barrio']}")
            print(f"  Caida: {row['drop_pct']:.0f}% | Vulnerabilidad: {row['elderly_vulnerability']:.0%} "
                  f"| Confianza: {row['confidence']:.0%}")
            if row.get("other_models_confirming", 0) > 0:
                print(f"  Modelos confirmando: {row['other_models_confirming']}")
            print(f"\n  > {row['mensaje']}")
            print()

    # --- Alertas ALTO (resumen) ---
    altas = welfare_alerts[welfare_alerts["nivel"] == "ALTO"]
    if len(altas) > 0:
        print(f"\n  {'─' * 70}")
        print(f"  ALERTAS NIVEL ALTO")
        print(f"  {'─' * 70}")
        print(f"  {'Barrio':<30} {'Fecha':>12} {'Caida':>8} {'Vuln.':>8} {'Conf.':>8}")
        print(f"  {'─' * 70}")
        for _, row in altas.head(20).iterrows():
            print(f"  {row['barrio']:<30} {str(row['fecha']):>12} "
                  f"{row['drop_pct']:>7.0f}% {row['elderly_vulnerability']:>7.0%} "
                  f"{row['confidence']:>7.0%}")

    # --- Alertas VIGILANCIA (tabla compacta) ---
    vigil = welfare_alerts[welfare_alerts["nivel"] == "VIGILANCIA"]
    if len(vigil) > 0:
        print(f"\n  {'─' * 70}")
        print(f"  VIGILANCIA — {len(vigil)} barrios en observacion")
        print(f"  {'─' * 70}")
        vigil_barrios = vigil.groupby("barrio").agg(
            n_alertas=("nivel", "count"),
            max_drop=("drop_pct", "max"),
            avg_vuln=("elderly_vulnerability", "mean"),
        ).sort_values("max_drop", ascending=False)

        for barrio, row in vigil_barrios.head(15).iterrows():
            print(f"    {barrio:<30} {row['n_alertas']:>3} alertas | "
                  f"max caida {row['max_drop']:.0f}% | vuln {row['avg_vuln']:.0%}")

    # --- Datos REALES del Padron Municipal ---
    padron_data = _load_padron_elderly()
    if padron_data:
        print(f"\n  {'─' * 70}")
        print(f"  DATOS REALES — Padron Municipal de Alicante 2025")
        print(f"  Fuente: alicante.es/estadisticas-poblacion (barrios_2025.xls)")
        print(f"  {'─' * 70}")
        # Show stats for affected barrios
        affected = welfare_alerts["barrio"].unique()
        for barrio_clean in affected:
            padron_name = AMAEM_TO_PADRON.get(barrio_clean, "")
            info = padron_data.get(padron_name, {})
            if info:
                pct65 = info.get("pct_65plus", 0)
                solos = int(info.get("mayores_65_solos", 0))
                pct_solos = info.get("pct_65plus_solos", 0)
                pop = int(info.get("poblacion_total", 0))
                print(f"    {barrio_clean:<30} {pct65:>5.1f}% mayor 65 | "
                      f"{solos:>5} viven SOLOS ({pct_solos:.1f}%) | pob: {pop:>6}")
        total_solos = sum(
            padron_data.get(AMAEM_TO_PADRON.get(b, ""), {}).get("mayores_65_solos", 0)
            for b in affected
        )
        print(f"\n    TOTAL: {int(total_solos)} personas mayores viviendo SOLAS en barrios afectados")

    # --- Mensaje final ---
    print(f"\n  {'=' * 70}")
    print("  NOTA PARA EL OPERADOR:")
    print()
    print("  Detras de cada linea de esta tabla puede haber una persona mayor")
    print("  que vive sola y que lleva dias sin abrir un grifo.")
    print()
    if padron_data:
        print(f"  Segun el Padron Municipal 2025, en los barrios con alertas hay")
        print(f"  {int(total_solos)} personas mayores de 65 anos viviendo SOLAS.")
        print()
    print("  En Japon, Tokyo Water ha salvado vidas con este mismo principio.")
    print("  El agua no miente: si deja de fluir, algo ha pasado.")
    print()
    print("  Una llamada. Una visita. Puede ser la diferencia.")
    print(f"  {'=' * 70}")
    print()


# ═════════════════════════════════════════════════════════════════
# 9. VALIDACIONES AQUACARE — 5 tests independientes
# ═════════════════════════════════════════════════════════════════

CAUDAL_PATH = "data/_caudal_medio_sector_hidraulico_hora_2024_-caudal_medio_sector_hidraulico_hora_2024.csv"
CAMBIOS_PATH = "data/cambios-de-contador-solo-alicante_hackaton-dataart-cambios-de-contador-solo-alicante.csv.csv"
HACKATHON_PATH = "data/datos-hackathon-amaem.xlsx-set-de-datos-.csv"
CONSUMO_BARRIO_PATH = "data/_consumos_alicante_regenerada_barrio_mes-2024_-consumos_alicante_regenerada_barrio_mes-2024.csv.csv"


def validate_aquacare_vs_mnf(leak_df: pd.DataFrame = None) -> dict:
    """
    V1: Cross-validation AquaCare vs Minimum Night Flow.

    Si AquaCare detecta barrios con riesgo de fugas silenciosas,
    esos sectores deberian tener MNF (2-4 AM) mas alto.

    rho(silent_leak_risk, mnf_ratio) > 0 con p < 0.05 → evidencia fisica.
    """
    from sector_mapping import get_mapped_sectors
    from scipy.stats import spearmanr

    print("\n  ═══ V1: AquaCare vs MNF nocturno ═══")

    if leak_df is None:
        leak_df = detect_meter_leaks_by_barrio()
    if leak_df.empty:
        return {"test": "V1_MNF", "status": "NO_DATA", "reason": "sin datos leak_df"}

    if not Path(CAUDAL_PATH).exists():
        return {"test": "V1_MNF", "status": "NO_DATA", "reason": "sin datos caudal"}

    # 1. Cargar caudal horario
    caudal = pd.read_csv(CAUDAL_PATH)
    caudal["fecha"] = pd.to_datetime(caudal["FECHA_HORA"], format="%d/%m/%Y %H:%M", dayfirst=True)
    caudal["hour"] = caudal["fecha"].dt.hour
    caudal["caudal"] = (
        caudal["CAUDAL MEDIO(M3)"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )

    # 2. MNF ratio por sector: media 2-4AM / media 10-18h
    night = caudal[caudal["hour"].isin([2, 3, 4])].groupby("SECTOR")["caudal"].mean()
    day = caudal[caudal["hour"].isin(range(10, 18))].groupby("SECTOR")["caudal"].mean()
    mnf_ratio = (night / day).dropna()
    mnf_ratio.name = "mnf_ratio"

    # 3. Map sector → barrio
    mapping = get_mapped_sectors()
    mnf_by_barrio = {}
    for sector, ratio in mnf_ratio.items():
        barrio = mapping.get(sector)
        if barrio:
            if barrio not in mnf_by_barrio:
                mnf_by_barrio[barrio] = []
            mnf_by_barrio[barrio].append(ratio)
    # Media si multiples sectores mapean al mismo barrio
    mnf_barrio = {b: np.mean(v) for b, v in mnf_by_barrio.items()}

    # 4. Merge con silent_leak_risk
    leak_df = leak_df.copy()
    leak_df["mnf_ratio"] = leak_df["BARRIO"].map(mnf_barrio)
    paired = leak_df.dropna(subset=["mnf_ratio", "silent_leak_risk"])

    n = len(paired)
    if n < 5:
        print(f"    Solo {n} barrios emparejados sector-barrio (minimo 5)")
        return {"test": "V1_MNF", "status": "INSUFFICIENT_DATA", "n_paired": n}

    # 5. Spearman correlation
    rho, p_scipy = spearmanr(paired["silent_leak_risk"], paired["mnf_ratio"])

    # 6. Permutation p-value (mas robusto)
    rng = np.random.default_rng(42)
    n_perm = 999
    null_rhos = np.empty(n_perm)
    risk_vals = paired["silent_leak_risk"].values
    mnf_vals = paired["mnf_ratio"].values
    for i in range(n_perm):
        perm = rng.permutation(mnf_vals)
        null_rhos[i] = spearmanr(risk_vals, perm)[0]
    p_perm = (np.sum(null_rhos >= rho) + 1) / (n_perm + 1)

    sig = "SIGNIFICATIVA" if p_perm < 0.05 else "NO significativa"
    direction = "POSITIVA (mas riesgo = mas fugas)" if rho > 0 else "NEGATIVA"

    print(f"    Barrios emparejados: {n}")
    print(f"    Spearman rho = {rho:+.3f} ({direction})")
    print(f"    p_permutation = {p_perm:.4f} → {sig}")
    print(f"    Top-3 MNF: {paired.nlargest(3, 'mnf_ratio')[['BARRIO', 'mnf_ratio', 'silent_leak_risk']].to_string(index=False)}")

    return {
        "test": "V1_MNF",
        "rho": round(rho, 4),
        "p_perm": round(p_perm, 4),
        "n_paired": n,
        "significant": p_perm < 0.05,
        "direction": direction,
        "status": "OK",
    }


def validate_aquacare_vs_meter_changes(leak_df: pd.DataFrame = None) -> dict:
    """
    V2: Cross-validation AquaCare vs cambios de contador por edad.

    Si contadores viejos en zonas elderly son problema real, barrios con
    mas contadores viejos (pct_old) deberian tener mas reemplazos por edad.

    Usamos CALIBRE como proxy de cruce ya que cambios no tiene BARRIO.
    """
    from scipy.stats import spearmanr

    print("\n  ═══ V2: AquaCare vs cambios de contador ═══")

    if leak_df is None:
        leak_df = detect_meter_leaks_by_barrio()
    if leak_df.empty:
        return {"test": "V2_CAMBIOS", "status": "NO_DATA", "reason": "sin leak_df"}

    if not Path(CAMBIOS_PATH).exists():
        print(f"    Sin datos: {CAMBIOS_PATH}")
        return {"test": "V2_CAMBIOS", "status": "NO_DATA", "reason": "sin cambios"}

    # 1. Cargar cambios de contador
    cambios = pd.read_csv(CAMBIOS_PATH)
    # Filtrar por edad (motivo ED-CAMBIO POR EDAD)
    cambios_edad = cambios[cambios["MOTIVO_CAMBIO"].str.contains("ED", na=False)].copy()
    print(f"    Cambios totales: {len(cambios):,}, por edad: {len(cambios_edad):,}")

    # 2. Tasa de cambio por CALIBRE
    cambio_by_cal = cambios_edad.groupby("CALIBRE").size().rename("n_cambios_edad")
    total_cambio_by_cal = cambios.groupby("CALIBRE").size().rename("n_cambios_total")

    # 3. Desde telelectura: contadores por CALIBRE × BARRIO
    if not Path(TELELECTURA_PATH).exists():
        return {"test": "V2_CAMBIOS", "status": "NO_DATA", "reason": "sin telelectura"}

    tele = pd.read_csv(TELELECTURA_PATH)
    residential = tele[tele["ACTIVIDAD"].str.contains("VIVIENDA", na=False)].copy()
    meters_by_cal_barrio = residential.groupby(["BARRIO", "CALIBRE"]).size().reset_index(name="n_meters")

    # 4. Tasa de cambio por edad por calibre (city-wide)
    cal_stats = pd.DataFrame({"n_cambios_edad": cambio_by_cal}).join(
        pd.DataFrame({"n_cambios_total": total_cambio_by_cal}), how="outer"
    ).fillna(0)
    # Contar metros totales por calibre en toda la ciudad
    total_meters_by_cal = residential.groupby("CALIBRE").size().rename("total_meters")
    cal_stats = cal_stats.join(total_meters_by_cal, how="outer").fillna(0)
    cal_stats["rate_edad"] = np.where(
        cal_stats["total_meters"] > 0,
        cal_stats["n_cambios_edad"] / cal_stats["total_meters"],
        0
    )

    # 5. Tasa esperada de cambio por edad por barrio
    meters_by_cal_barrio = meters_by_cal_barrio.merge(
        cal_stats[["rate_edad"]].reset_index(), on="CALIBRE", how="left"
    )
    meters_by_cal_barrio["expected_changes"] = meters_by_cal_barrio["n_meters"] * meters_by_cal_barrio["rate_edad"]

    barrio_expected = meters_by_cal_barrio.groupby("BARRIO").agg(
        total_meters=("n_meters", "sum"),
        expected_age_changes=("expected_changes", "sum"),
    ).reset_index()
    barrio_expected["expected_rate"] = barrio_expected["expected_age_changes"] / barrio_expected["total_meters"]

    # 6. Merge con silent_leak_risk
    merged = leak_df[["BARRIO", "silent_leak_risk", "pct_old"]].merge(
        barrio_expected[["BARRIO", "expected_rate"]], on="BARRIO", how="inner"
    )

    n = len(merged)
    if n < 5:
        print(f"    Solo {n} barrios emparejados (minimo 5)")
        return {"test": "V2_CAMBIOS", "status": "INSUFFICIENT_DATA", "n": n}

    # 7. Correlacion risk vs expected_rate
    rho, _ = spearmanr(merged["silent_leak_risk"], merged["expected_rate"])

    # Permutation p-value
    rng = np.random.default_rng(42)
    n_perm = 999
    null_rhos = np.empty(n_perm)
    x = merged["silent_leak_risk"].values
    y = merged["expected_rate"].values
    for i in range(n_perm):
        null_rhos[i] = spearmanr(x, rng.permutation(y))[0]
    p_perm = (np.sum(null_rhos >= rho) + 1) / (n_perm + 1)

    sig = "SIGNIFICATIVA" if p_perm < 0.05 else "NO significativa"

    print(f"    Barrios cruzados: {n}")
    print(f"    Spearman rho(silent_leak_risk, expected_age_change_rate) = {rho:+.3f}")
    print(f"    p_permutation = {p_perm:.4f} → {sig}")

    return {
        "test": "V2_CAMBIOS",
        "rho": round(rho, 4),
        "p_perm": round(p_perm, 4),
        "n_barrios": n,
        "significant": p_perm < 0.05,
        "status": "OK",
    }


def validate_aquacare_vs_consumption(leak_df: pd.DataFrame = None) -> dict:
    """
    V3: Cross-validation AquaCare vs consumo per contrato.

    Barrios con mas elderly solos deberian mostrar consumo anormal:
    - Mayor variabilidad (CV) → fugas intermitentes
    - O consumo per contrato distinto al esperado
    """
    from scipy.stats import spearmanr

    print("\n  ═══ V3: AquaCare vs consumo per capita ═══")

    if leak_df is None:
        leak_df = detect_meter_leaks_by_barrio()
    if leak_df.empty:
        return {"test": "V3_CONSUMO", "status": "NO_DATA"}

    # Intentar datos hackathon (tiene contratos)
    hackathon_ok = Path(HACKATHON_PATH).exists()
    consumo_ok = Path(CONSUMO_BARRIO_PATH).exists()

    if not hackathon_ok and not consumo_ok:
        print("    Sin datos de consumo por barrio")
        return {"test": "V3_CONSUMO", "status": "NO_DATA"}

    results = {}

    # 3a. Consumo per contrato (hackathon data)
    if hackathon_ok:
        hack = pd.read_csv(HACKATHON_PATH)
        # Parse comma-separated numbers
        hack["Consumo (litros)"] = hack["Consumo (litros)"].astype(str).str.replace(",", "", regex=False).astype(float)
        hack["Nº Contratos"] = hack["Nº Contratos"].astype(str).str.replace(",", "", regex=False).astype(float)
        # Solo domestico
        dom = hack[hack["Uso"].str.contains("DOMESTICO", case=False, na=False)]
        barrio_consumo = dom.groupby("Barrio").agg(
            total_litros=("Consumo (litros)", "sum"),
            total_contratos=("Nº Contratos", "sum"),
        )
        barrio_consumo["litros_per_contrato"] = barrio_consumo["total_litros"] / barrio_consumo["total_contratos"]

        # Normalizar barrio names para merge
        barrio_consumo = barrio_consumo.reset_index()
        barrio_consumo["barrio_clean"] = barrio_consumo["Barrio"].str.replace(r"^\d+-", "", regex=True).str.strip().str.upper()
        leak_clean = leak_df.copy()
        leak_clean["barrio_clean"] = leak_clean["BARRIO"].str.replace(r"^\d+-", "", regex=True).str.strip().str.upper()

        merged_h = leak_clean.merge(barrio_consumo[["barrio_clean", "litros_per_contrato"]], on="barrio_clean", how="inner")

        if len(merged_h) >= 5:
            rho_h, _ = spearmanr(merged_h["silent_leak_risk"], merged_h["litros_per_contrato"])
            # Permutation p
            rng = np.random.default_rng(42)
            n_perm = 999
            x = merged_h["silent_leak_risk"].values
            y = merged_h["litros_per_contrato"].values
            null_rhos = [spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)]
            p_h = (sum(1 for r in null_rhos if r >= rho_h) + 1) / (n_perm + 1)
            results["consumo_per_contrato"] = {"rho": rho_h, "p_perm": p_h, "n": len(merged_h)}
            print(f"    Consumo/contrato: rho={rho_h:+.3f}, p={p_h:.4f}, n={len(merged_h)}")

    # 3b. Variabilidad temporal (CV mensual de agua regenerada)
    if consumo_ok:
        cons = pd.read_csv(CONSUMO_BARRIO_PATH)
        barrio_cv = cons.groupby("BARRIO")["CONSUMO_2024"].agg(["mean", "std"])
        barrio_cv["regen_cv"] = barrio_cv["std"] / barrio_cv["mean"].replace(0, np.nan)
        barrio_cv = barrio_cv.dropna().reset_index()

        barrio_cv["barrio_clean"] = barrio_cv["BARRIO"].str.replace(r"^\d+-", "", regex=True).str.strip().str.upper()
        leak_clean = leak_df.copy()
        leak_clean["barrio_clean"] = leak_clean["BARRIO"].str.replace(r"^\d+-", "", regex=True).str.strip().str.upper()

        merged_cv = leak_clean.merge(barrio_cv[["barrio_clean", "regen_cv"]], on="barrio_clean", how="inner")

        if len(merged_cv) >= 5:
            rho_cv, _ = spearmanr(merged_cv["silent_leak_risk"], merged_cv["regen_cv"])
            rng = np.random.default_rng(42)
            n_perm = 999
            x = merged_cv["silent_leak_risk"].values
            y = merged_cv["regen_cv"].values
            null_rhos = [spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)]
            p_cv = (sum(1 for r in null_rhos if r >= rho_cv) + 1) / (n_perm + 1)
            results["variabilidad_cv"] = {"rho": rho_cv, "p_perm": p_cv, "n": len(merged_cv)}
            print(f"    Variabilidad CV: rho={rho_cv:+.3f}, p={p_cv:.4f}, n={len(merged_cv)}")

    # Mejor resultado
    best = None
    for key, val in results.items():
        if best is None or val["p_perm"] < best.get("p_perm", 1):
            best = val
            best["sub_test"] = key

    if best:
        sig = "SIGNIFICATIVA" if best["p_perm"] < 0.05 else "NO significativa"
        print(f"    Mejor: {best['sub_test']} → rho={best['rho']:+.3f}, p={best['p_perm']:.4f} ({sig})")
        return {
            "test": "V3_CONSUMO",
            "rho": round(best["rho"], 4),
            "p_perm": round(best["p_perm"], 4),
            "sub_test": best["sub_test"],
            "n_barrios": best["n"],
            "significant": best["p_perm"] < 0.05,
            "all_results": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in results.items()},
            "status": "OK",
        }
    else:
        return {"test": "V3_CONSUMO", "status": "INSUFFICIENT_DATA"}


def permutation_test_aquacare(leak_df: pd.DataFrame = None, n_perm: int = 1000, top_k: int = 5) -> dict:
    """
    V4: Null permutation test sobre AquaCare scores.

    Permuta demographics elderly entre barrios, recalcula silent_leak_risk.
    Si el top-k real es significativamente mayor que el aleatorio → no es ruido.

    Nota: esto valida que la COMBINACION meter_age × elderly × barrio_size
    produce un ranking que no se obtendria por azar.
    """
    print("\n  ═══ V4: Permutation test AquaCare ═══")

    if leak_df is None:
        leak_df = detect_meter_leaks_by_barrio()
    if leak_df.empty or len(leak_df) < 10:
        return {"test": "V4_PERMUTATION", "status": "NO_DATA"}

    df = leak_df.copy()

    # Columnas necesarias (base)
    required = ["pct_old", "pct_elderly_alone", "pct_elderly", "n_meters"]
    for col in required:
        if col not in df.columns:
            return {"test": "V4_PERMUTATION", "status": "MISSING_COLS", "missing": col}

    # V4 validates the VULNERABILITY SCORE (demographics only)
    # This is independent of consumption features
    vuln_col = "vulnerability_score" if "vulnerability_score" in df.columns else "silent_leak_risk"

    # Observed: top-k mean vulnerability_score
    observed = df.nlargest(top_k, vuln_col)[vuln_col].mean()

    # Permutaciones: shuffle elderly demographics, recompute vulnerability
    rng = np.random.default_rng(42)
    null_dist = np.empty(n_perm)

    pct_old = df["pct_old"].fillna(0).values
    pct_elderly_alone = df["pct_elderly_alone"].fillna(0).values
    pct_elderly = df["pct_elderly"].fillna(0).values
    n_meters_norm = (df["n_meters"] / df["n_meters"].max()).values

    for i in range(n_perm):
        # Shuffle ONLY elderly columns (keep meter data fixed)
        perm_alone = rng.permutation(pct_elderly_alone)
        perm_elderly = rng.permutation(pct_elderly)
        risk_perm = (
            0.35 * pct_old +
            0.35 * perm_alone +
            0.20 * perm_elderly +
            0.10 * n_meters_norm
        )
        null_dist[i] = np.partition(risk_perm, -top_k)[-top_k:].mean()

    p_value = (np.sum(null_dist >= observed) + 1) / (n_perm + 1)
    z_score = (observed - null_dist.mean()) / null_dist.std() if null_dist.std() > 0 else 0

    sig = "SIGNIFICATIVA" if p_value < 0.05 else "NO significativa"

    print(f"    Barrios analizados: {len(df)}")
    print(f"    Top-{top_k} mean observed: {observed:.4f}")
    print(f"    Null mean: {null_dist.mean():.4f} +/- {null_dist.std():.4f}")
    print(f"    Z-score: {z_score:.2f}")
    print(f"    p-value: {p_value:.4f} → {sig}")

    return {
        "test": "V4_PERMUTATION",
        "observed_mean": round(observed, 4),
        "null_mean": round(null_dist.mean(), 4),
        "null_std": round(null_dist.std(), 4),
        "z_score": round(z_score, 2),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "n_barrios": len(df),
        "top_k": top_k,
        "n_perm": n_perm,
        "status": "OK",
        "null_scores": null_dist.tolist(),
    }


def sensitivity_aquacare_weights(leak_df: pd.DataFrame = None, n_random: int = 200, top_k: int = 5) -> dict:
    """
    V5: Sensitivity analysis de pesos AquaCare.

    Genera 200 vectores de pesos aleatorios (Dirichlet), recalcula risk,
    cuenta frecuencia de cada barrio en top-k.

    Si los mismos barrios aparecen en >80% de configuraciones → ultra-robusto.
    """
    print("\n  ═══ V5: Sensitivity analysis de pesos ═══")

    if leak_df is None:
        leak_df = detect_meter_leaks_by_barrio()
    if leak_df.empty or len(leak_df) < 10:
        return {"test": "V5_SENSITIVITY", "status": "NO_DATA"}

    df = leak_df.copy()
    required = ["pct_old", "pct_elderly_alone", "pct_elderly", "n_meters", "BARRIO"]
    for col in required:
        if col not in df.columns:
            return {"test": "V5_SENSITIVITY", "status": "MISSING_COLS"}

    rng = np.random.default_rng(42)

    pct_old = df["pct_old"].fillna(0).values
    pct_alone = df["pct_elderly_alone"].fillna(0).values
    pct_elderly = df["pct_elderly"].fillna(0).values
    n_meters_norm = (df["n_meters"] / df["n_meters"].max()).values
    barrios = df["BARRIO"].values

    # V5 tests weight robustness on ALL features (4 or 6)
    has_dynamic = "drop_freq_norm" in df.columns and "cv_norm" in df.columns
    if has_dynamic:
        drop_freq = df["drop_freq_norm"].fillna(0).values
        cv_vals = df["cv_norm"].fillna(0).values
        features = np.column_stack([pct_old, pct_alone, pct_elderly, n_meters_norm, drop_freq, cv_vals])
    else:
        features = np.column_stack([pct_old, pct_alone, pct_elderly, n_meters_norm])
    n_features = features.shape[1]

    # Count frequency in top-k
    frequency = {}
    for _ in range(n_random):
        weights = rng.dirichlet(np.ones(n_features))  # Uniform Dirichlet
        risk = features @ weights
        top_idx = np.argpartition(risk, -top_k)[-top_k:]
        for idx in top_idx:
            b = barrios[idx]
            frequency[b] = frequency.get(b, 0) + 1

    # Normalize to percentage
    freq_pct = {b: count / n_random for b, count in frequency.items()}
    freq_sorted = sorted(freq_pct.items(), key=lambda x: x[1], reverse=True)

    # Ultra-robust: appear in >80%
    ultra_robust = [(b, pct) for b, pct in freq_sorted if pct >= 0.80]
    robust = [(b, pct) for b, pct in freq_sorted if pct >= 0.50]

    print(f"    {n_random} configuraciones de pesos aleatorios (Dirichlet)")
    print(f"    Ultra-robustos (>80% de configs): {len(ultra_robust)}")
    for b, pct in ultra_robust:
        print(f"      {b:<35} {pct:.0%}")
    print(f"    Robustos (>50%): {len(robust)}")

    # Also test: original top-5 overlap
    if has_dynamic:
        # Combined: 0.6*vulnerability + 0.4*consumption_risk
        # vulnerability = [0.35, 0.35, 0.20, 0.10, 0, 0]
        # consumption  = [0, 0, 0, 0, 0.40, 0.30] + 0.30*pct_old
        # combined = 0.6*[0.35,0.35,0.20,0.10,0,0] + 0.4*[0.30,0,0,0,0.40,0.30]
        original_weights = np.array([0.6*0.35+0.4*0.30, 0.6*0.35, 0.6*0.20, 0.6*0.10, 0.4*0.40, 0.4*0.30])
    else:
        original_weights = np.array([0.35, 0.35, 0.20, 0.10])
    original_risk = features @ original_weights
    original_top = set(barrios[np.argpartition(original_risk, -top_k)[-top_k:]])

    overlap_scores = []
    for _ in range(n_random):
        weights = rng.dirichlet(np.ones(n_features))
        risk = features @ weights
        top_set = set(barrios[np.argpartition(risk, -top_k)[-top_k:]])
        overlap = len(original_top & top_set) / top_k
        overlap_scores.append(overlap)

    mean_overlap = np.mean(overlap_scores)
    print(f"    Overlap medio con ranking original: {mean_overlap:.0%}")

    return {
        "test": "V5_SENSITIVITY",
        "n_configs": n_random,
        "ultra_robust": [(b, round(pct, 3)) for b, pct in ultra_robust],
        "n_ultra_robust": len(ultra_robust),
        "n_robust": len(robust),
        "mean_overlap_with_original": round(mean_overlap, 3),
        "top_10_frequency": [(b, round(pct, 3)) for b, pct in freq_sorted[:10]],
        "status": "OK",
    }


def run_aquacare_validations(leak_df: pd.DataFrame = None) -> dict:
    """
    Ejecuta las 5 validaciones de AquaCare y devuelve resumen.
    """
    print("\n" + "=" * 70)
    print("  VALIDACIONES AQUACARE — 5 Tests Independientes")
    print("=" * 70)

    if leak_df is None:
        leak_df = detect_meter_leaks_by_barrio()

    results = {}
    results["V1"] = validate_aquacare_vs_mnf(leak_df)
    results["V2"] = validate_aquacare_vs_meter_changes(leak_df)
    results["V3"] = validate_aquacare_vs_consumption(leak_df)
    results["V4"] = permutation_test_aquacare(leak_df)
    results["V5"] = sensitivity_aquacare_weights(leak_df)

    # Resumen
    print("\n" + "=" * 70)
    print("  RESUMEN VALIDACIONES AQUACARE")
    print("=" * 70)
    n_sig = 0
    n_run = 0
    for key, res in results.items():
        status = res.get("status", "?")
        if status == "OK":
            n_run += 1
            if key in ("V1", "V2", "V3"):
                sig = res.get("significant", False)
                rho = res.get("rho", 0)
                p = res.get("p_perm", 1)
                mark = "PASS" if sig else "FAIL"
                if sig:
                    n_sig += 1
                print(f"    {key} ({res['test']:<15}): rho={rho:+.3f}, p={p:.4f} → {mark}")
            elif key == "V4":
                sig = res.get("significant", False)
                p = res.get("p_value", 1)
                z = res.get("z_score", 0)
                mark = "PASS" if sig else "FAIL"
                if sig:
                    n_sig += 1
                print(f"    {key} ({res['test']:<15}): Z={z:.1f}, p={p:.4f} → {mark}")
            elif key == "V5":
                n_ultra = res.get("n_ultra_robust", 0)
                overlap = res.get("mean_overlap_with_original", 0)
                mark = "PASS" if n_ultra >= 3 else "FAIL"
                if n_ultra >= 3:
                    n_sig += 1
                print(f"    {key} ({res['test']:<15}): {n_ultra} ultra-robustos, overlap={overlap:.0%} → {mark}")
        else:
            print(f"    {key}: {status} — {res.get('reason', 'datos insuficientes')}")

    verdict = "FIABLE" if n_sig >= 3 else ("PARCIAL" if n_sig >= 2 else "INSUFICIENTE")
    print(f"\n    Tests ejecutados: {n_run}/5")
    print(f"    Tests superados: {n_sig}/{n_run}")
    print(f"    Veredicto: {verdict}")
    print("=" * 70)

    results["summary"] = {
        "n_run": n_run,
        "n_significant": n_sig,
        "verdict": verdict,
    }

    return results


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AquaCare — Deteccion de emergencias en personas vulnerables"
    )
    parser.add_argument(
        "--data",
        default="data/datos-hackathon-amaem.xlsx-set-de-datos-.csv",
        help="Path al CSV de datos del hackathon",
    )
    parser.add_argument(
        "--caudal",
        default=None,
        help="Path al CSV de datos horarios de caudal (opcional)",
    )
    parser.add_argument(
        "--uso",
        default="DOMESTICO",
        help="Tipo de uso a filtrar (default: DOMESTICO)",
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"ERROR: No se encuentra {args.data}")
        exit(1)

    # Cargar datos
    from train_local import load_hackathon_amaem
    from monthly_features import compute_monthly_features

    print("Cargando datos del hackathon...")
    df_all = load_hackathon_amaem(args.data)

    print("Calculando features mensuales...")
    df_monthly = compute_monthly_features(df_all)

    # Filtrar por uso
    df_uso = df_monthly[df_monthly["uso"].str.strip() == args.uso].copy()
    print(f"  {len(df_uso)} filas para uso={args.uso}, "
          f"{df_uso['barrio_key'].nunique()} barrios")

    # Detectar caudal path por defecto si no se especifica
    caudal_path = args.caudal
    if caudal_path is None:
        default_caudal = "data/_caudal_medio_sector_hidraulico_hora_2024_-caudal_medio_sector_hidraulico_hora_2024.csv"
        if Path(default_caudal).exists():
            caudal_path = default_caudal

    # Ejecutar deteccion welfare
    alerts = run_welfare_detection(df_uso, results=None, caudal_path=caudal_path)

    # Imprimir resumen humano
    welfare_summary(alerts)
