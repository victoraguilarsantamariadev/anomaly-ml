"""
synthetic_external_data.py
Genera datasets sinteticos de fuentes externas open-source para demostrar
que el pipeline detecta anomalias con senales fisicas del mundo real.

Fuentes reales equivalentes:
  1. InSAR Subsidencia  -> Copernicus EGMS (egms.land.copernicus.eu)
  2. Landsat Thermal    -> USGS EarthExplorer / NASA ECOSTRESS
  3. Airbnb Density     -> Inside Airbnb (insideairbnb.com)
  4. IGME Piezometria   -> IGME-SINAS (info.igme.es/BDAguas)
  5. Electricidad/Agua  -> REE (ree.es) + AMAEM ratio
  6. Catastro Vivienda  -> DGC Catastro INSPIRE WFS + IDAE
  7. Perfiles Hogar     -> Padron Municipal (anonimizado)

Cada dataset tiene anomalias inyectadas en los barrios correspondientes.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Barrios del dataset sintético con sus anomalías ──────────────
BARRIO_TIPO = {
    "1-BENALUA":                        "normal",
    "2-SAN ANTON":                      "normal",
    "3-CENTRO":                         "turismo",
    "4-MERCADO":                        "normal",
    "5-CAMPOAMOR":                      "normal",
    "6-LOS ANGELES":                    "normal",
    "8-ALIPARK":                        "normal",
    "9-FLORIDA ALTA":                   "normal",
    "10-FLORIDA BAJA":                  "normal",
    "14-ENSANCHE DIPUTACION":           "normal",
    "16-PLA DEL BON REPOS":             "normal",
    "17-CAROLINAS ALTAS":               "fuga_silenciosa",
    "18-CAROLINAS BAJAS":               "normal",
    "19-GARBINET":                      "normal",
    "20-RABASA":                        "normal",
    "22-CASCO ANTIGUO - SANTA CRUZ":    "normal",
    "24-SAN BLAS - SANTO DOMINGO":      "normal",
    "25-ALTOZANO - CONDE LUMIARES":     "normal",
    "28-EL PALMERAL":                   "normal",
    "31-CIUDAD JARDIN":                 "normal",
    "32-VIRGEN DEL REMEDIO":            "fraude",
    "33- MORANT -SAN NICOLAS BARI":     "normal",
    "34-COLONIA REQUENA":               "fuga_fisica",
    "35-VIRGEN DEL CARMEN":             "reparacion",
    "40-CABO DE LAS HUERTAS":           "normal",
    "41-PLAYA DE SAN JUAN":             "estacionalidad",
    "54-POLIGONO VALLONGA":             "normal",
    "55-PUERTO":                        "normal",
    "56-DISPERSOS":                     "enganche",
    "TABARCA":                          "contador_roto",
    "VILLAFRANQUEZA":                   "normal",
}

ALL_BARRIOS = list(BARRIO_TIPO.keys())


def _month_index(year: int, month: int, start_year=2022) -> int:
    return (year - start_year) * 12 + (month - 1)


# ═══════════════════════════════════════════════════════════════════
# DATASET 1: InSAR Subsidencia del Terreno (Copernicus EGMS)
# Mide hundimiento del suelo mm/año — fuga subterránea -> más hundimiento
# ═══════════════════════════════════════════════════════════════════
def generate_insar_subsidence(output_path: str) -> pd.DataFrame:
    """
    Dataset estático por barrio: velocidad de subsidencia (mm/año).
    Baseline urbano Alicante: -0.5 a -1.8 mm/año (consolidación normal).
    Anomalía: -5 a -10 mm/año (erosión subterránea por fuga).
    """
    rng = np.random.RandomState(101)
    rows = []

    for barrio, tipo in BARRIO_TIPO.items():
        # Baseline: hundimiento urbano normal
        base = rng.uniform(-1.8, -0.5)
        std = rng.uniform(0.2, 0.5)
        n_ps = rng.randint(40, 200)
        accel = round(rng.uniform(-0.05, 0.05), 3)

        if tipo == "fuga_fisica":
            # Tubería rota -> erosión activa -> hundimiento acelerado
            base = rng.uniform(-9.8, -8.5)
            std = rng.uniform(1.5, 2.0)
            accel = round(rng.uniform(-0.9, -0.6), 3)
            n_ps = rng.randint(80, 150)

        elif tipo == "fuga_silenciosa":
            # Fuga lenta -> hundimiento moderado, creciente
            base = rng.uniform(-4.2, -3.2)
            std = rng.uniform(0.7, 1.1)
            accel = round(rng.uniform(-0.35, -0.20), 3)

        elif tipo == "reparacion":
            # Fuga activa durante 40 días -> hundimiento elevado pero recuperando
            base = rng.uniform(-5.5, -4.2)
            std = rng.uniform(1.0, 1.5)
            accel = round(rng.uniform(0.5, 0.8), 3)  # positivo = recuperación

        elif tipo == "estacionalidad":
            # Costa: suelo más blando, hundimiento ligeramente mayor
            base = rng.uniform(-2.0, -1.2)
            std = rng.uniform(0.5, 0.8)

        elif tipo == "enganche":
            # Pozo ilegal extrae agua del acuífero -> ligera subsidencia
            base = rng.uniform(-2.2, -1.4)
            std = rng.uniform(0.5, 0.9)
            accel = round(rng.uniform(-0.15, -0.05), 3)

        subsidence = round(base + rng.normal(0, 0.2), 2)
        zscore = round((subsidence - (-1.1)) / 1.3, 2)  # z-score vs población

        rows.append({
            "barrio":                 barrio,
            "tipo_anomalia":          tipo,
            "subsidence_mm_yr":       subsidence,
            "subsidence_std_mm_yr":   round(std, 2),
            "trend_acceleration":     accel,
            "n_ps_points":            n_ps,
            "insar_zscore":           zscore,
            "insar_anomaly_flag":     int(subsidence < -4.0 and accel < -0.15),
            "campaign_year":          2023,
            "source":                 "Copernicus EGMS L3 (synthetic)",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    n_anom = df["insar_anomaly_flag"].sum()
    print(f"  [Ext-1] InSAR Subsidencia: {len(df)} barrios, {n_anom} con anomalía -> {output_path}")
    return df


# ═══════════════════════════════════════════════════════════════════
# DATASET 2: Landsat / ECOSTRESS Thermal Anomaly
# Temperatura superficial por barrio/mes — fuga -> cold spot por evaporación
# ═══════════════════════════════════════════════════════════════════
def generate_thermal_anomaly(output_path: str, start_year=2022, end_year=2024) -> pd.DataFrame:
    """
    Serie mensual por barrio: temperatura superficial y anomalía vs vecinos.
    Baseline Alicante verano (julio): ~39-42°C. Fugas crean cold spots -3 a -6°C.
    """
    rng = np.random.RandomState(102)

    # Temperatura base mensual Alicante (LST urbano, °C)
    LST_BASE = {1: 14.5, 2: 15.8, 3: 18.2, 4: 21.5, 5: 26.3, 6: 32.1,
                7: 39.4, 8: 38.7, 9: 31.2, 10: 24.8, 11: 18.5, 12: 15.2}

    rows = []

    for barrio, tipo in BARRIO_TIPO.items():
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                mi = _month_index(year, month)
                lst_base = LST_BASE[month]

                # Ruido normal (nubes, variación espacial)
                lst_mean = lst_base + rng.normal(0, 1.2)
                coldspot_delta = rng.normal(-1.2, 0.5)   # siempre ligeramente frío
                neighbor_delta = rng.normal(0.0, 0.8)
                cloud_pct = rng.uniform(5, 35)
                n_pixels = rng.randint(80, 250)

                # ── Inyectar anomalías ──────────────────────────────
                if tipo == "fuga_fisica" and mi >= 18:
                    # Evaporación constante -> cold spot persistente en verano
                    coldspot_delta = rng.normal(-5.8 if month in range(5, 10) else -2.5, 0.6)
                    neighbor_delta = rng.normal(-4.1 if month in range(5, 10) else -1.8, 0.5)
                    lst_mean -= abs(coldspot_delta) * 0.4

                elif tipo == "fuga_silenciosa" and mi >= 15:
                    # Efecto gradual creciente
                    months_since = mi - 15
                    intensity = min(0.08 * months_since, 2.0)
                    coldspot_delta = rng.normal(-1.4 - intensity, 0.4)
                    neighbor_delta = rng.normal(-0.8 - intensity * 0.6, 0.3)

                elif tipo == "reparacion" and 10 <= mi < 30:
                    # Fuga activa -> cold spot temporal
                    coldspot_delta = rng.normal(-3.4 if month in range(5, 10) else -1.8, 0.5)
                    neighbor_delta = rng.normal(-2.3 if month in range(5, 10) else -1.1, 0.4)

                elif tipo == "turismo":
                    # UHI por densidad urbana (ligeramente más caliente)
                    neighbor_delta = rng.normal(0.4, 0.5)

                elif tipo == "estacionalidad":
                    # Costa: brisa marina -> ligeramente más frío
                    lst_mean -= 1.5
                    neighbor_delta = rng.normal(-0.6, 0.5)

                elif tipo == "enganche":
                    # Vegetación irrigada -> ligero enfriamiento
                    coldspot_delta = rng.normal(-2.0, 0.5)
                    neighbor_delta = rng.normal(-0.9, 0.4)

                coldspot_zscore = round((coldspot_delta - (-1.2)) / 1.1, 2)
                neighbor_zscore = round(neighbor_delta / 1.0, 2)

                rows.append({
                    "barrio":                   barrio,
                    "tipo_anomalia":            tipo,
                    "year":                     year,
                    "month":                    month,
                    "lst_mean_c":               round(lst_mean, 1),
                    "lst_coldspot_delta_c":     round(coldspot_delta, 2),
                    "lst_neighbor_delta_c":     round(neighbor_delta, 2),
                    "thermal_coldspot_zscore":  coldspot_zscore,
                    "thermal_neighbor_zscore":  neighbor_zscore,
                    "cloud_cover_pct":          round(cloud_pct, 1),
                    "n_valid_pixels":           n_pixels,
                    "thermal_leak_flag":        int(coldspot_zscore < -2.0 and neighbor_zscore < -1.5),
                    "source":                   "Landsat 8/9 Band10 + ECOSTRESS (synthetic)",
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    n_flags = df["thermal_leak_flag"].sum()
    print(f"  [Ext-2] Landsat Thermal: {len(df)} barrio-meses, {n_flags} cold spots -> {output_path}")
    return df


# ═══════════════════════════════════════════════════════════════════
# DATASET 3: Airbnb / Viviendas Turísticas (Inside Airbnb)
# Densidad turística real -> deconfounding del consumo veraniego
# ═══════════════════════════════════════════════════════════════════
def generate_airbnb_density(output_path: str) -> pd.DataFrame:
    """
    Dataset estático por barrio: densidad de viviendas turísticas.
    CENTRO y PLAYA DE SAN JUAN tienen alta densidad -> consumo veraniego esperado.
    Barrios residenciales: <50 listings -> picos de consumo = anomalía real.
    """
    rng = np.random.RandomState(103)
    rows = []

    for barrio, tipo in BARRIO_TIPO.items():
        # Baseline residencial
        n_listings = int(rng.uniform(8, 45))
        n_licensed = int(n_listings * rng.uniform(0.5, 0.75))
        occ_summer = round(rng.uniform(18, 35), 1)
        occ_winter = round(rng.uniform(12, 25), 1)
        avg_guests = round(rng.uniform(1.8, 2.5), 1)

        if tipo == "turismo":
            # CENTRO: masiva oferta turística
            n_listings = int(rng.uniform(480, 560))
            n_licensed = int(rng.uniform(310, 370))
            occ_summer = round(rng.uniform(74, 82), 1)
            occ_winter = round(rng.uniform(35, 48), 1)
            avg_guests = round(rng.uniform(2.8, 3.4), 1)

        elif tipo == "estacionalidad":
            # PLAYA DE SAN JUAN: mayor densidad turística
            n_listings = int(rng.uniform(620, 740))
            n_licensed = int(rng.uniform(380, 440))
            occ_summer = round(rng.uniform(84, 92), 1)
            occ_winter = round(rng.uniform(28, 40), 1)
            avg_guests = round(rng.uniform(3.8, 4.6), 1)

        elif tipo == "contador_roto":
            # TABARCA: isla turística pequeña
            n_listings = int(rng.uniform(28, 42))
            n_licensed = int(rng.uniform(18, 28))
            occ_summer = round(rng.uniform(65, 75), 1)
            occ_winter = round(rng.uniform(20, 35), 1)

        # Tourist Water Pressure Index: turistas equivalentes / (contratos + turistas)
        # Alto en CENTRO y PLAYA -> consumo elevado ESPERADO (no anomalia)
        estimated_contracts = max(100, int(rng.uniform(400, 2500)))
        tourist_equiv = n_listings * (occ_summer / 100) * avg_guests
        twpi = round(tourist_equiv / (estimated_contracts + tourist_equiv), 3)

        rows.append({
            "barrio":                       barrio,
            "tipo_anomalia":                tipo,
            "n_airbnb_listings":            n_listings,
            "n_licensed_vt":                n_licensed,
            "avg_occupancy_summer_pct":     occ_summer,
            "avg_occupancy_winter_pct":     occ_winter,
            "avg_guests_per_listing":       avg_guests,
            "tourist_water_pressure_index": twpi,
            "snapshot_year":                2024,
            "is_tourism_barrio":            int(twpi > 0.30),
            "source":                       "Inside Airbnb + GVA Registro Turismo (synthetic)",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    n_tourist = df["is_tourism_barrio"].sum()
    print(f"  [Ext-3] Airbnb Density: {len(df)} barrios, {n_tourist} con alta presión turística -> {output_path}")
    return df


# ═══════════════════════════════════════════════════════════════════
# DATASET 4: IGME Piezometría (Nivel Freático)
# Si sube el agua subterránea sin lluvia -> fuga alimenta el acuífero
# ═══════════════════════════════════════════════════════════════════
def generate_piezometry(output_path: str, start_year=2022, end_year=2024) -> pd.DataFrame:
    """
    Serie mensual: profundidad del nivel freático (m).
    Baseline: 3.5-5.0m. Subida sin lluvia -> fuga subterránea.
    """
    rng = np.random.RandomState(104)

    # Precipitación mensual Alicante (mm)
    PRECIP = {1: 22, 2: 19, 3: 22, 4: 27, 5: 27, 6: 9,
              7: 4, 8: 6, 9: 38, 10: 49, 11: 36, 12: 24}

    rows = []

    for barrio, tipo in BARRIO_TIPO.items():
        base_depth = rng.uniform(3.8, 5.2)  # profundidad inicial

        if tipo == "estacionalidad":
            base_depth = rng.uniform(2.5, 3.2)  # costa: más superficial
        elif tipo == "contador_roto":
            base_depth = rng.uniform(5.0, 6.5)  # isla: roca, muy profundo

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                mi = _month_index(year, month)
                precip = PRECIP[month] + rng.normal(0, 5)

                # Variación estacional natural (lluvia otoño -> sube tabla)
                seasonal_change = -0.015 * precip + rng.normal(0, 0.08)
                depth = base_depth + seasonal_change

                monthly_change = seasonal_change
                precip_adjusted = round(monthly_change + 0.012 * precip, 3)

                # ── Inyectar anomalías ──────────────────────────────
                if tipo == "enganche" and mi >= 0:
                    # Extracción de pozo ilegal + fuga -> tabla sube cronicamente
                    depth -= 0.35 * (mi / 36 + 0.5)  # profundidad baja (sube tabla)
                    monthly_change = round(rng.normal(-0.35, 0.08), 3)
                    precip_adjusted = round(monthly_change + 0.012 * precip, 3)

                elif tipo == "fuga_fisica" and mi >= 18:
                    # Fuga activa -> tabla sube rápidamente
                    leak_months = mi - 18
                    depth -= 0.55 * min(leak_months * 0.1 + 0.5, 2.5)
                    monthly_change = round(rng.normal(-0.55, 0.10), 3)
                    precip_adjusted = round(monthly_change + 0.012 * precip, 3)

                elif tipo == "fuga_silenciosa" and mi >= 15:
                    months_since = mi - 15
                    depth -= 0.12 * months_since * 0.08
                    monthly_change = round(rng.normal(-0.12, 0.05), 3)
                    precip_adjusted = round(monthly_change + 0.012 * precip, 3)

                elif tipo == "reparacion":
                    if 10 <= mi < 30:
                        depth -= rng.uniform(0.8, 1.4)
                        monthly_change = round(rng.normal(-0.30, 0.08), 3)
                        precip_adjusted = round(monthly_change + 0.012 * precip, 3)
                    elif mi >= 30:
                        # Recuperación post-reparación
                        depth += rng.uniform(0.3, 0.6)
                        monthly_change = round(rng.normal(+0.25, 0.07), 3)
                        precip_adjusted = round(monthly_change + 0.012 * precip, 3)

                depth = max(0.5, round(depth + rng.normal(0, 0.05), 2))
                wt_zscore = round((depth - 4.0) / 0.9, 2)

                rows.append({
                    "barrio":                    barrio,
                    "tipo_anomalia":             tipo,
                    "year":                      year,
                    "month":                     month,
                    "water_table_depth_m":       depth,
                    "monthly_change_m":          round(monthly_change, 3),
                    "precip_mm":                 round(precip, 1),
                    "precip_adjusted_wt_change": precip_adjusted,
                    "water_table_zscore":        wt_zscore,
                    "wt_rising_anomaly":         int(precip_adjusted < -0.20 and depth < 2.8),
                    "source":                    "IGME SINAS Masa 080.006 (synthetic)",
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    n_rising = df["wt_rising_anomaly"].sum()
    print(f"  [Ext-4] IGME Piezometría: {len(df)} barrio-meses, {n_rising} subidas anómalas -> {output_path}")
    return df


# ═══════════════════════════════════════════════════════════════════
# DATASET 5: Ratio Electricidad / Agua (REE + AMAEM)
# Pozo ilegal: bomba eléctrica + sin agua registrada -> ratio explota
# ═══════════════════════════════════════════════════════════════════
def generate_electricity_water_ratio(output_path: str, start_year=2022, end_year=2024) -> pd.DataFrame:
    """
    Serie mensual: kWh consumidos por m3 de agua REGISTRADA.
    Normal: 1.5-2.8 kWh/m3. Pozo ilegal o fraude: 5-14 kWh/m3.
    """
    rng = np.random.RandomState(105)

    # Factor estacional electricidad (verano sube por AC)
    ELEC_SEASONAL = {1: 1.05, 2: 1.02, 3: 0.98, 4: 0.95, 5: 0.97, 6: 1.08,
                     7: 1.25, 8: 1.22, 9: 1.05, 10: 0.97, 11: 1.00, 12: 1.07}

    rows = []

    for barrio, tipo in BARRIO_TIPO.items():
        base_ratio = rng.uniform(1.6, 2.8)
        base_kwh = rng.uniform(15000, 80000)

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                mi = _month_index(year, month)
                seasonal = ELEC_SEASONAL[month]

                kwh_total = base_kwh * seasonal * (1 + rng.normal(0, 0.06))
                ratio = base_ratio * seasonal * (1 + rng.normal(0, 0.08))

                # ── Inyectar anomalías ──────────────────────────────
                if tipo == "enganche":
                    # Bomba de pozo ilegal -> alto consumo eléctrico,
                    # bajo agua registrada -> ratio EXTREMO
                    kwh_total *= rng.uniform(2.5, 3.5)
                    ratio = rng.uniform(10.5, 13.2) * (1 + rng.normal(0, 0.08))

                elif tipo == "fraude":
                    # Contador manipulado -> bajo agua registrada -> ratio ALTO
                    # pero no tanto como pozo (no hay bomba extra)
                    ratio = rng.uniform(4.8, 6.2) * (1 + rng.normal(0, 0.07))

                elif tipo == "contador_roto" and mi >= 20:
                    # Contador en 0 -> denominador casi 0 -> ratio explota
                    ratio = rng.uniform(6.0, 9.5)

                elif tipo == "estacionalidad" and month in (6, 7, 8):
                    # Más agua por piscinas -> ratio BAJA en verano
                    ratio = base_ratio * rng.uniform(0.55, 0.72)

                # Histórico YoY
                ratio_yoy = round(ratio / (base_ratio * seasonal), 2)
                ratio_zscore = round((ratio - 2.1) / 1.1, 2)

                rows.append({
                    "barrio":                    barrio,
                    "tipo_anomalia":             tipo,
                    "year":                      year,
                    "month":                     month,
                    "electricity_kwh_total":     round(kwh_total, 0),
                    "electricity_kwh_per_m3":    round(ratio, 2),
                    "ratio_zscore":              ratio_zscore,
                    "ratio_yoy":                 ratio_yoy,
                    "elec_water_anomaly_flag":   int(ratio > 5.5 and ratio_yoy > 1.8),
                    "source":                    "REE distribución + AMAEM ratio (synthetic)",
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    n_flags = df["elec_water_anomaly_flag"].sum()
    print(f"  [Ext-5] Electricidad/Agua: {len(df)} barrio-meses, {n_flags} anomalías -> {output_path}")
    return df


# ═══════════════════════════════════════════════════════════════════
# DATASET 6: Catastro INSPIRE WFS — Datos de vivienda individual
# Fuente real: Dirección General del Catastro (DGC)
# API pública: catastro.meh.es (WFS INSPIRE)
# Granularidad: por contrato_id (dirección individual)
# ═══════════════════════════════════════════════════════════════════

# Factores de corrección por década de construcción (antigüedad tuberías)
AGE_FACTOR = {
    1950: 1.25,  # tuberías pre-1960: muy deterioradas, +25% pérdidas
    1960: 1.20,  # años 60: galvanizado corroído
    1970: 1.15,  # años 70: amianto/fibrocemento
    1980: 1.08,  # años 80: primeras normativas modernas
    1990: 1.04,
    2000: 1.00,
    2010: 1.00,
}

# Características constructivas típicas por tipo de barrio
BARRIO_BUILDING_PROFILE = {
    # (year_range, m2_range_domestic, m2_range_commercial)
    "fuga_fisica":    ((1968, 1978), (55, 85), (80, 200)),   # COLONIA REQUENA, 1970s
    "fuga_silenciosa":((1962, 1982), (48, 75), (70, 180)),   # CAROLINAS ALTAS, 1960-80
    "fraude":         ((1985, 2005), (60, 95), (90, 220)),   # VIRGEN DEL REMEDIO, mixto
    "enganche":       ((1975, 1995), (45, 65), (60, 150)),   # DISPERSOS, humilde
    "contador_roto":  ((1970, 1990), (50, 80), (70, 160)),   # TABARCA
    "reparacion":     ((1980, 2000), (60, 90), (80, 180)),   # VIRGEN DEL CARMEN
    "turismo":        ((1990, 2015), (55, 90), (100, 300)),  # CENTRO, moderno-turístico
    "estacionalidad": ((1985, 2010), (55, 95), (100, 250)),  # PLAYA SAN JUAN, coastal
    "normal":         ((1975, 2010), (55, 85), (80, 200)),   # resto
}

# Consumo IDAE: 128 L/persona/día (media española oficial)
IDAE_L_PER_PERSON_PER_DAY = 128.0


def _get_age_factor(year: int) -> float:
    """Devuelve el factor de corrección por antigüedad del edificio."""
    decade = (year // 10) * 10
    decade = max(1950, min(decade, 2010))
    return AGE_FACTOR[decade]


def generate_catastro_households(
    output_path: str = "data/synthetic_catastro_households.csv",
    hourly_csv: str = "data/synthetic_hourly_domicilio.csv",
) -> pd.DataFrame:
    """
    Genera datos sintéticos de Catastro por contrato_id individual.

    Para cada vivienda/contrato calcula:
      - building_m2:                Superficie construida (m²)
      - construction_year:          Año de construcción
      - age_factor:                 Corrección por antigüedad de tuberías
      - occupancy_estimate:         Personas estimadas (m² / 30)
      - expected_monthly_L:         Consumo teórico IDAE × age_factor
      - actual_monthly_L:           Consumo real (promedio mensual del CSV horario)
      - consumption_efficiency_ratio: actual / expected
      - pipe_risk_score:             0-1, riesgo por antigüedad

    Anomalías inyectadas:
      - COLONIA REQUENA (fuga_fisica):      ratio > 2.0 (fuga en instalación)
      - CAROLINAS ALTAS (fuga_silenciosa):  ratio > 1.5 sostenido
      - VIRGEN DEL REMEDIO (fraude):        ratio < 0.4 (contadores manipulados)
      - DISPERSOS (enganche):               ratio > 2.5 (extracción ilegal)
      - TABARCA (contador_roto):            ratio ~0 (no registra)
    """
    rng = np.random.RandomState(42)

    # Cargar datos horarios para calcular consumo real por contrato
    hourly_path = Path(hourly_csv)
    if hourly_path.exists():
        df_h = pd.read_csv(hourly_path, parse_dates=["timestamp"])
        df_h["month"] = df_h["timestamp"].dt.month
        # Consumo total en 2 meses -> promedio mensual
        actual_monthly = (
            df_h.groupby(["contrato_id", "barrio", "uso", "month"])["consumo_litros"]
            .sum()
            .groupby(["contrato_id", "barrio", "uso"])
            .mean()
            .reset_index()
            .rename(columns={"consumo_litros": "actual_monthly_L"})
        )
    else:
        actual_monthly = pd.DataFrame()

    # Cargar ground truth de fugas para inyectar anomalías en los contratos correctos
    leak_path = Path(hourly_csv).parent / "synthetic_leak_labels.csv"
    leak_tipo = {}
    if leak_path.exists():
        labels = pd.read_csv(leak_path)
        fuga_map = {
            "rotura_tuberia":            "fuga_fisica",
            "consumo_nocturno_anomalo":  "fuga_fisica",
            "fuga_lenta_continua":       "fuga_fisica",
            "fuga_intermitente":         "fuga_silenciosa",
            "degradacion_gradual":       "fuga_silenciosa",
        }
        for _, r in labels.iterrows():
            leak_tipo[r["contrato_id"]] = fuga_map.get(r["tipo_fuga"], "fuga_fisica")

    rows = []

    for _, row in actual_monthly.iterrows():
        cid    = row["contrato_id"]
        barrio = row["barrio"]
        uso    = row["uso"]
        actual = row["actual_monthly_L"]

        # Detectar si este contrato tiene fuga conocida
        leak_type = leak_tipo.get(cid, None)

        # Determinar perfil constructivo del barrio (para m² y año)
        tipo_barrio = "normal"
        for bname, btype in BARRIO_TIPO.items():
            if bname in barrio or barrio in bname:
                tipo_barrio = btype
                break

        # Si el barrio tiene un perfil específico, usarlo; si no, usar "normal"
        profile_key = tipo_barrio if tipo_barrio in BARRIO_BUILDING_PROFILE else "normal"
        profile = BARRIO_BUILDING_PROFILE[profile_key]
        year_range, m2_dom, m2_com = profile

        # Asignar características constructivas
        construction_year = int(rng.randint(year_range[0], year_range[1] + 1))
        if uso == "DOMESTICO":
            building_m2 = round(rng.uniform(*m2_dom), 1)
        elif uso == "COMERCIAL":
            building_m2 = round(rng.uniform(*m2_com), 1)
        else:
            building_m2 = round(rng.uniform(m2_dom[0], m2_com[1]) * 0.7, 1)

        age_factor = _get_age_factor(construction_year)
        pipe_risk  = round((construction_year < 1985) * 0.6 + (construction_year < 1975) * 0.3, 2)

        # Consumo teórico IDAE
        occupancy         = round(building_m2 / 30.0, 2)
        expected_monthly  = round(occupancy * IDAE_L_PER_PERSON_PER_DAY * age_factor * 30, 1)

        # Ratio real/esperado — anomalías según ground truth de fugas
        base_ratio = actual / (expected_monthly + 1e-9)

        if leak_type == "fuga_fisica":
            # Rotura/fuga constante: consume 1.9–2.8x lo esperado
            efficiency_ratio = round(rng.uniform(1.9, 2.8), 3)
        elif leak_type == "fuga_silenciosa":
            # Fuga gradual/intermitente: 1.4–2.0x
            efficiency_ratio = round(rng.uniform(1.4, 2.0), 3)
        elif tipo_barrio == "fraude":
            # Fraude contador: registra sólo el 25-50% del consumo real
            efficiency_ratio = round(rng.uniform(0.25, 0.50), 3)
        elif tipo_barrio == "enganche":
            # Enganche ilegal: ratio muy alto
            efficiency_ratio = round(rng.uniform(2.3, 3.5), 3)
        elif tipo_barrio == "contador_roto":
            # Contador roto: ratio casi cero
            efficiency_ratio = round(rng.uniform(0.0, 0.08), 3)
        else:
            # Normal: usar el ratio real con pequeño ruido
            efficiency_ratio = round(
                max(0.4, min(base_ratio + rng.uniform(-0.10, 0.10), 1.35)), 3
            )

        rows.append({
            "contrato_id":               cid,
            "barrio":                    barrio,
            "uso":                       uso,
            "building_m2":               building_m2,
            "construction_year":         construction_year,
            "age_factor":                age_factor,
            "occupancy_estimate":        occupancy,
            "expected_monthly_L":        expected_monthly,
            "actual_monthly_L":          round(actual, 1),
            "consumption_efficiency_ratio": efficiency_ratio,
            "pipe_risk_score":           pipe_risk,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    n_anomaly = (df["consumption_efficiency_ratio"] > 1.5).sum() + \
                (df["consumption_efficiency_ratio"] < 0.45).sum()
    print(f"  [Catastro] {len(df)} viviendas, {n_anomaly} con ratio anomalo -> {output_path}")
    print(f"    Ratio medio: {df['consumption_efficiency_ratio'].mean():.2f} "
          f"(fugas > 1.5: {(df['consumption_efficiency_ratio'] > 1.5).sum()}, "
          f"fraude < 0.45: {(df['consumption_efficiency_ratio'] < 0.45).sum()})")
    return df


# ═══════════════════════════════════════════════════════════════════
# DATASET 7: Perfiles Demograficos Individuales (Padron Municipal)
# Fuente real: Padron Municipal de Alicante (datos anonimizados)
# Granularidad: por contrato_id (titular de la vivienda)
# ═══════════════════════════════════════════════════════════════════

# Nombres espanoles sinteticos (pool realista)
_NOMBRES_M = [
    "Maria Garcia Lopez", "Carmen Martinez Perez", "Josefa Fernandez Sanchez",
    "Ana Lopez Gonzalez", "Pilar Ruiz Diaz", "Teresa Moreno Jimenez",
    "Rosa Alvarez Hernandez", "Dolores Romero Torres", "Concepcion Navarro Dominguez",
    "Francisca Gutierrez Vazquez", "Isabel Serrano Ramos", "Mercedes Muñoz Flores",
    "Manuela Blanco Molina", "Antonia Suarez Ortega", "Luisa Castro Delgado",
    "Encarnacion Gil Rubio", "Amparo Medina Castillo", "Emilia Ortiz Marin",
    "Consuelo Herrera Iglesias", "Aurora Garrido Santos",
]
_NOMBRES_H = [
    "Jose Garcia Lopez", "Antonio Martinez Perez", "Manuel Fernandez Sanchez",
    "Francisco Lopez Gonzalez", "Juan Ruiz Diaz", "Pedro Moreno Jimenez",
    "Miguel Alvarez Hernandez", "Angel Romero Torres", "Rafael Navarro Dominguez",
    "Ramon Gutierrez Vazquez", "Carlos Serrano Ramos", "Fernando Muñoz Flores",
    "Luis Blanco Molina", "Joaquin Suarez Ortega", "Andres Castro Delgado",
    "Salvador Gil Rubio", "Enrique Medina Castillo", "Emilio Ortiz Marin",
    "Vicente Herrera Iglesias", "Alejandro Garrido Santos",
]

# Calles reales por barrio en Alicante
_CALLES_POR_BARRIO = {
    "CAROLINAS ALTAS": ["C/ Azorin", "C/ Pintor Murillo", "C/ San Isidro", "C/ Doctor Just"],
    "CAROLINAS BAJAS": ["C/ Italia", "C/ Reyes Catolicos", "C/ Garbinet", "C/ Padre Mariana"],
    "CASCO ANTIGUO":   ["C/ Mayor", "C/ Labradores", "C/ San Rafael", "C/ Maldonado"],
    "CENTRO TRADICIONAL": ["C/ San Fernando", "C/ Gerona", "C/ Castaños", "C/ Bazán"],
    "BENALUA":         ["C/ Arquitecto Morell", "C/ Churruca", "C/ Isabel la Católica"],
    "ALIPARK":         ["C/ Deportista Moro", "Av. Locutor Vicente Hipólito", "C/ Poeta Zorrilla"],
    "ALTOZANO":        ["C/ Teniente Alvarez Soto", "C/ Bono Guarner", "C/ Alfonso el Sabio"],
    "SAN GABRIEL":     ["C/ Ingeniero Canales", "C/ Virgen del Socorro", "C/ Jaime I"],
    "CIUDAD DE ASIS":  ["C/ Rio Turia", "C/ Rio Segura", "C/ Rio Jucar", "C/ Rio Mundo"],
    "POLIGONO BABEL":  ["C/ Peru", "C/ Chile", "C/ Ecuador", "C/ Uruguay"],
    "ENSANCHE DIPUTACION": ["C/ Foglietti", "C/ Doctor Gadea", "Av. Alfonso el Sabio"],
    "PLA DEL BON REPOS": ["C/ Virgen de la Cabeza", "C/ Escritor Azorin"],
    "GARBINET":        ["C/ Jose Maria Py", "C/ Catedratico Soler", "C/ Alcalde Suárez Llanos"],
    "POLIGONO SAN BLAS": ["C/ Vistahermosa", "C/ Rio Manzanares", "C/ Escritor Azorín"],
    "FLORIDA BAJA":    ["C/ Canalejas", "C/ Poeta Vila y Blanco", "C/ Garcia Morato"],
}

# Barrios con mas poblacion elderly (Padron real de Alicante 2025)
_BARRIOS_ELDERLY = {
    "CAROLINAS ALTAS":   0.30,  # 30% probabilidad de titular >70 anos
    "CAROLINAS BAJAS":   0.28,
    "CASCO ANTIGUO":     0.25,
    "CENTRO TRADICIONAL": 0.22,
    "BENALUA":           0.20,
    "PLA DEL BON REPOS": 0.18,
    "ALTOZANO":          0.18,
}


def generate_household_profiles(
    output_path: str = "data/synthetic_household_profiles.csv",
    hourly_csv: str = "data/synthetic_hourly_domicilio.csv",
) -> pd.DataFrame:
    """
    Genera perfiles demograficos sinteticos por contrato_id.

    Columnas: contrato_id, barrio, nombre_titular, edad_titular, sexo,
              vive_solo, n_personas_hogar, telefono_contacto, direccion_sintetica

    Inyecta titulares mayores/vulnerables en contratos que:
      a) tienen fuga detectada (leak_labels)
      b) estan en barrios con alta poblacion elderly

    Fuente real equivalente: Padron Municipal (datos proteccion datos, anonimizados)
    """
    rng = np.random.RandomState(77)

    # Cargar contratos unicos del hourly data
    hourly_path = Path(hourly_csv)
    if not hourly_path.exists():
        return pd.DataFrame()

    df_h = pd.read_csv(hourly_path, usecols=["contrato_id", "barrio", "uso"])
    contratos = df_h.drop_duplicates("contrato_id")[["contrato_id", "barrio", "uso"]]

    # Cargar leak labels para saber que contratos tienen fuga
    leak_path = hourly_path.parent / "synthetic_leak_labels.csv"
    leak_ids = set()
    if leak_path.exists():
        labels = pd.read_csv(leak_path)
        leak_ids = set(labels["contrato_id"])

    # Casos criticos forzados (elderly + fuga + vive solo)
    FORCED_ELDERLY = {
        "CTR-17-00070": ("Maria Garcia Lopez", 78, "F", True, "C/ Azorin 14, 2o B"),
        "CTR-18-00185": ("Antonio Navarro Dominguez", 82, "M", True, "C/ Italia 23, 3o A"),
        "CTR-18-00112": ("Josefa Fernandez Sanchez", 75, "F", True, "C/ Reyes Catolicos 8, 1o D"),
        "CTR-22-00117": ("Manuel Romero Torres", 80, "M", True, "C/ Teniente Alvarez Soto 5, 4o C"),
        "CTR-14-00161": ("Carmen Martinez Perez", 73, "F", True, "C/ Foglietti 11, 5o A"),
        "CTR-23-00098": ("Rosa Alvarez Hernandez", 77, "F", True, "C/ Mayor 31, 2o B"),
        "CTR-24-00077": ("Francisco Lopez Gonzalez", 84, "M", True, "C/ Labradores 7, 1o A"),
    }

    rows = []
    for _, row in contratos.iterrows():
        cid = row["contrato_id"]
        barrio = row["barrio"]
        uso = row["uso"]

        # Extraer nombre limpio del barrio
        barrio_clean = barrio.split("-", 1)[1].strip() if "-" in barrio else barrio

        # Caso forzado
        if cid in FORCED_ELDERLY:
            nombre, edad, sexo, solo, direccion = FORCED_ELDERLY[cid]
            n_personas = 1 if solo else rng.randint(2, 4)
            tel = f"+34 6{rng.randint(10, 99):02d} {rng.randint(100, 999):03d} {rng.randint(100, 999):03d}"
            rows.append({
                "contrato_id": cid, "barrio": barrio,
                "nombre_titular": nombre, "edad_titular": edad, "sexo": sexo,
                "vive_solo": solo, "n_personas_hogar": n_personas,
                "telefono_contacto": tel, "direccion_sintetica": direccion,
            })
            continue

        # Generar perfil aleatorio
        sexo = rng.choice(["M", "F"])
        nombre = rng.choice(_NOMBRES_H if sexo == "M" else _NOMBRES_M)

        # Edad: segun barrio + si tiene fuga = mas probable elderly
        elderly_prob = _BARRIOS_ELDERLY.get(barrio_clean, 0.10)
        if cid in leak_ids:
            elderly_prob = min(elderly_prob + 0.25, 0.60)

        if rng.random() < elderly_prob:
            edad = int(rng.randint(68, 92))
        elif uso == "COMERCIAL":
            edad = int(rng.randint(30, 55))
        else:
            edad = int(rng.randint(25, 70))

        # Vive solo: mas probable si elderly
        if edad >= 70:
            vive_solo = bool(rng.random() < 0.55)
        elif edad >= 60:
            vive_solo = bool(rng.random() < 0.25)
        else:
            vive_solo = bool(rng.random() < 0.10)

        n_personas = 1 if vive_solo else int(rng.randint(2, 5))

        # Direccion sintetica
        calles = _CALLES_POR_BARRIO.get(barrio_clean, ["C/ Principal"])
        calle = rng.choice(calles)
        num = rng.randint(1, 45)
        piso = rng.randint(1, 6)
        letra = rng.choice(["A", "B", "C", "D"])
        direccion = f"{calle} {num}, {piso}o {letra}"

        # Telefono
        tel = f"+34 6{rng.randint(10, 99):02d} {rng.randint(100, 999):03d} {rng.randint(100, 999):03d}"

        rows.append({
            "contrato_id": cid, "barrio": barrio,
            "nombre_titular": nombre, "edad_titular": edad, "sexo": sexo,
            "vive_solo": vive_solo, "n_personas_hogar": n_personas,
            "telefono_contacto": tel, "direccion_sintetica": direccion,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    n_elderly = (df["edad_titular"] >= 65).sum()
    n_solo = df["vive_solo"].sum()
    n_elderly_solo = ((df["edad_titular"] >= 65) & (df["vive_solo"])).sum()
    print(f"  [Perfiles] {len(df)} viviendas, {n_elderly} titulares >65 anos, "
          f"{n_solo} viven solos, {n_elderly_solo} mayores solos -> {output_path}")
    return df


# ═══════════════════════════════════════════════════════════════════
# MAIN — Genera todos los datasets
# ═══════════════════════════════════════════════════════════════════
def generate_all_external_data(output_dir: str = "data") -> dict:
    """Genera los 5 datasets de fuentes externas sintéticas."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  GENERANDO DATOS EXTERNOS SINTÉTICOS (5 fuentes)")
    print("=" * 70)
    print(f"  Barrios: {len(BARRIO_TIPO)}")
    print(f"  Anomalías inyectadas en:")
    for b, t in BARRIO_TIPO.items():
        if t != "normal":
            print(f"    {b:40s} : {t}")
    print()

    datasets = {
        "insar":       generate_insar_subsidence(
                           os.path.join(output_dir, "synthetic_insar_subsidence.csv")),
        "thermal":     generate_thermal_anomaly(
                           os.path.join(output_dir, "synthetic_thermal_anomaly.csv")),
        "airbnb":      generate_airbnb_density(
                           os.path.join(output_dir, "synthetic_airbnb_density.csv")),
        "piezometry":  generate_piezometry(
                           os.path.join(output_dir, "synthetic_piezometry.csv")),
        "electricity": generate_electricity_water_ratio(
                           os.path.join(output_dir, "synthetic_electricity_water_ratio.csv")),
        "catastro":    generate_catastro_households(
                           os.path.join(output_dir, "synthetic_catastro_households.csv"),
                           os.path.join(output_dir, "synthetic_hourly_domicilio.csv")),
        "profiles":    generate_household_profiles(
                           os.path.join(output_dir, "synthetic_household_profiles.csv"),
                           os.path.join(output_dir, "synthetic_hourly_domicilio.csv")),
    }

    print()
    print("  DONE. Para verificar cobertura:")
    print("  python check_coverage.py")
    print()
    print("  Para integrar en el pipeline:")
    print("  python run_all_models.py --csv data/synthetic_monthly.csv --with-physical-sensors"
)

    return datasets


if __name__ == "__main__":
    generate_all_external_data()
