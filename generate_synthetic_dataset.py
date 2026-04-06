"""
generate_synthetic_dataset.py
Genera datasets sinteticos completos para entrenar y demostrar el pipeline.
Los datos se generan en el formato EXACTO que espera run_all_models.py.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from calendar import monthrange

# ── Curva diurna española (AEAS 2022 / MITECO) ──
DIURNAL = np.array([
    0.35, 0.25, 0.20, 0.18, 0.18, 0.25, 0.55, 1.45, 1.70, 1.45, 1.20, 1.10,
    1.20, 1.35, 1.10, 0.80, 0.75, 0.80, 0.95, 1.15, 1.25, 1.15, 0.90, 0.55,
])
WEEKDAY_F = np.array([1.0, 1.0, 1.0, 1.0, 1.05, 1.15, 1.15])

# Estacionalidad mensual (1.0 = media anual)
SEASONAL = {1: 0.85, 2: 0.88, 3: 0.95, 4: 1.00, 5: 1.08, 6: 1.18,
            7: 1.25, 8: 1.22, 9: 1.10, 10: 0.98, 11: 0.88, 12: 0.82}

# ── Barrios con perfiles definidos ──
BARRIOS = {
    # barrio: (n_contratos_base, consumo_L_mes_por_contrato, tipo)
    "1-BENALUA": (3200, 6200, "normal"),
    "2-SAN ANTON": (1800, 5800, "normal"),
    "3-CENTRO": (6300, 6500, "turismo"),  # <-- anomalia turismo
    "4-MERCADO": (2100, 5600, "normal"),
    "5-CAMPOAMOR": (4500, 6800, "normal"),
    "6-LOS ANGELES": (2800, 5500, "normal"),
    "8-ALIPARK": (1500, 6000, "normal"),
    "9-FLORIDA ALTA": (2600, 6100, "normal"),
    "10-FLORIDA BAJA": (4700, 6300, "normal"),
    "14-ENSANCHE DIPUTACION": (3400, 6700, "normal"),
    "16-PLA DEL BON REPOS": (7500, 6400, "normal"),
    "17-CAROLINAS ALTAS": (10400, 5900, "fuga_silenciosa"),  # <-- anomalia
    "18-CAROLINAS BAJAS": (3600, 5700, "normal"),
    "19-GARBINET": (8200, 7200, "normal"),
    "20-RABASA": (5100, 6600, "normal"),
    "22-CASCO ANTIGUO - SANTA CRUZ": (2700, 5400, "normal"),
    "24-SAN BLAS - SANTO DOMINGO": (4300, 6100, "normal"),
    "25-ALTOZANO - CONDE LUMIARES": (3900, 5800, "normal"),
    "28-EL PALMERAL": (1200, 7000, "normal"),
    "31-CIUDAD JARDIN": (2200, 6900, "normal"),
    "32-VIRGEN DEL REMEDIO": (8900, 5200, "fraude"),  # <-- anomalia
    "33- MORANT -SAN NICOLAS BARI": (3100, 5600, "normal"),
    "34-COLONIA REQUENA": (4100, 5100, "fuga_fisica"),  # <-- anomalia
    "35-VIRGEN DEL CARMEN": (6800, 5300, "reparacion"),  # <-- anomalia
    "40-CABO DE LAS HUERTAS": (3500, 7500, "normal"),
    "41-PLAYA DE SAN JUAN": (9800, 8200, "estacionalidad"),  # <-- piscinas
    "54-POLIGONO VALLONGA": (800, 12000, "normal"),
    "55-PUERTO": (400, 15000, "normal"),
    "56-DISPERSOS": (600, 4800, "enganche"),  # <-- anomalia
    "TABARCA": (200, 5000, "contador_roto"),  # <-- anomalia
    "VILLAFRANQUEZA": (1100, 6500, "normal"),
}

# Sectores hidraulicos con mapeo a barrio
SECTORS = {
    "COLONIA REQUENA": "34-COLONIA REQUENA",
    "VIRGEN DEL REMEDIO": "32-VIRGEN DEL REMEDIO",
    "1 CIUDAD JARDÍN": "31-CIUDAD JARDIN",
    "ALIPARK DL": "8-ALIPARK",
    "ALTOZANO": "25-ALTOZANO - CONDE LUMIARES",
    "BENALÚA DL": "1-BENALUA",
    "Bº LOS ÁNGELES": "6-LOS ANGELES",
    "CABO HUERTAS - PLAYA": "40-CABO DE LAS HUERTAS",
    "Campoamor Alto": "5-CAMPOAMOR",
    "DIPUTACIÓN DL": "14-ENSANCHE DIPUTACION",
    "GARBINET NORTE 1": "19-GARBINET",
    "LONJA": "4-MERCADO",
    "Les Palmeretes": "28-EL PALMERAL",
    "P.A.U. 1 (norte+sur)": "41-PLAYA DE SAN JUAN",
    "PLAYA DE SAN JUAN 1": "41-PLAYA DE SAN JUAN",
    "PZA. MONTAÑETA": "3-CENTRO",
    "Pla-Hospital": "16-PLA DEL BON REPOS",
    "Postiguet": "22-CASCO ANTIGUO - SANTA CRUZ",
    "RABASA DL": "20-RABASA",
    "SANTO DOMINGO DL": "24-SAN BLAS - SANTO DOMINGO",
    "VALLONGA GLOBAL": "54-POLIGONO VALLONGA",
    "VILLAFRANQUEZA": "VILLAFRANQUEZA",
    "VIRGEN DEL CARMEN 1000 Viv": "35-VIRGEN DEL CARMEN",
    "PARQUE LO MORANT": "33- MORANT -SAN NICOLAS BARI",
}


def _month_index(year, month, start_year=2022, start_month=1):
    return (year - start_year) * 12 + (month - start_month)


# ═══════════════════════════════════════════════════════════════
# DATASET 1: Consumo mensual (formato hackathon)
# ═══════════════════════════════════════════════════════════════
def generate_monthly_consumption(output_path, start_year=2022, end_year=2024):
    """Genera CSV mensual en formato exacto del hackathon AMAEM."""
    rng = np.random.RandomState(42)
    rows = []

    for barrio, (n_cont, base_cpc, tipo) in BARRIOS.items():
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                mi = _month_index(year, month)
                last_day = monthrange(year, month)[1]
                fecha = f"{year}/{month:02d}/{last_day:02d}"

                # Contratos con crecimiento leve
                contratos = int(n_cont * (1 + 0.003 * mi) + rng.normal(0, n_cont * 0.01))

                # Consumo base con estacionalidad + ruido
                seasonal = SEASONAL[month]
                cpc = base_cpc * seasonal * (1 + rng.normal(0, 0.05))

                # ── Inyectar anomalías según tipo ──
                if tipo == "fuga_fisica" and mi >= 18:
                    cpc *= 1.30 + rng.normal(0, 0.03)  # +30% por fuga

                elif tipo == "fraude" and mi >= 12:
                    cpc *= 0.60 + rng.normal(0, 0.02)  # -40% por contador manipulado

                elif tipo == "fuga_silenciosa" and mi >= 15:
                    months_since = mi - 15
                    cpc *= 1.0 + 0.05 * months_since  # +5% acumulativo por mes

                elif tipo == "turismo" and month in (6, 7, 8, 9):
                    cpc *= 2.0 + rng.normal(0, 0.1)  # 2x en verano (turismo)

                elif tipo == "estacionalidad" and month in (6, 7, 8):
                    cpc *= 2.5 + rng.normal(0, 0.15)  # 2.5x piscinas

                elif tipo == "enganche":
                    cpc *= 0.5  # Solo registra la mitad (el resto va por enganche)

                elif tipo == "contador_roto" and mi >= 20:
                    cpc = rng.uniform(0, 50)  # Casi 0

                elif tipo == "reparacion":
                    if 10 <= mi < 30:
                        cpc *= 1.25 + rng.normal(0, 0.02)  # Fuga activa
                    # Despues del mes 30: vuelve a normal

                consumo_total = max(0, int(cpc * contratos))

                # Formato DOMESTICO
                rows.append({
                    "Barrio": barrio,
                    "Uso": "DOMESTICO",
                    "Fecha (aaaa/mm/dd)": fecha,
                    "Consumo (litros)": f"{consumo_total:,}",
                    "Nº Contratos": f"{contratos:,}",
                })

                # Tambien NO DOMESTICO (10% de contratos, 3x consumo)
                cont_nd = max(1, int(contratos * 0.1))
                consumo_nd = max(0, int(cpc * 3 * cont_nd * (1 + rng.normal(0, 0.08))))
                rows.append({
                    "Barrio": barrio,
                    "Uso": "NO DOMESTICO",
                    "Fecha (aaaa/mm/dd)": fecha,
                    "Consumo (litros)": f"{consumo_nd:,}",
                    "Nº Contratos": f"{cont_nd:,}",
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  [Dataset 1] Consumo mensual: {len(df)} filas → {output_path}")
    return df


# ═══════════════════════════════════════════════════════════════
# DATASET 2: Caudal horario por sector (formato AMAEM)
# ═══════════════════════════════════════════════════════════════
def generate_hourly_caudal(output_path, year=2024):
    """Genera CSV horario de caudal por sector en formato exacto AMAEM."""
    rng = np.random.RandomState(43)
    rows = []

    start = datetime(year, 1, 1, 1)  # Empieza a la 1:00
    n_hours = 365 * 24

    for sector, barrio in SECTORS.items():
        if barrio is None:
            continue

        barrio_info = BARRIOS.get(barrio)
        if not barrio_info:
            continue

        n_cont, base_cpc, tipo = barrio_info
        # Caudal base del sector en m3/hora
        daily_m3 = (base_cpc * n_cont) / 1000 / 30  # m3/dia promedio
        hourly_base = daily_m3 / 24

        for h_idx in range(n_hours):
            dt = start + timedelta(hours=h_idx)
            hour = dt.hour
            month = dt.month
            weekday = dt.weekday()
            mi = _month_index(dt.year, month)

            factor = DIURNAL[hour] * WEEKDAY_F[weekday] * SEASONAL[month]
            caudal = hourly_base * factor + rng.normal(0, hourly_base * 0.1)

            # ── Inyectar anomalias horarias ──
            if tipo == "fuga_fisica" and month >= 7:
                # Fuga constante 24h (mas notable de noche)
                caudal += hourly_base * 0.8

            elif tipo == "fraude":
                pass  # Caudal del sector es normal; el fraude solo afecta facturacion

            elif tipo == "fuga_silenciosa" and month >= 4:
                months_since = month - 4
                caudal += hourly_base * 0.02 * months_since  # Crece gradualmente

            elif tipo == "enganche":
                # El sector tiene caudal REAL (incluyendo enganche)
                caudal += hourly_base * 1.5  # El enganche añade 150% extra

            elif tipo == "reparacion":
                if 3 <= month <= 8:
                    caudal += hourly_base * 0.6  # Fuga activa
                # Despues de agosto: reparado, caudal normal

            caudal = max(0.1, caudal)

            # Formato AMAEM: DD/MM/YYYY H:MM, coma decimal
            fecha_str = dt.strftime("%d/%m/%Y %-H:%M") if os.name != "nt" else dt.strftime("%d/%m/%Y %#H:%M")
            caudal_str = f"{caudal:.1f}".replace(".", ",")

            rows.append(f"{fecha_str},{sector},{caudal_str}")

    # Escribir directamente para rendimiento (245K+ filas)
    header = "FECHA_HORA,SECTOR,CAUDAL MEDIO(M3)"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(rows))
    print(f"  [Dataset 2] Caudal horario: {len(rows)} filas → {output_path}")


# ═══════════════════════════════════════════════════════════════
# DATASET 3: Cambios de contador (ground truth)
# ═══════════════════════════════════════════════════════════════
def generate_counter_changes(output_path, start_year=2020, end_year=2024):
    """Genera CSV de cambios de contador con fraude en barrios especificos."""
    rng = np.random.RandomState(44)
    rows = []

    motivos_normales = ["ED-CAMBIO POR EDAD", "CM-CAMBIO CLASE O MODELO", "TE-INSTALAR TELELECTURA"]
    motivos_fraud = ["FP-FRAUDE POSIBLE", "MR-MARCHA AL REVES", "PA-PARADO", "RO-CONTADOR ROTO", "RB-ROBO"]
    fraud_weights = [0.3, 0.3, 0.2, 0.15, 0.05]

    for barrio, (n_cont, _, tipo) in BARRIOS.items():
        n_changes = int(n_cont * 0.15)  # 15% de contadores cambiados en 5 años

        for _ in range(n_changes):
            year = rng.randint(start_year, end_year + 1)
            month = rng.randint(1, 13)
            day = rng.randint(1, 29)
            fecha = f"{year}-{month:02d}-{day:02d}"

            # Barrios con anomalias tienen mas eventos sospechosos
            if tipo in ("fraude", "enganche") and rng.random() < 0.08:
                motivo = rng.choice(motivos_fraud, p=fraud_weights)
            elif tipo == "contador_roto" and rng.random() < 0.15:
                motivo = "PA-PARADO"
            elif tipo == "fuga_fisica" and rng.random() < 0.05:
                motivo = "RO-CONTADOR ROTO"
            else:
                motivo = rng.choice(motivos_normales)

            calibre = rng.choice([13, 15, 20, 25], p=[0.7, 0.15, 0.1, 0.05])
            emplaz = rng.choice(["BATERIA", "GALERIA", "FACHADA", "ARQUETA"])

            rows.append({
                "EXPLOTACION": "ALICANTE",
                "FECHA": fecha,
                "MOTIVO_CAMBIO": motivo,
                "EMPLAZAMIENTO": emplaz,
                "CALIBRE": calibre,
            })

    df = pd.DataFrame(rows).sort_values("FECHA").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    n_fraud = df["MOTIVO_CAMBIO"].isin(motivos_fraud).sum()
    print(f"  [Dataset 3] Cambios contador: {len(df)} filas, {n_fraud} sospechosos → {output_path}")
    return df


# ═══════════════════════════════════════════════════════════════
# DATASET 4: Contadores telelectura
# ═══════════════════════════════════════════════════════════════
def generate_telelectura(output_path):
    """Genera CSV de contadores instalados con % smart meters variable."""
    rng = np.random.RandomState(45)
    rows = []

    for barrio, (n_cont, _, tipo) in BARRIOS.items():
        # Barrios anomalos tienen menos smart meters (mas riesgo)
        if tipo in ("fraude", "enganche", "contador_roto"):
            smart_pct = rng.uniform(0.60, 0.80)
        elif tipo == "fuga_fisica":
            smart_pct = rng.uniform(0.75, 0.90)
        else:
            smart_pct = rng.uniform(0.90, 0.99)

        for i in range(n_cont):
            is_smart = rng.random() < smart_pct
            year_inst = rng.randint(2015, 2025)
            month_inst = rng.randint(1, 13)
            day_inst = rng.randint(1, 29)

            rows.append({
                "EXPLOTACION": "ALICANTE",
                "LOCALIDAD": "ALICANTE",
                "BARRIO": barrio,
                "CALLE": f"CALLE SINTETICA {rng.randint(1, 500)}",
                "CALIBRE": rng.choice([13, 15, 20], p=[0.75, 0.15, 0.10]),
                "FECHA INSTALACION": f"{day_inst:02d}/{month_inst:02d}/{year_inst}",
                "SISTEMA": "Leer por telelectura" if is_smart else "Leer manualmente",
                "USO": "DOMÉSTICO",
                "ACTIVIDAD": "VIVIENDA",
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  [Dataset 4] Telelectura: {len(df)} contadores → {output_path}")
    return df


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def generate_full_synthetic_dataset(output_dir=None):
    """Genera TODOS los datasets sinteticos."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  GENERANDO DATASETS SINTETICOS COMPLETOS")
    print("=" * 70)
    print(f"  Directorio: {output_dir}")
    print(f"  Barrios: {len(BARRIOS)}")
    print(f"  Anomalias inyectadas:")
    for b, (_, _, t) in BARRIOS.items():
        if t != "normal":
            print(f"    {b}: {t}")
    print()

    generate_monthly_consumption(os.path.join(output_dir, "synthetic_monthly.csv"))
    generate_hourly_caudal(os.path.join(output_dir, "synthetic_hourly_caudal.csv"))
    generate_counter_changes(os.path.join(output_dir, "synthetic_counter_changes.csv"))
    generate_telelectura(os.path.join(output_dir, "synthetic_telelectura.csv"))

    print()
    print("  DONE. Para ejecutar el pipeline:")
    print(f"  python run_all_models.py --csv {os.path.join(output_dir, 'synthetic_monthly.csv')}")


if __name__ == "__main__":
    generate_full_synthetic_dataset()
