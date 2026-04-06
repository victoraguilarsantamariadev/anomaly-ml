"""
synthetic_scenarios.py
Genera datos horarios sinteticos para 8 escenarios de anomalias hidricas.
Cada escenario simula 90 dias (2160 horas) con consumo realista basado
en la curva diurna española (AEAS 2022 / MITECO).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Curva diurna española (24 factores, suma=24, media=1.0)
DIURNAL_CURVE = np.array([
    0.35, 0.25, 0.20, 0.18, 0.18, 0.25, 0.55, 1.45, 1.70, 1.45, 1.20, 1.10,
    1.20, 1.35, 1.10, 0.80, 0.75, 0.80, 0.95, 1.15, 1.25, 1.15, 0.90, 0.55,
])
WEEKDAY_FACTORS = np.array([1.0, 1.0, 1.0, 1.0, 1.05, 1.15, 1.15])

COSTE_M3 = 1.5  # EUR/m3 tarifa media AMAEM


def _base_hourly(n_days, daily_liters=200, noise=0.08, seed=42, start_date=None):
    """Genera consumo horario base realista para un hogar/contrato."""
    rng = np.random.RandomState(seed)
    if start_date is None:
        start_date = datetime(2024, 6, 1)

    timestamps = []
    consumptions = []
    for d in range(n_days):
        dt = start_date + timedelta(days=d)
        weekday = dt.weekday()
        for h in range(24):
            ts = dt + timedelta(hours=h)
            factor = DIURNAL_CURVE[h] * WEEKDAY_FACTORS[weekday]
            hourly_base = daily_liters * factor / DIURNAL_CURVE.sum()
            hourly = max(0, hourly_base + rng.normal(0, noise * hourly_base))
            timestamps.append(ts)
            consumptions.append(hourly)

    return pd.DataFrame({"timestamp": timestamps, "consumption_liters": consumptions})


def scenario_fuga_fisica(n_days=90):
    """Escenario 1: Fuga fisica — tuberia rota a las 3AM del dia 20."""
    df = _base_hourly(n_days, daily_liters=200, seed=1)
    leak_start = 20 * 24 + 3  # Dia 20, hora 3AM
    leak_liters_per_hour = 25  # ~600 L/dia de fuga

    df["scenario_phase"] = "normal"
    df["is_anomaly_true"] = False
    df["caudal_sector_m3"] = df["consumption_liters"] / 1000 * 1.05  # Sector = ligeramente mas

    for i in range(leak_start, len(df)):
        df.loc[i, "consumption_liters"] += leak_liters_per_hour
        df.loc[i, "caudal_sector_m3"] += leak_liters_per_hour / 1000
        df.loc[i, "scenario_phase"] = "fuga_activa"
        df.loc[i, "is_anomaly_true"] = True

    df["barrio"] = "34-COLONIA REQUENA"
    df["scenario"] = "fuga_fisica"
    df["title"] = "Fuga fisica: tuberia rota"
    df["description"] = (
        "Una tuberia se rompe a las 3AM del dia 20. La fuga añade ~600 litros/dia "
        "de forma constante (dia y noche). El consumo nocturno sube un 300%. "
        "El sistema lo detecta por el ratio noche/dia anormalmente alto."
    )
    return df


def scenario_fraude(n_days=90):
    """Escenario 2: Fraude — manipulacion de contador, consumo diurno baja 40%."""
    df = _base_hourly(n_days, daily_liters=250, seed=2)
    fraud_start = 15 * 24  # Dia 15

    df["scenario_phase"] = "normal"
    df["is_anomaly_true"] = False
    df["caudal_sector_m3"] = df["consumption_liters"] / 1000 * 1.05

    for i in range(fraud_start, len(df)):
        hour = df.loc[i, "timestamp"].hour
        if 7 <= hour <= 22:  # Solo baja de dia (cuando el contador esta manipulado)
            df.loc[i, "consumption_liters"] *= 0.6  # Registra 60% del real
        df.loc[i, "scenario_phase"] = "fraude_activo"
        df.loc[i, "is_anomaly_true"] = True
        # Caudal del sector NO baja (el agua sigue fluyendo)

    df["barrio"] = "32-VIRGEN DEL REMEDIO"
    df["scenario"] = "fraude"
    df["title"] = "Fraude: manipulacion de contador"
    df["description"] = (
        "El dia 15 alguien manipula el contador. El agua sigue fluyendo igual, "
        "pero el contador solo registra el 60% del consumo diurno. De noche no se nota "
        "porque el consumo ya era bajo. La pista: el ratio noche/dia SUBE porque "
        "el dia parece mas bajo de lo normal."
    )
    return df


def scenario_fuga_silenciosa_mayor(n_days=90):
    """Escenario 3: Fuga silenciosa en casa de persona mayor de 82 años."""
    df = _base_hourly(n_days, daily_liters=120, seed=3)  # Persona sola, bajo consumo
    leak_start = 10 * 24  # Dia 10

    df["scenario_phase"] = "normal"
    df["is_anomaly_true"] = False
    df["caudal_sector_m3"] = df["consumption_liters"] / 1000 * 1.05

    # Fuga lenta que crece gradualmente (cisterna del WC que pierde)
    for i in range(leak_start, len(df)):
        days_since = (i - leak_start) / 24
        leak_rate = 2.0 + days_since * 0.05  # Empieza en 2 L/h, crece 0.05 L/h por dia
        df.loc[i, "consumption_liters"] += leak_rate
        df.loc[i, "caudal_sector_m3"] += leak_rate / 1000
        df.loc[i, "scenario_phase"] = "fuga_silenciosa"
        df.loc[i, "is_anomaly_true"] = True

    df["barrio"] = "17-CAROLINAS ALTAS"
    df["scenario"] = "fuga_silenciosa_mayor"
    df["title"] = "Fuga silenciosa: persona mayor sola (82 años)"
    df["description"] = (
        "Maria, 82 años, vive sola. La cisterna del WC empieza a perder agua el dia 10. "
        "Son solo 2 litros por hora al principio — ella no lo nota. Pero la fuga crece "
        "poco a poco. Al cabo de 2 meses gasta 50% mas agua y la factura sube. "
        "AquaCare detecta: barrio con alta poblacion mayor + consumo creciente = alerta social."
    )
    df["persona_edad"] = 82
    df["persona_sola"] = True
    df["pct_elderly_barrio"] = 28.5
    return df


def scenario_turismo(n_days=90):
    """Escenario 4: Piso turistico (falsa alarma) — patron Airbnb."""
    df = _base_hourly(n_days, daily_liters=0, seed=4)  # Empieza vacio
    df["consumption_liters"] = 0  # Piso vacio por defecto

    df["scenario_phase"] = "vacio"
    df["is_anomaly_true"] = False
    df["caudal_sector_m3"] = 0.001  # Minimo del sector

    rng = np.random.RandomState(4)
    # 3 estancias turisticas en 90 dias
    stays = [(10, 14), (35, 39), (65, 72)]  # (dia_inicio, dia_fin)
    for start_day, end_day in stays:
        for d in range(start_day, end_day + 1):
            for h in range(24):
                idx = d * 24 + h
                if idx < len(df):
                    factor = DIURNAL_CURVE[h] * 1.3  # Turistas gastan mas
                    daily = 350 + rng.normal(0, 30)  # ~350 L/dia (mas que residente)
                    df.loc[idx, "consumption_liters"] = max(0, daily * factor / DIURNAL_CURVE.sum())
                    df.loc[idx, "caudal_sector_m3"] = df.loc[idx, "consumption_liters"] / 1000
                    df.loc[idx, "scenario_phase"] = "turistas"

    df["barrio"] = "3-CENTRO"
    df["scenario"] = "turismo"
    df["title"] = "Turismo: piso Airbnb (falsa alarma)"
    df["description"] = (
        "Un piso en el centro de Alicante. Vacio 20 dias (0 litros), luego llegan turistas "
        "y gastan 350 L/dia durante 4-7 dias, y luego vuelve a 0. Este patron PARECE anomalia "
        "(consumo muy irregular) pero es turismo normal. El sistema lo filtra porque el barrio "
        "tiene 781 viviendas turisticas registradas."
    )
    df["n_viviendas_turisticas"] = 781
    return df


def scenario_contador_roto(n_days=90):
    """Escenario 5: Contador parado — deja de registrar."""
    df = _base_hourly(n_days, daily_liters=180, seed=5)
    break_start = 25 * 24  # Dia 25

    df["scenario_phase"] = "normal"
    df["is_anomaly_true"] = False
    df["caudal_sector_m3"] = df["consumption_liters"] / 1000 * 1.05

    for i in range(break_start, len(df)):
        real_consumption = df.loc[i, "consumption_liters"]
        df.loc[i, "consumption_liters"] = 0  # Contador no registra
        # Pero el agua sigue fluyendo en el sector
        df.loc[i, "caudal_sector_m3"] = real_consumption / 1000 * 1.05
        df.loc[i, "scenario_phase"] = "contador_parado"
        df.loc[i, "is_anomaly_true"] = True

    df["barrio"] = "TABARCA"
    df["scenario"] = "contador_roto"
    df["title"] = "Contador parado: registra 0 pero el agua fluye"
    df["description"] = (
        "El dia 25 el contador se para (mecanismo atascado). La persona sigue usando agua "
        "normalmente, pero el contador marca 0. La pista: el ANR del sector sube mucho "
        "(entra agua pero no se registra consumo). AMAEM tiene 1.401 casos reales de "
        "contadores parados en sus registros."
    )
    return df


def scenario_reparacion(n_days=90):
    """Escenario 6: Antes/despues de reparacion de fuga."""
    df = _base_hourly(n_days, daily_liters=200, seed=6)
    leak_start = 5 * 24   # Fuga empieza dia 5
    repair_day = 45 * 24  # Reparacion dia 45
    leak_rate = 20  # L/hora de fuga

    df["scenario_phase"] = "normal_pre"
    df["is_anomaly_true"] = False
    df["caudal_sector_m3"] = df["consumption_liters"] / 1000 * 1.05

    for i in range(leak_start, len(df)):
        if i < repair_day:
            df.loc[i, "consumption_liters"] += leak_rate
            df.loc[i, "caudal_sector_m3"] += leak_rate / 1000
            df.loc[i, "scenario_phase"] = "fuga_activa"
            df.loc[i, "is_anomaly_true"] = True
        else:
            df.loc[i, "scenario_phase"] = "reparado"

    df["barrio"] = "35-VIRGEN DEL CARMEN"
    df["scenario"] = "reparacion"
    df["title"] = "Antes/despues: reparacion de fuga"
    df["description"] = (
        "La fuga empieza el dia 5. Durante 40 dias pierde 480 litros/dia (20 L/hora). "
        "El dia 45, AMAEM repara la tuberia. El consumo vuelve a la normalidad INMEDIATAMENTE. "
        "Agua total perdida: 19.200 litros. Coste: 28,80 EUR. "
        "Si se hubiera detectado el dia 6, se habrian ahorrado 18.720 litros."
    )
    total_lost = leak_rate * (repair_day - leak_start)
    df["agua_perdida_litros"] = total_lost
    df["coste_perdida_eur"] = total_lost / 1000 * COSTE_M3
    df["ahorro_si_deteccion_rapida_litros"] = leak_rate * ((repair_day - leak_start) - 24)
    return df


def scenario_estacionalidad(n_days=180):
    """Escenario 7: Piscina + jardin en verano (NO es anomalia)."""
    start = datetime(2024, 4, 1)  # Abril a septiembre
    df = _base_hourly(n_days, daily_liters=200, seed=7, start_date=start)

    df["scenario_phase"] = "primavera"
    df["is_anomaly_true"] = False
    df["caudal_sector_m3"] = df["consumption_liters"] / 1000 * 1.05

    for i in range(len(df)):
        month = df.loc[i, "timestamp"].month
        hour = df.loc[i, "timestamp"].hour
        if month in (6, 7, 8):  # Verano
            # Riego jardin por la mañana (6-8h) y noche (21-23h)
            riego_factor = 2.5 if hour in (6, 7, 21, 22) else 1.3
            df.loc[i, "consumption_liters"] *= riego_factor
            df.loc[i, "caudal_sector_m3"] = df.loc[i, "consumption_liters"] / 1000 * 1.05
            df.loc[i, "scenario_phase"] = "verano_piscina"

    df["barrio"] = "41-PLAYA DE SAN JUAN"
    df["scenario"] = "estacionalidad"
    df["title"] = "Estacionalidad: piscina + jardin (NO es anomalia)"
    df["description"] = (
        "Chalet en Playa de San Juan. En primavera gasta 200 L/dia. En verano sube a 500+ L/dia "
        "por la piscina y el riego del jardin. Parece anomalia (consumo sube 150%) pero es "
        "estacionalidad normal. El satelite Sentinel-2 confirma: NDVI alto (jardin verde) "
        "y la renta del barrio es alta (25.000 EUR/persona). Consumo explicado, no sospechoso."
    )
    df["ndvi_summer"] = 0.35
    df["renta_media"] = 25000
    return df


def scenario_enganche_ilegal(n_days=90):
    """Escenario 8: Enganche ilegal — conexion directa a la red."""
    df = _base_hourly(n_days, daily_liters=150, seed=8)  # Consumo facturado bajo

    df["scenario_phase"] = "normal"
    df["is_anomaly_true"] = False

    # El consumo facturado es bajo, pero el sector inyecta MUCHO mas
    real_consumption = 400  # L/dia reales (tiene jardin grande, piscina)
    df["caudal_sector_m3"] = df["consumption_liters"] / 1000 * 1.05

    enganche_start = 0  # Desde el principio
    for i in range(len(df)):
        hour = df.loc[i, "timestamp"].hour
        factor = DIURNAL_CURVE[hour]
        extra = (real_consumption - 150) * factor / DIURNAL_CURVE.sum()
        # El agua extra fluye por el sector pero NO pasa por el contador
        df.loc[i, "caudal_sector_m3"] += max(0, extra) / 1000
        df.loc[i, "is_anomaly_true"] = True
        df.loc[i, "scenario_phase"] = "enganche_activo"

    df["barrio"] = "56-DISPERSOS"
    df["scenario"] = "enganche_ilegal"
    df["title"] = "Enganche ilegal: conexion directa a la red"
    df["description"] = (
        "Una vivienda en zona dispersa tiene una conexion directa a la tuberia principal, "
        "saltandose el contador. El contador registra 150 L/dia (lo minimo para no levantar "
        "sospechas) pero la casa realmente gasta 400 L/dia (piscina + jardin). "
        "La pista: el ANR del sector es altisimo (entra mucha mas agua de la que se factura) "
        "y el NDVI satelital muestra un jardin verde en plena sequia sin facturar agua de riego."
    )
    df["ndvi_summer"] = 0.40
    df["anr_ratio_esperado"] = real_consumption / 150
    return df


def generate_all_scenarios():
    """Genera todos los escenarios sinteticos.

    Returns:
        dict[str, pd.DataFrame]: Clave = nombre, valor = DataFrame horario
    """
    scenarios = {
        "fuga_fisica": scenario_fuga_fisica(),
        "fraude": scenario_fraude(),
        "fuga_silenciosa_mayor": scenario_fuga_silenciosa_mayor(),
        "turismo": scenario_turismo(),
        "contador_roto": scenario_contador_roto(),
        "reparacion": scenario_reparacion(),
        "estacionalidad": scenario_estacionalidad(),
        "enganche_ilegal": scenario_enganche_ilegal(),
    }
    return scenarios


if __name__ == "__main__":
    scenarios = generate_all_scenarios()
    for name, df in scenarios.items():
        n_anom = df["is_anomaly_true"].sum()
        print(f"  {name:<30} {len(df):>5} horas, {n_anom:>4} anomalas, barrio={df['barrio'].iloc[0]}")
