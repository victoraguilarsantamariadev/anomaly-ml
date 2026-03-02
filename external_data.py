"""
Datos externos open source para mejorar la deteccion de anomalias.

Fuentes:
  - AEMET: temperatura y precipitacion mensual de Alicante
  - INE: ocupacion hotelera de Alicante provincia
  - SPEI: indice de sequia
  - Calendario: festivos nacionales + locales de Alicante

Cada fuente tiene datos estaticos de fallback (medias climatologicas oficiales)
para que el sistema funcione SIN API keys ni conexion a internet.

Uso:
  from external_data import load_external_data
  df_ext = load_external_data("2022-01-01", "2024-12-31")
"""

import numpy as np
import pandas as pd
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# Datos estaticos de Alicante (medias climatologicas oficiales)
# Fuente: AEMET, estacion Alicante/Alacant (8025)
# ─────────────────────────────────────────────────────────────────

ALICANTE_MONTHLY_TEMP = {
    1: 11.5, 2: 12.2, 3: 14.0, 4: 16.0, 5: 19.5, 6: 23.5,
    7: 26.2, 8: 26.8, 9: 24.0, 10: 20.0, 11: 15.5, 12: 12.5,
}

ALICANTE_MONTHLY_PRECIP_MM = {
    1: 22, 2: 19, 3: 22, 4: 27, 5: 27, 6: 9,
    7: 4, 8: 6, 9: 38, 10: 49, 11: 36, 12: 24,
}

# Ocupacion hotelera Alicante provincia (patron estacional tipico, %)
# Fuente: INE, Encuesta de Ocupacion Hotelera
ALICANTE_MONTHLY_TOURISM = {
    1: 38, 2: 42, 3: 48, 4: 55, 5: 60, 6: 72,
    7: 82, 8: 85, 9: 68, 10: 55, 11: 42, 12: 40,
}

# Festivos nacionales + locales de Alicante
ALICANTE_HOLIDAYS = {
    # (mes, dia): nombre del festivo
    (1, 1): "Ano Nuevo",
    (1, 6): "Epifania",
    (3, 19): "San Jose (Comunitat Valenciana)",
    (5, 1): "Dia del Trabajador",
    (6, 24): "Hogueras de San Juan (Alicante)",
    (8, 15): "Asuncion de la Virgen",
    (10, 9): "Dia de la Comunitat Valenciana",
    (10, 12): "Fiesta Nacional",
    (11, 1): "Todos los Santos",
    (12, 6): "Dia de la Constitucion",
    (12, 8): "Inmaculada Concepcion",
    (12, 25): "Navidad",
}

# Festivos moviles (Semana Santa) — fechas aproximadas por ano
SEMANA_SANTA = {
    2022: (4, 14),  # Jueves Santo 2022
    2023: (4, 6),
    2024: (3, 28),
    2025: (4, 17),
    2026: (4, 2),
}

# Santa Faz (jueves despues de Semana Santa + 2 semanas, fiesta local Alicante)
SANTA_FAZ = {
    2022: (4, 28),
    2023: (4, 20),
    2024: (4, 11),
    2025: (5, 1),
    2026: (4, 16),
}


def load_aemet_data(start_date: str, end_date: str,
                    api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Temperatura media y precipitacion mensual para Alicante.

    Si se proporciona api_key, intenta descargar datos reales de AEMET OpenData.
    Si no, usa las medias climatologicas oficiales como fallback.

    Returns:
        DataFrame con columnas: [fecha, avg_temp, total_precip]
    """
    dates = _generate_monthly_dates(start_date, end_date)

    if api_key:
        real_data = _fetch_aemet_api(dates, api_key)
        if real_data is not None:
            return real_data

    # Fallback: medias climatologicas
    rows = []
    for d in dates:
        month = d.month
        rows.append({
            "fecha": d,
            "avg_temp": ALICANTE_MONTHLY_TEMP[month],
            "total_precip": ALICANTE_MONTHLY_PRECIP_MM[month],
        })
    return pd.DataFrame(rows)


def _fetch_aemet_api(dates: list, api_key: str) -> Optional[pd.DataFrame]:
    """Intenta descargar datos reales de AEMET. Retorna None si falla."""
    try:
        import requests
        # AEMET OpenData API — valores climatologicos mensuales
        # Estacion 8025 = Alicante/Alacant
        url = "https://opendata.aemet.es/opendata/api/valores/climatologicos/mensualesanuales/datos/anioini/{}/aniofin/{}/estacion/8025"
        start_year = min(d.year for d in dates)
        end_year = max(d.year for d in dates)

        resp = requests.get(
            url.format(start_year, end_year),
            headers={"api_key": api_key},
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        data_url = resp.json().get("datos")
        if not data_url:
            return None

        data_resp = requests.get(data_url, timeout=10)
        if data_resp.status_code != 200:
            return None

        records = data_resp.json()
        rows = []
        for r in records:
            try:
                year = int(r.get("anio", r.get("año", 0)))
                month = int(r.get("mes", 0))
                temp = float(str(r.get("tm_mes", "0")).replace(",", "."))
                precip = float(str(r.get("p_mes", "0")).replace(",", "."))
                fecha = pd.Timestamp(year=year, month=month, day=1)
                rows.append({"fecha": fecha, "avg_temp": temp, "total_precip": precip})
            except (ValueError, TypeError):
                continue

        if rows:
            df = pd.DataFrame(rows)
            # Filtrar al rango solicitado
            date_set = set(dates)
            df = df[df["fecha"].isin(date_set)]
            if len(df) > 0:
                return df

    except Exception:
        pass
    return None


def load_ine_tourism(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Ocupacion hotelera de Alicante provincia (%).

    Usa patron estacional tipico como datos base.
    Se podria mejorar con la API del INE para datos reales por ano.

    Returns:
        DataFrame con columnas: [fecha, tourist_occupancy_pct]
    """
    dates = _generate_monthly_dates(start_date, end_date)
    rows = []
    for d in dates:
        rows.append({
            "fecha": d,
            "tourist_occupancy_pct": float(ALICANTE_MONTHLY_TOURISM[d.month]),
        })
    return pd.DataFrame(rows)


def load_spei_index(start_date: str, end_date: str,
                    csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Indice de sequia SPEI para Alicante.

    Si se proporciona csv_path, carga datos reales descargados de spei.csic.es.
    Si no, retorna 0.0 (neutral) para todos los meses.

    SPEI interpretation:
        > 2.0   extremely wet
        1.0-2.0 moderately wet
        -1.0-1.0 normal
        -2.0--1.0 moderately dry
        < -2.0  extremely dry

    Returns:
        DataFrame con columnas: [fecha, spei_index]
    """
    dates = _generate_monthly_dates(start_date, end_date)

    if csv_path:
        try:
            df_spei = pd.read_csv(csv_path)
            # Intentar parsear — formato tipico de SPEI: columna de fecha + valor
            if "fecha" in df_spei.columns and "spei" in df_spei.columns:
                df_spei["fecha"] = pd.to_datetime(df_spei["fecha"])
                df_spei = df_spei.rename(columns={"spei": "spei_index"})
                date_set = set(dates)
                df_spei = df_spei[df_spei["fecha"].isin(date_set)]
                if len(df_spei) > 0:
                    return df_spei[["fecha", "spei_index"]]
        except Exception:
            pass

    # Fallback: 0.0 (neutral)
    return pd.DataFrame({"fecha": dates, "spei_index": 0.0})


def build_event_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Calendario de festivos y eventos de Alicante.

    Calcula por mes: cuantos festivos hay y si es un mes con evento importante
    (Hogueras de San Juan en junio, Semana Santa en marzo/abril).

    Returns:
        DataFrame con columnas: [fecha, n_holidays, is_holiday_month, has_major_event]
    """
    dates = _generate_monthly_dates(start_date, end_date)
    rows = []

    for d in dates:
        year = d.year
        month = d.month

        # Contar festivos fijos en este mes
        n_holidays = sum(1 for (m, _), _ in ALICANTE_HOLIDAYS.items() if m == month)

        # Semana Santa (variable)
        ss = SEMANA_SANTA.get(year)
        if ss and ss[0] == month:
            n_holidays += 2  # Jueves Santo + Viernes Santo

        # Santa Faz
        sf = SANTA_FAZ.get(year)
        if sf and sf[0] == month:
            n_holidays += 1

        # Eventos importantes (turismo alto)
        has_major_event = False
        if month == 6:  # Hogueras de San Juan
            has_major_event = True
        if ss and ss[0] == month:  # Semana Santa
            has_major_event = True

        rows.append({
            "fecha": d,
            "n_holidays": n_holidays,
            "is_holiday_month": n_holidays >= 2,
            "has_major_event": has_major_event,
        })

    return pd.DataFrame(rows)


def load_external_data(start_date: str, end_date: str,
                       aemet_api_key: Optional[str] = None,
                       spei_csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Punto de entrada principal. Carga y combina todas las fuentes externas.

    Args:
        start_date: fecha inicio (YYYY-MM-DD)
        end_date:   fecha fin (YYYY-MM-DD)
        aemet_api_key: API key de AEMET OpenData (opcional)
        spei_csv_path: ruta a CSV con datos SPEI (opcional)

    Returns:
        DataFrame indexado por mes con columnas:
        [fecha, avg_temp, total_precip, tourist_occupancy_pct,
         is_holiday_month, has_major_event, n_holidays, spei_index]
    """
    dates = _generate_monthly_dates(start_date, end_date)
    base = pd.DataFrame({"fecha": dates})

    # Cargar cada fuente independientemente
    df_aemet = load_aemet_data(start_date, end_date, api_key=aemet_api_key)
    df_tourism = load_ine_tourism(start_date, end_date)
    df_spei = load_spei_index(start_date, end_date, csv_path=spei_csv_path)
    df_calendar = build_event_calendar(start_date, end_date)

    # Merge todo sobre la fecha
    result = base
    for df_source in [df_aemet, df_tourism, df_spei, df_calendar]:
        if df_source is not None and len(df_source) > 0:
            result = result.merge(df_source, on="fecha", how="left")

    return result


def _generate_monthly_dates(start_date: str, end_date: str) -> list:
    """Genera lista de primer-dia-de-mes entre start y end."""
    start = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date).replace(day=1)
    return list(pd.date_range(start=start, end=end, freq="MS"))


# ─────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Cargando datos externos para Alicante (2022-2024)...\n")
    df = load_external_data("2022-01-01", "2024-12-31")
    print(f"Shape: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    print(f"\nPrimeros 6 meses:")
    print(df.head(6).to_string(index=False))
    print(f"\nUltimos 6 meses:")
    print(df.tail(6).to_string(index=False))
    print(f"\nMeses con evento importante:")
    events = df[df["has_major_event"] == True]
    print(events[["fecha", "n_holidays", "has_major_event"]].to_string(index=False))
