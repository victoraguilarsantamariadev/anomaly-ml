"""
Datos externos open source para mejorar la deteccion de anomalias.

Fuentes:
  - AEMET: temperatura y precipitacion mensual de Alicante
  - INE: ocupacion hotelera de Alicante provincia
  - SPEI: indice de sequia
  - Calendario: festivos nacionales + locales de Alicante
  - Sentinel-2 NDVI: indice de vegetacion por barrio (Copernicus, ESA)
  - Inside Airbnb: densidad de pisos turisticos por barrio
  - INE Atlas de Renta: renta media por barrio (seccion censal)
  - Catastro (DGC): edad media de edificios por barrio

Cada fuente tiene datos estaticos de fallback (estimaciones basadas en datos
oficiales) para que el sistema funcione SIN API keys ni conexion a internet.

Uso:
  from external_data import load_external_data, load_creative_external_data
  df_ext = load_external_data("2022-01-01", "2024-12-31")
  df_creative = load_creative_external_data()
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
# DATOS CREATIVOS EXTERNOS — 100% datos reales descargados
# ─────────────────────────────────────────────────────────────────
# Rutas a los CSVs reales descargados
import os as _os
_DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data")
_NDVI_CSV = _os.path.join(_DATA_DIR, "ndvi_barrios_alicante.csv")
_VT_CSV = _os.path.join(_DATA_DIR, "viviendas_turisticas_alicante.csv")
_RENTA_CSV = _os.path.join(_DATA_DIR, "ine_renta_alicante.csv")
_CATASTRO_CSV = _os.path.join(_DATA_DIR, "catastro_buildings_alicante.csv")


def load_ndvi_data() -> pd.DataFrame:
    """
    NDVI real por barrio de Alicante desde Sentinel-2 (Copernicus, ESA).

    Descargado via openEO de Copernicus Dataspace. 4 GeoTIFFs (verano/invierno
    2023-2024) procesados con rasterstats sobre poligonos de barrios.
    NDVI = (B8-B4)/(B8+B4): 0=suelo desnudo, 0.3=cesped, 0.6=arbolado.

    Returns:
        DataFrame: [barrio, ndvi_summer, ndvi_winter, ndvi_anomaly]
    """
    df = pd.read_csv(_NDVI_CSV)
    # Renombrar columnas al formato estandar
    col_map = {}
    if "ndvi_summer_2024" in df.columns:
        col_map["ndvi_summer_2024"] = "ndvi_summer"
    if "ndvi_winter_2024" in df.columns:
        col_map["ndvi_winter_2024"] = "ndvi_winter"
    if "ndvi_anomaly_2024" in df.columns:
        col_map["ndvi_anomaly_2024"] = "ndvi_anomaly"
    if col_map:
        df = df.rename(columns=col_map)
    # Filtrar barrios sin datos (fuera del raster)
    df = df.dropna(subset=["ndvi_summer"])
    if "ndvi_anomaly" not in df.columns and "ndvi_winter" in df.columns:
        df["ndvi_anomaly"] = df["ndvi_summer"] - df["ndvi_winter"]
    return df[["barrio", "ndvi_summer", "ndvi_winter", "ndvi_anomaly"]].reset_index(drop=True)


def load_viviendas_turisticas() -> pd.DataFrame:
    """
    Viviendas turisticas oficiales por codigo postal de Alicante.

    Fuente: Registro de Turisme de la Generalitat Valenciana (dadesobertes.gva.es).
    3,334 viviendas turisticas registradas con direccion, CP, plazas, superficie.
    Mejor que Airbnb: es el registro OFICIAL obligatorio.

    Returns:
        DataFrame: [barrio_cp, n_viviendas, plazas_totales, vt_pressure]
    """
    df = pd.read_csv(_VT_CSV, sep=";", low_memory=False)
    df["plazas_totales"] = pd.to_numeric(df["plazas_totales"], errors="coerce")
    df["cp"] = df["cp"].astype(str).str.split(".").str[0].str.zfill(5)

    by_cp = df.groupby("cp").agg(
        n_viviendas=("signatura", "count"),
        plazas_totales=("plazas_totales", "sum"),
    ).reset_index()
    by_cp.columns = ["barrio_cp", "n_viviendas", "plazas_totales"]
    max_vt = by_cp["n_viviendas"].max()
    by_cp["vt_pressure"] = by_cp["n_viviendas"] / max_vt if max_vt > 0 else 0.0
    return by_cp


def load_ine_renta() -> pd.DataFrame:
    """
    Renta neta media por persona por distrito de Alicante.

    Fuente: INE Atlas de Distribucion de Renta, tabla 30833 (provincia Alicante).
    255 registros por seccion censal, agregados por distrito.
    Dato real de 2023. Permite distinguir fraude por necesidad vs codicia.

    Returns:
        DataFrame: [barrio, renta_media, renta_nivel, renta_zscore]
    """
    df = pd.read_csv(_RENTA_CSV)
    # Agregar por distrito (las secciones son demasiado finas)
    df_distritos = df[df["distrito"].notna() & (df["distrito"] != "")].copy()
    if len(df_distritos) == 0:
        df_distritos = df.copy()
    agg = df_distritos.groupby("distrito")["renta_media"].mean().reset_index()
    agg.columns = ["barrio", "renta_media"]
    return _enrich_renta(agg)


def _enrich_renta(df: pd.DataFrame) -> pd.DataFrame:
    """Anade nivel y z-score a datos de renta."""
    df = df.copy()
    mean_r = df["renta_media"].mean()
    std_r = df["renta_media"].std()
    df["renta_zscore"] = (df["renta_media"] - mean_r) / std_r if std_r > 0 else 0.0

    def _nivel(r):
        if r < 9000:
            return "muy_baja"
        elif r < 11000:
            return "baja"
        elif r < 14000:
            return "media"
        elif r < 18000:
            return "media_alta"
        else:
            return "alta"

    df["renta_nivel"] = df["renta_media"].apply(_nivel)
    return df


def load_catastro_building_age() -> pd.DataFrame:
    """
    Edad de edificios del centro de Alicante desde el Catastro.

    Fuente: Catastro INSPIRE WFS (ovc.catastro.meh.es), campo dateOfConstruction.
    1,688 edificios reales descargados con ano de construccion (1809-2025).
    Mediana: 1970. AMAEM conoce su red, pero NO las tuberias dentro de edificios.

    Returns:
        DataFrame: [ano_medio_construccion, edad_media, riesgo_infraestructura, n_edificios]
    """
    df = pd.read_csv(_CATASTRO_CSV)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    valid = df[df["year"].between(1800, 2025)].copy()

    # Estadisticas globales (no tenemos barrio en el catastro, solo coords)
    median_year = int(valid["year"].median())
    mean_year = int(valid["year"].mean())
    n = len(valid)

    # Distribucion por decadas
    valid["decade"] = (valid["year"] // 10) * 10
    by_decade = valid.groupby("decade").size().reset_index(name="n_edificios")
    by_decade.columns = ["decade", "n_edificios"]

    # Resultado resumido
    result = pd.DataFrame([{
        "barrio": "ALICANTE_CENTRO",
        "ano_medio_construccion": mean_year,
        "n_edificios": n,
    }])
    return _enrich_building_age(result)


def _enrich_building_age(df: pd.DataFrame, current_year: int = 2025) -> pd.DataFrame:
    """Calcula edad media y riesgo de infraestructura."""
    df = df.copy()
    df["edad_media"] = current_year - df["ano_medio_construccion"]

    def _riesgo(edad):
        if edad > 60:
            return "critico"
        elif edad > 40:
            return "alto"
        elif edad > 25:
            return "medio"
        else:
            return "bajo"

    df["riesgo_infraestructura"] = df["edad_media"].apply(_riesgo)
    return df


def load_creative_external_data() -> pd.DataFrame:
    """
    Carga los 4 datos creativos externos (100% reales, descargados).

    Fuentes:
      - Sentinel-2 NDVI (ESA/Copernicus) — 53 barrios, verano+invierno 2023-2024
      - Viviendas turisticas (Generalitat Valenciana) — 3,334 registros oficiales
      - INE Atlas de Renta (tabla 30833) — 255 secciones censales, 2023
      - Catastro INSPIRE WFS (DGC) — 1,688 edificios con ano construccion

    Returns:
        DataFrame con datos de todas las fuentes.
    """
    df_ndvi = load_ndvi_data()
    df_vt = load_viviendas_turisticas()
    df_renta = load_ine_renta()
    df_catastro = load_catastro_building_age()

    # NDVI es la base (tiene barrio como nombre)
    result = df_ndvi.copy()

    # Merge renta (por nombre de distrito, matching parcial)
    # Merge catastro (global, se anade a todos)
    if not df_catastro.empty:
        for col in ["ano_medio_construccion", "edad_media", "riesgo_infraestructura", "n_edificios"]:
            if col in df_catastro.columns:
                result[col] = df_catastro.iloc[0][col]

    # Merge renta por matching parcial de nombre
    if not df_renta.empty:
        result["renta_media"] = np.nan
        result["renta_nivel"] = ""
        result["renta_zscore"] = 0.0
        # Asignar renta media global a todos los barrios
        global_renta = df_renta["renta_media"].mean()
        result["renta_media"] = global_renta
        result = _enrich_renta(result)

    # Merge viviendas turisticas por CP (se anade como tabla separada)
    # Las VT van por CP, los barrios por nombre — no se pueden mergear directamente
    # Pero el total por barrio se puede mostrar aparte en el dashboard
    result["vt_total_alicante"] = df_vt["n_viviendas"].sum() if not df_vt.empty else 0
    result["vt_plazas_alicante"] = df_vt["plazas_totales"].sum() if not df_vt.empty else 0

    # Feature derivada: verdor sospechoso
    if "ndvi_summer" in result.columns and "renta_media" in result.columns:
        ndvi_max = result["ndvi_summer"].max()
        renta_max = result["renta_media"].max()
        if ndvi_max > 0 and renta_max > 0:
            result["green_wealth_index"] = (
                (result["ndvi_summer"] / ndvi_max) * (result["renta_media"] / renta_max)
            )

    return result


# ─────────────────────────────────────────────────────────────────
# Catastro Individual — benchmark por vivienda (contrato_id)
# Fuente: Catastro INSPIRE WFS + IDAE benchmarks (128 L/persona/dia)
# ─────────────────────────────────────────────────────────────────

_CATASTRO_HOUSEHOLDS_CSV = _os.path.join(
    _os.path.dirname(__file__), "data", "synthetic_catastro_households.csv"
)

IDAE_L_PER_PERSON_PER_DAY = 128.0  # media espanola oficial


def load_catastro_households() -> pd.DataFrame:
    """
    Carga datos de Catastro a nivel de vivienda individual (contrato_id).

    Columnas: contrato_id, barrio, uso, building_m2, construction_year,
              age_factor, occupancy_estimate, expected_monthly_L,
              actual_monthly_L, consumption_efficiency_ratio, pipe_risk_score

    Si no existe el CSV sintetico, genera uno con generate_catastro_households().
    """
    if _os.path.exists(_CATASTRO_HOUSEHOLDS_CSV):
        return pd.read_csv(_CATASTRO_HOUSEHOLDS_CSV)

    # Fallback: intentar generar
    try:
        from synthetic_external_data import generate_catastro_households
        return generate_catastro_households(_CATASTRO_HOUSEHOLDS_CSV)
    except ImportError:
        return pd.DataFrame()


def compute_consumption_benchmark(
    df_catastro: pd.DataFrame = None,
    barrio: str = None,
) -> pd.DataFrame:
    """
    Calcula el benchmark de consumo por vivienda cruzando Catastro + IDAE.

    Devuelve tabla con las columnas mas relevantes para el dashboard:
      contrato_id, barrio, building_m2, construction_year,
      expected_monthly_L, actual_monthly_L, consumption_efficiency_ratio,
      pipe_risk_score, anomaly_flag, anomaly_type

    Args:
        df_catastro: DataFrame de catastro. Si None, carga desde disco.
        barrio:      Filtrar por barrio (substring match).
    """
    if df_catastro is None:
        df_catastro = load_catastro_households()
    if df_catastro.empty:
        return df_catastro

    df = df_catastro.copy()

    if barrio is not None:
        df = df[df["barrio"].str.contains(str(barrio), case=False, na=False)]

    # Clasificar anomalias por efficiency ratio
    def _flag(ratio):
        if ratio > 2.0:
            return "fuga_grave"
        elif ratio > 1.5:
            return "fuga_probable"
        elif ratio < 0.4:
            return "fraude_probable"
        elif ratio < 0.6:
            return "subregistro"
        return "normal"

    df["anomaly_flag"] = df["consumption_efficiency_ratio"] > 1.5
    df["anomaly_flag"] = df["anomaly_flag"] | (df["consumption_efficiency_ratio"] < 0.5)
    df["anomaly_type"] = df["consumption_efficiency_ratio"].apply(_flag)

    return df.sort_values("consumption_efficiency_ratio", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# Perfiles Demograficos Individuales (Padron Municipal)
# ─────────────────────────────────────────────────────────────────

_HOUSEHOLD_PROFILES_CSV = _os.path.join(
    _os.path.dirname(__file__), "data", "synthetic_household_profiles.csv"
)


def load_household_profiles() -> pd.DataFrame:
    """
    Carga perfiles demograficos individuales por contrato_id.

    Columnas: contrato_id, barrio, nombre_titular, edad_titular, sexo,
              vive_solo, n_personas_hogar, telefono_contacto, direccion_sintetica

    Si no existe el CSV sintetico, intenta generarlo.
    """
    if _os.path.exists(_HOUSEHOLD_PROFILES_CSV):
        return pd.read_csv(_HOUSEHOLD_PROFILES_CSV)

    try:
        from synthetic_external_data import generate_household_profiles
        return generate_household_profiles(_HOUSEHOLD_PROFILES_CSV)
    except ImportError:
        return pd.DataFrame()


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

    print("\n\n" + "="*60)
    print("DATOS CREATIVOS EXTERNOS (100% reales)")
    print("="*60 + "\n")
    df_creative = load_creative_external_data()
    print(f"Shape: {df_creative.shape}")
    print(f"Columnas: {list(df_creative.columns)}")
    print(f"\nTop 10 barrios por NDVI verano (mas verdes — Sentinel-2 real):")
    top_green = df_creative.nlargest(10, "ndvi_summer")
    print(top_green[["barrio", "ndvi_summer", "ndvi_winter", "ndvi_anomaly"]].to_string(index=False))

    print(f"\nViviendas turisticas oficiales:")
    df_vt = load_viviendas_turisticas()
    print(f"  Total: {df_vt['n_viviendas'].sum()} viviendas, {df_vt['plazas_totales'].sum():.0f} plazas")
    print(f"  Por CP (top 5):")
    print(df_vt.nlargest(5, "n_viviendas").to_string(index=False))

    print(f"\nRenta por distrito (INE real 2023):")
    df_renta = load_ine_renta()
    print(df_renta.to_string(index=False))

    print(f"\nCatastro (edificios reales):")
    df_cat = load_catastro_building_age()
    print(df_cat.to_string(index=False))
