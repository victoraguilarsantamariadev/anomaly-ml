"""
Extraccion de features de infraestructura GIS y datos auxiliares.

Carga datos de:
  - sectores_de_consumo.json: sectores con poligonos, elevacion, AMAEMID
  - tuberias.json: red de saneamiento (diametro, clasificacion)
  - redes_primarias.json: red de abastecimiento primaria (diametro)
  - redes_arteriales.json: red arterial (diametro)
  - bocasriego_hidrantes.json: hidrantes y bocas de riego (puntos)
  - centros_de_bombeo.json: estaciones de bombeo (poligonos)
  - depositos.json: depositos de almacenamiento (capacidad)
  - altas-por-poblacion: altas/bajas de contratos por mes
  - longitud-red: km de red y cobertura de inspeccion

Todos los GeoJSON estan en formato ESRI JSON (EPSG:25830 = UTM zona 30N).
Las coordenadas estan en metros, lo que facilita calculos de distancia/longitud.

Uso:
  from gis_features import load_infrastructure_features, load_contract_growth
  infra_df = load_infrastructure_features("data/")
  growth_df = load_contract_growth("data/altas-por-poblacion-....csv")
"""

import json
import math
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# Mapeo sector de consumo → barrio del hackathon
# Multiples sectores pueden mapear al mismo barrio.
# Sectores sin match = None (zonas industriales, arteriales, etc.)
# ─────────────────────────────────────────────────────────────────

SECTOR_TO_BARRIO = {
    "SAN GABRIEL": "13-SAN GABRIEL",
    "SANTO DOMINGO": "24-SAN BLAS - SANTO DOMINGO",
    "INFORMACION": "14-ENSANCHE DIPUTACION",
    "FLORIDA": "10-FLORIDA BAJA",
    "REBOLLEDO": "REBOLLEDO",
    "PLA DE LA VALLONGA": "54-POLIGONO VALLONGA",
    "SAN AGUSTIN": "7-SAN AGUSTIN",
    "RABASA": "20-RABASA",
    "ATALAYAS": "53-POLIGONO ATALAYAS",
    "BENALUA": "1-BENALUA",
    "CABO HUERTAS": "40-CABO DE LAS HUERTAS",
    "CABO INTERIOR": "40-CABO DE LAS HUERTAS",
    "CAMPOAMOR ALTO": "5-CAMPOAMOR",
    "CAMPOAMOR BAJO": "5-CAMPOAMOR",
    "CAROLINAS ALTAS": "17-CAROLINAS ALTAS",
    "CASCO ANTIGUO": "22-CASCO ANTIGUO - SANTA CRUZ",
    "CIUDAD DE ASIS": "11-CIUDAD DE ASIS",
    "CIUDAD JARDIN": "31-CIUDAD JARDIN",
    "COLONIA REQUENA": "34-COLONIA REQUENA",
    "DIPUTACION": "14-ENSANCHE DIPUTACION",
    "DIVINA PASTORA": "30-DIVINA PASTORA",
    "GARBINET ESTE": "19-GARBINET",
    "GARBINET NORTE": "19-GARBINET",
    "GARBINET OESTE": "19-GARBINET",
    "ALBUFERETA": "39-ALBUFERETA",
    "ALBUFERETA - ADOC": "39-ALBUFERETA",
    "ALBUFERETA - CONCHA ESPINA": "39-ALBUFERETA",
    "ALBUFERETA - TELEFONICA": "39-ALBUFERETA",
    "ALBUFERETA - VIA PARQUE": "39-ALBUFERETA",
    "ALIPARK": "8-ALIPARK",
    "ALTOZANO": "25-ALTOZANO - CONDE LUMIARES",
    "JUAN XXIII 2 FASE (MIRADORES)": "37-JUAN XXIII",
    "JUAN XXIII 2 FASE (MODULOS)": "37-JUAN XXIII",
    "LA TOMBOLA": "21-TOMBOLA",
    "LES PALMERETES": "28-EL PALMERAL",
    "PALMERAL": "28-EL PALMERAL",
    "LO MORANT": "33- MORANT -SAN NICOLAS BARI",
    "LOS ANGELES": "6-LOS ANGELES",
    "MERCADO CENTRAL": "4-MERCADO",
    "MIL VIVIENDAS": "36-CUATROCIENTAS VIVIENDAS",
    "NOU ALACANT": "26-SIDI IFNI - NOU ALACANT",
    "PLA DEL BON REPOS": "16-PLA DEL BON REPOS",
    "PLA HOSPITAL": "16-PLA DEL BON REPOS",
    "PLAYA SAN JUAN": "41-PLAYA DE SAN JUAN",
    "PAU 1 NORTE": "41-PLAYA DE SAN JUAN",
    "PAU 1 SUR": "41-PLAYA DE SAN JUAN",
    "PAU 2": "41-PLAYA DE SAN JUAN",
    "PAU 3": "41-PLAYA DE SAN JUAN",
    "POSTIGUET": "22-CASCO ANTIGUO - SANTA CRUZ",
    "SAN BLAS": "24-SAN BLAS - SANTO DOMINGO",
    "URBANOVA": "29-URBANOVA",
    "VERDEGAS": "VERDEGAS",
    "VIRGEN DEL REMEDIO": "32-VIRGEN DEL REMEDIO",
    "VISTAHERMOSA": "38-VISTAHERMOSA",
    "VISTAHERMOSA-EL CHOPO": "38-VISTAHERMOSA",
    "BACAROT": "BACAROT",
    "MORALET": "MORALET",
    "ALCORAYA": "LA ALCORAYA",
    "POLIGONO D (INDUSTRIAL)": "12-POLIGONO BABEL",
    "POLIGONO D (RESIDENCIAL)": "12-POLIGONO BABEL",
    "PLAZA DE LA MONTANETA": "3-CENTRO",
    "ALFONSO EL SABIO": "3-CENTRO",
    "RAMBLA": "3-CENTRO",
    "AVDA. MARVA": "3-CENTRO",
    "AVDA. SALAMANCA": "2-SAN ANTON",
    "BOTANICO": "2-SAN ANTON",
    "FENOLLAR": "2-SAN ANTON",
    "BARRIO OBRERO": "18-CAROLINAS BAJAS",
    "CASTALLA - DON JAIME": "17-CAROLINAS ALTAS",
    "CASTALLA - POLVORINES": "17-CAROLINAS ALTAS",
    "CUBETA REQUENA": "34-COLONIA REQUENA",
    "DR BERGEZ": "9-FLORIDA ALTA",
    "DR SAPENA": "9-FLORIDA ALTA",
    "SANTA FELICITAS": "9-FLORIDA ALTA",
    "HOTEL MAYA - V. SOCORRO": "23-RAVAL ROIG -V. DEL SOCORRO",
    "LA GOTETA": "23-RAVAL ROIG -V. DEL SOCORRO",
    "MUELLE DE GRANELES": "55-PUERTO",
    "MUELLE DE LEVANTE": "55-PUERTO",
    "MUELLE DE PONIENTE": "55-PUERTO",
    "CONDOMINA": "5-CAMPOAMOR",
    "GRAN VIA - SANTA POLA": "10-FLORIDA BAJA",
    "GRAN VIA JESUITAS": "10-FLORIDA BAJA",
    "JESUITAS": "10-FLORIDA BAJA",
    "NAZARET": "33- MORANT -SAN NICOLAS BARI",
    "VALLONGA VIVIENDAS": "54-POLIGONO VALLONGA",
    "MONTOTO": "19-GARBINET",
    "HOSPITAL GENERAL": "16-PLA DEL BON REPOS",
    "POLIGONO GARRACHICO": "15-POLIGONO SAN BLAS",
    "RICO PEREZ": "15-POLIGONO SAN BLAS",
    "MONTEMAR": "41-PLAYA DE SAN JUAN",
    "ORGEGIA": "27-SAN FERNANDO-PRIN. MERCEDES",
    "CRUZ DE PIEDRA": "27-SAN FERNANDO-PRIN. MERCEDES",
    "PISTILO": "28-EL PALMERAL",
    "LODE DIE": "5-CAMPOAMOR",
    "MONCHET": "5-CAMPOAMOR",
    "EL AGUILA": "6-LOS ANGELES",
    "ZONA INDUSTRIAL RABASA": "20-RABASA",
    "POLITECNICO": "20-RABASA",
    "CASAFUS": "56-DISPERSOS",
    "CROSS": "56-DISPERSOS",
    "PIEL DEL OSO": "56-DISPERSOS",
    "ISLA DE CORFU": "56-DISPERSOS",
}


# ─────────────────────────────────────────────────────────────────
# Geometria basica (sin dependencia de shapely)
# ─────────────────────────────────────────────────────────────────

def _polygon_area(ring: list[list[float]]) -> float:
    """Area de un poligono usando la formula del zapato (Shoelace).
    Coordenadas en metros (UTM) → area en m²."""
    n = len(ring)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += ring[i][0] * ring[j][1]
        area -= ring[j][0] * ring[i][1]
    return abs(area) / 2.0


def _polygon_centroid(ring: list[list[float]]) -> tuple[float, float]:
    """Centroide de un poligono. Retorna (x, y)."""
    n = len(ring)
    if n == 0:
        return (0.0, 0.0)
    cx = sum(p[0] for p in ring) / n
    cy = sum(p[1] for p in ring) / n
    return (cx, cy)


def _point_in_polygon(px: float, py: float, ring: list[list[float]]) -> bool:
    """Test punto-en-poligono via ray casting."""
    n = len(ring)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _polyline_length(paths: list[list[list[float]]]) -> float:
    """Longitud total de una polylinea (paths = lista de segmentos).
    Coordenadas en metros → longitud en metros."""
    total = 0.0
    for path in paths:
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            total += math.sqrt(dx * dx + dy * dy)
    return total


def _polyline_midpoint(paths: list[list[list[float]]]) -> tuple[float, float]:
    """Punto medio aproximado de una polylinea."""
    all_pts = [p for path in paths for p in path]
    if not all_pts:
        return (0.0, 0.0)
    mid_idx = len(all_pts) // 2
    return (all_pts[mid_idx][0], all_pts[mid_idx][1])


# ─────────────────────────────────────────────────────────────────
# Carga de datos ESRI JSON
# ─────────────────────────────────────────────────────────────────

def _load_esri_json(path: str) -> dict:
    """Carga un archivo ESRI JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_sector_polygons(data_dir: str) -> list[dict]:
    """
    Carga sectores de consumo como lista de dicts con:
      name, barrio, amaemid, elevation_min/mean/max,
      area_m2, centroid_x, centroid_y, rings
    """
    path = os.path.join(data_dir, "sectores_de_consumo.json")
    if not os.path.exists(path):
        return []

    data = _load_esri_json(path)
    sectors = []

    for feat in data["features"]:
        attr = feat["attributes"]
        geom = feat.get("geometry", {})
        rings = geom.get("rings", [])
        name = attr.get("DCONS_PO_2", "")

        # Usar el primer anillo (exterior) para area y centroide
        outer_ring = rings[0] if rings else []
        area = _polygon_area(outer_ring)
        cx, cy = _polygon_centroid(outer_ring)

        sectors.append({
            "name": name,
            "barrio": SECTOR_TO_BARRIO.get(name),
            "amaemid": attr.get("AMAEMID", 0),
            "elevation_min": attr.get("COTA_MINIM", 0),
            "elevation_mean": attr.get("COTA_MEDIA", 0),
            "elevation_max": attr.get("COTA_MAXIM", 0),
            "area_m2": area,
            "centroid_x": cx,
            "centroid_y": cy,
            "outer_ring": outer_ring,
        })

    return sectors


def _find_sector_for_point(px: float, py: float,
                            sectors: list[dict]) -> Optional[str]:
    """Encuentra el sector que contiene un punto. Retorna nombre o None."""
    for sector in sectors:
        ring = sector["outer_ring"]
        if ring and _point_in_polygon(px, py, ring):
            return sector["name"]
    return None


def count_hydrants_per_sector(data_dir: str,
                               sectors: list[dict]) -> dict[str, dict]:
    """
    Cuenta hidrantes y bocas de riego por sector.
    Retorna dict[sector_name] → {n_hydrants, n_bocas_riego, avg_diameter}
    """
    path = os.path.join(data_dir, "bocasriego_hidrantes.json")
    if not os.path.exists(path):
        return {}

    data = _load_esri_json(path)
    counts: dict[str, list] = {}

    for feat in data["features"]:
        attr = feat["attributes"]
        geom = feat.get("geometry", {})
        x, y = geom.get("x", 0), geom.get("y", 0)
        if x == 0 and y == 0:
            continue

        sector_name = _find_sector_for_point(x, y, sectors)
        if sector_name:
            if sector_name not in counts:
                counts[sector_name] = []
            counts[sector_name].append({
                "tipo": attr.get("d_TIPO", ""),
                "diametro": attr.get("DIAMETRO", 0),
            })

    result = {}
    for sector_name, items in counts.items():
        n_hydrants = sum(1 for i in items if "HIDRANTE" in str(i["tipo"]).upper())
        n_bocas = sum(1 for i in items if "BOCA" in str(i["tipo"]).upper() or "HIDRANTE" not in str(i["tipo"]).upper())
        diameters = [i["diametro"] for i in items if i["diametro"] > 0]
        result[sector_name] = {
            "n_hydrants": n_hydrants,
            "n_bocas_riego": n_bocas,
            "n_total_riego": len(items),
            "avg_hydrant_diameter": np.mean(diameters) if diameters else 0,
        }

    return result


def count_pumping_stations_per_sector(data_dir: str,
                                       sectors: list[dict]) -> dict[str, int]:
    """Cuenta estaciones de bombeo por sector."""
    path = os.path.join(data_dir, "centros_de_bombeo.json")
    if not os.path.exists(path):
        return {}

    data = _load_esri_json(path)
    counts: dict[str, int] = {}

    for feat in data["features"]:
        geom = feat.get("geometry", {})
        # Bombeos son poligonos — usar centroide
        rings = geom.get("rings", [])
        if not rings or not rings[0]:
            continue
        cx, cy = _polygon_centroid(rings[0])
        sector_name = _find_sector_for_point(cx, cy, sectors)
        if sector_name:
            counts[sector_name] = counts.get(sector_name, 0) + 1

    return counts


def sum_deposit_capacity_per_sector(data_dir: str,
                                     sectors: list[dict]) -> dict[str, float]:
    """Suma capacidad de depositos (m³) por sector."""
    path = os.path.join(data_dir, "depositos.json")
    if not os.path.exists(path):
        return {}

    data = _load_esri_json(path)
    caps: dict[str, float] = {}

    for feat in data["features"]:
        attr = feat["attributes"]
        geom = feat.get("geometry", {})
        x, y = geom.get("x", 0), geom.get("y", 0)

        # depositos pueden ser puntos o poligonos
        if x == 0 and y == 0:
            rings = geom.get("rings", [])
            if rings and rings[0]:
                x, y = _polygon_centroid(rings[0])

        if x == 0 and y == 0:
            continue

        sector_name = _find_sector_for_point(x, y, sectors)
        if sector_name:
            caps[sector_name] = caps.get(sector_name, 0) + attr.get("CAPACIDAD", 0)

    return caps


def compute_pipe_stats_per_sector(data_dir: str,
                                   sectors: list[dict]) -> dict[str, dict]:
    """
    Calcula estadisticas de tuberias de abastecimiento por sector.
    Usa redes_primarias.json y redes_arteriales.json (red de agua potable).

    Retorna dict[sector_name] → {total_length_m, avg_diameter, n_segments}
    """
    result: dict[str, dict] = {}

    for filename, label in [
        ("redes_primarias.json", "primaria"),
        ("redes_arteriales.json", "arterial"),
    ]:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            continue

        data = _load_esri_json(path)
        for feat in data["features"]:
            attr = feat["attributes"]
            geom = feat.get("geometry", {})
            paths = geom.get("paths", [])
            if not paths:
                continue

            length = _polyline_length(paths)
            mx, my = _polyline_midpoint(paths)
            diameter = attr.get("DIAMETRO", 0)

            sector_name = _find_sector_for_point(mx, my, sectors)
            if sector_name:
                if sector_name not in result:
                    result[sector_name] = {
                        "total_length_m": 0,
                        "diameters": [],
                        "n_segments": 0,
                    }
                result[sector_name]["total_length_m"] += length
                result[sector_name]["n_segments"] += 1
                if diameter > 0:
                    result[sector_name]["diameters"].append(diameter)

    # Calcular promedios
    for sector_name, stats in result.items():
        diams = stats.pop("diameters")
        stats["avg_diameter_mm"] = np.mean(diams) if diams else 0

    return result


def count_imbornales_per_sector(data_dir: str,
                                 sectors: list[dict]) -> dict[str, int]:
    """
    Cuenta imbornales (sumideros de drenaje) por sector.
    imbornal.json tiene 11,956 puntos.
    """
    path = os.path.join(data_dir, "imbornal.json")
    if not os.path.exists(path):
        return {}

    data = _load_esri_json(path)
    counts: dict[str, int] = {}

    for feat in data["features"]:
        geom = feat.get("geometry", {})
        x, y = geom.get("x", 0), geom.get("y", 0)
        if x == 0 and y == 0:
            continue

        sector_name = _find_sector_for_point(x, y, sectors)
        if sector_name:
            counts[sector_name] = counts.get(sector_name, 0) + 1

    return counts


def compute_colector_stats_per_sector(data_dir: str,
                                       sectors: list[dict]) -> dict[str, dict]:
    """
    Estadisticas de grandes colectores por sector.
    grandes_colectores.json tiene 1,741 poligonos con d_CLASIFIC.
    """
    path = os.path.join(data_dir, "grandes_colectores.json")
    if not os.path.exists(path):
        return {}

    data = _load_esri_json(path)
    result: dict[str, dict] = {}

    for feat in data["features"]:
        attr = feat["attributes"]
        geom = feat.get("geometry", {})
        rings = geom.get("rings", [])
        if not rings or not rings[0]:
            continue

        # Usar centroide del poligono para asignar a sector
        cx, cy = _polygon_centroid(rings[0])
        area = _polygon_area(rings[0])
        clasif = attr.get("d_CLASIFIC", "")

        sector_name = _find_sector_for_point(cx, cy, sectors)
        if sector_name:
            if sector_name not in result:
                result[sector_name] = {
                    "colector_area_m2": 0,
                    "n_colectores": 0,
                    "n_colector_pluvial": 0,
                    "n_colector_unitaria": 0,
                }
            result[sector_name]["colector_area_m2"] += area
            result[sector_name]["n_colectores"] += 1
            if "PLUVIAL" in clasif:
                result[sector_name]["n_colector_pluvial"] += 1
            elif "UNITARIA" in clasif:
                result[sector_name]["n_colector_unitaria"] += 1

    return result


def compute_sewer_stats_per_sector(data_dir: str,
                                    sectors: list[dict]) -> dict[str, dict]:
    """
    Estadisticas de red de saneamiento (tuberias.json) por sector.
    Retorna dict[sector_name] → {sewer_length_m, n_sewer_segments, pct_unitaria}
    """
    path = os.path.join(data_dir, "tuberias.json")
    if not os.path.exists(path):
        return {}

    data = _load_esri_json(path)
    result: dict[str, dict] = {}

    for feat in data["features"]:
        attr = feat["attributes"]
        geom = feat.get("geometry", {})
        paths = geom.get("paths", [])
        if not paths:
            continue

        length = _polyline_length(paths)
        mx, my = _polyline_midpoint(paths)
        clasif = attr.get("d_CLASIFIC", "")

        sector_name = _find_sector_for_point(mx, my, sectors)
        if sector_name:
            if sector_name not in result:
                result[sector_name] = {
                    "sewer_length_m": 0,
                    "n_sewer_segments": 0,
                    "n_unitaria": 0,
                    "n_pluvial": 0,
                }
            result[sector_name]["sewer_length_m"] += length
            result[sector_name]["n_sewer_segments"] += 1
            if clasif == "UNITARIA":
                result[sector_name]["n_unitaria"] += 1
            elif clasif == "PLUVIAL":
                result[sector_name]["n_pluvial"] += 1

    # Calcular % unitaria
    for stats in result.values():
        total = stats["n_sewer_segments"]
        stats["pct_unitaria"] = stats["n_unitaria"] / total if total > 0 else 0
        del stats["n_unitaria"]
        del stats["n_pluvial"]

    return result


# ─────────────────────────────────────────────────────────────────
# Agregacion por barrio
# ─────────────────────────────────────────────────────────────────

def load_infrastructure_features(data_dir: str = "data/") -> pd.DataFrame:
    """
    Punto de entrada principal. Carga todos los GeoJSON y agrega por barrio.

    Retorna DataFrame con una fila por barrio y columnas:
      - elevation_mean, elevation_range: topografia
      - area_km2: area total del barrio (sum de sectores)
      - pipe_density_km_per_km2: km de red de abastecimiento / km² de area
      - avg_pipe_diameter_mm: diametro medio de tuberias de abastecimiento
      - sewer_density_km_per_km2: km de red de saneamiento / km²
      - pct_sewer_unitaria: % de red unitaria (vs separativa pluvial)
      - n_hydrants: numero de hidrantes
      - n_bocas_riego: numero de bocas de riego
      - hydrant_density_per_km2: hidrantes + bocas por km²
      - n_pumping_stations: estaciones de bombeo
      - deposit_capacity_m3: capacidad total de depositos
    """
    print(f"  Cargando infraestructura GIS de {data_dir}...")

    # 1. Cargar sectores (poligonos base)
    sectors = load_sector_polygons(data_dir)
    if not sectors:
        print("    WARNING: no se encontro sectores_de_consumo.json")
        return pd.DataFrame()

    mapped = sum(1 for s in sectors if s["barrio"])
    print(f"    {len(sectors)} sectores, {mapped} mapeados a barrios")

    # 2. Cargar infraestructura por sector
    print("    Contando hidrantes/bocas de riego...")
    hydrants = count_hydrants_per_sector(data_dir, sectors)

    print("    Contando estaciones de bombeo...")
    pumps = count_pumping_stations_per_sector(data_dir, sectors)

    print("    Calculando capacidad de depositos...")
    deposits = sum_deposit_capacity_per_sector(data_dir, sectors)

    print("    Calculando longitud de red de abastecimiento...")
    pipes = compute_pipe_stats_per_sector(data_dir, sectors)

    print("    Calculando longitud de red de saneamiento...")
    sewers = compute_sewer_stats_per_sector(data_dir, sectors)

    print("    Contando imbornales (sumideros)...")
    imbornales = count_imbornales_per_sector(data_dir, sectors)

    print("    Calculando grandes colectores...")
    colectores = compute_colector_stats_per_sector(data_dir, sectors)

    # 3. Agregar todo por sector
    sector_data = []
    for s in sectors:
        name = s["name"]
        barrio = s["barrio"]
        if not barrio:
            continue

        h = hydrants.get(name, {})
        p = pipes.get(name, {})
        sw = sewers.get(name, {})
        col = colectores.get(name, {})

        sector_data.append({
            "barrio": barrio,
            "sector_name": name,
            "area_m2": s["area_m2"],
            "elevation_min": s["elevation_min"],
            "elevation_mean": s["elevation_mean"],
            "elevation_max": s["elevation_max"],
            # Hidrantes
            "n_hydrants": h.get("n_hydrants", 0),
            "n_bocas_riego": h.get("n_bocas_riego", 0),
            "n_total_riego": h.get("n_total_riego", 0),
            # Bombeos
            "n_pumping_stations": pumps.get(name, 0),
            # Depositos
            "deposit_capacity_m3": deposits.get(name, 0),
            # Tuberias de abastecimiento
            "pipe_length_m": p.get("total_length_m", 0),
            "avg_pipe_diameter_mm": p.get("avg_diameter_mm", 0),
            "n_pipe_segments": p.get("n_segments", 0),
            # Saneamiento
            "sewer_length_m": sw.get("sewer_length_m", 0),
            "pct_sewer_unitaria": sw.get("pct_unitaria", 0),
            # Imbornales (sumideros de drenaje)
            "n_imbornales": imbornales.get(name, 0),
            # Grandes colectores
            "colector_area_m2": col.get("colector_area_m2", 0),
            "n_colectores": col.get("n_colectores", 0),
        })

    if not sector_data:
        return pd.DataFrame()

    df_sectors = pd.DataFrame(sector_data)

    # 4. Agregar por barrio (sum areas, sum longitudes, mean elevaciones)
    barrio_agg = df_sectors.groupby("barrio").agg(
        area_m2=("area_m2", "sum"),
        elevation_mean=("elevation_mean", "mean"),
        elevation_min=("elevation_min", "min"),
        elevation_max=("elevation_max", "max"),
        n_hydrants=("n_hydrants", "sum"),
        n_bocas_riego=("n_bocas_riego", "sum"),
        n_total_riego=("n_total_riego", "sum"),
        n_pumping_stations=("n_pumping_stations", "sum"),
        deposit_capacity_m3=("deposit_capacity_m3", "sum"),
        pipe_length_m=("pipe_length_m", "sum"),
        avg_pipe_diameter_mm=("avg_pipe_diameter_mm", "mean"),
        n_pipe_segments=("n_pipe_segments", "sum"),
        sewer_length_m=("sewer_length_m", "sum"),
        pct_sewer_unitaria=("pct_sewer_unitaria", "mean"),
        n_imbornales=("n_imbornales", "sum"),
        colector_area_m2=("colector_area_m2", "sum"),
        n_colectores=("n_colectores", "sum"),
        n_sectors=("sector_name", "count"),
    ).reset_index()

    # 5. Calcular densidades
    barrio_agg["area_km2"] = barrio_agg["area_m2"] / 1e6
    barrio_agg["elevation_range"] = barrio_agg["elevation_max"] - barrio_agg["elevation_min"]

    barrio_agg["pipe_density_km_per_km2"] = np.where(
        barrio_agg["area_km2"] > 0,
        (barrio_agg["pipe_length_m"] / 1000) / barrio_agg["area_km2"],
        0,
    )
    barrio_agg["sewer_density_km_per_km2"] = np.where(
        barrio_agg["area_km2"] > 0,
        (barrio_agg["sewer_length_m"] / 1000) / barrio_agg["area_km2"],
        0,
    )
    barrio_agg["hydrant_density_per_km2"] = np.where(
        barrio_agg["area_km2"] > 0,
        barrio_agg["n_total_riego"] / barrio_agg["area_km2"],
        0,
    )
    barrio_agg["imbornal_density_per_km2"] = np.where(
        barrio_agg["area_km2"] > 0,
        barrio_agg["n_imbornales"] / barrio_agg["area_km2"],
        0,
    )
    barrio_agg["colector_coverage_pct"] = np.where(
        barrio_agg["area_m2"] > 0,
        barrio_agg["colector_area_m2"] / barrio_agg["area_m2"] * 100,
        0,
    )

    # Columnas finales
    cols = [
        "barrio", "area_km2", "elevation_mean", "elevation_range",
        "pipe_density_km_per_km2", "avg_pipe_diameter_mm",
        "sewer_density_km_per_km2", "pct_sewer_unitaria",
        "n_hydrants", "n_bocas_riego", "hydrant_density_per_km2",
        "n_pumping_stations", "deposit_capacity_m3",
        "n_imbornales", "imbornal_density_per_km2",
        "n_colectores", "colector_coverage_pct",
        "n_pipe_segments", "n_sectors",
    ]
    result = barrio_agg[[c for c in cols if c in barrio_agg.columns]]

    print(f"    Features de infraestructura para {len(result)} barrios")
    return result


# ─────────────────────────────────────────────────────────────────
# Datos auxiliares: Altas de contratos
# ─────────────────────────────────────────────────────────────────

def load_contract_growth(csv_path: str) -> pd.DataFrame:
    """
    Carga altas/bajas de contratos y calcula tasa de crecimiento mensual.

    Input: altas-por-poblacion CSV con columnas:
      EXPLOTACION, AÑO, MES, MOTIVO, TIPO CLIENTE, USO, ACTIVIDAD, CANTIDAD

    Output: DataFrame con columnas:
      fecha, net_new_contracts, new_domestic, new_non_domestic,
      growth_momentum (media movil 3 meses de altas netas)

    Datos a nivel ciudad (no por barrio), util como feature temporal.
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # El año viene como 2.020 → float 2.02, 2.021 → 2.021, etc.
    # Formato español: punto como separador de miles. Multiplicar por 1000.
    df["AÑO"] = df["AÑO"].apply(lambda x: int(round(float(x) * 1000)))

    df["MES"] = df["MES"].astype(int)
    df["CANTIDAD"] = pd.to_numeric(df["CANTIDAD"], errors="coerce").fillna(0)

    # Calcular altas netas por mes (altas - bajas)
    df["is_alta"] = df["MOTIVO"].str.contains("Alta", case=False, na=False)
    df["is_baja"] = df["MOTIVO"].str.contains("baja", case=False, na=False)
    df["signed_qty"] = np.where(df["is_baja"], -df["CANTIDAD"], df["CANTIDAD"])

    monthly = df.groupby(["AÑO", "MES"]).agg(
        net_new_contracts=("signed_qty", "sum"),
        total_altas=("CANTIDAD", lambda x: x[df.loc[x.index, "is_alta"]].sum()),
        n_domestic=("CANTIDAD", lambda x: x[
            df.loc[x.index, "USO"].str.contains("DOMÉSTICO", case=False, na=False) &
            df.loc[x.index, "is_alta"]
        ].sum()),
    ).reset_index()

    monthly["fecha"] = pd.to_datetime(
        monthly["AÑO"].astype(str) + "-" + monthly["MES"].astype(str) + "-01"
    )
    monthly = monthly.sort_values("fecha")

    # Growth momentum: media movil de 3 meses
    monthly["growth_momentum"] = monthly["net_new_contracts"].rolling(3, min_periods=1).mean()

    # Renombrar para claridad
    monthly["new_domestic"] = monthly["n_domestic"]
    monthly["new_non_domestic"] = monthly["total_altas"] - monthly["n_domestic"]

    return monthly[["fecha", "net_new_contracts", "new_domestic",
                     "new_non_domestic", "growth_momentum"]]


# ─────────────────────────────────────────────────────────────────
# Datos auxiliares: Longitud de red + inspeccion
# ─────────────────────────────────────────────────────────────────

def load_network_stats(csv_path: str) -> pd.DataFrame:
    """
    Carga longitud de red de abastecimiento e inspeccion buscafugas por año.

    Output: DataFrame con columnas:
      year, network_length_km, inspected_km, inspection_coverage_pct
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    # Columnas: AÑO, EXPLOTACION, LONGITUD RED ABASTECIMIENTO (km),
    #           LONGITUD RED ABASTECIMIENTO INSPECCIONADA BUSCAFUGAS (km)

    rows = []
    for _, row in df.iterrows():
        year = int(row.iloc[0])
        # Valores con coma decimal española
        net_len = float(str(row.iloc[2]).replace(".", "").replace(",", "."))
        insp_len = float(str(row.iloc[3]).replace(".", "").replace(",", "."))

        rows.append({
            "year": year,
            "network_length_km": net_len,
            "inspected_km": insp_len,
            "inspection_coverage_pct": min(insp_len / net_len * 100, 200) if net_len > 0 else 0,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Datos auxiliares: EDAR (depuradora)
# ─────────────────────────────────────────────────────────────────

def load_edar_data(csv_path: str) -> pd.DataFrame:
    """
    Carga datos de la depuradora EDAR Rincon de Leon.

    Output: DataFrame con columnas:
      fecha, treated_m3, reused_m3, reuse_ratio
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    # Columnas: AÑO, MES, CAUDAL TRATADO (m3), TOTAL AGUA REUTILIZADA (m3)

    mes_map = {
        "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
        "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
        "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12,
    }

    rows = []
    for _, row in df.iterrows():
        year = int(row.iloc[0])
        month_str = str(row.iloc[1]).strip()
        month = mes_map.get(month_str, 0)
        if month == 0:
            continue

        treated = float(row.iloc[2])
        reused = float(row.iloc[3])

        rows.append({
            "fecha": pd.Timestamp(year=year, month=month, day=1),
            "treated_m3": treated,
            "reused_m3": reused,
            "reuse_ratio": reused / treated if treated > 0 else 0,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Sector adjacency (para M11 modelo espacial)
# ─────────────────────────────────────────────────────────────────

def compute_barrio_adjacency(data_dir: str = "data/") -> dict[str, list[str]]:
    """
    Calcula que barrios son adyacentes basandose en la proximidad de los
    centroides de sus sectores.

    Dos barrios son adyacentes si tienen sectores cuya distancia
    centroide-centroide es menor que un umbral (500m).

    Retorna dict[barrio] → lista de barrios vecinos.
    """
    sectors = load_sector_polygons(data_dir)
    if not sectors:
        return {}

    DISTANCE_THRESHOLD = 800  # metros

    # Agrupar centroides por barrio
    barrio_centroids: dict[str, list[tuple[float, float]]] = {}
    for s in sectors:
        barrio = s["barrio"]
        if not barrio:
            continue
        if barrio not in barrio_centroids:
            barrio_centroids[barrio] = []
        barrio_centroids[barrio].append((s["centroid_x"], s["centroid_y"]))

    # Calcular adyacencia
    barrios = list(barrio_centroids.keys())
    adjacency: dict[str, list[str]] = {b: [] for b in barrios}

    for i in range(len(barrios)):
        for j in range(i + 1, len(barrios)):
            b1, b2 = barrios[i], barrios[j]
            # Minima distancia entre cualquier par de centroides
            min_dist = float("inf")
            for c1 in barrio_centroids[b1]:
                for c2 in barrio_centroids[b2]:
                    d = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                    min_dist = min(min_dist, d)

            if min_dist < DISTANCE_THRESHOLD:
                adjacency[b1].append(b2)
                adjacency[b2].append(b1)

    return adjacency


# ─────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  GIS Features — Extraccion de infraestructura")
    print("=" * 70)

    # Infrastructure features
    infra = load_infrastructure_features("data/")
    if len(infra) > 0:
        print(f"\n  {len(infra)} barrios con datos de infraestructura:")
        print(infra.to_string(index=False))

    # Contract growth
    altas_path = "data/altas-por-poblacion-solo-alicante_hackaton-dataart-altas-por-poblacion-solo-alicante.csv.csv"
    growth = load_contract_growth(altas_path)
    if len(growth) > 0:
        print(f"\n  Crecimiento de contratos ({len(growth)} meses):")
        print(growth.tail(12).to_string(index=False))

    # Network stats
    net_path = "data/amaem-pda-longitud-red-abastecimiento-explotacion-solo-alicante-amaem-pda-longitud-red-abastecim.csv"
    net = load_network_stats(net_path)
    if len(net) > 0:
        print(f"\n  Red de abastecimiento:")
        print(net.to_string(index=False))

    # EDAR
    edar_path = "data/amaem-pda-depuracion-edar-rincon-de-leon_hackaton-dataart-2.0-amaem-pda-depuracion-edar-rincon-d.csv"
    edar = load_edar_data(edar_path)
    if len(edar) > 0:
        print(f"\n  EDAR ({len(edar)} meses):")
        print(edar.tail(6).to_string(index=False))

    # Adjacency
    print("\n  Calculando adyacencia de barrios...")
    adj = compute_barrio_adjacency("data/")
    for barrio in sorted(adj.keys())[:5]:
        neighbors = adj[barrio]
        print(f"    {barrio}: {len(neighbors)} vecinos → {', '.join(sorted(neighbors)[:4])}")
