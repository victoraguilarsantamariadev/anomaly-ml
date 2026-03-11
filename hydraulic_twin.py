"""
Digital Twin Hidraulico Simplificado — Modelo de la red de agua de Alicante.

Usa datos GIS reales (tuberias, sectores, depositos, centros de bombeo)
para construir un modelo hidraulico simplificado que:
  1. Modela la topologia de la red (que sectores estan conectados)
  2. Estima presion por elevacion (agua fluye cuesta abajo)
  3. Estima capacidad de flujo por diametro de tuberias (Hazen-Williams)
  4. Detecta anomalias explicadas por la infraestructura hidraulica

No es un modelo EPANET completo — es una aproximacion que demuestra el concepto
y aporta features explicativos basados en fisica real.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

__all__ = ["run_hydraulic_twin", "hydraulic_summary"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_arcgis_json(path):
    """Load ArcGIS JSON file with latin-1 encoding."""
    with open(path, encoding="latin-1") as f:
        data = json.load(f)
    return data


def _polygon_centroid(rings):
    """Compute centroid of polygon rings."""
    all_pts = []
    for ring in rings:
        all_pts.extend(ring)
    if not all_pts:
        return (0.0, 0.0)
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    return (np.mean(xs), np.mean(ys))


def _polyline_endpoints(paths):
    """Extract start and end points from polyline paths."""
    if not paths or not paths[0]:
        return None, None
    first_path = paths[0]
    last_path = paths[-1]
    start = tuple(first_path[0][:2])
    end = tuple(last_path[-1][:2])
    return start, end


def _polyline_length(paths):
    """Compute total length of polyline in meters (UTM coords)."""
    total = 0.0
    for path in paths:
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            total += np.sqrt(dx*dx + dy*dy)
    return total


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_hydraulic_model(
    sectores_path="data/sectores_de_consumo.json",
    tuberias_path="data/tuberias.json",
    depositos_path="data/depositos.json",
    bombeo_path="data/centros_de_bombeo.json",
):
    """Build simplified hydraulic model from GIS data."""
    model = {
        "sectors": [],
        "pipes": [],
        "sources": [],
        "pumps": [],
        "adjacency": None,
        "sector_names": [],
    }

    # --- Load sectors ---
    sectores_data = _load_arcgis_json(sectores_path)
    sectors = []
    centroids = []
    for feat in sectores_data["features"]:
        attrs = feat["attributes"]
        geom = feat.get("geometry", {})
        rings = geom.get("rings", [])
        cx, cy = _polygon_centroid(rings)
        sector = {
            "fid": attrs.get("FID", 0),
            "name": attrs.get("DCONS_PO_2", "UNKNOWN"),
            "amaemid": attrs.get("AMAEMID", 0),
            "cota_min": attrs.get("COTA_MINIM", 0.0),
            "cota_media": attrs.get("COTA_MEDIA", 0.0),
            "cota_max": attrs.get("COTA_MAXIM", 0.0),
            "centroid_x": cx,
            "centroid_y": cy,
        }
        sectors.append(sector)
        centroids.append((cx, cy))

    model["sectors"] = sectors
    model["sector_names"] = [s["name"] for s in sectors]
    n_sectors = len(sectors)

    # --- Build centroid array for spatial queries ---
    centroids_arr = np.array(centroids)  # (n_sectors, 2)

    # --- Load pipes and build adjacency ---
    tuberias_data = _load_arcgis_json(tuberias_path)
    adjacency = np.zeros((n_sectors, n_sectors), dtype=float)
    pipe_count = np.zeros((n_sectors, n_sectors), dtype=int)
    pipes_info = []

    for feat in tuberias_data["features"]:
        attrs = feat["attributes"]
        geom = feat.get("geometry", {})
        paths = geom.get("paths", [])
        if not paths:
            continue

        diameter = attrs.get("DIMEN1", 100)
        clasificacion = attrs.get("d_CLASIFIC", "UNITARIA")

        # Skip pluvial (rainwater) pipes
        if clasificacion == "PLUVIAL":
            continue

        start, end = _polyline_endpoints(paths)
        if start is None or end is None:
            continue

        length = _polyline_length(paths)
        if length < 1.0:
            length = 1.0

        # Find nearest sectors to start and end
        start_arr = np.array(start)
        end_arr = np.array(end)
        dist_start = np.sqrt(np.sum((centroids_arr - start_arr) ** 2, axis=1))
        dist_end = np.sqrt(np.sum((centroids_arr - end_arr) ** 2, axis=1))
        sec_a = int(np.argmin(dist_start))
        sec_b = int(np.argmin(dist_end))

        if sec_a == sec_b:
            continue  # Internal pipe

        # Hazen-Williams capacity approximation
        # Q = 0.2785 * C * D^2.63 * S^0.54
        # C = 120 (typical), D in meters, S = slope
        d_m = diameter / 1000.0  # mm to m
        elev_a = sectors[sec_a]["cota_media"]
        elev_b = sectors[sec_b]["cota_media"]
        slope = abs(elev_a - elev_b) / length if length > 0 else 0.001
        slope = max(slope, 0.0001)  # Minimum slope

        capacity = 0.2785 * 120 * (d_m ** 2.63) * (slope ** 0.54)

        adjacency[sec_a, sec_b] += capacity
        adjacency[sec_b, sec_a] += capacity
        pipe_count[sec_a, sec_b] += 1
        pipe_count[sec_b, sec_a] += 1

        pipes_info.append({
            "from": sec_a,
            "to": sec_b,
            "diameter": diameter,
            "length": length,
            "capacity": capacity,
            "clasificacion": clasificacion,
        })

    model["pipes"] = pipes_info
    model["adjacency"] = adjacency
    model["pipe_count"] = pipe_count

    # --- Load depositos (sources) ---
    if Path(depositos_path).exists():
        try:
            dep_data = _load_arcgis_json(depositos_path)
            for feat in dep_data["features"]:
                attrs = feat["attributes"]
                geom = feat.get("geometry", {})
                # Could be point or polygon
                if "x" in geom and "y" in geom:
                    x, y = geom["x"], geom["y"]
                elif "rings" in geom:
                    x, y = _polygon_centroid(geom["rings"])
                else:
                    continue
                # Find nearest sector
                pt = np.array([x, y])
                dists = np.sqrt(np.sum((centroids_arr - pt) ** 2, axis=1))
                nearest = int(np.argmin(dists))
                model["sources"].append({
                    "sector_idx": nearest,
                    "sector_name": sectors[nearest]["name"],
                    "attrs": {k: v for k, v in attrs.items() if isinstance(v, (int, float, str))},
                    "x": x, "y": y,
                })
        except Exception:
            pass

    # --- Load centros de bombeo (pumps) ---
    if Path(bombeo_path).exists():
        try:
            bomb_data = _load_arcgis_json(bombeo_path)
            for feat in bomb_data["features"]:
                attrs = feat["attributes"]
                geom = feat.get("geometry", {})
                if "x" in geom and "y" in geom:
                    x, y = geom["x"], geom["y"]
                elif "rings" in geom:
                    x, y = _polygon_centroid(geom["rings"])
                elif "paths" in geom:
                    paths = geom["paths"]
                    if paths and paths[0]:
                        x, y = paths[0][0][0], paths[0][0][1]
                    else:
                        continue
                else:
                    continue
                pt = np.array([x, y])
                dists = np.sqrt(np.sum((centroids_arr - pt) ** 2, axis=1))
                nearest = int(np.argmin(dists))
                model["pumps"].append({
                    "sector_idx": nearest,
                    "sector_name": sectors[nearest]["name"],
                    "attrs": {k: v for k, v in attrs.items() if isinstance(v, (int, float, str))},
                })
        except Exception:
            pass

    return model


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_steady_state(model, demand_by_sector=None):
    """
    Simple steady-state flow simulation based on elevation and capacity.

    Water flows from high to low elevation. Each sector's supply depends on
    the total capacity of incoming pipes from higher-elevation neighbors.
    """
    sectors = model["sectors"]
    adj = model["adjacency"]
    n = len(sectors)

    elevations = np.array([s["cota_media"] for s in sectors])

    # Compute pressure head at each sector (relative to max elevation source)
    source_indices = [s["sector_idx"] for s in model.get("sources", [])]
    if source_indices:
        max_source_elev = max(elevations[i] for i in source_indices)
    else:
        max_source_elev = elevations.max()

    # Pressure = (source_elevation - sector_elevation) * rho * g
    # Simplified: head = source_elev - sector_elev (in meters of water column)
    pressure_head = max_source_elev - elevations
    pressure_head = np.maximum(pressure_head, 0)  # Can't have negative pressure

    # Pump boost: sectors near pumps get additional head
    pump_indices = [p["sector_idx"] for p in model.get("pumps", [])]
    pump_boost = np.zeros(n)
    for pi in pump_indices:
        pump_boost[pi] = 20.0  # Typical pump adds ~20m head
        # Also boost neighbors
        for j in range(n):
            if adj[pi, j] > 0:
                pump_boost[j] = max(pump_boost[j], 10.0)
    pressure_head += pump_boost

    # Supply capacity: sum of pipe capacity from upstream (higher) neighbors
    supply_capacity = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if adj[j, i] > 0 and elevations[j] >= elevations[i]:
                supply_capacity[i] += adj[j, i]
        # Sources have unlimited supply
        if i in source_indices:
            supply_capacity[i] = 1e6

    # Demand
    if demand_by_sector is None:
        demand = np.ones(n) * 100.0  # Placeholder
    else:
        demand = demand_by_sector

    # Capacity ratio
    capacity_ratio = np.where(demand > 0, supply_capacity / demand, 999.0)
    capacity_ratio = np.clip(capacity_ratio, 0, 100)

    # Pressure zones
    pressure_zones = []
    for i in range(n):
        if pressure_head[i] < 15:
            pressure_zones.append("LOW")
        elif pressure_head[i] < 40:
            pressure_zones.append("NORMAL")
        else:
            pressure_zones.append("HIGH")

    # Bottlenecks: low capacity and low pressure
    bottlenecks = []
    for i in range(n):
        if capacity_ratio[i] < 1.0 and pressure_zones[i] == "LOW":
            bottlenecks.append(i)

    return {
        "pressure_head": pressure_head,
        "supply_capacity": supply_capacity,
        "capacity_ratio": capacity_ratio,
        "pressure_zones": pressure_zones,
        "bottlenecks": bottlenecks,
        "pump_boost": pump_boost,
    }


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def _fuzzy_match_sector_barrio(sector_name, barrio_key):
    """Fuzzy match sector name to barrio name."""
    barrio = barrio_key.split("__")[0]
    # Remove leading number prefix from barrio (e.g., "35-VIRGEN DEL CARMEN" -> "VIRGEN DEL CARMEN")
    parts = barrio.split("-", 1)
    barrio_clean = parts[1].strip().upper() if len(parts) > 1 else barrio.strip().upper()
    sector_clean = sector_name.strip().upper()

    if sector_clean == barrio_clean:
        return True
    if sector_clean in barrio_clean or barrio_clean in sector_clean:
        return True
    # Common abbreviations
    replacements = {
        "SAN": "S.", "SANTA": "STA.", "POLIGONO": "POL.",
        "PLAYA": "PL.", "VIRGEN": "V.",
    }
    for full, abbr in replacements.items():
        if sector_clean.replace(full, abbr) == barrio_clean.replace(full, abbr):
            return True
    return False


def detect_hydraulic_anomalies(model, simulation, results_df):
    """Detect anomalies explained by hydraulic properties."""
    sectors = model["sectors"]
    n_sectors = len(sectors)
    pressure_head = simulation["pressure_head"]
    pressure_zones = simulation["pressure_zones"]
    capacity_ratio = simulation["capacity_ratio"]
    adj = model["adjacency"]

    # Map barrios to sectors
    barrio_keys = results_df["barrio_key"].unique() if "barrio_key" in results_df.columns else []
    barrio_to_sector = {}
    for bk in barrio_keys:
        for i, sec in enumerate(sectors):
            if _fuzzy_match_sector_barrio(sec["name"], bk):
                barrio_to_sector[bk] = i
                break

    # Compute hydraulic features per barrio
    records = []
    for bk in barrio_keys:
        sec_idx = barrio_to_sector.get(bk)
        if sec_idx is None:
            records.append({
                "barrio_key": bk,
                "hydraulic_pressure": np.nan,
                "hydraulic_zone": "UNKNOWN",
                "hydraulic_capacity_ratio": np.nan,
                "hydraulic_elevation": np.nan,
                "hydraulic_connectivity": 0,
                "hydraulic_anomaly_type": "UNMAPPED",
                "near_pump": False,
                "near_source": False,
            })
            continue

        sec = sectors[sec_idx]
        connectivity = int((adj[sec_idx] > 0).sum())

        # Check proximity to infrastructure
        source_indices = {s["sector_idx"] for s in model.get("sources", [])}
        pump_indices = {p["sector_idx"] for p in model.get("pumps", [])}
        near_source = sec_idx in source_indices or any(
            adj[sec_idx, si] > 0 for si in source_indices
        )
        near_pump = sec_idx in pump_indices or any(
            adj[sec_idx, pi] > 0 for pi in pump_indices
        )

        # Determine anomaly type
        anomaly_type = "NORMAL"
        if pressure_zones[sec_idx] == "LOW" and capacity_ratio[sec_idx] < 2.0:
            anomaly_type = "PRESSURE_DEFICIT"
        elif sec["cota_media"] > 80 and not near_pump:
            anomaly_type = "HIGH_ELEVATION_NO_PUMP"
        elif capacity_ratio[sec_idx] < 0.5:
            anomaly_type = "CAPACITY_CONSTRAINED"

        # Check upstream anomalies (propagation)
        upstream_anomalous = False
        for j in range(n_sectors):
            if adj[j, sec_idx] > 0 and sectors[j]["cota_media"] > sec["cota_media"]:
                j_bk = None
                for bk2, sidx2 in barrio_to_sector.items():
                    if sidx2 == j:
                        j_bk = bk2
                        break
                if j_bk and j_bk in results_df["barrio_key"].values:
                    j_score = results_df[results_df["barrio_key"] == j_bk]["anomaly_score"].mean()
                    if j_score > 0.3:
                        upstream_anomalous = True
                        break

        if upstream_anomalous and anomaly_type == "NORMAL":
            anomaly_type = "UPSTREAM_PROPAGATION"

        records.append({
            "barrio_key": bk,
            "hydraulic_pressure": float(pressure_head[sec_idx]),
            "hydraulic_zone": pressure_zones[sec_idx],
            "hydraulic_capacity_ratio": float(capacity_ratio[sec_idx]),
            "hydraulic_elevation": float(sec["cota_media"]),
            "hydraulic_connectivity": connectivity,
            "hydraulic_anomaly_type": anomaly_type,
            "near_pump": near_pump,
            "near_source": near_source,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_hydraulic_twin(results_df):
    """Run the full hydraulic digital twin analysis."""
    print(f"\n  [HYDRAULIC TWIN] Construyendo modelo hidraulico...")

    model = build_hydraulic_model()
    n_sectors = len(model["sectors"])
    n_pipes = len(model["pipes"])
    n_sources = len(model["sources"])
    n_pumps = len(model["pumps"])

    print(f"    Red: {n_sectors} sectores, {n_pipes} tuberias "
          f"(excl. pluviales), {n_sources} depositos, {n_pumps} bombeos")

    # Extract demand per sector from results
    demand = np.ones(n_sectors) * 100.0
    if "consumption_per_contract" in results_df.columns:
        for bk in results_df["barrio_key"].unique():
            for i, sec in enumerate(model["sectors"]):
                if _fuzzy_match_sector_barrio(sec["name"], bk):
                    mean_cons = results_df[results_df["barrio_key"] == bk][
                        "consumption_per_contract"
                    ].mean()
                    if pd.notna(mean_cons) and mean_cons > 0:
                        demand[i] = mean_cons
                    break

    print(f"    Simulando estado estacionario...")
    simulation = simulate_steady_state(model, demand_by_sector=demand)

    n_low = sum(1 for z in simulation["pressure_zones"] if z == "LOW")
    n_normal = sum(1 for z in simulation["pressure_zones"] if z == "NORMAL")
    n_high = sum(1 for z in simulation["pressure_zones"] if z == "HIGH")
    print(f"    Zonas de presion: {n_low} baja, {n_normal} normal, {n_high} alta")
    print(f"    Cuellos de botella: {len(simulation['bottlenecks'])} sectores")

    print(f"    Detectando anomalias hidraulicas...")
    hydraulic_df = detect_hydraulic_anomalies(model, simulation, results_df)

    n_mapped = (hydraulic_df["hydraulic_anomaly_type"] != "UNMAPPED").sum()
    n_pressure = (hydraulic_df["hydraulic_anomaly_type"] == "PRESSURE_DEFICIT").sum()
    n_capacity = (hydraulic_df["hydraulic_anomaly_type"] == "CAPACITY_CONSTRAINED").sum()
    n_elevation = (hydraulic_df["hydraulic_anomaly_type"] == "HIGH_ELEVATION_NO_PUMP").sum()
    n_propagation = (hydraulic_df["hydraulic_anomaly_type"] == "UPSTREAM_PROPAGATION").sum()
    print(f"    Mapeados: {n_mapped}/{len(hydraulic_df)} barrios a sectores hidraulicos")
    print(f"    Anomalias: {n_pressure} presion baja, {n_capacity} capacidad limitada, "
          f"{n_elevation} elevacion sin bomba, {n_propagation} propagacion upstream")

    # Elevation stats
    elevations = [s["cota_media"] for s in model["sectors"]]
    diameters = [p["diameter"] for p in model["pipes"]] if model["pipes"] else [0]

    return {
        "model": model,
        "simulation": simulation,
        "hydraulic_df": hydraulic_df,
        "stats": {
            "n_sectors": n_sectors,
            "n_pipes": n_pipes,
            "n_sources": n_sources,
            "n_pumps": n_pumps,
            "elevation_range": (min(elevations), max(elevations)),
            "diameter_range": (min(diameters), max(diameters)),
            "diameter_mean": np.mean(diameters),
            "n_low_pressure": n_low,
            "n_normal_pressure": n_normal,
            "n_high_pressure": n_high,
            "n_bottlenecks": len(simulation["bottlenecks"]),
        },
    }


def hydraulic_summary(twin_results):
    """Print summary of hydraulic digital twin analysis."""
    stats = twin_results["stats"]
    hdf = twin_results["hydraulic_df"]

    print(f"\n{'='*80}")
    print(f"  DIGITAL TWIN HIDRAULICO — Modelo fisico de la red de Alicante")
    print(f"{'='*80}")

    print(f"\n  Infraestructura modelada:")
    print(f"    Sectores de consumo:  {stats['n_sectors']}")
    print(f"    Tuberias (no pluvial): {stats['n_pipes']}")
    print(f"    Depositos (fuentes):   {stats['n_sources']}")
    print(f"    Centros de bombeo:     {stats['n_pumps']}")

    elev_min, elev_max = stats["elevation_range"]
    print(f"\n  Topografia:")
    print(f"    Elevacion: {elev_min:.1f}m — {elev_max:.1f}m sobre nivel del mar")
    print(f"    Diametro tuberias: {stats['diameter_range'][0]}-{stats['diameter_range'][1]}mm "
          f"(media: {stats['diameter_mean']:.0f}mm)")

    print(f"\n  Zonas de presion simulada:")
    print(f"    BAJA (<15m cabeza):  {stats['n_low_pressure']} sectores")
    print(f"    NORMAL (15-40m):     {stats['n_normal_pressure']} sectores")
    print(f"    ALTA (>40m):         {stats['n_high_pressure']} sectores")
    print(f"    Cuellos de botella:  {stats['n_bottlenecks']} sectores")

    # Show anomalies by type
    anom_types = hdf[hdf["hydraulic_anomaly_type"].isin([
        "PRESSURE_DEFICIT", "CAPACITY_CONSTRAINED",
        "HIGH_ELEVATION_NO_PUMP", "UPSTREAM_PROPAGATION",
    ])]

    if len(anom_types) > 0:
        print(f"\n  Anomalias hidraulicas detectadas:")
        print(f"  {'Barrio':<35} {'Tipo':<25} {'Presion':>8} {'Elev.':>7} {'Conex.':>6}")
        print(f"  {'─'*85}")
        for _, row in anom_types.sort_values("hydraulic_pressure").iterrows():
            name = row["barrio_key"].split("__")[0]
            print(f"  {name:<35} {row['hydraulic_anomaly_type']:<25} "
                  f"{row['hydraulic_pressure']:>7.1f}m {row['hydraulic_elevation']:>6.1f}m "
                  f"{row['hydraulic_connectivity']:>5}")
    else:
        print(f"\n  Ningun barrio mapeado presenta anomalia hidraulica estructural.")

    # Cross-reference with detection results
    mapped = hdf[hdf["hydraulic_anomaly_type"] != "UNMAPPED"]
    if len(mapped) > 0:
        print(f"\n  Barrios mapeados al modelo hidraulico: {len(mapped)}")
        low_p = mapped[mapped["hydraulic_zone"] == "LOW"]["barrio_key"].tolist()
        if low_p:
            names = [bk.split("__")[0] for bk in low_p[:5]]
            print(f"    En zona presion BAJA: {', '.join(names)}")

    # Key finding
    print(f"\n  HALLAZGO CLAVE:")
    print(f"    El modelo hidraulico permite distinguir entre anomalias por")
    print(f"    infraestructura (presion baja, capacidad limitada) y anomalias")
    print(f"    por consumo real (fraude, fugas despues del contador).")
    print(f"    Esto es critico para priorizar inversiones de AMAEM.")
