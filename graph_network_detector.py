"""
M14 -- Detector de anomalias basado en topologia de red de agua.

Construye un grafo de la red real de tuberias de Alicante (21.638 tuberias,
183 sectores de consumo) y usa la topologia para:
  - Detectar propagacion de anomalias (fuga aguas arriba)
  - Identificar anomalias aisladas (posible fraude local)
  - Detectar anomalias en zonas de presion extrema

Input: resultados de los otros modelos (results_df) + GIS data (tuberias.json,
       sectores_de_consumo.json).
Output: graph features + network anomaly classification.

Uso:
  from graph_network_detector import run_graph_analysis, graph_network_summary
  analysis = run_graph_analysis(results_df)
  graph_network_summary(analysis)
"""

import json
import os
import math
import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

__all__ = ["run_graph_analysis", "graph_network_summary"]

# ---------------------------------------------------------------------------
# Mapeo sector de consumo -> barrio del hackathon
# Reutilizamos el mismo mapeo de gis_features.py
# ---------------------------------------------------------------------------

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

# Barrio inverso: barrio -> lista de sectores
_BARRIO_TO_SECTORS = defaultdict(list)
for _sec, _bar in SECTOR_TO_BARRIO.items():
    if _bar is not None:
        _BARRIO_TO_SECTORS[_bar].append(_sec)


# ---------------------------------------------------------------------------
# Geometria basica
# ---------------------------------------------------------------------------

def _polygon_centroid(ring):
    """Centroide de un poligono (media de coordenadas)."""
    if not ring:
        return (0.0, 0.0)
    cx = sum(p[0] for p in ring) / len(ring)
    cy = sum(p[1] for p in ring) / len(ring)
    return (cx, cy)


def _distance(x1, y1, x2, y2):
    """Distancia euclidea en metros (coordenadas UTM)."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ---------------------------------------------------------------------------
# Fuzzy matching: sector name <-> barrio name
# ---------------------------------------------------------------------------

def _normalize_name(name):
    """Normaliza nombre para fuzzy matching."""
    if not name:
        return ""
    import re
    s = name.upper().strip()
    # Quitar prefijo numerico tipo "13-"
    s = re.sub(r'^\d+\s*-\s*', '', s)
    # Quitar acentos basicos
    replacements = {
        'A': 'A', 'E': 'E', 'I': 'I', 'O': 'O', 'U': 'U',
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # Quitar caracteres no alfanumericos
    s = re.sub(r'[^A-Z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _fuzzy_match_barrio(barrio_key, sector_names_normalized, sector_names_original):
    """
    Intenta hacer match de un barrio_key a un sector.
    barrio_key tiene formato "13-SAN GABRIEL__DOMESTICO"
    Retorna indice del mejor match en sector_names, o None.
    """
    import re
    # Extraer barrio limpio (quitar __USO)
    barrio_clean = barrio_key.split("__")[0] if "__" in barrio_key else barrio_key
    barrio_norm = _normalize_name(barrio_clean)

    # Primero intentar match exacto via SECTOR_TO_BARRIO inverso
    for i, sname in enumerate(sector_names_original):
        mapped_barrio = SECTOR_TO_BARRIO.get(sname)
        if mapped_barrio and mapped_barrio == barrio_clean:
            return i

    # Segundo: match por nombre normalizado
    best_score = 0
    best_idx = None
    barrio_words = set(barrio_norm.split())

    for i, snorm in enumerate(sector_names_normalized):
        sector_words = set(snorm.split())
        if not barrio_words or not sector_words:
            continue
        # Jaccard similarity
        intersection = barrio_words & sector_words
        union = barrio_words | sector_words
        score = len(intersection) / len(union) if union else 0
        # Bonus: substring match
        if barrio_norm in snorm or snorm in barrio_norm:
            score += 0.3
        if score > best_score:
            best_score = score
            best_idx = i

    if best_score >= 0.3:
        return best_idx
    return None


# ---------------------------------------------------------------------------
# Function 1: build_water_network
# ---------------------------------------------------------------------------

def build_water_network(
    tuberias_path: str = "data/tuberias.json",
    sectores_path: str = "data/sectores_de_consumo.json",
) -> dict:
    """
    Construye un grafo de la red de agua de Alicante.

    1. Carga sectores -> extrae centroides de poligonos
    2. Carga tuberias -> para cada tuberia, encuentra los 2 sectores mas
       cercanos a sus endpoints
    3. Construye adjacencia: sector A conectado a sector B si una tuberia
       los une
    4. Atributos de arista: diametro (proxy de capacidad), conteo de tuberias
    5. Atributos de nodo: nombre, COTA_MEDIA, COTA_MAXIM, AMAEMID

    Returns:
        dict con: nodes, edges, adjacency_matrix, sector_names, sector_to_idx,
                  barrio_to_node_indices, stats
    """
    # --- Cargar sectores ---
    with open(sectores_path, encoding="latin-1") as f:
        sectores_data = json.load(f)

    nodes = []
    sector_centroids = []

    for feat in sectores_data["features"]:
        attr = feat["attributes"]
        geom = feat.get("geometry", {})
        rings = geom.get("rings", [])
        outer_ring = rings[0] if rings else []
        cx, cy = _polygon_centroid(outer_ring)
        name = attr.get("DCONS_PO_2", "")
        barrio = SECTOR_TO_BARRIO.get(name)

        nodes.append({
            "idx": len(nodes),
            "name": name,
            "barrio": barrio,
            "cota_min": attr.get("COTA_MINIM", 0) or 0,
            "cota_media": attr.get("COTA_MEDIA", 0) or 0,
            "cota_max": attr.get("COTA_MAXIM", 0) or 0,
            "amaemid": attr.get("AMAEMID", 0) or 0,
            "centroid_x": cx,
            "centroid_y": cy,
        })
        sector_centroids.append([cx, cy])

    n_sectors = len(nodes)
    sector_names = [n["name"] for n in nodes]
    centroid_array = np.array(sector_centroids)

    # Construir KDTree para busqueda espacial rapida
    if HAS_SCIPY:
        tree = KDTree(centroid_array)
    else:
        tree = None

    # Mapeo sector_name -> idx
    sector_to_idx = {}
    for i, n in enumerate(nodes):
        sector_to_idx[n["name"]] = i

    # Mapeo barrio -> lista de node indices
    barrio_to_node_indices = defaultdict(list)
    for i, n in enumerate(nodes):
        if n["barrio"]:
            barrio_to_node_indices[n["barrio"]].append(i)

    # --- Cargar tuberias ---
    with open(tuberias_path, encoding="latin-1") as f:
        tuberias_data = json.load(f)

    # Acumular aristas: (sector_i, sector_j) -> {diameters: [], count: int}
    edge_data = defaultdict(lambda: {"diameters": [], "count": 0})
    n_pipes = len(tuberias_data.get("features", []))
    all_diameters = []

    # Umbral: distancia maxima entre endpoint de tuberia y centroide de sector
    # para considerar que la tuberia "pertenece" a ese sector.
    # Usamos 2000m como umbral (sectores en Alicante son relativamente pequenos)
    MAX_DIST = 2000.0

    for feat in tuberias_data["features"]:
        attr = feat["attributes"]
        geom = feat.get("geometry", {})
        paths = geom.get("paths", [])
        diam = attr.get("DIMEN1", 0) or 0
        if diam > 0:
            all_diameters.append(diam)

        if not paths:
            continue

        # Obtener primer y ultimo punto de la tuberia
        all_coords = [p for path in paths for p in path]
        if len(all_coords) < 2:
            continue

        start_pt = np.array(all_coords[0][:2])
        end_pt = np.array(all_coords[-1][:2])

        # Encontrar sectores mas cercanos a cada endpoint
        if tree is not None:
            d_start, idx_start = tree.query(start_pt)
            d_end, idx_end = tree.query(end_pt)
        else:
            # Fallback sin scipy: busqueda lineal
            dists_start = np.sqrt(np.sum((centroid_array - start_pt) ** 2, axis=1))
            dists_end = np.sqrt(np.sum((centroid_array - end_pt) ** 2, axis=1))
            idx_start = int(np.argmin(dists_start))
            idx_end = int(np.argmin(dists_end))
            d_start = dists_start[idx_start]
            d_end = dists_end[idx_end]

        # Solo crear arista si ambos endpoints estan suficientemente cerca
        # de un sector Y los sectores son diferentes
        if d_start < MAX_DIST and d_end < MAX_DIST and idx_start != idx_end:
            key = (min(idx_start, idx_end), max(idx_start, idx_end))
            edge_data[key]["diameters"].append(diam if diam > 0 else 200)
            edge_data[key]["count"] += 1

    # Construir lista de aristas
    edges = []
    for (i, j), data in edge_data.items():
        edges.append({
            "source": i,
            "target": j,
            "n_pipes": data["count"],
            "diameters": data["diameters"],
            "mean_diameter": np.mean(data["diameters"]) if data["diameters"] else 200,
            "max_diameter": max(data["diameters"]) if data["diameters"] else 200,
            "total_capacity": sum(d ** 2 for d in data["diameters"]),  # ~proporcional a area seccion
        })

    # Construir matriz de adyacencia
    adj_matrix = np.zeros((n_sectors, n_sectors), dtype=np.float64)
    for e in edges:
        adj_matrix[e["source"], e["target"]] = e["mean_diameter"]
        adj_matrix[e["target"], e["source"]] = e["mean_diameter"]

    # Estadisticas
    stats = {
        "n_sectors": n_sectors,
        "n_pipes": n_pipes,
        "n_edges": len(edges),
        "diameter_mean": float(np.mean(all_diameters)) if all_diameters else 0,
        "diameter_min": float(min(all_diameters)) if all_diameters else 0,
        "diameter_max": float(max(all_diameters)) if all_diameters else 0,
        "sectors_with_barrio": sum(1 for n in nodes if n["barrio"] is not None),
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "adjacency_matrix": adj_matrix,
        "sector_names": sector_names,
        "sector_to_idx": sector_to_idx,
        "barrio_to_node_indices": barrio_to_node_indices,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Graph algorithms (sin networkx)
# ---------------------------------------------------------------------------

def _build_adjacency_list(n_nodes, edges):
    """Construye lista de adyacencia desde lista de aristas."""
    adj = defaultdict(list)
    for e in edges:
        adj[e["source"]].append((e["target"], e["mean_diameter"]))
        adj[e["target"]].append((e["source"], e["mean_diameter"]))
    return adj


def _bfs_shortest_paths(adj_list, n_nodes, source):
    """BFS desde source. Retorna distancias (en saltos) y predecesores."""
    dist = [-1] * n_nodes
    dist[source] = 0
    queue = [source]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v, _ in adj_list[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _betweenness_centrality_approx(n_nodes, edges, sample_size=50):
    """
    Calcula betweenness centrality aproximada usando BFS desde un subconjunto
    de nodos (mas rapido que O(V*E) completo).
    """
    adj_list = _build_adjacency_list(n_nodes, edges)
    centrality = np.zeros(n_nodes)

    # Seleccionar nodos fuente (usar todos si pocos, o sample)
    if n_nodes <= sample_size:
        sources = list(range(n_nodes))
    else:
        rng = np.random.RandomState(42)
        sources = rng.choice(n_nodes, size=sample_size, replace=False).tolist()

    for s in sources:
        # BFS
        dist = [-1] * n_nodes
        dist[s] = 0
        sigma = [0] * n_nodes  # numero de caminos mas cortos
        sigma[s] = 1
        pred = [[] for _ in range(n_nodes)]
        queue = [s]
        order = []  # orden de descubrimiento
        head = 0

        while head < len(queue):
            u = queue[head]
            head += 1
            order.append(u)
            for v, _ in adj_list[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                if dist[v] == dist[u] + 1:
                    sigma[v] += sigma[u]
                    pred[v].append(u)

        # Backpropagation de dependencias
        delta = [0.0] * n_nodes
        for w in reversed(order):
            if sigma[w] == 0:
                continue
            for p in pred[w]:
                frac = sigma[p] / sigma[w]
                delta[p] += frac * (1 + delta[w])
            if w != s:
                centrality[w] += delta[w]

    # Normalizar
    scale = 1.0 / (len(sources) * max(1, (n_nodes - 1) * (n_nodes - 2)))
    centrality *= scale

    return centrality


# ---------------------------------------------------------------------------
# Function 2: compute_graph_features
# ---------------------------------------------------------------------------

def compute_graph_features(network: dict, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapea resultados de anomalia al grafo y calcula features basadas en la red.

    Para cada nodo/sector:
      - degree: numero de sectores conectados
      - weighted_degree: suma de diametros de tuberias conectando este sector
      - elevation_diff: max(COTA) - min(COTA) de vecinos (diferencial de presion)
      - anomaly_neighbor_score: media de anomaly_score de sectores vecinos
      - anomaly_propagation: media ponderada por capacidad de anomaly de vecinos
      - graph_centrality: betweenness centrality
      - upstream_anomaly: anomaly_score de sectores a mayor elevacion
      - downstream_count: numero de sectores a menor elevacion que alimenta
    """
    nodes = network["nodes"]
    edges = network["edges"]
    adj_matrix = network["adjacency_matrix"]
    barrio_to_nodes = network["barrio_to_node_indices"]
    n_nodes = len(nodes)

    # --- Calcular anomaly_score medio por barrio ---
    barrio_scores = {}
    if "barrio_key" in results_df.columns and "anomaly_score" in results_df.columns:
        barrio_col = "barrio_key"
        # Extraer barrio limpio
        temp = results_df.copy()
        temp["_barrio"] = temp[barrio_col].str.split("__").str[0]
        score_by_barrio = temp.groupby("_barrio")["anomaly_score"].mean()
        barrio_scores = score_by_barrio.to_dict()
    elif "barrio" in results_df.columns and "anomaly_score" in results_df.columns:
        barrio_scores = results_df.groupby("barrio")["anomaly_score"].mean().to_dict()

    # Mapear anomaly_score a cada nodo
    node_anomaly = np.zeros(n_nodes)
    for i, node in enumerate(nodes):
        barrio = node["barrio"]
        if barrio and barrio in barrio_scores:
            node_anomaly[i] = barrio_scores[barrio]

    # --- Construir lista de adyacencia ---
    adj_list = _build_adjacency_list(n_nodes, edges)

    # --- Features por nodo ---
    degree = np.zeros(n_nodes)
    weighted_degree = np.zeros(n_nodes)
    elevation_diff = np.zeros(n_nodes)
    anomaly_neighbor_score = np.zeros(n_nodes)
    anomaly_propagation = np.zeros(n_nodes)
    upstream_anomaly = np.zeros(n_nodes)
    downstream_count = np.zeros(n_nodes)

    for i in range(n_nodes):
        neighbors = adj_list[i]
        degree[i] = len(neighbors)

        if not neighbors:
            continue

        # Weighted degree
        weighted_degree[i] = sum(diam for _, diam in neighbors)

        # Elevation diff de vecinos
        neighbor_cotas = [nodes[j]["cota_media"] for j, _ in neighbors]
        if neighbor_cotas:
            elevation_diff[i] = max(neighbor_cotas) - min(neighbor_cotas)

        # Anomaly neighbor score
        neighbor_anomalies = [node_anomaly[j] for j, _ in neighbors]
        anomaly_neighbor_score[i] = np.mean(neighbor_anomalies)

        # Anomaly propagation (weighted by pipe capacity)
        total_weight = sum(diam for _, diam in neighbors)
        if total_weight > 0:
            anomaly_propagation[i] = sum(
                node_anomaly[j] * diam / total_weight
                for j, diam in neighbors
            )

        # Upstream/downstream analysis (water flows downhill)
        my_cota = nodes[i]["cota_media"]
        upstream_scores = []
        n_downstream = 0
        for j, diam in neighbors:
            neighbor_cota = nodes[j]["cota_media"]
            if neighbor_cota > my_cota + 1:  # +1m tolerancia
                upstream_scores.append(node_anomaly[j])
            elif neighbor_cota < my_cota - 1:
                n_downstream += 1

        if upstream_scores:
            upstream_anomaly[i] = np.mean(upstream_scores)
        downstream_count[i] = n_downstream

    # --- Betweenness centrality ---
    if HAS_NX:
        G = nx.Graph()
        for e in edges:
            G.add_edge(e["source"], e["target"], weight=e["mean_diameter"])
        nx_centrality = nx.betweenness_centrality(G, weight="weight")
        graph_centrality = np.array([nx_centrality.get(i, 0) for i in range(n_nodes)])
    else:
        graph_centrality = _betweenness_centrality_approx(n_nodes, edges)

    # --- Construir DataFrame con features por barrio ---
    # Agregar nodos por barrio (muchos sectores -> 1 barrio)
    barrio_features = []

    # Obtener lista unica de barrios del results_df
    if "barrio_key" in results_df.columns:
        all_barrio_keys = results_df["barrio_key"].unique()
    else:
        all_barrio_keys = []

    barrios_processed = set()
    for bk in all_barrio_keys:
        barrio_clean = bk.split("__")[0] if "__" in bk else bk
        if barrio_clean in barrios_processed:
            continue
        barrios_processed.add(barrio_clean)

        node_indices = barrio_to_nodes.get(barrio_clean, [])
        if not node_indices:
            # Intentar fuzzy match
            sector_names_norm = [_normalize_name(n["name"]) for n in nodes]
            sector_names_orig = [n["name"] for n in nodes]
            idx = _fuzzy_match_barrio(barrio_clean, sector_names_norm, sector_names_orig)
            if idx is not None:
                node_indices = [idx]

        if not node_indices:
            barrio_features.append({
                "barrio": barrio_clean,
                "degree": 0,
                "weighted_degree": 0.0,
                "elevation_diff": 0.0,
                "anomaly_neighbor_score": 0.0,
                "anomaly_propagation": 0.0,
                "graph_centrality": 0.0,
                "upstream_anomaly": 0.0,
                "downstream_count": 0,
                "n_sectors": 0,
                "mean_elevation": 0.0,
            })
            continue

        barrio_features.append({
            "barrio": barrio_clean,
            "degree": float(np.mean([degree[j] for j in node_indices])),
            "weighted_degree": float(np.sum([weighted_degree[j] for j in node_indices])),
            "elevation_diff": float(np.max([elevation_diff[j] for j in node_indices])),
            "anomaly_neighbor_score": float(np.mean([anomaly_neighbor_score[j] for j in node_indices])),
            "anomaly_propagation": float(np.mean([anomaly_propagation[j] for j in node_indices])),
            "graph_centrality": float(np.mean([graph_centrality[j] for j in node_indices])),
            "upstream_anomaly": float(np.mean([upstream_anomaly[j] for j in node_indices])),
            "downstream_count": float(np.sum([downstream_count[j] for j in node_indices])),
            "n_sectors": len(node_indices),
            "mean_elevation": float(np.mean([nodes[j]["cota_media"] for j in node_indices])),
        })

    graph_features_df = pd.DataFrame(barrio_features)
    return graph_features_df


# ---------------------------------------------------------------------------
# Function 3: detect_network_anomalies
# ---------------------------------------------------------------------------

def detect_network_anomalies(
    graph_features: pd.DataFrame,
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detecta anomalias explicadas por la topologia de la red.

    Tipos:
      - PROPAGATION: sector anomalo cuyos vecinos aguas arriba tambien son anomalos
        -> probable fuga en tuberia aguas arriba
      - ISOLATED: sector anomalo sin vecinos anomalos -> posible fraude local
      - PRESSURE: sector en zona de elevacion extrema con anomalia -> perdidas
        por presion
      - NORMAL: sin anomalia de red significativa

    Retorna DataFrame con columnas: barrio, network_anomaly_type, network_confidence
    """
    # Obtener anomaly_score medio por barrio
    if "barrio_key" in results_df.columns and "anomaly_score" in results_df.columns:
        temp = results_df.copy()
        temp["_barrio"] = temp["barrio_key"].str.split("__").str[0]
        barrio_score = temp.groupby("_barrio")["anomaly_score"].mean()
    elif "barrio" in results_df.columns and "anomaly_score" in results_df.columns:
        barrio_score = results_df.groupby("barrio")["anomaly_score"].mean()
    else:
        barrio_score = pd.Series(dtype=float)

    # Obtener n_models_detecting medio por barrio
    if "barrio_key" in results_df.columns and "n_models_detecting" in results_df.columns:
        temp = results_df.copy()
        temp["_barrio"] = temp["barrio_key"].str.split("__").str[0]
        barrio_n_models = temp.groupby("_barrio")["n_models_detecting"].mean()
    else:
        barrio_n_models = pd.Series(dtype=float)

    # Umbral para considerar "anomalo"
    if len(barrio_score) > 0:
        anomaly_threshold = barrio_score.quantile(0.75)
    else:
        anomaly_threshold = 0.5

    results = []
    for _, row in graph_features.iterrows():
        barrio = row["barrio"]
        score = barrio_score.get(barrio, 0)
        n_models = barrio_n_models.get(barrio, 0)
        is_anomalous = score > anomaly_threshold

        net_type = "NORMAL"
        confidence = 0.0

        if is_anomalous:
            upstream_anom = row.get("upstream_anomaly", 0)
            neighbor_anom = row.get("anomaly_neighbor_score", 0)
            propagation = row.get("anomaly_propagation", 0)
            elevation = row.get("mean_elevation", 0)
            elev_diff = row.get("elevation_diff", 0)

            # Check propagation: upstream anomaly is high
            if upstream_anom > anomaly_threshold * 0.8:
                net_type = "PROPAGATION"
                confidence = min(1.0, (upstream_anom / max(anomaly_threshold, 1e-6)) * 0.5
                                + (propagation / max(anomaly_threshold, 1e-6)) * 0.3
                                + (score / max(anomaly_threshold, 1e-6)) * 0.2)

            # Check isolated: no anomalous neighbors
            elif neighbor_anom < anomaly_threshold * 0.3:
                net_type = "ISOLATED"
                confidence = min(1.0, (score / max(anomaly_threshold, 1e-6)) * 0.5
                                + (1 - neighbor_anom / max(score, 1e-6)) * 0.3
                                + (n_models / 10.0) * 0.2)

            # Check pressure zone: extreme elevation with high elevation diff
            elif elev_diff > 30 or elevation > graph_features["mean_elevation"].quantile(0.9):
                net_type = "PRESSURE"
                confidence = min(1.0, (elev_diff / 100.0) * 0.4
                                + (score / max(anomaly_threshold, 1e-6)) * 0.4
                                + (elevation / max(graph_features["mean_elevation"].max(), 1)) * 0.2)

            else:
                # Anomalous but no clear network pattern
                net_type = "PROPAGATION" if propagation > neighbor_anom else "ISOLATED"
                confidence = min(1.0, score / max(anomaly_threshold, 1e-6) * 0.3)

        results.append({
            "barrio": barrio,
            "anomaly_score": score,
            "network_anomaly_type": net_type,
            "network_confidence": round(confidence, 4),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Function 4: run_graph_analysis
# ---------------------------------------------------------------------------

def run_graph_analysis(
    results_df: pd.DataFrame,
    tuberias_path: str = "data/tuberias.json",
    sectores_path: str = "data/sectores_de_consumo.json",
) -> dict:
    """
    Punto de entrada principal.

    1. Construye la red de agua desde datos GIS
    2. Calcula features de grafo
    3. Detecta anomalias de red
    4. Retorna dict con resultados

    Args:
        results_df: DataFrame con resultados de otros modelos
                    (barrio_key, anomaly_score, n_models_detecting, etc.)
        tuberias_path: ruta al archivo tuberias.json
        sectores_path: ruta al archivo sectores_de_consumo.json

    Returns:
        dict con: network_stats, graph_features_df, network_anomalies_df, network
    """
    print("=" * 60)
    print("M14 - GRAPH NETWORK DETECTOR")
    print("=" * 60)

    # 1. Build network
    print("\n[1/3] Construyendo red de agua desde datos GIS...")
    network = build_water_network(tuberias_path, sectores_path)
    stats = network["stats"]
    print(f"  Red: {stats['n_sectors']} sectores, {stats['n_pipes']} tuberias, "
          f"{stats['n_edges']} conexiones entre sectores")
    print(f"  Diametro medio: {stats['diameter_mean']:.0f}mm "
          f"(rango: {stats['diameter_min']:.0f}-{stats['diameter_max']:.0f}mm)")
    print(f"  Sectores con barrio mapeado: {stats['sectors_with_barrio']}")

    # 2. Compute graph features
    print("\n[2/3] Calculando features de grafo...")
    graph_features_df = compute_graph_features(network, results_df)
    n_with_connections = (graph_features_df["degree"] > 0).sum()
    print(f"  Barrios con conexiones de red: {n_with_connections}/{len(graph_features_df)}")
    if len(graph_features_df) > 0 and "graph_centrality" in graph_features_df.columns:
        top_central = graph_features_df.nlargest(3, "graph_centrality")
        print("  Top 3 barrios mas centrales en la red:")
        for _, r in top_central.iterrows():
            print(f"    - {r['barrio']}: centralidad={r['graph_centrality']:.4f}, "
                  f"grado={r['degree']:.0f}")

    # 3. Detect network anomalies
    print("\n[3/3] Detectando anomalias de red...")
    network_anomalies_df = detect_network_anomalies(graph_features_df, results_df)

    n_propagation = (network_anomalies_df["network_anomaly_type"] == "PROPAGATION").sum()
    n_isolated = (network_anomalies_df["network_anomaly_type"] == "ISOLATED").sum()
    n_pressure = (network_anomalies_df["network_anomaly_type"] == "PRESSURE").sum()
    print(f"  Anomalias por propagacion: {n_propagation}")
    print(f"  Anomalias aisladas: {n_isolated}")
    print(f"  Anomalias por presion: {n_pressure}")

    return {
        "network_stats": stats,
        "graph_features_df": graph_features_df,
        "network_anomalies_df": network_anomalies_df,
        "network": network,
    }


# ---------------------------------------------------------------------------
# Function 5: graph_network_summary
# ---------------------------------------------------------------------------

def graph_network_summary(analysis_results: dict) -> None:
    """
    Imprime resumen legible del analisis de red.

    Incluye:
      - Estadisticas de la red
      - Clasificacion de anomalias
      - Visualizacion ASCII de los caminos mas anomalos
      - Hallazgo clave sobre Colonia Requena
    """
    stats = analysis_results["network_stats"]
    gf = analysis_results["graph_features_df"]
    na = analysis_results["network_anomalies_df"]
    network = analysis_results["network"]

    print()
    print("=" * 60)
    print("RESUMEN: ANALISIS DE RED DE AGUA")
    print("=" * 60)

    # --- Estadisticas de red ---
    print(f"\nRed de agua: {stats['n_sectors']} sectores conectados por "
          f"{stats['n_pipes']} tuberias")
    print(f"Diametro medio: {stats['diameter_mean']:.0f}mm, "
          f"rango: {stats['diameter_min']:.0f}-{stats['diameter_max']:.0f}mm")
    print(f"Conexiones unicas entre sectores: {stats['n_edges']}")

    # --- Conteos de anomalias ---
    n_propagation = (na["network_anomaly_type"] == "PROPAGATION").sum()
    n_isolated = (na["network_anomaly_type"] == "ISOLATED").sum()
    n_pressure = (na["network_anomaly_type"] == "PRESSURE").sum()

    print(f"\n{n_propagation} anomalias por propagacion (fuga aguas arriba)")
    print(f"{n_isolated} anomalias aisladas (posible fraude local)")
    print(f"{n_pressure} anomalias en zonas de presion extrema")

    # --- Top anomalias por tipo ---
    if n_propagation > 0:
        print("\n--- PROPAGACION (fuga aguas arriba) ---")
        prop = na[na["network_anomaly_type"] == "PROPAGATION"].nlargest(
            min(5, n_propagation), "network_confidence")
        for _, r in prop.iterrows():
            print(f"  {r['barrio']}: score={r['anomaly_score']:.3f}, "
                  f"confianza={r['network_confidence']:.2f}")

    if n_isolated > 0:
        print("\n--- AISLADAS (posible fraude local) ---")
        isol = na[na["network_anomaly_type"] == "ISOLATED"].nlargest(
            min(5, n_isolated), "network_confidence")
        for _, r in isol.iterrows():
            print(f"  {r['barrio']}: score={r['anomaly_score']:.3f}, "
                  f"confianza={r['network_confidence']:.2f}")

    if n_pressure > 0:
        print("\n--- PRESION EXTREMA ---")
        pres = na[na["network_anomaly_type"] == "PRESSURE"].nlargest(
            min(5, n_pressure), "network_confidence")
        for _, r in pres.iterrows():
            print(f"  {r['barrio']}: score={r['anomaly_score']:.3f}, "
                  f"confianza={r['network_confidence']:.2f}")

    # --- Visualizacion ASCII: top 5 caminos anomalos ---
    print("\n--- RED: Caminos mas anomalos ---")
    _print_anomalous_paths(network, na, gf, top_n=5)

    # --- Hallazgo clave ---
    _print_key_finding(network, na, gf)

    print()


def _print_anomalous_paths(network, anomalies_df, graph_features, top_n=5):
    """
    Muestra los caminos mas anomalos a traves de la red en formato ASCII.
    Busca pares de nodos anomalos conectados y muestra la cadena.
    """
    nodes = network["nodes"]
    edges = network["edges"]
    adj_list = _build_adjacency_list(len(nodes), edges)
    barrio_to_nodes = network["barrio_to_node_indices"]

    # Obtener score por nodo
    anomaly_by_barrio = dict(zip(anomalies_df["barrio"], anomalies_df["anomaly_score"]))

    # Buscar pares de barrios anomalos conectados
    anomalous_pairs = []
    seen = set()

    for e in edges:
        src_node = nodes[e["source"]]
        tgt_node = nodes[e["target"]]
        src_barrio = src_node["barrio"]
        tgt_barrio = tgt_node["barrio"]

        if not src_barrio or not tgt_barrio:
            continue
        if src_barrio == tgt_barrio:
            continue

        pair_key = tuple(sorted([src_barrio, tgt_barrio]))
        if pair_key in seen:
            continue
        seen.add(pair_key)

        src_score = anomaly_by_barrio.get(src_barrio, 0)
        tgt_score = anomaly_by_barrio.get(tgt_barrio, 0)
        combined = src_score + tgt_score

        if combined > 0:
            # Determinar flujo por elevacion
            src_elev = src_node["cota_media"]
            tgt_elev = tgt_node["cota_media"]
            if src_elev >= tgt_elev:
                upstream, downstream = src_barrio, tgt_barrio
                up_score, down_score = src_score, tgt_score
                up_elev, down_elev = src_elev, tgt_elev
            else:
                upstream, downstream = tgt_barrio, src_barrio
                up_score, down_score = tgt_score, src_score
                up_elev, down_elev = tgt_elev, src_elev

            anomalous_pairs.append({
                "upstream": upstream,
                "downstream": downstream,
                "up_score": up_score,
                "down_score": down_score,
                "up_elev": up_elev,
                "down_elev": down_elev,
                "combined": combined,
                "diameter": e["mean_diameter"],
            })

    # Ordenar por score combinado
    anomalous_pairs.sort(key=lambda x: x["combined"], reverse=True)

    if not anomalous_pairs:
        print("  (sin caminos anomalos detectados)")
        return

    for i, p in enumerate(anomalous_pairs[:top_n]):
        elev_arrow = f"{p['up_elev']:.0f}m" if p['up_elev'] else "?"
        elev_arrow2 = f"{p['down_elev']:.0f}m" if p['down_elev'] else "?"
        diam_str = f"{p['diameter']:.0f}mm"

        # Marcador de anomalia
        up_mark = "(!)" if p["up_score"] > 0.1 else "   "
        down_mark = "(!)" if p["down_score"] > 0.1 else "   "

        print(f"  {i+1}. {up_mark} {p['upstream'][:25]:<25} ({elev_arrow}) "
              f"--[{diam_str}]--> "
              f"{down_mark} {p['downstream'][:25]:<25} ({elev_arrow2})")
        print(f"     score: {p['up_score']:.3f} {'>>>' if p['up_score'] > 0.1 else '---'} "
              f"{p['down_score']:.3f}")


def _print_key_finding(network, anomalies_df, graph_features):
    """Imprime hallazgo clave sobre Colonia Requena y su contexto de red."""
    print("\n--- HALLAZGO CLAVE ---")

    nodes = network["nodes"]
    edges = network["edges"]
    barrio_to_nodes = network["barrio_to_node_indices"]
    adj_list = _build_adjacency_list(len(nodes), edges)

    # Buscar Colonia Requena
    requena_indices = barrio_to_nodes.get("34-COLONIA REQUENA", [])
    if not requena_indices:
        print("  Colonia Requena no encontrada en la red.")
        return

    # Encontrar vecinos aguas arriba de Colonia Requena
    requena_elev = np.mean([nodes[i]["cota_media"] for i in requena_indices])
    upstream_barrios = set()

    for ri in requena_indices:
        for neighbor_idx, diam in adj_list[ri]:
            neighbor_node = nodes[neighbor_idx]
            if neighbor_node["cota_media"] > requena_elev and neighbor_node["barrio"]:
                upstream_barrios.add(neighbor_node["barrio"])

    # Buscar ANR de Colonia Requena en anomalies
    requena_row = anomalies_df[anomalies_df["barrio"] == "34-COLONIA REQUENA"]
    requena_score = requena_row["anomaly_score"].values[0] if len(requena_row) > 0 else 0

    # Buscar en graph_features para la elevacion
    gf_requena = graph_features[graph_features["barrio"] == "34-COLONIA REQUENA"]
    requena_upstream_anom = gf_requena["upstream_anomaly"].values[0] if len(gf_requena) > 0 else 0

    if upstream_barrios:
        upstream_list = ", ".join(sorted(upstream_barrios)[:3])
        print(f"  Colonia Requena esta aguas abajo de {upstream_list}")
        print(f"  Score anomalia Colonia Requena: {requena_score:.3f}")
        print(f"  Score anomalia aguas arriba: {requena_upstream_anom:.3f}")
        if requena_score > 0.1:
            print(f"  -> La anomalia en Colonia Requena puede explicarse por "
                  f"propagacion desde sectores aguas arriba")
    else:
        print("  Colonia Requena: sin sectores aguas arriba detectados en la red")
        if requena_score > 0.1:
            print(f"  Score anomalia: {requena_score:.3f} "
                  f"-> anomalia aislada, posible problema local")


# ---------------------------------------------------------------------------
# Main entry point (para test directo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Cargar resultados
    results_path = "results_full.csv"
    if not os.path.exists(results_path):
        results_path = "results_all_models.csv"
    if not os.path.exists(results_path):
        print(f"ERROR: No se encuentra archivo de resultados.")
        sys.exit(1)

    print(f"Cargando resultados desde {results_path}...")
    results_df = pd.read_csv(results_path)

    analysis = run_graph_analysis(results_df)
    graph_network_summary(analysis)
