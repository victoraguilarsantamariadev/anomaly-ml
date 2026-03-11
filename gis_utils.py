"""
gis_utils.py
Convierte ESRI JSON (UTM EPSG:25830) a GeoJSON (WGS84 EPSG:4326) para Folium.
"""

import json
from pyproj import Transformer

_transformer = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)


def esri_to_geojson(esri_path, name_field=None):
    """Convierte ESRI JSON con rings (UTM 25830) a GeoJSON (WGS84).

    Parameters
    ----------
    esri_path : str
        Path to ESRI JSON file.
    name_field : str, optional
        Attribute field to use as feature name.

    Returns
    -------
    dict — GeoJSON FeatureCollection
    """
    with open(esri_path, encoding="latin-1") as f:
        data = json.load(f)

    features_in = data.get("features", data) if isinstance(data, dict) else data
    features_out = []

    for feat in features_in:
        attrs = feat.get("attributes", {})
        geom = feat.get("geometry", {})

        if "rings" in geom:
            coords_wgs84 = []
            for ring in geom["rings"]:
                coords_wgs84.append([
                    list(_transformer.transform(x, y)) for x, y in ring
                ])
            geometry = {"type": "Polygon", "coordinates": coords_wgs84}
        elif "paths" in geom:
            coords_wgs84 = []
            for path in geom["paths"]:
                coords_wgs84.append([
                    list(_transformer.transform(x, y)) for x, y in path
                ])
            geometry = {"type": "MultiLineString", "coordinates": coords_wgs84}
        elif "x" in geom and "y" in geom:
            lon, lat = _transformer.transform(geom["x"], geom["y"])
            geometry = {"type": "Point", "coordinates": [lon, lat]}
        else:
            continue

        features_out.append({
            "type": "Feature",
            "properties": attrs,
            "geometry": geometry,
        })

    return {"type": "FeatureCollection", "features": features_out}


def get_sector_centroids(geojson):
    """Get centroids (lat, lon) for each sector from GeoJSON."""
    centroids = {}
    for feat in geojson["features"]:
        name = feat["properties"].get("DCONS_PO_2", f"FID_{feat['properties'].get('FID', '?')}")
        coords = feat["geometry"]["coordinates"][0]  # outer ring
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        centroids[name] = (sum(lats) / len(lats), sum(lons) / len(lons))
    return centroids
