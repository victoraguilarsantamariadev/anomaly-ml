"""
Descarga NDVI de Sentinel-2 para Alicante via Copernicus Dataspace (openEO).

Requisitos:
  1. Cuenta en https://dataspace.copernicus.eu (gratis)
  2. pip install openeo

Uso:
  python download_ndvi.py

Se abrira el navegador para login. Despues descarga automaticamente
el NDVI medio mensual para el bbox de Alicante y lo guarda como CSV.
"""

import openeo
import json
import csv
import os

# Bounding box de Alicante municipio (lon/lat)
ALICANTE_BBOX = {
    "west": -0.56,
    "south": 38.28,
    "east": -0.38,
    "north": 38.43,
}

# Periodos: verano (jul-ago) e invierno (ene-feb) para comparar
PERIODS = {
    "summer_2024": ("2024-07-01", "2024-08-31"),
    "winter_2024": ("2024-01-01", "2024-02-28"),
    "summer_2023": ("2023-07-01", "2023-08-31"),
    "winter_2023": ("2023-01-01", "2023-02-28"),
}


def main():
    print("Conectando a Copernicus Dataspace (openEO)...")
    connection = openeo.connect("openeo.dataspace.copernicus.eu")

    print("Autenticando (se abrira el navegador)...")
    connection.authenticate_oidc()
    print("Autenticado OK\n")

    results = {}

    for period_name, (start, end) in PERIODS.items():
        print(f"Procesando {period_name} ({start} a {end})...")

        # Cargar Sentinel-2 L2A (bandas B04=rojo, B08=NIR)
        s2 = connection.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=ALICANTE_BBOX,
            temporal_extent=[start, end],
            bands=["B04", "B08"],
            max_cloud_cover=30,
        )

        # Calcular NDVI = (B08 - B04) / (B08 + B04)
        ndvi = s2.ndvi(nir="B08", red="B04")

        # Media temporal del periodo
        ndvi_mean = ndvi.reduce_dimension(dimension="t", reducer="mean")

        # Descargar como GeoTIFF
        output_path = f"data/ndvi_{period_name}.tif"
        print(f"  Descargando a {output_path}...")
        ndvi_mean.download(output_path, format="GTiff")
        print(f"  OK: {os.path.getsize(output_path) / 1024:.0f} KB")

        results[period_name] = output_path

    print("\nDescarga completada!")
    print("Archivos generados:")
    for name, path in results.items():
        print(f"  {name}: {path}")

    print("\nPara calcular NDVI por barrio, ejecuta:")
    print("  python process_ndvi.py")


if __name__ == "__main__":
    main()
