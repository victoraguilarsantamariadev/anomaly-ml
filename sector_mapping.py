"""
Mapeo entre sectores hidraulicos (caudal horario) y barrios (hackathon).

Los sectores hidraulicos no coinciden 1:1 con barrios. Este mapeo es aproximado
basado en nombres y conocimiento del area de Alicante.

Sectores sin mapeo se ignoran (son zonas industriales, puertos, etc.).
"""

# Sector hidraulico → barrio del hackathon
# Un sector puede mapear a un barrio. Sectores sin match = None.
SECTOR_TO_BARRIO = {
    "1 CIUDAD JARDÍN": "31-CIUDAD JARDIN",
    "ALIPARK DL": "8-ALIPARK",
    "ALTOZANO": "25-ALTOZANO - CONDE LUMIARES",
    "BAHÍA LOS PINOS": None,  # zona costera sin barrio claro
    "BENALÚA DL": "1-BENALUA",
    "Bº GRANADA 1": None,  # subbario sin match directo
    "Bº LOS ÁNGELES": "6-LOS ANGELES",
    "CABO HUERTAS - PLAYA": "40-CABO DE LAS HUERTAS",
    "CENTRO COMERCIAL GRAN VÍA": None,  # zona comercial
    "CIUDAD DEPORTIVA DL": None,  # zona deportiva
    "COLONIA REQUENA": "34-COLONIA REQUENA",
    "COLONIA ROMANA": None,
    "CONDOMINA": None,
    "Campoamor Alto": "5-CAMPOAMOR",
    "DIPUTACIÓN DL": "14-ENSANCHE DIPUTACION",
    "Depósito Los Ángeles": None,  # infraestructura, no barrio
    "GARBINET NORTE 1": "19-GARBINET",
    "INFORMACIÓN DL": None,
    "LONJA": "4-MERCADO",
    "LONJA DL": "4-MERCADO",
    "Les Palmeretes": "28-EL PALMERAL",
    "MATADERO": None,
    "MERCADO DL": "4-MERCADO",
    "MUCHAVISTA - P.A.U. 5": None,  # fuera del area de barrios
    "MUELLE GRANELES DL": "55-PUERTO",
    "MUELLE LEVANTE DL": "55-PUERTO",
    "O.A.M.I 1": None,
    "P.A.U. 1 (norte+sur)": "41-PLAYA DE SAN JUAN",
    "P.A.U. 2": "41-PLAYA DE SAN JUAN",
    "PARQUE LO MORANT": "33- MORANT -SAN NICOLAS BARI",
    "PLAYA DE SAN JUAN 1": "41-PLAYA DE SAN JUAN",
    "PZA. MONTAÑETA": "3-CENTRO",
    "Pla-Hospital": "16-PLA DEL BON REPOS",
    "Postiguet": "22-CASCO ANTIGUO - SANTA CRUZ",
    "RABASA DL": "20-RABASA",
    "SANTO DOMINGO DL": "24-SAN BLAS - SANTO DOMINGO",
    "SH_Demo": None,  # sector demo/test
    "TOBO": None,
    "VALLONGA GLOBAL": "54-POLIGONO VALLONGA",
    "VALLONGA-TOLON DL": "54-POLIGONO VALLONGA",
    "VILLAFRANQUEZA": "VILLAFRANQUEZA",
    "VIRGEN DEL CARMEN 1000 Viv": "35-VIRGEN DEL CARMEN",
    "VIRGEN DEL REMEDIO": "32-VIRGEN DEL REMEDIO",
}


def get_mapped_sectors() -> dict[str, str]:
    """Devuelve solo los sectores que tienen mapeo a barrio."""
    return {k: v for k, v in SECTOR_TO_BARRIO.items() if v is not None}


def get_unmapped_sectors() -> list[str]:
    """Devuelve sectores sin mapeo (para verificacion)."""
    return [k for k, v in SECTOR_TO_BARRIO.items() if v is None]
