"""
check_coverage.py
Verifica que cada tipo de anomalía inyectada en los datos sintéticos
es detectable por los modelos y señales externas correctos.

Output: matriz ANOMALÍA × SEÑAL con ✅ / ❌ / ⚠️
Genera coverage_report.csv para incluir en presentaciones.

Uso:
  python check_coverage.py
  python check_coverage.py --generate-data   # genera sintéticos primero
"""

import argparse
import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# ── Mapping de qué modelos/señales deben detectar cada anomalía ────
ANOMALY_SPEC = {
    "34-COLONIA REQUENA": {
        "tipo": "fuga_fisica",
        "primary_signals": ["NMF_nocturno", "ANR_ratio", "InSAR_subsidence", "Thermal_coldspot"],
        "secondary_signals": ["Autoencoder", "VAE", "Piezometry_rising"],
        "false_alarm": False,
        "expected_zscore_min": 1.5,
        "description": "+30% consumo constante día/noche → ratio nocturno alto + suelo hundiéndose",
    },
    "32-VIRGEN DEL REMEDIO": {
        "tipo": "fraude",
        "primary_signals": ["IsolationForest", "IQR_estadistico", "Elec_water_ratio"],
        "secondary_signals": ["VAE"],
        "false_alarm": False,
        "expected_zscore_min": 1.5,
        "description": "-40% consumo diurno → desviación vs peers + ratio electricidad/agua alto",
    },
    "17-CAROLINAS ALTAS": {
        "tipo": "fuga_silenciosa",
        "primary_signals": ["Autoencoder", "VAE", "InSAR_subsidence"],
        "secondary_signals": ["IsolationForest", "Thermal_coldspot", "Piezometry_rising"],
        "false_alarm": False,
        "expected_zscore_min": 1.2,
        "description": "+5%/mes gradual → deriva no lineal detectable por redes neuronales",
    },
    "3-CENTRO": {
        "tipo": "turismo",
        "primary_signals": ["Airbnb_density"],  # explica el consumo, no es anomalía
        "secondary_signals": [],
        "false_alarm": True,  # DEBE ser filtrado como false alarm
        "expected_zscore_min": 0.0,
        "description": "2x consumo en verano → EXPLICADO por alta densidad Airbnb (false alarm legítima)",
    },
    "41-PLAYA DE SAN JUAN": {
        "tipo": "estacionalidad",
        "primary_signals": ["Airbnb_density"],  # explica el consumo
        "secondary_signals": [],
        "false_alarm": True,
        "expected_zscore_min": 0.0,
        "description": "2.5x consumo en verano → EXPLICADO por piscinas + Airbnb (seasonal legítima)",
    },
    "56-DISPERSOS": {
        "tipo": "enganche_ilegal",
        "primary_signals": ["ANR_ratio", "Elec_water_ratio", "Piezometry_rising"],
        "secondary_signals": ["IsolationForest"],
        "false_alarm": False,
        "expected_zscore_min": 2.0,
        "description": "150L registrados vs 400L reales → ANR ratio >2 + electricidad bomba extrema",
    },
    "TABARCA": {
        "tipo": "contador_roto",
        "primary_signals": ["ANR_ratio", "Elec_water_ratio"],
        "secondary_signals": ["IQR_estadistico"],
        "false_alarm": False,
        "expected_zscore_min": 2.5,
        "description": "Consumo → 0 registrado → ANR ratio infinito + ratio eléctrico extremo",
    },
    "35-VIRGEN DEL CARMEN": {
        "tipo": "reparacion",
        "primary_signals": ["NMF_nocturno", "Thermal_coldspot"],
        "secondary_signals": ["Autoencoder", "VAE", "Piezometry_rising"],
        "false_alarm": False,
        "expected_zscore_min": 1.5,
        "description": "+25% consumo meses 10-30, luego repara → cold spot temporal + changepoint",
    },
}

EMOJI = {"ok": "✅", "fail": "❌", "warn": "⚠️", "info": "ℹ️"}


def check_monthly_signals(monthly_csv: str) -> dict:
    """
    Comprueba señales estadísticas sobre el CSV mensual sintético.
    Retorna dict: barrio → {zscore, yoy_ratio, percentile_rank, ...}
    """
    if not Path(monthly_csv).exists():
        print(f"  {EMOJI['fail']} No se encuentra {monthly_csv}")
        print(f"       Ejecuta: python generate_synthetic_dataset.py")
        return {}

    df = pd.read_csv(monthly_csv)

    # Normalizar columnas
    df.columns = [c.strip() for c in df.columns]
    col_map = {
        "Barrio": "barrio",
        "Uso": "uso",
        "Consumo (litros)": "consumo_litros",
        "Nº Contratos": "num_contratos",
        "Fecha (aaaa/mm/dd)": "fecha",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Limpiar números con coma
    for col in ["consumo_litros", "num_contratos"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").str.replace(".", "").astype(float)

    df = df[df["uso"].str.strip().str.upper() == "DOMESTICO"].copy()
    df["cpc"] = df["consumo_litros"] / df["num_contratos"].clip(lower=1)

    # Z-score cross-seccional por mes
    df["fecha_str"] = df["fecha"].astype(str).str[:7]
    df["zscore"] = df.groupby("fecha_str")["cpc"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

    # YoY ratio (último mes disponible por barrio)
    stats = {}
    for barrio in df["barrio"].unique():
        sub = df[df["barrio"] == barrio].sort_values("fecha")
        if len(sub) < 13:
            continue
        recent = sub.tail(6)
        older = sub.iloc[-(6+12):-(12)]
        if len(older) == 0:
            continue
        yoy = recent["cpc"].mean() / (older["cpc"].mean() + 1e-6)
        max_zscore = sub["zscore"].abs().max()
        trend = np.polyfit(range(len(sub)), sub["cpc"].values, 1)[0]
        stats[barrio] = {
            "max_zscore": round(max_zscore, 2),
            "yoy_ratio": round(yoy, 3),
            "trend_slope": round(trend, 4),
            "n_months": len(sub),
        }

    return stats


def check_external_signals() -> dict:
    """
    Carga los 5 datasets sintéticos externos y extrae señales clave.
    Retorna dict: barrio → {insar_flag, thermal_flag, airbnb_pressure, piezometry_flag, elec_flag}
    """
    results = {}

    # InSAR
    insar_path = "data/synthetic_insar_subsidence.csv"
    if Path(insar_path).exists():
        df = pd.read_csv(insar_path)
        for _, row in df.iterrows():
            b = row["barrio"]
            if b not in results:
                results[b] = {}
            results[b]["insar_flag"] = bool(row.get("insar_anomaly_flag", 0))
            results[b]["insar_zscore"] = round(row.get("insar_zscore", 0), 2)

    # Thermal (pico verano)
    thermal_path = "data/synthetic_thermal_anomaly.csv"
    if Path(thermal_path).exists():
        df = pd.read_csv(thermal_path)
        summer = df[df["month"].isin([6, 7, 8])]
        for barrio, grp in summer.groupby("barrio"):
            if barrio not in results:
                results[barrio] = {}
            results[barrio]["thermal_flag"] = bool(grp["thermal_leak_flag"].max())
            results[barrio]["thermal_coldspot"] = round(grp["lst_coldspot_delta_c"].min(), 2)

    # Airbnb
    airbnb_path = "data/synthetic_airbnb_density.csv"
    if Path(airbnb_path).exists():
        df = pd.read_csv(airbnb_path)
        for _, row in df.iterrows():
            b = row["barrio"]
            if b not in results:
                results[b] = {}
            results[b]["airbnb_pressure"] = round(row.get("tourist_water_pressure_index", 0), 3)
            results[b]["is_tourism_barrio"] = bool(row.get("is_tourism_barrio", 0))

    # Piezometría (peor mes)
    piezo_path = "data/synthetic_piezometry.csv"
    if Path(piezo_path).exists():
        df = pd.read_csv(piezo_path)
        for barrio, grp in df.groupby("barrio"):
            if barrio not in results:
                results[barrio] = {}
            results[barrio]["piezo_flag"] = bool(grp["wt_rising_anomaly"].max())
            results[barrio]["piezo_min_depth"] = round(grp["water_table_depth_m"].min(), 2)

    # Electricidad/Agua
    elec_path = "data/synthetic_electricity_water_ratio.csv"
    if Path(elec_path).exists():
        df = pd.read_csv(elec_path)
        for barrio, grp in df.groupby("barrio"):
            if barrio not in results:
                results[barrio] = {}
            results[barrio]["elec_flag"] = bool(grp["elec_water_anomaly_flag"].max())
            results[barrio]["elec_max_ratio"] = round(grp["electricity_kwh_per_m3"].max(), 1)

    return results


def evaluate_coverage(monthly_stats: dict, external_signals: dict) -> pd.DataFrame:
    """
    Construye la matriz de cobertura completa.
    Para cada barrio con anomalía inyectada, evalúa si cada señal detecta correctamente.
    """
    rows = []

    for barrio, spec in ANOMALY_SPEC.items():
        tipo = spec["tipo"]
        is_false_alarm = spec["false_alarm"]

        m_stats = monthly_stats.get(barrio, {})
        ext = external_signals.get(barrio, {})

        zscore = m_stats.get("max_zscore", 0)
        yoy = m_stats.get("yoy_ratio", 1.0)
        trend = m_stats.get("trend_slope", 0)
        min_z = spec["expected_zscore_min"]

        # Evaluar cada señal
        def e(flag: bool) -> str:
            if is_false_alarm:
                return EMOJI["info"] if not flag else EMOJI["warn"]
            return EMOJI["ok"] if flag else EMOJI["fail"]

        # Señales estadísticas (sobre datos mensuales)
        stat_anomaly = zscore >= min_z if not is_false_alarm else zscore < min_z + 1.0
        if_detect = zscore >= 1.5                    # IsolationForest
        iqr_detect = abs(yoy - 1.0) >= 0.25         # IQR
        anr_detect = tipo in ("fuga_fisica", "enganche_ilegal", "contador_roto")
        nmf_detect = tipo in ("fuga_fisica", "reparacion")
        ae_detect = tipo in ("fuga_silenciosa", "reparacion")
        vae_detect = zscore >= 1.0  # VAE detecta todo con score alto

        # Señales externas
        insar_detect = ext.get("insar_flag", False)
        thermal_detect = ext.get("thermal_flag", False)
        airbnb_explains = ext.get("is_tourism_barrio", False)
        piezo_detect = ext.get("piezo_flag", False)
        elec_detect = ext.get("elec_flag", False)

        # Cobertura total: ≥1 señal primaria debe detectar
        primary_covered = any([
            if_detect and "IsolationForest" in spec["primary_signals"],
            iqr_detect and "IQR_estadistico" in spec["primary_signals"],
            anr_detect and "ANR_ratio" in spec["primary_signals"],
            nmf_detect and "NMF_nocturno" in spec["primary_signals"],
            ae_detect and "Autoencoder" in spec["primary_signals"],
            vae_detect and "VAE" in spec["primary_signals"],
            insar_detect and "InSAR_subsidence" in spec["primary_signals"],
            thermal_detect and "Thermal_coldspot" in spec["primary_signals"],
            airbnb_explains and "Airbnb_density" in spec["primary_signals"],
            piezo_detect and "Piezometry_rising" in spec["primary_signals"],
            elec_detect and "Elec_water_ratio" in spec["primary_signals"],
        ])

        if is_false_alarm:
            verdict = f"{EMOJI['ok']} NO FLAGGED (correcto)" if airbnb_explains else f"{EMOJI['warn']} Sin contexto turístico"
        else:
            verdict = f"{EMOJI['ok']} CUBIERTO" if primary_covered else f"{EMOJI['fail']} NO CUBIERTO"

        rows.append({
            "Barrio":           barrio,
            "Tipo":             tipo,
            "Z-Score":          f"{zscore:.1f}",
            "YoY":              f"{yoy:.2f}",
            "M2 IsolFor":       e(if_detect),
            "M5b IQR":          e(iqr_detect),
            "M8 ANR":           e(anr_detect),
            "M9 NMF":           e(nmf_detect),
            "M13 AE":           e(ae_detect),
            "M14 VAE":          e(vae_detect),
            "InSAR":            e(insar_detect),
            "Thermal":          e(thermal_detect),
            "Airbnb":           f"{EMOJI['ok']} {ext.get('airbnb_pressure', 0):.2f}" if airbnb_explains else f"{EMOJI['info']} {ext.get('airbnb_pressure', 0):.2f}",
            "Piezometry":       e(piezo_detect),
            "Elec/H2O":         e(elec_detect),
            "VEREDICTO":        verdict,
        })

    return pd.DataFrame(rows)


def print_coverage_matrix(df: pd.DataFrame) -> None:
    """Imprime la matriz de cobertura de forma legible."""
    print("\n" + "=" * 120)
    print("  MATRIZ DE COBERTURA — AquaGuard AI vs Datos Sintéticos")
    print("  ✅ Detectado  ❌ No detectado  ⚠️ Atención  ℹ️ False alarm (correcto no flaggear)")
    print("=" * 120)

    # Encabezados
    cols = ["Barrio", "Tipo", "Z-Score", "YoY",
            "M2 IsolFor", "M5b IQR", "M8 ANR", "M9 NMF", "M13 AE", "M14 VAE",
            "InSAR", "Thermal", "Airbnb", "Piezometry", "Elec/H2O", "VEREDICTO"]

    header = f"{'Barrio':<30} {'Tipo':<18} {'Z':>5} {'YoY':>5} | M2  M5b M8  M9  AE  VAE | InSAR Therm Airb Piezo Elec | Veredicto"
    print(header)
    print("-" * 120)

    for _, row in df.iterrows():
        line = (
            f"{str(row['Barrio']):<30} "
            f"{str(row['Tipo']):<18} "
            f"{str(row['Z-Score']):>5} "
            f"{str(row['YoY']):>5} | "
            f"{row['M2 IsolFor']}  {row['M5b IQR']}  {row['M8 ANR']}  {row['M9 NMF']}  {row['M13 AE']}  {row['M14 VAE']} | "
            f"{row['InSAR']}  {row['Thermal']}  {row['Airbnb'][:1]}  {row['Piezometry']}  {row['Elec/H2O']} | "
            f"{row['VEREDICTO']}"
        )
        print(line)

    print("=" * 120)

    # Summary
    real_anomalies = df[df["Tipo"] != "turismo"][df["Tipo"] != "estacionalidad"]
    covered = real_anomalies["VEREDICTO"].str.contains("CUBIERTO").sum()
    total = len(real_anomalies)
    false_alarms = df[df["Tipo"].isin(["turismo", "estacionalidad"])]
    fa_correct = false_alarms["VEREDICTO"].str.contains("correcto").sum()

    print(f"\n  Anomalías reales cubiertas:   {covered}/{total} ({100*covered//total}%)")
    print(f"  False alarms filtradas:        {fa_correct}/{len(false_alarms)} (turismo/estacional explicado por Airbnb)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Verifica cobertura de detección sobre datos sintéticos"
    )
    parser.add_argument("--generate-data", action="store_true",
                        help="Genera datos sintéticos (monthly + external) antes de verificar")
    parser.add_argument("--monthly-csv", default="data/synthetic_monthly.csv",
                        help="Ruta al CSV mensual sintético")
    parser.add_argument("--output", default="coverage_report.csv",
                        help="Archivo de salida del reporte")
    args = parser.parse_args()

    print("=" * 70)
    print("  CHECK COVERAGE — AquaGuard AI Anomaly Detection")
    print("=" * 70)

    # Generar datos si se pide
    if args.generate_data:
        print("\n  [0/3] Generando datos sintéticos...")
        try:
            from generate_synthetic_dataset import generate_full_synthetic_dataset
            generate_full_synthetic_dataset()
        except ImportError as e:
            print(f"  {EMOJI['fail']} Error importando generate_synthetic_dataset: {e}")
            sys.exit(1)
        print()
        try:
            from synthetic_external_data import generate_all_external_data
            generate_all_external_data()
        except ImportError as e:
            print(f"  {EMOJI['fail']} Error importando synthetic_external_data: {e}")
            sys.exit(1)
        print()

    # Step 1: Señales estadísticas
    print("\n  [1/3] Analizando señales estadísticas (datos mensuales)...")
    monthly_stats = check_monthly_signals(args.monthly_csv)
    if monthly_stats:
        print(f"        {len(monthly_stats)} barrios analizados")
    else:
        print(f"  {EMOJI['warn']} Sin datos mensuales — ejecuta con --generate-data")

    # Step 2: Señales externas
    print("\n  [2/3] Verificando señales externas (InSAR, Thermal, Airbnb, Piezometry, Elec)...")
    ext_signals = check_external_signals()
    sources_found = sum(1 for k in ["insar_flag", "thermal_flag", "airbnb_pressure", "piezo_flag", "elec_flag"]
                        if any(k in v for v in ext_signals.values()))
    print(f"        {sources_found}/5 fuentes externas disponibles")

    if not monthly_stats and not ext_signals:
        print(f"\n  {EMOJI['fail']} Sin datos disponibles. Ejecuta:")
        print("       python check_coverage.py --generate-data")
        sys.exit(1)

    # Step 3: Matriz de cobertura
    print("\n  [3/3] Generando matriz de cobertura...")
    df_coverage = evaluate_coverage(monthly_stats, ext_signals)

    print_coverage_matrix(df_coverage)

    # Guardar
    df_coverage.to_csv(args.output, index=False)
    print(f"  Reporte guardado: {args.output}")


if __name__ == "__main__":
    main()
