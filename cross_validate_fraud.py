"""
Cross-validacion: compara detecciones de nuestros modelos vs fraude real.

El dataset de cambios-de-contador tiene motivos sospechosos:
  - FP-FRAUDE POSIBLE (77 casos)
  - RB-ROBO (29 casos)
  - MR-MARCHA AL REVES (84 casos)

No tienen columna BARRIO, pero podemos correlacionar temporalmente:
  ¿En los meses con mas fraude real, nuestros modelos detectan mas anomalias?

Tambien genera estadisticas utiles para la presentacion del hackathon.

Uso:
  python cross_validate_fraud.py
  python cross_validate_fraud.py --results results.csv  # usar CSV previo de run_all_models.py
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

CAMBIOS_PATH = "data/cambios-de-contador-solo-alicante_hackaton-dataart-cambios-de-contador-solo-alicante.csv.csv"
DATA_PATH = "data/datos-hackathon-amaem.xlsx-set-de-datos-.csv"

SUSPICIOUS_MOTIVOS = [
    "FP-FRAUDE POSIBLE",
    "RB-ROBO",
    "MR-MARCHA AL REVES",
]


def load_fraud_timeline(cambios_path: str) -> pd.DataFrame:
    """Carga y agrega casos sospechosos por mes."""
    df = pd.read_csv(cambios_path)
    df["FECHA"] = pd.to_datetime(df["FECHA"])

    suspicious = df[df["MOTIVO_CAMBIO"].isin(SUSPICIOUS_MOTIVOS)].copy()
    suspicious["year_month"] = suspicious["FECHA"].dt.to_period("M")

    # Contar por mes y tipo
    by_month = (
        suspicious.groupby(["year_month", "MOTIVO_CAMBIO"])
        .size()
        .unstack(fill_value=0)
    )
    by_month["total_suspicious"] = by_month.sum(axis=1)

    return by_month, suspicious


def load_all_motivos_stats(cambios_path: str) -> pd.DataFrame:
    """Estadisticas generales de motivos de cambio."""
    df = pd.read_csv(cambios_path)
    return df["MOTIVO_CAMBIO"].value_counts()


def run_models_and_get_monthly_anomalies() -> pd.DataFrame:
    """Ejecuta pipeline rapido (M2+M5) y agrega anomalias por mes."""
    from train_local import load_hackathon_amaem
    from run_all_models import run_m2, run_m5, load_data

    df_all, _ = load_data(DATA_PATH)

    m2 = run_m2(df_all, contamination=0.01)
    m5 = run_m5(df_all, iqr_multiplier=3.0)

    # Agregar por mes
    results = []
    for name, df_model, col in [("M2", m2, "is_anomaly_m2"),
                                  ("M5_3sigma", m5, "is_anomaly_3sigma"),
                                  ("M5_iqr", m5, "is_anomaly_iqr")]:
        if df_model.empty or col not in df_model.columns:
            continue
        df_model = df_model.copy()
        df_model["year_month"] = pd.to_datetime(df_model["fecha"]).dt.to_period("M")
        monthly = df_model.groupby("year_month")[col].sum().rename(f"anomalies_{name}")
        results.append(monthly)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, axis=1).fillna(0)


def load_results_csv(path: str) -> pd.DataFrame:
    """Carga resultados previos de run_all_models.py y agrega por mes."""
    df = pd.read_csv(path)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["year_month"] = df["fecha"].dt.to_period("M")

    results = []
    for col in df.columns:
        if col.startswith("is_anomaly"):
            monthly = df.groupby("year_month")[col].sum().rename(f"anomalies_{col.replace('is_anomaly_', '')}")
            results.append(monthly)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, axis=1).fillna(0)


def compute_correlation(fraud_monthly: pd.DataFrame,
                        anomaly_monthly: pd.DataFrame) -> dict:
    """Calcula correlacion temporal entre fraude real y detecciones."""
    # Alinear por meses comunes
    common = fraud_monthly.index.intersection(anomaly_monthly.index)
    if len(common) < 3:
        return {"error": "Menos de 3 meses en comun"}

    fraud_vals = fraud_monthly.loc[common, "total_suspicious"].values.astype(float)
    results = {}

    for col in anomaly_monthly.columns:
        anom_vals = anomaly_monthly.loc[common, col].values.astype(float)
        if np.std(fraud_vals) > 0 and np.std(anom_vals) > 0:
            corr = np.corrcoef(fraud_vals, anom_vals)[0, 1]
        else:
            corr = 0.0
        results[col] = round(corr, 3)

    return results


def print_report(fraud_monthly, anomaly_monthly, motivos_stats, suspicious_df, correlations):
    """Imprime reporte completo."""
    print("=" * 70)
    print("  CROSS-VALIDACION: Fraude Real vs Detecciones AquaGuard AI")
    print("=" * 70)

    print("\n1. ESTADISTICAS DE CAMBIOS DE CONTADOR")
    print("-" * 50)
    total = motivos_stats.sum()
    print(f"   Total cambios de contador: {total:,}")
    for motivo in SUSPICIOUS_MOTIVOS:
        count = motivos_stats.get(motivo, 0)
        print(f"   {motivo}: {count} ({count/total*100:.2f}%)")
    suspicious_total = sum(motivos_stats.get(m, 0) for m in SUSPICIOUS_MOTIVOS)
    print(f"   TOTAL SOSPECHOSOS: {suspicious_total} ({suspicious_total/total*100:.2f}%)")

    print("\n2. TIMELINE DE FRAUDE REAL (periodo hackathon 2022-2024)")
    print("-" * 50)
    hackathon_period = fraud_monthly[
        (fraud_monthly.index >= pd.Period("2022-01", "M")) &
        (fraud_monthly.index <= pd.Period("2024-12", "M"))
    ]
    if len(hackathon_period) > 0:
        for idx, row in hackathon_period.iterrows():
            bar = "█" * int(row["total_suspicious"])
            print(f"   {idx}: {int(row['total_suspicious']):2d} casos {bar}")
        total_hp = int(hackathon_period["total_suspicious"].sum())
        print(f"   Total en periodo hackathon: {total_hp} casos sospechosos")
    else:
        print("   Sin datos en periodo hackathon")

    # Meses con fraude vs sin fraude
    print("\n3. CORRELACION TEMPORAL: Fraude Real vs Detecciones")
    print("-" * 50)
    if correlations:
        for model, corr in sorted(correlations.items(), key=lambda x: -abs(x[1])):
            if corr > 0.3:
                indicator = "POSITIVA"
            elif corr > 0:
                indicator = "debil positiva"
            elif corr > -0.3:
                indicator = "debil negativa"
            else:
                indicator = "NEGATIVA"
            print(f"   {model}: r={corr:+.3f} ({indicator})")

        print("\n   Interpretacion:")
        print("   r > 0.3 → nuestros modelos detectan MAS en meses con fraude real")
        print("   r ~ 0   → sin correlacion (puede ser que fraude es individual, no barrio)")
        print("   r < -0.3 → anticorrelacion (improbable)")
    else:
        print("   No hay suficientes datos para correlacion")

    # Analisis por emplazamiento
    print("\n4. PERFIL DE FRAUDE (para la presentacion)")
    print("-" * 50)
    if len(suspicious_df) > 0:
        print("   Emplazamiento mas comun en fraudes:")
        emp_counts = suspicious_df["EMPLAZAMIENTO"].value_counts().head(5)
        for emp, count in emp_counts.items():
            print(f"     {emp}: {count} ({count/len(suspicious_df)*100:.0f}%)")

        print(f"\n   Calibre mas comun en fraudes:")
        cal_counts = suspicious_df["CALIBRE"].value_counts().head(3)
        for cal, count in cal_counts.items():
            print(f"     Calibre {cal}: {count} ({count/len(suspicious_df)*100:.0f}%)")

    print("\n5. ARGUMENTO PARA EL JURADO")
    print("-" * 50)
    print(f"   De {total:,} cambios de contador en Alicante (2020-2025):")
    print(f"   - {suspicious_total} fueron por motivos sospechosos (fraude, robo, manipulacion)")
    print(f"   - Esto representa ~{suspicious_total/total*100:.1f}% del total")
    print(f"   - La mayoria en contadores EXTERIORES ({emp_counts.iloc[0] if len(emp_counts) > 0 else '?'} casos)")
    print(f"   - AquaGuard AI detecta anomalias a nivel de BARRIO que complementan")
    print(f"     la deteccion individual de Aguas de Alicante")


def main():
    parser = argparse.ArgumentParser(description="Cross-validacion fraude real vs detecciones")
    parser.add_argument("--results", type=str, default=None,
                        help="CSV de resultados previos de run_all_models.py")
    parser.add_argument("--skip-models", action="store_true",
                        help="Solo analizar fraude, no ejecutar modelos")
    args = parser.parse_args()

    if not Path(CAMBIOS_PATH).exists():
        print(f"ERROR: No se encuentra {CAMBIOS_PATH}")
        return

    # 1. Cargar datos de fraude
    fraud_monthly, suspicious_df = load_fraud_timeline(CAMBIOS_PATH)
    motivos_stats = load_all_motivos_stats(CAMBIOS_PATH)

    # 2. Obtener detecciones de nuestros modelos
    anomaly_monthly = pd.DataFrame()
    correlations = {}

    if args.results:
        anomaly_monthly = load_results_csv(args.results)
    elif not args.skip_models:
        print("Ejecutando M2+M5 para obtener detecciones...\n")
        anomaly_monthly = run_models_and_get_monthly_anomalies()

    # 3. Calcular correlacion
    if not anomaly_monthly.empty:
        correlations = compute_correlation(fraud_monthly, anomaly_monthly)

    # 4. Imprimir reporte
    print_report(fraud_monthly, anomaly_monthly, motivos_stats, suspicious_df, correlations)


if __name__ == "__main__":
    main()
