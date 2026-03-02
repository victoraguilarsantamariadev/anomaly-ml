"""
Script para entrenar y probar el modelo en local.

Usa este script para:
  - Practicar con datos de Datadis, Kaggle, o el CSV del hackathon
  - Ver qué anomalías detecta
  - Ajustar parámetros antes de conectar al microservicio

Uso:
  python train_local.py --file data/mi_contador.csv --id contador_001
  python train_local.py --file data/HACKATHON_AMAEM.csv --mode hackathon
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # sin display (funciona en servidor)
import matplotlib.pyplot as plt
from pathlib import Path

from feature_engineering import compute_daily_features, add_rolling_features
from model import train, score_batch, model_exists


# =============================================================
# ADAPTADORES DE FORMATO — añade el tuyo si tienes otro formato
# =============================================================

def load_datadis(file_path: str) -> pd.DataFrame:
    """
    Carga datos de Datadis (electricidad española).
    Formato: CUPS;Fecha;Hora;Consumo_KWh;Metodo_obtencion
    """
    df = pd.read_csv(file_path, sep=";", encoding="utf-8")
    df.columns = df.columns.str.strip()

    # Datadis usa Fecha (YYYY/MM/DD) y Hora (1-24)
    df["timestamp"] = pd.to_datetime(
        df["Fecha"] + " " + (df["Hora"].astype(int) - 1).astype(str) + ":00",
        format="%Y/%m/%d %H:%M"
    )
    df = df.rename(columns={"Consumo_KWh": "consumption"})
    return df[["timestamp", "consumption"]]


def load_kaggle_smart_meters(file_path: str) -> pd.DataFrame:
    """
    Carga datos de Kaggle Smart Meters London.
    Formato: LCLid,tstp,energy(kWh/hh)
    """
    df = pd.read_csv(file_path)
    df = df.rename(columns={"tstp": "timestamp", "energy(kWh/hh)": "consumption"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce").fillna(0)
    return df[["timestamp", "consumption"]]


def load_hackathon_amaem(file_path: str) -> pd.DataFrame:
    """
    Carga el CSV mensual del hackathon de Aguas de Alicante.
    Formato real: Barrio,Uso,"Fecha (aaaa/mm/dd)","Consumo (litros)","Nº Contratos"
    Separador: coma. Los números grandes van entre comillas: "29,205,005"

    NOTA: datos mensuales, no horarios. El pipeline de features
    se adapta automáticamente pero la detección es menos precisa.
    """
    df = pd.read_csv(file_path, sep=",", encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    # Renombrar columnas según el formato real del CSV
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if "barrio" in lower or "zona" in lower:
            col_map[col] = "barrio"
        elif lower == "uso":
            col_map[col] = "uso"
        elif "fecha" in lower:
            col_map[col] = "fecha"
        elif "consumo" in lower:
            col_map[col] = "consumo_litros"
        elif "contrato" in lower or "nº" in lower or "n°" in lower:
            col_map[col] = "num_contratos"
    df = df.rename(columns=col_map)

    # Limpiar números con formato europeo entre comillas: "29,205,005" → 29205005
    for col in ["consumo_litros", "num_contratos"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('"', '', regex=False)
                .str.replace(',', '', regex=False)  # quitar separador de miles
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Parsear fecha
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], format="%Y/%m/%d", errors="coerce")

    print("Columnas detectadas:", df.columns.tolist())
    print(f"Barrios únicos: {df['barrio'].nunique() if 'barrio' in df.columns else 'N/A'}")
    print(f"Rango fechas: {df['fecha'].min()} → {df['fecha'].max()}" if "fecha" in df.columns else "")
    print(f"Filas: {len(df)}")

    return df


# =============================================================
# PIPELINE PRINCIPAL
# =============================================================

def run_pipeline(df: pd.DataFrame, meter_id: str, output_dir: str = "output"):
    """
    Pipeline completo: datos → features → entrenamiento → detección → resultados.
    """
    Path(output_dir).mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Contador: {meter_id}")
    print(f"Registros cargados: {len(df)}")
    print(f"Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # 1. Feature engineering
    print("\n[1/4] Calculando features diarios...")
    daily = compute_daily_features(df)
    daily = add_rolling_features(daily)
    print(f"      {len(daily)} días procesados")
    print(daily[["date", "daily_total", "nocturnal_min", "zscore"]].tail(7).to_string())

    # 2. Separar entrenamiento (primeras 8 semanas) vs detección (resto)
    n_train = min(56, int(len(daily) * 0.7))  # 56 días o 70% si hay menos datos
    train_data = daily.iloc[:n_train]
    test_data = daily.iloc[n_train:]

    print(f"\n[2/4] Entrenando con {len(train_data)} días...")
    result = train(meter_id, train_data)
    print(f"      {result}")

    if result.get("status") == "error":
        print(f"ERROR: {result.get('reason')}")
        return

    # 3. Detectar anomalías en datos de test
    print(f"\n[3/4] Detectando anomalías en {len(test_data)} días...")
    test_scored = score_batch(meter_id, test_data)

    anomalies = test_scored[test_scored["is_anomaly"] == True].sort_values("anomaly_score")
    print(f"\n      Anomalías detectadas: {len(anomalies)} de {len(test_data)} días")

    if len(anomalies) > 0:
        print("\n      Top anomalías:")
        cols = ["date", "daily_total", "nocturnal_min", "zscore", "anomaly_score", "severity"]
        available_cols = [c for c in cols if c in anomalies.columns]
        print(anomalies[available_cols].head(10).to_string(index=False))

    # 4. Visualización
    print(f"\n[4/4] Generando gráfica...")
    _plot_results(train_data, test_scored, meter_id, output_dir)

    return test_scored


def _plot_results(train_data, test_scored, meter_id, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    all_data = pd.concat([train_data, test_scored]).sort_values("date")
    all_data["date"] = pd.to_datetime(all_data["date"])

    train_end = pd.to_datetime(train_data["date"].max())

    # Gráfica 1: consumo diario con anomalías
    ax1 = axes[0]
    ax1.plot(all_data["date"], all_data["daily_total"], "b-", linewidth=1.5, label="Consumo diario")

    # Media móvil si existe
    if "moving_avg_7d" in all_data.columns:
        ax1.plot(all_data["date"], all_data["moving_avg_7d"], "g--",
                 linewidth=1, alpha=0.7, label="Media móvil 7d")

    # Línea vertical separando entrenamiento de detección
    ax1.axvline(x=train_end, color="gray", linestyle=":", alpha=0.7, label="Fin entrenamiento")
    ax1.axvspan(all_data["date"].min(), train_end, alpha=0.05, color="blue", label="Datos entrenamiento")

    # Anomalías en rojo
    if "is_anomaly" in test_scored.columns:
        anomalies = test_scored[test_scored["is_anomaly"] == True]
        anomalies_dates = pd.to_datetime(anomalies["date"])
        if len(anomalies) > 0:
            ax1.scatter(anomalies_dates, anomalies["daily_total"],
                        color="red", s=80, zorder=5, label=f"Anomalías ({len(anomalies)})")

    ax1.set_title(f"Consumo diario — {meter_id}")
    ax1.set_ylabel("Consumo")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Gráfica 2: anomaly score
    ax2 = axes[1]
    if "anomaly_score" in test_scored.columns:
        test_dates = pd.to_datetime(test_scored["date"])
        ax2.plot(test_dates, test_scored["anomaly_score"], "purple", linewidth=1.5)
        ax2.axhline(y=-0.1, color="orange", linestyle="--", label="Umbral MEDIUM (-0.1)")
        ax2.axhline(y=-0.3, color="red", linestyle="--", alpha=0.7, label="Umbral HIGH (-0.3)")
        ax2.axhline(y=-0.5, color="darkred", linestyle="--", alpha=0.7, label="Umbral CRITICAL (-0.5)")
        ax2.fill_between(test_dates, test_scored["anomaly_score"], -0.1,
                         where=test_scored["anomaly_score"] < -0.1,
                         alpha=0.2, color="red", label="Zona anómala")
        ax2.set_title("Anomaly Score (más negativo = más anómalo)")
        ax2.set_ylabel("Score")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f"{output_dir}/{meter_id}_anomalies.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Gráfica guardada: {output_path}")


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento local de detección de anomalías")
    parser.add_argument("--file", required=True, help="Ruta al CSV de datos")
    parser.add_argument("--id", default="meter_001", help="ID del contador")
    parser.add_argument(
        "--mode",
        default="datadis",
        choices=["datadis", "kaggle", "hackathon"],
        help="Formato del CSV"
    )
    args = parser.parse_args()

    print(f"Cargando datos desde: {args.file} (modo: {args.mode})")

    if args.mode == "datadis":
        df = load_datadis(args.file)
        run_pipeline(df, meter_id=args.id)

    elif args.mode == "kaggle":
        df = load_kaggle_smart_meters(args.file)
        # Kaggle tiene varios contadores en el mismo CSV, procesar uno
        if "LCLid" in df.columns:
            first_meter = df["LCLid"].iloc[0]
            df = df[df["LCLid"] == first_meter]
            print(f"Procesando contador: {first_meter}")
        run_pipeline(df, meter_id=args.id)

    elif args.mode == "hackathon":
        # Para el CSV mensual del hackathon, procesamos por barrio
        df_hackathon = load_hackathon_amaem(args.file)
        if "barrio" not in df_hackathon.columns:
            print("ERROR: No se detectó columna de barrio en el CSV")
            exit(1)

        # Tomar el primer barrio doméstico como ejemplo
        barrio = df_hackathon["barrio"].iloc[0]
        uso = "DOMESTICO"

        subset = df_hackathon[
            (df_hackathon["barrio"] == barrio) &
            (df_hackathon.get("uso", pd.Series(["DOMESTICO"] * len(df_hackathon))) == uso)
        ].copy()

        # Para datos mensuales, creamos un timestamp ficticio de inicio de mes
        subset["timestamp"] = pd.to_datetime(subset["fecha"])
        subset["consumption"] = pd.to_numeric(
            subset["consumo_litros"].astype(str).str.replace(".", "").str.replace(",", "."),
            errors="coerce"
        ).fillna(0)

        print(f"\nProcesando barrio: {barrio} ({uso})")
        print(f"Meses disponibles: {len(subset)}")
        print("\nNOTA: Con datos mensuales el modelo es menos preciso.")
        print("Úsalo para aprender el pipeline. Espera datos horarios para resultados reales.\n")

        run_pipeline(subset[["timestamp", "consumption"]], meter_id=f"{barrio}_{uso}")
