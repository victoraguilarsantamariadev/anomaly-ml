"""
M13 — Autoencoder para deteccion de anomalias.

Red neuronal que aprende a reconstruir el patron de consumo normal.
Error de reconstruccion alto = anomalia (el punto no se parece a lo normal).

Ventajas sobre IsolationForest (M2):
  - Captura relaciones NO LINEALES entre features
  - El error de reconstruccion es interpretable (que features se reconstruyen mal)
  - Funciona bien con datos de alta dimension

Arquitectura: Input(N) -> 32 -> 16 -> 8 -> 16 -> 32 -> Output(N)
  El bottleneck de 8 neuronas fuerza al modelo a aprender solo los patrones
  mas importantes. Si un punto necesita mas de 8 dimensiones para representarse,
  es probablemente anomalo.

Uso:
  from autoencoder_detector import run_autoencoder
  results = run_autoencoder(df_features, contamination=0.05)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor


def run_autoencoder(df_features: pd.DataFrame,
                    feature_cols: list,
                    uso_filter: str = "DOMESTICO",
                    contamination: float = 0.05,
                    hidden_layers: tuple = (32, 16, 8, 16, 32),
                    random_state: int = 42) -> pd.DataFrame:
    """
    Autoencoder anomaly detector.

    Entrena sobre los primeros 24 meses (datos normales),
    puntua los ultimos 12 meses por error de reconstruccion.

    Args:
        df_features: DataFrame con features calculados
        feature_cols: lista de columnas a usar como features
        uso_filter: tipo de uso a filtrar
        contamination: % esperado de anomalias (para threshold)
        hidden_layers: arquitectura del autoencoder
        random_state: semilla

    Returns:
        DataFrame con is_anomaly_autoencoder y reconstruction_error
    """
    print(f"\n  [M13] Autoencoder (layers={hidden_layers}, contamination={contamination})...")

    df_uso = df_features[df_features["uso"].str.strip() == uso_filter].copy()
    df_uso = df_uso.sort_values(["barrio_key", "fecha"]).reset_index(drop=True)

    available_cols = [c for c in feature_cols if c in df_uso.columns]
    if len(available_cols) < 5:
        print(f"    Insuficientes features ({len(available_cols)})")
        return pd.DataFrame()

    # Split temporal
    all_dates = sorted(df_uso["fecha"].unique())
    n_train_dates = min(24, int(len(all_dates) * 0.7))
    train_dates = set(all_dates[:n_train_dates])
    test_dates = set(all_dates[n_train_dates:])

    train_data = df_uso[df_uso["fecha"].isin(train_dates)]
    test_data = df_uso[df_uso["fecha"].isin(test_dates)]

    # Preparar matrices
    X_train = train_data[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_test = test_data[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

    if len(X_train) < 20:
        print(f"    Insuficientes datos de entrenamiento ({len(X_train)})")
        return pd.DataFrame()

    # Escalar con RobustScaler (resistente a outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar autoencoder (input -> hidden -> input)
    ae = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=500,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=random_state,
        verbose=False,
    )

    # El autoencoder aprende a reconstruir su propia entrada
    ae.fit(X_train_scaled, X_train_scaled)

    # Error de reconstruccion en train (para calibrar threshold)
    train_reconstructed = ae.predict(X_train_scaled)
    train_errors = np.mean((X_train_scaled - train_reconstructed) ** 2, axis=1)

    # Error de reconstruccion en test
    test_reconstructed = ae.predict(X_test_scaled)
    test_errors = np.mean((X_test_scaled - test_reconstructed) ** 2, axis=1)

    # Threshold basado en percentil del train (asumimos train es normal)
    threshold = np.percentile(train_errors, (1 - contamination) * 100)

    # Feature-level error para interpretabilidad
    feature_errors = (X_test_scaled - test_reconstructed) ** 2

    # Encontrar el feature con mayor error para cada punto
    worst_feature = []
    for i in range(len(X_test)):
        top_idx = np.argmax(feature_errors[i])
        worst_feature.append(available_cols[top_idx])

    # Construir resultado
    context_cols = ["barrio_key", "fecha"]
    result = test_data[context_cols].copy()
    result["reconstruction_error"] = test_errors
    result["is_anomaly_autoencoder"] = test_errors > threshold
    result["ae_worst_feature"] = worst_feature

    n_anomalies = result["is_anomaly_autoencoder"].sum()
    n_barrios = result["barrio_key"].nunique()
    print(f"    {n_anomalies} anomalias en {len(result)} puntos "
          f"({n_barrios} barrios, threshold={threshold:.4f})")
    print(f"    Train error: mean={train_errors.mean():.4f}, "
          f"max={train_errors.max():.4f}")
    print(f"    Test error:  mean={test_errors.mean():.4f}, "
          f"max={test_errors.max():.4f}")

    # Top anomalias por error
    if n_anomalies > 0:
        top_ae = result.nlargest(min(5, n_anomalies), "reconstruction_error")
        print(f"    Top anomalias:")
        for _, row in top_ae.iterrows():
            barrio = row["barrio_key"].split("__")[0]
            fecha = pd.to_datetime(row["fecha"]).strftime("%Y-%m")
            print(f"      {barrio} {fecha}: error={row['reconstruction_error']:.4f} "
                  f"(worst: {row['ae_worst_feature']})")

    return result
