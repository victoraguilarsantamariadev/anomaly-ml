"""
M13-PRO — Variational Autoencoder (VAE) con PyTorch.

Reemplaza el MLPRegressor-como-autoencoder con un VAE real que:
  1. Modela distribucion latente (mu, sigma) → incertidumbre nativa
  2. KL divergence + reconstruction loss → deteccion probabilistica
  3. Denoising opcional → robustez contra memorización
  4. Log-likelihood por punto → metrica probabilistica real (no MSE)

Arquitectura: Input(N) -> 64 -> 32 -> [mu(16), sigma(16)] -> 32 -> 64 -> Output(N)

Uso:
  from vae_detector import run_vae
  results = run_vae(df_features, feature_cols, contamination=0.05)
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler


# ─────────────────────────────────────────────────────────────────
# VAE Architecture
# ─────────────────────────────────────────────────────────────────

class VAE(nn.Module):
    """Variational Autoencoder con encoder/decoder simétrico."""

    def __init__(self, input_dim: int, hidden_dims: list = None,
                 latent_dim: int = 8):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space: mu y log_var
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims))
        prev_dim = latent_dim
        for h_dim in reversed_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(reversed_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def compute_loss(self, x, x_recon, mu, log_var, beta=1.0):
        """
        ELBO loss = Reconstruction + beta * KL divergence.
        beta > 1 → beta-VAE (disentangled representations).
        """
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='none').sum(dim=1)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

    def anomaly_score(self, x, n_samples: int = 10):
        """
        Score de anomalia basado en ELBO (mayor = mas anomalo).
        Promedia sobre n_samples del espacio latente para estabilidad.
        """
        self.eval()
        with torch.no_grad():
            scores = []
            for _ in range(n_samples):
                x_recon, mu, log_var = self(x)
                loss, recon, kl = self.compute_loss(x, x_recon, mu, log_var)
                scores.append(loss.cpu().numpy())
            return np.mean(scores, axis=0)

    def log_likelihood(self, x, n_samples: int = 50):
        """
        Estimacion de log-likelihood via importance sampling.
        log p(x) ≈ log(1/K * sum_k[ p(x|z_k) * p(z_k) / q(z_k|x) ])

        Metrica probabilistica REAL — no un threshold arbitrario.
        """
        self.eval()
        with torch.no_grad():
            mu, log_var = self.encode(x)
            log_weights = []
            for _ in range(n_samples):
                z = self.reparameterize(mu, log_var)
                x_recon = self.decode(z)

                # log p(x|z) - Gaussian reconstruction
                recon_ll = -0.5 * torch.sum((x - x_recon) ** 2, dim=1)

                # log p(z) - Standard normal prior
                prior_ll = -0.5 * torch.sum(z ** 2, dim=1)

                # log q(z|x) - Encoder posterior
                posterior_ll = -0.5 * torch.sum(
                    log_var + (z - mu) ** 2 / log_var.exp(), dim=1
                )

                log_w = recon_ll + prior_ll - posterior_ll
                log_weights.append(log_w)

            log_weights = torch.stack(log_weights, dim=0)
            # Log-sum-exp para estabilidad numerica
            ll = torch.logsumexp(log_weights, dim=0) - np.log(n_samples)
            return ll.cpu().numpy()


# ─────────────────────────────────────────────────────────────────
# Denoising: añade ruido al input para robustez
# ─────────────────────────────────────────────────────────────────

def add_noise(x: torch.Tensor, noise_factor: float = 0.1) -> torch.Tensor:
    """Gaussian noise injection para denoising VAE."""
    noise = torch.randn_like(x) * noise_factor
    return x + noise


# ─────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────

def train_vae(model: VAE, X_train: np.ndarray,
              epochs: int = 150, batch_size: int = 64,
              lr: float = 1e-3, beta: float = 1.0,
              denoising: bool = True, noise_factor: float = 0.1,
              patience: int = 15, verbose: bool = False):
    """
    Entrena el VAE con early stopping y opcion de denoising.
    """
    device = next(model.parameters()).device

    X_tensor = torch.FloatTensor(X_train).to(device)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=len(X_tensor) > batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )

    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for (batch_x,) in loader:
            # Denoising: input con ruido, target sin ruido
            if denoising:
                noisy_x = add_noise(batch_x, noise_factor)
            else:
                noisy_x = batch_x

            x_recon, mu, log_var = model(noisy_x)
            loss, _, _ = model.compute_loss(batch_x, x_recon, mu, log_var, beta)
            loss_mean = loss.mean()

            optimizer.zero_grad()
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss_mean.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"      Early stopping at epoch {epoch+1}")
            break

        if verbose and (epoch + 1) % 25 == 0:
            print(f"      Epoch {epoch+1}: loss={avg_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ─────────────────────────────────────────────────────────────────
# Main detector function
# ─────────────────────────────────────────────────────────────────

def run_vae(df_features: pd.DataFrame,
            feature_cols: list,
            uso_filter: str = "DOMESTICO",
            contamination: float = 0.05,
            latent_dim: int = 16,
            hidden_dims: list = None,
            beta: float = 2.0,
            denoising: bool = True,
            random_state: int = 42) -> pd.DataFrame:
    """
    VAE anomaly detector — reemplaza el MLPRegressor autoencoder.

    Mejoras sobre M13 original:
      1. Distribucion latente → incertidumbre nativa
      2. Log-likelihood por punto → probabilidad real de anomalia
      3. Denoising → no memoriza, aprende patrones robustos
      4. ELBO score → combina reconstruccion + regularizacion

    Args:
        df_features: DataFrame con features calculados
        feature_cols: columnas a usar
        uso_filter: tipo de uso
        contamination: % esperado de anomalias
        latent_dim: dimension del espacio latente
        hidden_dims: capas del encoder [32, 16] por defecto
        beta: peso del KL divergence (>1 = beta-VAE)
        denoising: usar denoising autoencoder
        random_state: semilla

    Returns:
        DataFrame con is_anomaly_vae, vae_score, vae_log_likelihood, etc.
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    mode = "Denoising " if denoising else ""
    print(f"\n  [M13-PRO] {mode}VAE (latent={latent_dim}, beta={beta}, "
          f"contamination={contamination})...")

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

    X_train = train_data[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_test = test_data[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

    if len(X_train) < 20:
        print(f"    Insuficientes datos de entrenamiento ({len(X_train)})")
        return pd.DataFrame()

    # Escalar
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Crear y entrenar VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_s.shape[1]

    model = VAE(input_dim=input_dim, hidden_dims=hidden_dims,
                latent_dim=latent_dim).to(device)

    model = train_vae(model, X_train_s, epochs=200, beta=beta,
                      denoising=denoising, verbose=False)

    # Scores en train (para calibrar threshold)
    X_train_t = torch.FloatTensor(X_train_s).to(device)
    X_test_t = torch.FloatTensor(X_test_s).to(device)

    train_scores = model.anomaly_score(X_train_t, n_samples=10)
    test_scores = model.anomaly_score(X_test_t, n_samples=10)

    # Log-likelihood (metrica probabilistica real)
    test_ll = model.log_likelihood(X_test_t, n_samples=50)

    # Threshold basado en percentil del train
    threshold = np.percentile(train_scores, (1 - contamination) * 100)

    # Reconstruccion para interpretabilidad (feature-level)
    model.eval()
    with torch.no_grad():
        x_recon, mu, log_var = model(X_test_t)
        feature_errors = (X_test_t - x_recon).pow(2).cpu().numpy()

    worst_feature = []
    for i in range(len(X_test)):
        top_idx = np.argmax(feature_errors[i])
        worst_feature.append(available_cols[top_idx])

    # Latent space analysis: distancia al centroide
    with torch.no_grad():
        test_mu, _ = model.encode(X_test_t)
        train_mu, _ = model.encode(X_train_t)
        centroid = train_mu.mean(dim=0)
        latent_dist = torch.norm(test_mu - centroid, dim=1).cpu().numpy()

    # Normalizar scores a [0, 1] para comparabilidad
    score_min, score_max = train_scores.min(), np.percentile(train_scores, 99)
    test_scores_norm = np.clip(
        (test_scores - score_min) / (score_max - score_min + 1e-10), 0, 1
    )

    # Construir resultado
    context_cols = ["barrio_key", "fecha"]
    result = test_data[context_cols].copy()
    result["vae_score"] = test_scores
    result["vae_score_norm"] = test_scores_norm
    result["vae_log_likelihood"] = test_ll
    result["vae_latent_distance"] = latent_dist
    result["is_anomaly_vae"] = test_scores > threshold
    result["vae_worst_feature"] = worst_feature

    # Separacion entre anomalos y normales
    anom_mask = result["is_anomaly_vae"]
    if anom_mask.sum() > 0 and (~anom_mask).sum() > 0:
        mean_anom = test_scores[anom_mask.values].mean()
        mean_norm = test_scores[~anom_mask.values].mean()
        separation = mean_anom / (mean_norm + 1e-10)
    else:
        separation = 0

    n_anomalies = anom_mask.sum()
    n_barrios = result["barrio_key"].nunique()

    print(f"    {n_anomalies} anomalias en {len(result)} puntos "
          f"({n_barrios} barrios)")
    print(f"    Train ELBO: mean={train_scores.mean():.4f}, "
          f"P95={np.percentile(train_scores, 95):.4f}")
    print(f"    Test ELBO:  mean={test_scores.mean():.4f}, "
          f"P95={np.percentile(test_scores, 95):.4f}")
    print(f"    Separacion anomalo/normal: {separation:.1f}x")
    print(f"    Log-likelihood: normal={test_ll[~anom_mask.values].mean():.1f}, "
          f"anomalo={test_ll[anom_mask.values].mean():.1f}" if anom_mask.sum() > 0 else "")

    # Top anomalias
    if n_anomalies > 0:
        top = result.nlargest(min(5, n_anomalies), "vae_score")
        print(f"    Top anomalias:")
        for _, row in top.iterrows():
            barrio = row["barrio_key"].split("__")[0]
            fecha = pd.to_datetime(row["fecha"]).strftime("%Y-%m")
            print(f"      {barrio} {fecha}: ELBO={row['vae_score']:.2f} "
                  f"LL={row['vae_log_likelihood']:.1f} "
                  f"dist={row['vae_latent_distance']:.2f} "
                  f"(worst: {row['vae_worst_feature']})")

    return result
