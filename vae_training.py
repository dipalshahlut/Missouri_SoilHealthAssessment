#!/usr/bin/env python3
# vae_training.py
"""
Stage 2 — VAE Training & Latent Representations (self-contained)

Inputs (read from OUTPUT_DIR):
  - data_scaled.npy            : numpy array from Stage 1 (rows aligned to prepared_df)
  - prepared_df.parquet        : (optional) only for alignment sanity-check

Outputs (written to OUTPUT_DIR):
  - vae_model.pt               : trained VAE weights (PyTorch state_dict)
  - train_losses.csv           : per-epoch training loss
  - z_mean.npy                 : latent representations (aligned to prepared_df rows)
  - figures/loss_curve.png     : training loss plot

Usage:
  python vae_training.py \
    --output-dir /path/to/data/aggResult \
    --latent-dim 2 \
    --epochs 100
"""

import argparse
import csv
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# -----------------------------
# Embedded plotting util
# -----------------------------
def plot_training_loss(train_losses, output_dir: str):
    import matplotlib.pyplot as plt

    figdir = Path(output_dir) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    outpath = figdir / "loss_curve.png"

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# -----------------------------
# Embedded minimal VAE (PyTorch)
# -----------------------------
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    torch = None
    _torch_import_error = e


class VAE(nn.Module):
    """
    Simple MLP VAE:
      encoder:  X -> hidden -> (mu, logvar)
      decoder:  z -> hidden -> X_hat
    Uses MSE recon + KL loss.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # z

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar):
    """
    Reconstruction (MSE) + KL divergence.
    """
    mse = torch.mean((recon_x - x) ** 2)
    # KL: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / N
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kl


def train_vae(X: np.ndarray, latent_dim: int, epochs: int = 100, batch_size: int = 128, lr: float = 1e-3):
    if torch is None:
        raise ImportError(
            "PyTorch is required for VAE training but is not available. "
            f"Original import error: {_torch_import_error}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(X_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = VAE(input_dim=X.shape[1], latent_dim=latent_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            optim.zero_grad()
            x_hat, mu, logvar = model(xb)
            loss = vae_loss_function(x_hat, xb, mu, logvar)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(ds)
        train_losses.append(epoch_loss)

    return model, train_losses


def extract_latent_representations(model, X: np.ndarray) -> np.ndarray:
    if torch is None:
        raise ImportError(
            "PyTorch is required for latent extraction but is not available. "
            f"Original import error: {_torch_import_error}"
        )
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        mu, logvar = model.encode(X_tensor)
        z_mean = mu.cpu().numpy()
    return z_mean


# -----------------------------
# Runner
# -----------------------------
def save_model(model, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_path = out_dir / "vae_model.pt"

    if torch is not None and hasattr(model, "state_dict"):
        torch.save(model.state_dict(), torch_path)
        return torch_path

    # If torch wasn't available we'd have errored earlier, but keep fallback:
    import pickle
    pkl_path = out_dir / "vae_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    return pkl_path

def train_and_save(output_dir: Path, latent_dim: int, epochs: int, seed: int = 42):
    """
    Programmatic convenience wrapper: trains a VAE on Stage-1 data and
    writes all Stage-2 artifacts into `output_dir`.

    Inputs (must already exist in output_dir):
      - data_scaled.npy

    Writes to output_dir:
      - train_losses.csv
      - figures/loss_curve.png
      - z_mean.npy
      - vae_model.pt

    Returns
    -------
    (model, z_mean) : (torch.nn.Module, np.ndarray)
        Trained VAE model and latent representations aligned to data rows.
    """

    # Ensure directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Reproducibility
    np.random.seed(seed)

    # ---- Load inputs from Stage 1
    X_path = output_dir / "data_scaled.npy"
    if not X_path.exists():
        raise FileNotFoundError(f"Missing input file: {X_path} (run data_preparation.py first).")
    X = np.load(X_path)

    # Optional sanity check with prepared_df
    df_path = output_dir / "prepared_df.parquet"
    if df_path.exists():
        try:
            df = pd.read_parquet(df_path)
            if len(df) != len(X):
                logging.warning(
                    "Row count mismatch: prepared_df (%d) vs data_scaled (%d). "
                    "Ensure alignment before proceeding.",
                    len(df), len(X)
                )
        except Exception as e:
            logging.warning("Could not read prepared_df.parquet for alignment check: %s", e)

    # ---- Train VAE
    model, train_losses = train_vae(X, latent_dim, epochs=epochs)

    # ---- Save training loss trace
    losses_csv = output_dir / "train_losses.csv"
    with open(losses_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for i, loss in enumerate(train_losses, start=1):
            writer.writerow([i, float(loss)])

    # ---- Plot loss curve (best-effort)
    try:
        plot_training_loss(train_losses, str(output_dir))
    except Exception as e:
        logging.warning("Could not plot training loss: %s", e)

    # ---- Extract and save latent representations
    z_mean = extract_latent_representations(model, X)
    np.save(output_dir / "z_mean.npy", np.asarray(z_mean))

    # ---- Save model weights
    save_model(model, output_dir)

    return model, z_mean


def run(output_dir: Path, latent_dim: int, epochs: int, seed: int) -> None:
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=== Stage 2: VAE Training & Latent Extraction ===")
    logging.info("Output dir  : %s", output_dir)
    logging.info("Latent dim  : %d", latent_dim)
    logging.info("Epochs      : %d", epochs)
    logging.info("Seed        : %d", seed)

    # ---- Load inputs from Stage 1
    X_path = output_dir / "data_scaled.npy"
    if not X_path.exists():
        logging.error("Missing %s (run data_preparation.py first).", X_path)
        sys.exit(1)

    X = np.load(X_path)
    logging.info("Loaded data_scaled.npy with shape %s", X.shape)

    # Optional sanity check with prepared_df
    df_path = output_dir / "prepared_df.parquet"
    if df_path.exists():
        df = pd.read_parquet(df_path)
        if len(df) != len(X):
            logging.warning(
                "Row count mismatch: prepared_df (%d) vs data_scaled (%d). "
                "Ensure alignment before proceeding.",
                len(df), len(X)
            )

    # ---- Train VAE
    logging.info("Training VAE...")
    model, train_losses = train_vae(X, latent_dim, epochs=epochs)
    logging.info("VAE training complete. Epochs: %d", len(train_losses))

    # ---- Save training loss trace
    losses_csv = output_dir / "train_losses.csv"
    with open(losses_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for i, loss in enumerate(train_losses, start=1):
            writer.writerow([i, float(loss)])
    logging.info("Saved: %s", losses_csv.name)

    # ---- Plot loss curve
    try:
        plot_training_loss(train_losses, str(output_dir))
        logging.info("Saved: figures/loss_curve.png")
    except Exception as e:
        logging.warning("Could not plot training loss: %s", e)

    # ---- Extract latent representations
    logging.info("Extracting latent representations (z_mean)...")
    z_mean = extract_latent_representations(model, X)
    z_path = output_dir / "z_mean.npy"
    np.save(z_path, np.asarray(z_mean))
    logging.info("Saved: %s (shape=%s)", z_path.name, np.asarray(z_mean).shape)

    # ---- Save model weights
    model_path = save_model(model, output_dir)
    logging.info("Saved model: %s", model_path.name)

    logging.info("Stage 2 complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 — VAE Training & Latent Representations (self-contained)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=False,
        default=Path("/Users/dscqv/Desktop/SHA_copy/data/aggResult"),
        help="Directory to read Stage-1 outputs from and write Stage-2 outputs to.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        required=False,
        default=2,
        help="Dimensionality of the VAE latent space.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run(args.output_dir, args.latent_dim, args.epochs, args.seed)
