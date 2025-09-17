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
    --latent-dim 2 --hidden-dim1 64 --hidden-dim2 32 \
    --epochs 50 --batch-size 256 --lr 1e-3

__author__ = "Dipal Shah"
__email__  = "dipalshah@missouri.edu"
__license__ = "MIT"
"""

import argparse
import csv
import logging
from pathlib import Path
import sys, os, random
import torch
import numpy as np
import pandas as pd
# -------------------
# Reproducibility setup
# -------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def load_stage1_artifacts(output_dir: str | Path):
    out = Path(output_dir)
    df = pd.read_parquet(out / "prepared_df.parquet")
    X  = np.load(out / "data_scaled.npy")
    assert len(df) == len(X), f"Row mismatch: df={len(df)} vs X={len(X)}"
    row_keys = df["mukey"].astype(str).to_numpy() if "mukey" in df.columns else df.index.to_numpy()
    return df, X, row_keys


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
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim2, latent_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
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
    mse = nn.functional.mse_loss(recon_x, x, reduction='sum')  # keep your original
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kl


def train_vae(X: np.ndarray, *, latent_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, epochs: int = 100, batch_size: int = 128, lr: float = 1e-3, beta: float = 1.0):
    if torch is None:
        raise ImportError(
            "PyTorch is required for VAE training but is not available. "
            f"Original import error: {_torch_import_error}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(X_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    model = VAE(input_dim=X.shape[1], latent_dim=latent_dim)#, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    #model.train()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (xb,) in dl:
            xb = xb[0]
            optim.zero_grad()
            x_hat, mu, logvar = model(xb)
            loss = vae_loss_function(x_hat, xb, mu, logvar)
            if torch.isnan(loss):
                logging.error(f"NaN loss encountered at Epoch {epoch+1}. Stopping training.")
                sys.exit(1)

            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(ds)
        train_losses.append(epoch_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss:.4f}")

    logging.info("VAE training finished.")
    return model, train_losses


def compute_z_mean(model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    Compute z_mean (μ) for every row in X using the trained VAE encoder.
    - Preserves row order (shuffle=False).
    - CPU-only by default; no .to(device) used.
    """
    model.eval()
    X_t = torch.from_numpy(X).float()
    ds = TensorDataset(X_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    mus = []
    for (xb,) in loader:
        mu, _ = model.encode(xb)     # encoder mean; no reparameterization
        mus.append(mu.detach().numpy())
    z = np.concatenate(mus, axis=0)
    assert z.shape[0] == X.shape[0], f"z rows {z.shape[0]} != X rows {X.shape[0]}"
    return z


def run_stage2(
    output_dir: str, *,
    latent_dim: int,
    hidden_dim1: int = 64, hidden_dim2: int = 32,
    epochs: int = 100, batch_size: int = 256, lr: float = 1e-3,
):
    df, X, row_keys = load_stage1_artifacts(output_dir)
    model = train_vae(X, latent_dim=latent_dim,
                      hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2,
                      epochs=epochs, batch_size=batch_size, lr=lr)
    z = compute_z_mean(model, X, batch_size=batch_size)
    out = Path(output_dir)
    np.save(out / "z_mean.npy", z)
    # optional: keep keys for audit
    np.save(out / "z_mean_row_keys.npy", row_keys.astype(str))


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
# NEW: global plotting helper so run() can call it
# -----------------------------
def plot_training_loss(losses: list[float], out_dir: Path | str):
    try:
        import matplotlib.pyplot as plt
        out_dir = Path(out_dir)
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(losses) + 1), losses, marker="o")
        plt.xlabel("Epoch"); plt.ylabel("Avg loss (per sample)")
        plt.title("VAE training loss")
        plt.grid(True, ls="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(fig_dir / "loss_curve.png", dpi=200)
        plt.close()
    except Exception as e:
        logging.warning("Could not plot training loss: %s", e)


# -----------------------------
# Runner helpers
# -----------------------------
def save_model(model, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_path = out_dir / "vae_model.pt"

    if torch is not None and hasattr(model, "state_dict"):
        torch.save(model.state_dict(), torch_path)
        return torch_path

    # Fallback (shouldn't happen)
    import pickle
    pkl_path = out_dir / "vae_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    return pkl_path


def train_and_save(output_dir: Path, latent_dim: int,
                   *, hidden_dim1: int = 64, hidden_dim2: int = 32,
                   epochs: int = 100, batch_size: int = 256, lr: float = 1e-3):
    """
    Programmatic convenience wrapper: trains a VAE on Stage-1 data and
    writes all Stage-2 artifacts into `output_dir`.
    """
    import csv
    import logging
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def _plot_training_loss(losses: list[float], out_dir: Path):
        # (kept as-is; separate from the new global one)
        try:
            import matplotlib.pyplot as plt
            fig_dir = out_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(7, 4))
            plt.plot(range(1, len(losses) + 1), losses, marker="o")
            plt.xlabel("Epoch"); plt.ylabel("Avg loss (per sample)")
            plt.title("VAE training loss")
            plt.grid(True, ls="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(fig_dir / "loss_curve.png", dpi=200)
            plt.close()
        except Exception as e:
            logging.warning("Could not plot training loss: %s", e)

    # Ensure directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    # ---- Load inputs from Stage 1
    X_path = output_dir / "data_scaled.npy"
    if not X_path.exists():
        raise FileNotFoundError(f"Missing input file: {X_path} (run data_preparation.py first).")
    X = np.load(X_path)

    # Optional sanity check with prepared_df + persist MUKEY order (if aligned)
    df_path = output_dir / "prepared_df.parquet"
    if df_path.exists():
        try:
            df = pd.read_parquet(df_path)
            if len(df) != len(X):
                logging.warning(
                    "Row count mismatch: prepared_df (%d) vs data_scaled (%d). "
                    "z_mean_row_keys.npy will NOT be written from prepared_df.",
                    len(df), len(X)
                )
            else:
                if "mukey" in df.columns:
                    keys = df["mukey"].astype(str).to_numpy()
                    np.save(output_dir / "z_mean_row_keys.npy", keys)
                    logging.info("Saved z_mean_row_keys.npy from prepared_df (len=%d).", len(keys))
                else:
                    logging.warning("'mukey' column not found in prepared_df.parquet; "
                                    "z_mean_row_keys.npy will not be written from prepared_df.")
        except Exception as e:
            logging.warning("Could not read prepared_df.parquet for alignment check: %s", e)

    # ---- Train VAE
    result = train_vae(
        X,
        latent_dim=latent_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    if isinstance(result, tuple) and len(result) == 2:
        model, train_losses = result
    else:
        model, train_losses = result, None

    # ---- Save training loss trace (if available)
    if train_losses is not None:
        losses_csv = output_dir / "train_losses.csv"
        try:
            with open(losses_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "loss"])
                for i, loss in enumerate(train_losses, start=1):
                    writer.writerow([i, float(loss)])
            _plot_training_loss(train_losses, output_dir)
        except Exception as e:
            logging.warning("Could not write/plot training losses: %s", e)

    # ---- Extract and save latent representations (μ)
    try:
        z_mean = compute_z_mean(model, X, batch_size=batch_size)
    except NameError:
        # fallback inline
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(torch.from_numpy(X).float())
            z_mean = mu.numpy()
    np.save(output_dir / "z_mean.npy", np.asarray(z_mean))

    # ---- Save model weights
    model_path = output_dir / "vae_model.pt"
    try:
        torch.save(model.state_dict(), model_path)
    except Exception as e:
        logging.warning("Could not save model weights: %s", e)

    return model, z_mean


# -----------------------------
# UPDATED: run() signature to match your bottom call
# -----------------------------
def run(
    output_dir: Path,
    latent_dim: int,
    hidden_dim1: int,
    hidden_dim2: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=== Stage 2: VAE Training & Latent Extraction ===")
    logging.info("Output dir  : %s", output_dir)
    logging.info("Latent dim  : %d", latent_dim)
    logging.info("Hidden dims : %d, %d", hidden_dim1, hidden_dim2)
    logging.info("Epochs      : %d", epochs)
    logging.info("Batch size  : %d", batch_size)
    logging.info("LR          : %g", lr)

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
        try:
            df = pd.read_parquet(df_path)
            if len(df) != len(X):
                logging.warning(
                    "Row count mismatch: prepared_df (%d) vs data_scaled (%d). "
                    "Ensure alignment before proceeding.",
                    len(df), len(X)
                )
        except Exception as e:
            logging.warning("Could not read prepared_df.parquet: %s", e)

    # ---- Train VAE
    logging.info("Training VAE...")
    model, train_losses = train_vae(
        X,
        latent_dim=latent_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    logging.info("VAE training complete. Epochs: %d", len(train_losses))

    # ---- Save training loss trace
    losses_csv = output_dir / "train_losses.csv"
    try:
        with open(losses_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for i, loss in enumerate(train_losses, start=1):
                writer.writerow([i, float(loss)])
        plot_training_loss(train_losses, output_dir)  # <-- now defined globally
        logging.info("Saved training losses and loss curve.")
    except Exception as e:
        logging.warning("Could not write/plot training losses: %s", e)

    # ---- Extract latent representations
    logging.info("Extracting latent representations (z_mean)...")
    try:
        z_mean = compute_z_mean(model, X, batch_size=batch_size)
    except Exception:
        with torch.no_grad():
            mu, _ = model.encode(torch.from_numpy(X).float())
            z_mean = mu.detach().numpy()

    z_path = output_dir / "z_mean.npy"
    np.save(z_path, np.asarray(z_mean))
    logging.info("Saved: %s (shape=%s)", z_path.name, np.asarray(z_mean).shape)

    # ---- Save model weights
    model_path = save_model(model, output_dir)
    logging.info("Saved model: %s", model_path.name)

    logging.info("Stage 2 complete.")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent / "data" / "aggResult"
    parser = argparse.ArgumentParser(description="Stage 2 — VAE Training & Latent Representations (self-contained)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=False,
        default=default_output,
        help="Directory to read Stage-1 outputs from and write Stage-2 outputs to.",
    )
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--hidden-dim1", type=int, default=64)
    parser.add_argument("--hidden-dim2", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run(args.output_dir, args.latent_dim, args.hidden_dim1, args.hidden_dim2, args.epochs, args.batch_size, args.lr)
