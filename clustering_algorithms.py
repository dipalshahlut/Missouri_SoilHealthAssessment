#!/usr/bin/env python3
"""
clustering_algorithms.py

Runs a single clustering algorithm (user-chosen method and k) on the
precomputed VAE latent space and saves results alongside the main dataframe.

Expected inputs (from Step 2: Data Prep + VAE):
  - <OUTPUT_DIR>/z_mean.npy
  - <OUTPUT_DIR>/main_df.csv  or <OUTPUT_DIR>/prepared_df.parquet
  - (optional) <OUTPUT_DIR>/row_ids.npy   # indices used to build data_scaled/z_mean

Outputs (written to OUTPUT_DIR):
  - labels_{method}_k{k}.csv              # row_id, label (row_id is 0..N-1 over z_mean)
  - main_df_with_{method}_k{k}.csv        # df joined with labels (if row counts align or row_ids.npy exists)
  - silhouette_{method}_k{k}.txt          # optional; only if silhouette can be computed

Usage :
python clustering_algorithms.py \
  --output-dir /path/to/data/aggResult \
  --method KMeans \
  --k 10
"""
from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Optional fuzzy c-means
try:
    import skfuzzy as fuzz
    _FUZZY_AVAILABLE = True
except Exception:
    _FUZZY_AVAILABLE = False


# --------------------------- Logging ---------------------------

def _setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.INFO
    if verbosity >= 3:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


# --------------------------- IO helpers ---------------------------

def _default_z_path(output_dir: str | Path) -> str:
    return str(Path(output_dir) / "z_mean.npy")

def _default_df_candidates(output_dir: str | Path) -> list[str]:
    out = Path(output_dir)
    return [
        str(out / "main_df.csv"),
        str(out / "prepared_df.parquet"),
    ]

def _load_z(z_path: str) -> np.ndarray:
    if not os.path.exists(z_path):
        raise FileNotFoundError(f"Missing latent file: {z_path}")
    z = np.load(z_path)
    if z.ndim != 2:
        raise ValueError(f"z_mean must be 2D (n_samples, latent_dim), got shape {z.shape}")
    logging.info("Loaded z_mean: %s", z.shape)
    return z

def _load_df(df_path: Optional[str], output_dir: str | Path) -> Optional[pd.DataFrame]:
    """
    Try to load a dataframe for label join. If df_path is provided, use it.
    Otherwise try defaults in this order: main_df.csv, prepared_df.parquet.
    Returns None if nothing found. Supports .csv or .parquet.
    """
    candidates: list[str] = []
    if df_path:
        candidates = [df_path]
    else:
        candidates = _default_df_candidates(output_dir)

    for path in candidates:
        if os.path.exists(path):
            try:
                if path.lower().endswith(".csv"):
                    df = pd.read_csv(path)
                elif path.lower().endswith(".parquet"):
                    df = pd.read_parquet(path)
                else:
                    logging.warning("Unsupported DF extension for %s; skipping.", path)
                    continue
                logging.info("Loaded dataframe: %s (shape=%s)", os.path.basename(path), df.shape)
                return df
            except Exception as e:
                logging.warning("Failed to load dataframe at %s: %s", path, e)
    logging.warning(
        "No dataframe found (looked for main_df.csv or prepared_df.parquet). "
        "Will proceed and write labels only."
    )
    return None


# --------------------------- Core clustering ---------------------------

def _fit_one(z_mean: np.ndarray, method: str, k: int, random_state: int = 42) -> np.ndarray:
    method = method.strip()
    logging.info("Fitting %s with k=%d on z_mean shape=%s", method, k, z_mean.shape)

    if method == "KMeans":
        return KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(z_mean)
    if method == "Agglomerative":
        return AgglomerativeClustering(n_clusters=k).fit_predict(z_mean)
    if method == "Birch":
        return Birch(n_clusters=k).fit_predict(z_mean)
    if method == "GMM":
        return GaussianMixture(n_components=k, random_state=random_state).fit_predict(z_mean)
    if method == "FuzzyCMeans":
        if not _FUZZY_AVAILABLE:
            raise ImportError("FuzzyCMeans requested but scikit-fuzzy (skfuzzy) is not installed.")
        # skfuzzy expects features x samples
        _, u, *_ = fuzz.cluster.cmeans(z_mean.T, c=k, m=2.0, error=0.005, maxiter=1000, init=None)
        return np.argmax(u, axis=0)

    raise ValueError("Unknown method: {0}. Choose from "
                     "{{'KMeans','Agglomerative','Birch','GMM','FuzzyCMeans'}}".format(method))


def _try_join_and_write(
    df: pd.DataFrame,
    labels: np.ndarray,
    col_name: str,
    output_dir: str | Path
) -> bool:
    """
    Try to write a joined dataframe with labels.
    Strategy:
      1) If len(df) == len(labels): join by position.
      2) Else, look for row_ids.npy in output_dir and prepared_df.parquet. If present and lengths match,
         subset df via df.iloc[row_ids] and join labels to that subset.
    Returns True if a CSV was written.
    """
    out_dir = Path(output_dir)

    # Case 1: perfect length match
    if len(df) == len(labels):
        joined = df.copy()
        joined[col_name] = labels
        out_df = out_dir / f"main_df_with_{col_name}.csv"
        joined.to_csv(out_df, index=False)
        logging.info("Wrote: %s", out_df)
        return True

    # Case 2: use row_ids.npy if available
    row_ids_path = out_dir / "row_ids.npy"
    if row_ids_path.exists():
        try:
            row_ids = np.load(row_ids_path)
            if len(row_ids) == len(labels):
                # Prefer prepared_df.parquet for alignment if available
                prep_path = out_dir / "prepared_df.parquet"
                base_df = pd.read_parquet(prep_path) if prep_path.exists() else df
                if len(base_df) >= row_ids.max() + 1:
                    sub = base_df.iloc[row_ids].copy()
                    sub[col_name] = labels
                    out_df = out_dir / f"main_df_with_{col_name}.csv"
                    sub.to_csv(out_df, index=False)
                    logging.info("Wrote (via row_ids alignment): %s", out_df)
                    return True
                else:
                    logging.warning(
                        "row_ids.npy found but indices exceed dataframe length (%d). "
                        "Skipping joined dataframe.", len(base_df)
                    )
            else:
                logging.warning(
                    "row_ids.npy length (%d) != labels length (%d). Skipping joined dataframe.",
                    len(row_ids), len(labels)
                )
        except Exception as e:
            logging.warning("Failed to use row_ids.npy for join: %s", e)

    logging.warning(
        "Row count mismatch: df has %d rows, z_mean/labels have %d rows. "
        "Wrote labels file only; skipped joined dataframe.",
        len(df), len(labels)
    )
    return False


def run_single_clustering_io(
    output_dir: str | Path,
    method: str,
    k: int,
    z_path: Optional[str] = None,
    df_path: Optional[str] = None,
    save_labels: bool = True,
    compute_silhouette: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, Optional[float], str]:
    """
    Load z_mean (and optionally a dataframe), fit one clustering, and write outputs.

    Returns
    -------
    labels : np.ndarray
    sil : Optional[float]
    col_name : str
    """
    output_dir = str(output_dir)

    # 1) Load latent space
    z = _load_z(z_path or _default_z_path(output_dir))

    # 2) Load dataframe if present
    df = _load_df(df_path, output_dir)

    # 3) Fit clustering
    labels = _fit_one(z, method, k, random_state=random_state)

    # 4) Silhouette (if feasible)
    sil: Optional[float] = None
    if compute_silhouette:
        uniq = np.unique(labels)
        if 1 < len(uniq) < len(labels):
            try:
                sil = float(silhouette_score(z, labels))
                logging.info("Silhouette (%s, k=%d) = %.6f", method, k, sil)
            except Exception as e:
                logging.warning("Silhouette computation failed: %s", e)

    # 5) Write outputs
    col_name = f"{method}_k{k}"

    if save_labels:
        # labels file (always)
        lab_path = os.path.join(output_dir, f"labels_{col_name}.csv")
        pd.DataFrame({"row_id": np.arange(len(labels)), "label": labels}).to_csv(lab_path, index=False)
        logging.info("Wrote: %s", lab_path)

        # silhouette file (if available)
        if sil is not None:
            sil_path = os.path.join(output_dir, f"silhouette_{col_name}.txt")
            with open(sil_path, "w") as f:
                f.write(f"{sil:.6f}")
            logging.info("Wrote: %s", sil_path)

        # joined dataframe (if df exists)
        if df is not None:
            _ = _try_join_and_write(df, labels, col_name, output_dir)

    return labels, sil, col_name


# --------------------------- CLI ---------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a single clustering method at a specified k on VAE latent space."
    )
    p.add_argument("-o", "--output-dir", required=True,
                   help="Directory for inputs/outputs (default z_mean.npy; tries main_df.csv then prepared_df.parquet).")
    p.add_argument("-m", "--method", required=True,
                   choices=["KMeans", "Agglomerative", "Birch", "GMM", "FuzzyCMeans"],
                   help="Clustering method to run.")
    p.add_argument("-k", "--k", type=int, required=True,
                   help="Number of clusters/components.")
    p.add_argument("--z-path", default=None,
                   help="Explicit path to z_mean.npy (overrides default).")
    p.add_argument("--df-path", default=None,
                   help="Explicit path to dataframe (CSV or Parquet). Overrides default search.")
    p.add_argument("--no-write", action="store_true",
                   help="Do not write outputs; only run and print summary.")
    p.add_argument("--no-silhouette", action="store_true",
                   help="Skip silhouette computation.")
    p.add_argument("--random-state", type=int, default=42,
                   help="Random seed for applicable algorithms.")
    p.add_argument("-v", "--verbose", action="count", default=1,
                   help="Increase verbosity (-v, -vv).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    _setup_logging(args.verbose)

    try:
        labels, sil, col = run_single_clustering_io(
            output_dir=args.output_dir,
            method=args.method,
            k=args.k,
            z_path=args.z_path,
            df_path=args.df_path,
            save_labels=(not args.no_write),
            compute_silhouette=(not args.no_silhouette),
            random_state=args.random_state,
        )

        print(f"Method: {args.method} | k: {args.k}")
        print(f"Labels shape: {labels.shape} | unique: {sorted(set(labels))}")
        if sil is not None:
            print(f"Silhouette: {sil:.6f}")
        else:
            print("Silhouette: (not computed)")
        print(f"Column name: {col}")
        return 0

    except Exception as e:
        logging.exception("Clustering failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
