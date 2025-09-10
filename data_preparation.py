#!/usr/bin/env python3
# data_preparation.py
"""
Stage 1 — Data Preparation

Outputs (written to OUTPUT_DIR):
  - prepared_df.parquet       : cleaned/filtered table used downstream
  - data_scaled.npy           : numpy array used to train the VAE
  - cluster_cols.json         : list of feature columns used for clustering
  - scaler.joblib (optional)  : fitted scaler (saved if available from utils)

Usage:
  python data_preparation.py \
    --base-dir /path/to/data \
    --output-dir /path/to/data/aggResult \
    --analysis-depth 30
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Optional dependency for saving the scaler, if we can fit one
try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

# Project utilities
from data_preprocessing import load_and_prepare_data, scale_features


def run(base_dir: Path, output_dir: Path, analysis_depth: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=== Stage 1: Data Preparation ===")
    logging.info("Base dir       : %s", base_dir)
    logging.info("Output dir     : %s", output_dir)
    logging.info("Analysis depth : %scm", analysis_depth)

    # 1) Load + prepare data (returns df, scaled matrix, and the feature list)
    df, data_scaled, cluster_cols = load_and_prepare_data(str(base_dir), analysis_depth)
    logging.info("Prepared dataframe shape: %s", df.shape)
    logging.info("Scaled matrix shape     : %s", getattr(data_scaled, "shape", None))
    logging.info("Num clustering features : %d", len(cluster_cols))

    # 2) Save primary artifacts
    df_path = output_dir / "prepared_df.parquet"
    npy_path = output_dir / "data_scaled.npy"
    cols_path = output_dir / "cluster_cols.json"

    df.to_parquet(df_path, index=False)
    np.save(npy_path, np.asarray(data_scaled))
    with open(cols_path, "w") as f:
        json.dump(list(cluster_cols), f, indent=2)

    logging.info("Saved: %s", df_path.name)
    logging.info("Saved: %s", npy_path.name)
    logging.info("Saved: %s", cols_path.name)

    # 3) (Optional) Save a fitted scaler for future inference, if possible
    # We re-fit using the same columns to persist the scaler object.
    scaler_path = output_dir / "scaler.joblib"
    try:
        X = df[cluster_cols].to_numpy()
        _, scaler = scale_features(X)  # expecting (scaled_array, fitted_scaler)
        if joblib is not None and scaler is not None:
            joblib.dump(scaler, scaler_path)
            logging.info("Saved: %s", scaler_path.name)
        else:
            logging.warning("Scaler not saved (joblib or scaler unavailable).")
    except Exception as e:  # pragma: no cover
        logging.warning("Could not persist scaler: %s", e)

    logging.info("Data preparation complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 — Data Preparation")
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=False,
        default=Path("/Users/dscqv/Desktop/SHA_copy/data"),
        help="Base data directory containing raw/processed inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=False,
        default=Path("/Users/dscqv/Desktop/SHA_copy/data/aggResult"),
        help="Directory to write intermediate artifacts.",
    )
    parser.add_argument(
        "--analysis-depth",
        type=int,
        required=False,
        default=30,
        help="Depth (cm) to select features for (e.g., 30).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run(args.base_dir, args.output_dir, args.analysis_depth)
