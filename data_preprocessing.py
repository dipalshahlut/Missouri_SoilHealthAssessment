#!/usr/bin/env python3
# data_preprocessing.py
"""
Shared preprocessing utilities for the VAE + clustering pipeline.

Functions:
  - load_and_prepare_data(base_dir: str, analysis_depth: int)
      -> tuple[pd.DataFrame, np.ndarray, list[str]]

  - scale_features(X: np.ndarray)
      -> tuple[np.ndarray, sklearn.base.TransformerMixin]

Notes:
- Expects a CSV named: aggResult/MO_{analysis_depth}cm_for_clustering.csv
- Selects the 10 clustering feature columns observed in your logs.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Default feature set you used at 30 cm (from your run logs)
DEFAULT_FEATURE_COLS_30CM: List[str] = [
    "MnRs_dep",
    "clay_30cm",
    "sand_30cm",
    "om_30cm",
    "cec_30cm",
    "bd_30cm",
    "ec_30cm",
    "pH_30cm",
    "ksat_30cm",
    "awc_30cm",
]


def scale_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale features with StandardScaler.

    Parameters
    ----------
    X : np.ndarray
        2D array (n_samples, n_features).

    Returns
    -------
    X_scaled : np.ndarray
        Scaled features.
    scaler : StandardScaler
        Fitted scaler (persist if needed).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def _resolve_input_csv(base_dir: str, analysis_depth: int) -> str:
    """
    Build the expected path to the clustering CSV.
    Example for depth=30:
      {base_dir}/aggResult/MO_30cm_for_clustering.csv
    """
    fname = f"MO_{analysis_depth}cm_for_clustering.csv"
    path = os.path.join(base_dir, "aggResult", fname)
    return path


def _pick_feature_columns(df: pd.DataFrame, analysis_depth: int) -> List[str]:
    """
    Choose clustering feature columns based on depth.
    Currently uses the 30cm list (observed in your logs).
    Extend here if you later add other depths.
    """
    if analysis_depth == 30:
        wanted = DEFAULT_FEATURE_COLS_30CM
    else:
        # Try depth-aware names like "clay_{depth}cm", "sand_{depth}cm", etc.,
        # and keep the common, depth-agnostic features if present.
        depth_suffix = f"_{analysis_depth}cm"
        candidates = [
            f"clay{depth_suffix}",
            f"sand{depth_suffix}",
            f"om{depth_suffix}",
            f"cec{depth_suffix}",
            f"bd{depth_suffix}",
            f"ec{depth_suffix}",
            f"pH{depth_suffix}",
            f"ksat{depth_suffix}",
            f"awc{depth_suffix}",
        ]
        wanted = ["MnRs_dep"] + candidates

    # Keep only columns that exist; warn if any are missing.
    cols = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        logging.warning("Some expected feature columns are missing: %s", missing)
    if not cols:
        raise ValueError("No clustering feature columns found in the input dataframe.")
    return cols


def load_and_prepare_data(
    base_dir: str,
    analysis_depth: int = 30,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load the clustering table, select features, scale them, and return artifacts.

    Parameters
    ----------
    base_dir : str
        Base data directory (the parent of 'aggResult').
    analysis_depth : int
        Depth in cm (e.g., 30).

    Returns
    -------
    df : pd.DataFrame
        The full dataframe loaded from the CSV (unchanged, for downstream merges).
    data_scaled : np.ndarray
        Scaled feature matrix for VAE/clustering (row order aligned with df).
    cluster_cols : list[str]
        The exact feature columns used (for reproducibility).
    """
    csv_path = _resolve_input_csv(base_dir, analysis_depth)
    logging.info("Loading data from: %s", csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logging.info("Data loaded. Shape: %s", df.shape)

    cluster_cols = _pick_feature_columns(df, analysis_depth)
    logging.info("Selected %d columns for clustering: %s", len(cluster_cols), cluster_cols)

    X = df[cluster_cols].to_numpy(dtype=float)
    data_scaled, _ = scale_features(X)

    return df, data_scaled, cluster_cols
