
#!/usr/bin/env python3
# data_preprocessing.py
"""
Shared preprocessing utilities for the VAE + clustering pipeline.

Functions:
  - load_and_prepare_data(base_dir: str, analysis_depth: int = 30)
      -> tuple[pd.DataFrame, np.ndarray, list[str]]

  - scale_features(X: np.ndarray)
      -> tuple[np.ndarray, sklearn.base.TransformerMixin]

Notes:
- Expects CSV: {base_dir}/aggResult/MO_{analysis_depth}cm_for_clustering.csv
- Matches the preprocessing used in Clustering-VAE_AlgorithmEval.py:
  • mean imputation (on area_ac, MnRs_dep, and depth-specific features)
  • RobustScaler on ['MnRs_dep'] + cluster_cols_base (excluding area_ac)
  • optional removal of known MUKEYs used by the pipeline
__author__ = "Dipal Shah"
__email__  = "dipalshah@missouri.edu"
__license__ = "MIT"
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

# Columns used at 30 cm (cluster_cols_base + MnRs_dep for scaling)
DEFAULT_FEATURE_COLS_30CM_BASE: List[str] = [
    "clay_30cm", "sand_30cm", "om_30cm", "cec_30cm",
    "bd_30cm", "ec_30cm", "pH_30cm", "ksat_30cm", "awc_30cm",
]
DEFAULT_FEATURES_TO_SCALE_30CM: List[str] = ["MnRs_dep"] + DEFAULT_FEATURE_COLS_30CM_BASE

# MUKEYs filtered in the main script before processing
DEFAULT_EXCLUDE_MUKEYS = [2498901, 2498902, 2500799, 2500800, 2571513, 2571527]

def _resolve_input_csv(base_dir: str, analysis_depth: int) -> str:
    """{base_dir}/aggResult/MO_{analysis_depth}cm_for_clustering.csv"""
    fname = f"MO_{analysis_depth}cm_for_clustering.csv"
    return os.path.join(base_dir, "aggResult", fname)

def _depth_cols(analysis_depth: int) -> Tuple[List[str], List[str]]:
    """
    Returns (cluster_cols_base, features_to_scale) for a given depth,
    mirroring the main script:
      cluster_cols_all = ['area_ac','MnRs_dep'] + cluster_cols_base
      features_to_scale = ['MnRs_dep'] + cluster_cols_base
    """
    if analysis_depth == 30:
        base = DEFAULT_FEATURE_COLS_30CM_BASE
        to_scale = DEFAULT_FEATURES_TO_SCALE_30CM
    else:
        suf = f"_{analysis_depth}cm"
        base = [
            f"clay{suf}", f"sand{suf}", f"om{suf}", f"cec{suf}",
            f"bd{suf}", f"ec{suf}", f"pH{suf}", f"ksat{suf}", f"awc{suf}",
        ]
        to_scale = ["MnRs_dep"] + base
    return base, to_scale

def load_and_prepare_data(
    base_dir: str,
    analysis_depth: int = 30,
    drop_mukeys: bool = True,
    excluded_mukeys: List[int] | None = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load the CSV, align columns, mean-impute, and Robust-scale features.

    Returns
    -------
    df : pd.DataFrame
        The dataframe after optional MUKEY filtering and imputation (unchanged columns kept).
    data_scaled : np.ndarray
        Scaled matrix for VAE/clustering (row order aligned with df).
    features_to_scale : list[str]
        The exact feature columns used for scaling (for reproducibility).
    """
    csv_path = _resolve_input_csv(base_dir, analysis_depth)
    logging.info("Loading data from: %s", csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logging.info("Data loaded. Shape: %s", df.shape)

    # Optional MUKEY filtering to mirror the main script
    if drop_mukeys and 'mukey' in df.columns:
        to_exclude = excluded_mukeys if excluded_mukeys is not None else DEFAULT_EXCLUDE_MUKEYS
        before = len(df)
        df = df[~df['mukey'].isin(to_exclude)]
        logging.info("Filtered MUKEYs: removed %d rows (%s)", before - len(df), to_exclude)

    # Build column sets to mirror cluster_cols_all and features_to_scale
    cluster_cols_base, features_to_scale = _depth_cols(analysis_depth)
    cluster_cols_all = ['MnRs_dep'] + cluster_cols_base

    # Validate presence (the main script requires area_ac in the file)
    missing_all = [c for c in cluster_cols_all if c not in df.columns]
    if missing_all:
        raise ValueError(f"Missing required data columns: {missing_all}")

    # Mean-impute across ALL clustering columns (including area_ac)
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(
        imputer.fit_transform(df[cluster_cols_all]),
        columns=cluster_cols_all,
        index=df.index
    )
    logging.info("Imputed missing values with mean across %d columns.", len(cluster_cols_all))

    # Robust-scale ONLY the features_to_scale (exclude area_ac)
    scaler = RobustScaler()
    X = data_imputed[features_to_scale].to_numpy(dtype=float)
    X_scaled = scaler.fit_transform(X)
    logging.info("Applied RobustScaler to %d features (excluding 'area_ac').", len(features_to_scale))

    return df, X_scaled, features_to_scale
