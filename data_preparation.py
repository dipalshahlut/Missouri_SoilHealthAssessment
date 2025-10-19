#!/usr/bin/env python3
# data_preparation.py
"""
Stage 1 — Data Preparation

Outputs (written to OUTPUT_DIR):
  - prepared_df.parquet       : cleaned/filtered table used downstream
  - data_scaled.npy           : numpy array used to train the VAE
  - cluster_cols.json         : list of feature columns used for clustering
  - scaler.joblib (optional)  : fitted scaler (if helper returns it)

Usage:
  # Minimal (uses defaults)
  python data_preparation.py \
    --base-dir /path/to/data \
    --output-dir /path/to/data/aggResult \
    --analysis-depth 30

  # With a custom MUKEY drop-list
  python data_preparation.py \
    --base-dir /path/to/data \
    --output-dir /path/to/data/aggResult \
    --analysis-depth 30 \
    --exclude-mukeys 2498901 2498902 2500799 2500800 2571513 2571527

__author__ = "Dipal Shah"
__email__  = "dipalshah@missouri.edu"
__license__ = "MIT"
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Optional dependency for saving the scaler
try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

# Project utilities — expected to perform filtering, imputation,
# scaling, returning (df, data_scaled, cluster_cols[, fitted_scaler])
from data_preprocessing import load_and_prepare_data

# -------------------------------
# Helpers for MUKEY order control
# -------------------------------
def _normalize_key_strings(arr_like) -> np.ndarray:
    """
    Coerce any array-like to a clean ndarray[str]:
    - convert to string
    - strip whitespace
    - drop trailing '.0'
    """
    s = pd.Series(list(arr_like), dtype="object").astype(str)
    s = s.str.strip().str.replace(r"\.0$", "", regex=True)
    return s.to_numpy(dtype=str)


def _load_prepared_keys(keys_path: Path) -> Optional[np.ndarray]:
    """
    Robustly load prepared_row_keys.npy:
    - first try allow_pickle=False (safe, preferred)
    - if ValueError (object array), retry with allow_pickle=True (legacy)
    - return normalized ndarray[str] or None if path missing
    """
    if not keys_path.exists():
        return None
    try:
        arr = np.load(keys_path, allow_pickle=False)
    except ValueError:
        # Legacy object array — load with pickle, then normalize to str
        arr = np.load(keys_path, allow_pickle=True)
    return _normalize_key_strings(arr)


def _reindex_to_prepared_keys(
    df: pd.DataFrame,
    data_scaled: np.ndarray,
    output_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, bool]:
    """
    If main.py wrote prepared_row_keys.npy, enforce that MUKEY order by
    reindexing BOTH df and data_scaled rows accordingly.

    Supports the case where keys are a SUBSET of df (e.g., 6080 vs 6624).
    Returns: (df_reindexed, data_scaled_reindexed, applied: bool)
    """
    keys = _load_prepared_keys(output_dir / "prepared_row_keys.npy")
    if keys is None:
        logging.info("No prepared_row_keys.npy found — keeping current row order.")
        return df, data_scaled, False

    if "mukey" not in df.columns:
        raise ValueError(
            "prepared_row_keys.npy exists, but 'mukey' column is missing in the dataframe."
        )

    mk = _normalize_key_strings(df["mukey"].to_numpy())
    pos = {m: i for i, m in enumerate(mk)}  # mukey -> row index in current df

    # Keep only keys that are present in df, preserving order
    idx_list = [pos[k] for k in keys if k in pos]
    missing = [k for k in keys if k not in pos]

    if not idx_list:
        raise ValueError(
            "None of the MUKEYs from prepared_row_keys.npy were found in the dataframe. "
            "Ensure main.py and this stage use the same inputs."
        )

    if missing:
        logging.warning(
            "prepared_row_keys.npy contains %d keys not present in df (showing up to 10): %s",
            len(missing), ", ".join(missing[:10])
        )

    before_rows = len(df)
    df2 = df.iloc[idx_list].reset_index(drop=True)

    # Reindex data_scaled rows if its length matches df
    if isinstance(data_scaled, np.ndarray) and data_scaled.shape[0] == before_rows:
        data_scaled = data_scaled[np.asarray(idx_list, dtype=np.int64), :]
    else:
        logging.warning(
            "data_scaled rows (%s) != df rows before reindex (%s); "
            "only df was reindexed.",
            getattr(data_scaled, "shape", None), before_rows
        )

    logging.info(
        "Reindexed to prepared_row_keys.npy (kept %d of %d rows).",
        len(df2), before_rows
    )
    return df2, data_scaled, True


# -------------------------------
# Main runner
# -------------------------------
def run(
    base_dir: Path,
    output_dir: Path,
    analysis_depth: int,
    exclude_mukeys: Optional[List[int]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=== Stage 1: Data Preparation ===")
    logging.info("Base dir       : %s", base_dir)
    logging.info("Output dir     : %s", output_dir)
    logging.info("Analysis depth : %scm", analysis_depth)
    if exclude_mukeys:
        logging.info("Exclude MUKEYs : %s", exclude_mukeys)

    # 1) Load + prepare (mirror monolith behavior where possible).
    # Try new-style helper signature first; fall back to legacy 2-arg call if needed.
    try:
        out = load_and_prepare_data(
            str(base_dir),
            analysis_depth,
            drop_mukeys=True,
            excluded_mukeys=exclude_mukeys,
        )
    except TypeError:
        logging.warning(
            "Your load_and_prepare_data(...) does not accept exclude args; "
            "falling back to legacy call. MUKEY filtering must be handled inside that function."
        )
        out = load_and_prepare_data(str(base_dir), analysis_depth)

    # Unpack (3-tuple legacy or 4-tuple with scaler)
    if len(out) == 3:
        df, data_scaled, cluster_cols = out  # type: ignore[misc]
        fitted_scaler = None
        logging.warning(
            "load_and_prepare_data returned 3 items; no scaler to persist from this stage."
        )
    else:
        df, data_scaled, cluster_cols, fitted_scaler = out  # type: ignore[misc]

    # 1b) Enforce canonical MUKEY order if main.py wrote it (supports subset)
    df, data_scaled, _applied = _reindex_to_prepared_keys(df, np.asarray(data_scaled), output_dir)

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

    # 3) (Optional) Persist the SAME fitted scaler used during prep
    if fitted_scaler is not None and joblib is not None:
        try:
            scaler_path = output_dir / "scaler.joblib"
            joblib.dump(fitted_scaler, scaler_path)
            logging.info("Saved: %s", scaler_path.name)
        except Exception as e:  # pragma: no cover
            logging.warning("Could not persist scaler: %s", e)
    else:
        logging.warning("Scaler not saved (helper did not return it or joblib unavailable).")

    logging.info("Data preparation complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 — Data Preparation")
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Base data directory containing raw/processed inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write intermediate artifacts.",
    )
    parser.add_argument(
        "--analysis-depth",
        type=int,
        required=True,
        default=30,
        help="Depth (cm) to select features for (e.g., 30).",
    )
    parser.add_argument(
        "--exclude-mukeys",
        nargs="*",
        type=int,
        default=None,
        help="List of MUKEYs to exclude (space-separated). Example: --exclude-mukeys 2498901 2498902 ...",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run(args.base_dir, args.output_dir, args.analysis_depth, exclude_mukeys=args.exclude_mukeys)
