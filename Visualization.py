#!/usr/bin/env python3
"""
visualization.py

Generate standard visualizations for a single clustering result on VAE latent space:
  1) 2D latent scatter colored by cluster labels
  2) (Optional) Boxplots of scaled variables by cluster (requires data_scaled.*)
  3) Area-by-cluster bar chart (requires area_ac in the dataframe)
  4) (Optional) Per-feature boxplots (one PNG per feature) (requires data_scaled.*)
  5) (Optional) Feature-importance heatmap via ANOVA F (requires data_scaled.*)
  6) (Optional) Centroid heatmap (clusters × features, scaled) (requires data_scaled.*)

Expected inputs (in OUTPUT_DIR):
  - z_mean.npy
  - main_df_with_{method}_k{k}.csv   (from clustering_algorithms.py)
  - (optional) data_scaled.npy/.csv/.parquet  (columns = scaled feature names; .npy preferred)

Outputs (written to <OUTPUT_DIR>/figures/):
  latent_<METHOD>_k<K>.png
  boxplot_scaled_vars_<METHOD>_k<K>.png              (if scaled data present)
  centroid_heatmap_<METHOD>_k<K>.png                 (if scaled data present)
  feature_importance_heatmap_<METHOD>_k<K>.png       (if scaled data present)
  area_by_cluster_<METHOD>_k<K>.png
  per_feature_boxplots_<METHOD>_k<K>/box_<feat>_...png  (if scaled data present)

# Example CLI:
# python visualization.py \
#   -o /path/to/data/aggResult \
#   -m KMeans -k 12 \
#   --scaled-path /path/to/data/aggResult/data_scaled.npy
"""

from __future__ import annotations

import os
import sys
import re
import json
import argparse
import logging
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _SEABORN = True
except Exception:
    _SEABORN = False

from sklearn.feature_selection import f_classif

# MUKEYs filtered during Stage-1; used to auto-align df with z if lengths differ
DEFAULT_EXCLUDE_MUKEYS = [2498901, 2498902, 2500799, 2500800, 2571513, 2571527]

# -----------------------
# Logging & small helpers
# -----------------------
def _setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING
    if verbosity >= 2: level = logging.INFO
    if verbosity >= 3: level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def _labels_col(method: str, k: int) -> str:
    return f"{method}_k{k}"

def _latent_path(output_dir: str) -> str:
    return os.path.join(output_dir, "z_mean.npy")

def _df_with_labels_path(output_dir: str, method: str, k: int) -> str:
    return os.path.join(output_dir, f"main_df_with_{method}_k{k}.csv")

def _scaled_path(output_dir: str) -> str:
    # Prefer npy if present; fallback to csv
    npy = os.path.join(output_dir, "data_scaled.npy")
    csv = os.path.join(output_dir, "data_scaled.csv")
    pq  = os.path.join(output_dir, "data_scaled.parquet")
    if os.path.exists(npy): return npy
    if os.path.exists(csv): return csv
    if os.path.exists(pq):  return pq
    return npy  # default hint

# -----------------------
# Loaders
# -----------------------
def _load_latent(z_path: str) -> np.ndarray:
    if not os.path.exists(z_path):
        raise FileNotFoundError(f"Missing {z_path}")
    z = np.load(z_path)
    if z.ndim != 2:
        raise ValueError(f"z_mean must be 2D, got {z.shape}")
    if z.shape[1] < 2:
        raise ValueError(f"z_mean must have at least 2 columns to scatter, got {z.shape[1]}")
    return z

def _load_scaled_df(scaled_path: str):
    """
    Load scaled features as a DataFrame from .npy/.csv/.parquet.
    If .npy, try to get column names from cluster_cols.json in the same folder.
    """
    if not scaled_path or not os.path.exists(scaled_path):
        return None
    try:
        ext = Path(scaled_path).suffix.lower()
        if ext == ".npy":
            arr = np.load(scaled_path)
            cols = None
            cc_path = Path(scaled_path).with_name("cluster_cols.json")
            if cc_path.exists():
                try:
                    with open(cc_path, "r") as f:
                        meta = json.load(f)
                    if isinstance(meta, dict) and "cluster_cols" in meta:
                        cols = meta["cluster_cols"]
                    elif isinstance(meta, list):
                        cols = meta
                except Exception:
                    cols = None
            if cols is None:
                cols = [f"feat{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols)

        if ext in (".csv", ".txt"):
            return pd.read_csv(scaled_path)

        if ext in (".parquet", ".pq"):
            return pd.read_parquet(scaled_path)
    except Exception as e:
        logging.warning("Could not load scaled data %s: %s", scaled_path, e)
        return None

# -----------------------
# Alignment & sanitation
# -----------------------
def _load_df_with_labels(df_path: str, col: str, output_dir: str | None = None, z_len: int | None = None) -> pd.DataFrame:
    """
    Loads a dataframe with cluster labels. If the expected legacy CSV is missing
    OR its row count does not match z_len (when provided), it falls back to
    parquet outputs, computes labels if needed, and writes a fresh CSV.
    """
    p = Path(df_path)
    out_dir = Path(output_dir) if output_dir else p.parent
    pq = out_dir / "main_with_best_labels_allAlgo.parquet"

    def _rebuild_from_parquet() -> pd.DataFrame:
        if not pq.exists():
            raise FileNotFoundError(
                f"Missing both legacy CSV {p} and fallback parquet {pq}. "
                "Run Stage 3 (clustering_selection.py) first."
            )
        dfp = pd.read_parquet(pq)

        # If requested label column missing, compute from z + method/k
        if col not in dfp.columns:
            z_path = out_dir / "z_mean.npy"
            if not z_path.exists():
                raise FileNotFoundError(
                    f"Column '{col}' not in parquet and '{z_path.name}' missing to compute it. "
                    "Run Stage 2 (vae_training.py) first."
                )
            z = np.load(z_path)

            m2 = re.match(r"([A-Za-z]+)_best(\d+)$", col)
            if not m2:
                m3 = re.match(r"([A-Za-z]+)_k(\d+)$", col)
                if not m3:
                    raise ValueError(f"Cannot infer (method,k) from col '{col}'.")
                method, k = m3.group(1), int(m3.group(2))
            else:
                method, k = m2.group(1), int(m2.group(2))

            try:
                from clustering_evaluation import _fit_predict_labels
            except Exception as e:
                raise ImportError(
                    "clustering_evaluation._fit_predict_labels not importable; ensure it is in PYTHONPATH."
                ) from e

            dfp[col] = _fit_predict_labels(method, int(k), np.asarray(z), random_state=42)
            dfp.to_parquet(pq, index=False)

        dfp["cluster"] = dfp[col]
        dfp.to_csv(p, index=False)
        return dfp

    if p.exists():
        df = pd.read_csv(p)
        if (z_len is None) or (len(df) == z_len):
            return df
        logging.warning("Legacy CSV %s has %d rows, but z has %s rows; rebuilding from parquet.",
                        p.name, len(df), z_len)
        return _rebuild_from_parquet()

    return _rebuild_from_parquet()

def _auto_align_df_to_z(
    df: pd.DataFrame,
    z: np.ndarray,
    exclude_mukeys: Optional[list[int]] = None
) -> pd.DataFrame:
    """
    If df length differs from z, attempt MUKEY-based alignment (Stage-1 filtering).
    If still mismatched, raise a clear error to avoid silent misalignment.
    """
    if len(df) == len(z):
        return df

    logging.warning("Mismatch: df=%d vs z=%d; attempting MUKEY-based alignment.", len(df), len(z))
    if "mukey" in df.columns:
        before = len(df)
        drop_list = exclude_mukeys if (exclude_mukeys and len(exclude_mukeys) > 0) else DEFAULT_EXCLUDE_MUKEYS
        df = df[~df["mukey"].isin(drop_list)].reset_index(drop=True)
        logging.info("Dropped %d rows using MUKEY filter: %s", before - len(df), drop_list)

    if len(df) != len(z):
        raise ValueError(
            f"Row mismatch after alignment: df={len(df)} vs z={len(z)}. "
            "Ensure you are using Stage-1/3 artifacts from the same run "
            "or pass the correct --df-with-labels-path."
        )
    return df

def _sanitize_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    if np.issubdtype(labels.dtype, np.floating) and np.all(np.isfinite(labels)):
        rounded = np.round(labels)
        if np.allclose(labels, rounded):
            labels = rounded.astype(int)
    return labels

# -----------------------
# Plotters
# -----------------------
def plot_latent_scatter_2d(z, labels, out_path, title):
    labels = _sanitize_labels(labels)
    if z.shape[0] != labels.shape[0]:
        raise ValueError(f"labels ({labels.shape[0]}) != z ({z.shape[0]})")
    if z.shape[1] < 2:
        raise ValueError(f"z requires at least 2 columns; got {z.shape[1]}")

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(z[:, 0], z[:, 1], c=labels, s=12, alpha=0.85, cmap="viridis")
    plt.colorbar(sc, label="Cluster")
    plt.title(title)
    plt.xlabel("Latent 1"); plt.ylabel("Latent 2")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    _ensure_parent_dir(out_path)
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_boxplots(df, labels_col, xs, out_path):
    if not _SEABORN:
        logging.info("Skipping boxplots: seaborn not available.")
        return
    if len(df) != len(xs):
        logging.warning("Skipping boxplots: scaled matrix rows != df rows (%d vs %d).", len(xs), len(df))
        return
    dfp = pd.concat([df[[labels_col]], xs], axis=1)
    dfp[labels_col] = dfp[labels_col].astype(int) + 1   # shift cluster IDs
    dfm = dfp.melt(id_vars=[labels_col], var_name="variable", value_name="value")
    g = sns.catplot(
        data=dfm, x=labels_col, y="value", col="variable",
        kind="box", col_wrap=4, sharey=False, height=3, aspect=1.05
    )
    g.set_axis_labels("Cluster", "Scaled value")
    g.set_titles("{col_name}")
    plt.tight_layout()
    _ensure_parent_dir(out_path)
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_area(df, labels_col, out_path, area_col="area_ac"):
    if area_col not in df.columns:
        logging.info("Skipping area plot: '%s' not in dataframe.", area_col)
        return
    df[labels_col] = df[labels_col].astype(int) + 1   # shift cluster IDs
    area = df.groupby(labels_col)[area_col].sum().sort_index()

    plt.figure(figsize=(8, 5))
    area.plot(kind="bar", edgecolor="k", alpha=0.85)
    plt.title("Total Area by Cluster")
    plt.xlabel("Cluster"); plt.ylabel("Total Area (ac)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    _ensure_parent_dir(out_path)
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_per_feature_boxplots(df, labels_col, X_scaled, out_dir):
    """Save one PNG per feature: feature (scaled) vs cluster labels."""
    if not _SEABORN:
        logging.info("Skipping per-feature boxplots: seaborn not available.")
        return
    if X_scaled is None:
        logging.info("Skipping per-feature boxplots: scaled data missing.")
        return
    if len(df) != len(X_scaled):
        logging.warning("Skipping per-feature boxplots: scaled matrix rows != df rows (%d vs %d).",
                        len(X_scaled), len(df))
        return

    os.makedirs(out_dir, exist_ok=True)  # ensure the directory itself exists
    dfp = pd.concat([df[[labels_col]], X_scaled], axis=1)
    dfp[labels_col] = dfp[labels_col].astype(int) + 1   # shift cluster IDs

    for feat in X_scaled.columns:
        plt.figure(figsize=(7.2, 5.0))
        sns.boxplot(x=dfp[labels_col], y=dfp[feat], showfliers=False)
        plt.xlabel("Cluster")
        plt.ylabel(f"{feat} (scaled)")
        plt.title(f"{feat} by cluster")
        plt.tight_layout()
        out_file = os.path.join(out_dir, f"box_{feat}_{labels_col}.png")
        plt.savefig(out_file, dpi=300)
        plt.close()

def plot_centroid_heatmap(df, labels_col, X_scaled, out_path):
    """Clusters × features mean heatmap (on scaled features)."""
    if not _SEABORN:
        logging.info("Skipping centroid heatmap: seaborn not available.")
        return
    if X_scaled is None:
        logging.info("Skipping centroid heatmap: scaled data missing.")
        return
    if len(df) != len(X_scaled):
        logging.warning("Skipping centroid heatmap: scaled matrix rows != df rows (%d vs %d).",
                        len(X_scaled), len(df))
        return

    mat = pd.concat([df[[labels_col]], X_scaled], axis=1).groupby(labels_col).mean().sort_index()
    mat_z = (mat - mat.mean(axis=0)) / (mat.std(axis=0, ddof=0) + 1e-12)

    plt.figure(figsize=(max(8, X_scaled.shape[1]*0.5), 6))
    ax = sns.heatmap(mat_z.T, cmap="YlGnBu", center=0, linewidths=0.3,
                     cbar_kws={"label": "Z-score of feature mean"})
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature")
    plt.title("Cluster centroids (scaled, z-normalized)")
    plt.tight_layout()
    _ensure_parent_dir(out_path)
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_feature_importance_heatmap(df, labels_col, X_scaled, out_path):
    """Feature importance via ANOVA F-score; 1-row heatmap (features as columns)."""
    if not _SEABORN:
        logging.info("Skipping feature-importance heatmap: seaborn not available.")
        return
    if X_scaled is None:
        logging.info("Skipping feature-importance heatmap: scaled data missing.")
        return
    if len(df) != len(X_scaled):
        logging.warning("Skipping feature-importance heatmap: scaled matrix rows != df rows (%d vs %d).",
                        len(X_scaled), len(df))
        return

    y = df[labels_col].astype(int).values
    X = X_scaled.values
    try:
        F, _ = f_classif(X, y)  # (n_features,)
    except Exception as e:
        logging.warning("ANOVA F computation failed: %s", e)
        return

    imp = pd.Series(F, index=X_scaled.columns)
    plt.figure(figsize=(max(6, len(imp)*0.4), 3.6))
    ax = sns.heatmap(imp.to_frame("ANOVA_F").T, cmap="YlGnBu",
                     cbar_kws={"label": "ANOVA F"})
    ax.set_xlabel("Feature")
    ax.set_ylabel("")
    plt.title("Feature importance vs clusters (ANOVA F)")
    plt.tight_layout()
    _ensure_parent_dir(out_path)
    plt.savefig(out_path, dpi=300)
    plt.close()

# -----------------------
# Orchestrator
# -----------------------
def make_all_plots(
    output_dir,
    method,
    k,
    z_path=None,
    df_with_labels_path=None,
    scaled_path=None,
    exclude_mukeys: Optional[list[int]] = None
):
    output_dir = str(output_dir)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    col = _labels_col(method, k)

    # Output paths (ALL under figures_dir)
    latent_out   = os.path.join(figures_dir, f"latent_{method}_k{k}.png")
    boxplot_out  = os.path.join(figures_dir, f"boxplot_scaled_vars_{method}_k{k}.png")
    per_feat_dir = os.path.join(figures_dir, f"per_feature_boxplots_{method}_k{k}")  # directory
    centroid_out = os.path.join(figures_dir, f"centroid_heatmap_{method}_k{k}.png")
    fimps_out    = os.path.join(figures_dir, f"feature_importance_heatmap_{method}_k{k}.png")
    area_out     = os.path.join(figures_dir, f"area_by_cluster_{method}_k{k}.png")

    # Load latent
    z = _load_latent(z_path or _latent_path(output_dir))

    # Load df with labels (CSV or fallback parquet) and align to z if needed
    df = _load_df_with_labels(
        df_with_labels_path or _df_with_labels_path(output_dir, method, k),
        col,
        output_dir=output_dir,
        z_len=z.shape[0],
    )
    df = _auto_align_df_to_z(df, z, exclude_mukeys=exclude_mukeys or DEFAULT_EXCLUDE_MUKEYS)

    # Latent scatter
    plot_latent_scatter_2d(z, df[col].values, latent_out, f"VAE Latent Space — {method} (k={k})")

    # Optional: boxplots & extras (if scaled data exists and row counts match)
    xs = _load_scaled_df(scaled_path or _scaled_path(output_dir))
    if xs is not None:
        plot_boxplots(df, col, xs, boxplot_out)
        plot_per_feature_boxplots(df, col, xs, per_feat_dir)
        plot_centroid_heatmap(df, col, xs, centroid_out)
        plot_feature_importance_heatmap(df, col, xs, fimps_out)
    else:
        logging.info("Scaled data not found; skipping boxplots/centroid/feature-importance/per-feature plots.")

    # Optional: area by cluster (if area_ac present)
    plot_area(df, col, area_out)

    return {
        "latent_path": latent_out,
        "boxplot_path": boxplot_out if xs is not None else None,
        "centroid_path": centroid_out if xs is not None else None,
        "feature_importance_path": fimps_out if xs is not None else None,
        "per_feature_dir": per_feat_dir if xs is not None else None,
        "area_path": area_out,
    }

# -----------------------
# CLI
# -----------------------
def _build_argparser():
    p = argparse.ArgumentParser(description="Generate visualizations for clustering results.")
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument(
        "-m", "--method", required=True,
        choices=["KMeans", "Agglomerative", "Birch", "GMM", "FuzzyCMeans"]
    )
    p.add_argument("-k", "--k", type=int, required=True)
    p.add_argument("--z-path", default=None)
    p.add_argument("--df-with-labels-path", default=None)
    p.add_argument("--scaled-path", default=None,
                   help="Path to data_scaled.npy/.csv/.parquet (defaults to files in output-dir).")
    p.add_argument(
        "--exclude-mukeys",
        nargs="*",
        type=int,
        default=None,
        help="MUKEYs to exclude if df and z lengths mismatch. Example: --exclude-mukeys 2498901 2498902 ...",
    )
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p

def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        out = make_all_plots(
            args.output_dir, args.method, args.k,
            z_path=args.z_path,
            df_with_labels_path=args.df_with_labels_path,
            scaled_path=args.scaled_path,
            exclude_mukeys=args.exclude_mukeys,
        )
        print("✅ Visualization completed successfully.")
        print("Results saved in:", os.path.join(str(args.output_dir), "figures"))
        for k_, v_ in out.items():
            if v_:
                print(f"  - {k_}: {v_}")
        return 0
    except Exception as e:
        logging.exception("Visualization failed: %s", e)
        return 1

if __name__ == "__main__":
    sys.exit(main())
