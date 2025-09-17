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

# Example CLI Usage:
python visualization.py \
    -o /path/to/data/aggResult \
    -m KMeans -k 12 \
    --scaled-path /path/to/data/aggResult/data_scaled.npy

__author__ = "Dipal Shah"
__email__  = "dipalshah@missouri.edu"
__license__ = "MIT"
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
import matplotlib.ticker as mticker
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

DEFAULT_COLOR_PALETTE = [
    "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
    "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a",
    # additional complementary colors
    "#b15928",  # earthy brown
    "#ffff99",  # soft yellow
    "#8dd3c7",  # teal cyan
    "#bebada",  # lavender purple
    "#fb8072",  # salmon pink
]

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


def plot_per_feature_boxplots(
    df,
    labels_col,
    X_scaled,
    out_dir,
    df_unscaled=None,          # optional: compute stats from unscaled values
    stats_from="scaled",       # "scaled" or "unscaled" (where to compute mean/median/Q1/Q3)
    showfliers=False,
    half=0.28,                 # half-width of the little stat bars
    draw_global_mean=True,
    global_mean_color="red",
    global_mean_ls="--",
    global_mean_lw=1.5,
    dpi=300,
):
    """
    Save one PNG per feature: feature (scaled) vs cluster labels, with overlays:
      - Global mean (horizontal dashed line)
      - Per-cluster mean (thick bar), median (diamond marker), Q1/Q3 (thin bars)

    If `df_unscaled` is provided and `stats_from="unscaled"`, the overlays are computed
    on the unscaled values while the boxplots are still drawn from X_scaled.
    """
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

    os.makedirs(out_dir, exist_ok=True)

    # ---- Build plotting frames ----
    dfp_scaled = pd.concat([df[[labels_col]].reset_index(drop=True),
                            X_scaled.reset_index(drop=True)], axis=1)

    # Shift labels to start at 1, ensure ints
    try:
        dfp_scaled[labels_col] = dfp_scaled[labels_col].astype(int) + 1
    except Exception:
        # if labels are strings, just leave them; don’t shift
        pass

    # Order is 1..k if labels are numeric; otherwise sorted unique
    if pd.api.types.is_integer_dtype(dfp_scaled[labels_col]):
        k = int(dfp_scaled[labels_col].nunique())
        order = list(range(1, k + 1))
    else:
        order = sorted(dfp_scaled[labels_col].unique(), key=lambda x: (isinstance(x, str), x))

    pos_by_cat = {cat: i for i, cat in enumerate(order)}  # x positions (0..n-1) for overlay bars

    # Choose palette
    try:
        DEFAULT_COLOR_PALETTE  # check if defined globally
        custom_palette = DEFAULT_COLOR_PALETTE
    except NameError:
        custom_palette = None

    if custom_palette is None:
        palette = sns.color_palette("tab20", len(order))
    else:
        if len(custom_palette) < len(order):
            logging.warning("custom palette has %d colors but k=%d; colors will repeat.",
                            len(custom_palette), len(order))
        palette = custom_palette[:len(order)]

    # Stats source: scaled vs unscaled
    if stats_from == "unscaled" and df_unscaled is not None and len(df_unscaled) == len(df):
        dfp_stats = pd.concat([df[[labels_col]].reset_index(drop=True),
                               df_unscaled.reset_index(drop=True)], axis=1).copy()
        # shift labels the same way for grouping
        try:
            dfp_stats[labels_col] = dfp_stats[labels_col].astype(int) + 1
        except Exception:
            pass
        y_label_suffix = ""   # raw units
    else:
        if stats_from == "unscaled":
            logging.warning("stats_from='unscaled' requested but df_unscaled missing/mismatched; using scaled.")
        dfp_stats = dfp_scaled
        y_label_suffix = " (scaled)"

    # Drop rows missing labels
    dfp_scaled = dfp_scaled.dropna(subset=[labels_col])
    dfp_stats  = dfp_stats.loc[dfp_scaled.index]

    # Ensure label dtype alignment
    try:
        dfp_stats[labels_col] = dfp_stats[labels_col].astype(dfp_scaled[labels_col].dtype)
    except Exception:
        pass

    # ---- Plot each feature ----
    for feat in X_scaled.columns:
        if feat not in dfp_scaled.columns or feat not in dfp_stats.columns:
            logging.warning("Feature %s not present in plotting/stats frames; skipping.", feat)
            continue

        fig, ax = plt.subplots(figsize=(7.4, 5.0))

        # Base boxplot (from scaled data)
        sns.boxplot(
            data=dfp_scaled,
            x=labels_col,
            y=feat,
            order=order,
            showfliers=showfliers,
            hue = labels_col,
            palette=palette,
            legend = False,
            ax=ax,
        )

        # Global mean (from chosen stats frame)
        if draw_global_mean:
            try:
                global_mean = float(dfp_stats[feat].mean())
                ax.axhline(global_mean, color=global_mean_color,
                           linestyle=global_mean_ls, linewidth=global_mean_lw, zorder=1)
            except Exception as e:
                logging.debug("Global mean line skipped for %s: %s", feat, e)

        # Per-cluster overlays: mean, median, Q1, Q3 (from chosen stats frame)
        try:
            stats = (
                dfp_stats.groupby(labels_col, observed=True)[feat]
                .agg(
                    mean="mean",
                    q1=lambda s: s.quantile(0.25),
                    med="median",
                    q3=lambda s: s.quantile(0.75),
                )
                .reindex(order)
            )
            #half = 0.28
            for cat, row in stats.iterrows():
                xpos = pos_by_cat[cat]
                # Draw centered bars at each category position
                m = row["mean"]
                if pd.notna(m):
                    ax.hlines(float(m), xpos - half, xpos + half,
                              colors="black", linewidth=2.0, zorder=4)

                med = row["med"]
                if pd.notna(med):
                    ax.scatter([xpos], [float(med)], s=40, zorder=5,
                               marker="D", facecolors="none", edgecolors="black")

                q1 = row["q1"]
                if pd.notna(q1):
                    ax.hlines(float(q1), xpos - half, xpos + half,
                              colors="black", linewidth=1.2, zorder=4)

                q3 = row["q3"]
                if pd.notna(q3):
                    ax.hlines(float(q3), xpos - half, xpos + half,
                              colors="black", linewidth=1.2, zorder=4)
        except Exception as e:
            logging.debug("Per-cluster stat overlays skipped for %s: %s", feat, e)

        # === Legend ===
        from matplotlib.lines import Line2D # For custom legends
        gm_handle = Line2D([0], [0], color="red", linestyle="--", lw=1.5,
                           label=f"Global Mean (Unscaled {global_mean:.1f})")
        mean_handle = Line2D([0], [0], color="black", lw=2.0, label="Cluster Mean")
        med_handle = Line2D([0], [0], marker="D", markersize=6,
                            markerfacecolor="none", markeredgecolor="black",
                            linestyle="None", label="Cluster Median")
        ax.legend(handles=[gm_handle, mean_handle, med_handle],
                  loc="upper right", fontsize=8, frameon=False)


        ax.set_xlabel("Cluster")
        ax.set_ylabel(f"{feat}{y_label_suffix}")
        ax.set_title(f"{feat} by cluster")
        plt.tight_layout()

        out_file = os.path.join(out_dir, f"box_{feat}_{labels_col}.png")
        plt.savefig(out_file, dpi=dpi)
        plt.close(fig)


def plot_centroid_heatmap(df, labels_col, X_scaled, out_path):
    """Clusters × features mean heatmap (z-score normalized, with numbers in cells)."""
    if not _SEABORN:
        logging.info("Skipping centroid heatmap: seaborn not available.")
        return
    if X_scaled is None:
        logging.info("Skipping centroid heatmap: scaled data missing.")
        return
    if len(df) != len(X_scaled):
        logging.warning(
            "Skipping centroid heatmap: scaled matrix rows != df rows (%d vs %d).",
            len(X_scaled), len(df),
        )
        return

    # Combine labels + features
    dfp = pd.concat([df[[labels_col]].reset_index(drop=True),
                     X_scaled.reset_index(drop=True)], axis=1)

    # Robust label remap → 1..k for grouping & display
    labs = dfp[labels_col]
    # if convertible to int, fine; else just keep as-is
    try:
        labs = labs.astype(int)
    except Exception:
        pass
    uniq = sorted(pd.unique(labs))
    disp_map = {orig: i+1 for i, orig in enumerate(uniq)}
    dfp["cluster_disp"] = labs.map(disp_map)

    # Centroids on scaled features
    # *** Only aggregate over the feature columns ***
    feature_cols = list(X_scaled.columns)          # <- excludes the label column
    centroids = dfp.groupby("cluster_disp")[feature_cols].mean().sort_index()

    # Z-score normalize per feature (column-wise)
    eps = 1e-12
    centroids_z = (centroids - centroids.mean(axis=0)) / (centroids.std(axis=0, ddof=0) + eps)

    # Rows = features, Cols = clusters 1..k
    centroids_plot = centroids_z.T.copy()
    k = centroids_plot.shape[1]
    centroids_plot.columns = [str(c) for c in centroids_plot.columns]  # already 1..k

    # Symmetric color scale (optional but nice for z-scores)
    vmax = float(np.nanmax(np.abs(centroids_plot.values)))
    vmin = -vmax

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        centroids_plot,
        cmap="YlGnBu",
        center=0,
        vmin=vmin, vmax=vmax,        # symmetric scale
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar=True,
        cbar_kws={"label": "Z-score of feature mean"},
    )

    ax.set_xlabel(f"Number of Clusters (k={k})", fontsize=14, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Colorbar styling
    try:
        cbar = ax.collections[0].colorbar
        cbar.set_label("Z-score of feature mean", fontsize=14, fontweight="bold")
        cbar.ax.tick_params(labelsize=12)
    except Exception:
        pass

    plt.tight_layout()
    _ensure_parent_dir(out_path)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance_heatmap(
    df,
    labels_col,
    X_scaled,
    out_path,
    random_state=42,
    test_size=0.2,
    n_estimators=200,
    annotate=True,
    dpi=300,
):
    """
    RF + SHAP feature-importance heatmap in PERCENT:
      - mean(|SHAP|) per (cluster, feature) on held-out set
      - column-normalized so each cluster sums to 100%
      - cells annotated as 'xx.x%'
    """
    if not _SEABORN:
        logging.info("Skipping FI heatmap: seaborn not available.")
        return
    try:
        import shap  # ensure available
        _ = shap.__version__
    except Exception:
        logging.info("Skipping FI heatmap: SHAP not available.")
        return
    if X_scaled is None:
        logging.info("Skipping FI heatmap: scaled data missing.")
        return
    if len(df) != len(X_scaled):
        logging.warning("Skipping FI heatmap: scaled matrix rows != df rows (%d vs %d).",
                        len(X_scaled), len(df))
        return

    # Ensure DataFrame to preserve feature names
    if not isinstance(X_scaled, pd.DataFrame):
        X_scaled = pd.DataFrame(X_scaled, columns=[f"feat{i}" for i in range(X_scaled.shape[1])])

    X = X_scaled.values
    y = df[labels_col].values
    feat_names = list(X_scaled.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200,
                                random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Accuracy (log only)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info("Random Forest accuracy on predicting clusters: %.3f", acc)

    # SHAP values → shape (n_test, n_features, n_classes)
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        shap_arr = np.transpose(np.stack(shap_vals, axis=0), (1, 2, 0))
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        if shap_vals.shape[0] == X_test.shape[0] and shap_vals.shape[1] == X_test.shape[1]:
            shap_arr = shap_vals
        elif shap_vals.shape[1] == X_test.shape[0] and shap_vals.shape[2] == X_test.shape[1]:
            shap_arr = np.transpose(shap_vals, (1, 2, 0))
        else:
            raise ValueError(f"Unexpected SHAP array shape: {shap_vals.shape}")
    else:
        raise ValueError("Unsupported SHAP output type/shape.")

    # mean(|SHAP|) across samples → (features × classes)
    mean_abs = np.abs(shap_arr).mean(axis=0)

    k = len(rf.classes_)
    col_labels = [f"Cluster {i+1}" for i in range(k)]  # 1..k display labels

    heat_df = pd.DataFrame(mean_abs, index=feat_names, columns=col_labels)

    # Column-normalize to 100%
    heat_pct = heat_df.div(heat_df.sum(axis=0), axis=1) * 100.0

    # Prepare string annotations like "12.3%"
    annot_strings = heat_pct.round(1).astype(str) + "%"

    # Plot
    plt.figure(figsize=(max(12, 0.6*len(feat_names)), max(6, 0.4*k + 4)))
    ax = sns.heatmap(
        heat_pct,
        cmap="YlGnBu",
        linewidths=0.5,
        cbar=True,
        cbar_kws={"label": "Mean |SHAP| (%)"},
        annot=False,   # we'll place our own strings for exact control
    )
    # Place our percent strings
    for (i, j), _ in np.ndenumerate(heat_pct.values):
        ax.text(j + 0.5, i + 0.5, annot_strings.iat[i, j],
                ha="center", va="center", fontsize=9)

    # Axes & ticks
    ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"Number of Clusters (k={k})", fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Colorbar as %
    try:
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        cbar.set_label("Mean |SHAP| (%)", fontsize=11, fontweight="bold", rotation=270, labelpad=14)
    except Exception:
        pass

    plt.tight_layout()
    _ensure_parent_dir(out_path)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return {"rf": rf, "heat_pct": heat_pct, "accuracy": acc}


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
        plot_feature_importance_heatmap(df, col, xs, per_feat_dir)
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
    p.add_argument("--base-dir", default=None, help="Project base directory; outputs go to BASE_DIR/data/aggResult")
    p.add_argument("-o", "--output-dir", default=None, help="Override output directory (if not using --base-dir).")
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
    # Derive output_dir
    if args.base_dir:
        output_dir = os.path.join(args.base_dir, "aggResult")
    elif args.output_dir:
        output_dir = args.output_dir
    else:
        raise SystemExit("Provide either --base-dir or --output-dir")
    try:
        out = make_all_plots(
            output_dir, args.method, args.k,
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
