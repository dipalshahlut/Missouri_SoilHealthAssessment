#!/usr/bin/env python3
"""
visualization.py

Generate standard visualizations for a single clustering result on VAE latent space:
  1) 2D latent scatter colored by cluster labels
  2) (Optional) Boxplots of scaled variables by cluster (requires data_scaled.csv)
  3) Area-by-cluster bar chart (requires area_ac in the dataframe)

Expected inputs (in OUTPUT_DIR):
  - z_mean.npy
  - main_df_with_{method}_k{k}.csv   (from clustering_algorithms.py)
  - (optional) data_scaled.csv        (columns = scaled feature names)

Outputs (written to OUTPUT_DIR):
  - latent_{method}_k{k}.png
  - boxplot_scaled_vars_{method}_k{k}.png   (only if data_scaled.csv exists)
  - area_by_cluster_{method}_k{k}.png       (only if area_ac exists)

Usage:
python visualization.py \
  ----output-dir /path/to/data/aggResult \
  --method KMeans \
  --k 10
or
python visualization.py \
  -o /path/to/data/aggResult \
  -m KMeans -k 10 \
  --z-path /custom/z_mean.npy \
  --df-with-labels-path /custom/main_df_with_KMeans_k10.csv \
  --scaled-path /custom/data_scaled.csv

"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _SEABORN = True
except Exception:
    _SEABORN = False


def _setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING
    if verbosity >= 2: level = logging.INFO
    if verbosity >= 3: level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def _labels_col(method: str, k: int) -> str: return f"{method}_k{k}"
def _latent_path(output_dir: str) -> str: return os.path.join(output_dir, "z_mean.npy")
def _df_with_labels_path(output_dir: str, method: str, k: int) -> str:
    return os.path.join(output_dir, f"main_df_with_{method}_k{k}.csv")
def _scaled_path(output_dir: str) -> str: return os.path.join(output_dir, "data_scaled.csv")


def _load_latent(z_path: str) -> np.ndarray:
    if not os.path.exists(z_path): raise FileNotFoundError(f"Missing {z_path}")
    z = np.load(z_path)
    if z.ndim != 2: raise ValueError(f"z_mean must be 2D, got {z.shape}")
    return z


def _load_df_with_labels(df_path: str, col: str) -> pd.DataFrame:
    if not os.path.exists(df_path): raise FileNotFoundError(f"Missing {df_path}")
    df = pd.read_csv(df_path)
    if col not in df.columns: raise KeyError(f"Missing column {col} in {df_path}")
    return df


def _load_scaled_df(scaled_path: str):
    if os.path.exists(scaled_path):
        try: return pd.read_csv(scaled_path)
        except Exception: return None
    return None


def plot_latent_scatter_2d(z, labels, out_path, title):
    plt.figure(figsize=(10,8))
    sc = plt.scatter(z[:,0], z[:,1], c=labels, s=12, alpha=0.85, cmap="viridis")
    plt.colorbar(sc, label="Cluster")
    plt.title(title); plt.xlabel("Latent 1"); plt.ylabel("Latent 2")
    plt.grid(True, ls="--", alpha=0.4); plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()


def plot_boxplots(df, labels_col, xs, out_path):
    if not _SEABORN: return
    if len(df) != len(xs): return
    dfp = pd.concat([df[[labels_col]], xs], axis=1)
    dfm = dfp.melt(id_vars=[labels_col], var_name="variable", value_name="value")
    g = sns.catplot(data=dfm, x=labels_col, y="value", col="variable",
                    kind="box", col_wrap=4, sharey=False, height=3, aspect=1.05)
    g.set_axis_labels("Cluster", "Scaled value")
    g.set_titles("{col_name}")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def plot_area(df, labels_col, out_path, area_col="area_ac"):
    if area_col not in df.columns: return
    area = df.groupby(labels_col)[area_col].sum().sort_index()
    plt.figure(figsize=(8,5))
    area.plot(kind="bar", edgecolor="k", alpha=0.85)
    plt.title("Total Area by Cluster"); plt.xlabel("Cluster"); plt.ylabel("Total Area (ac)")
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()


def make_all_plots(output_dir, method, k, z_path=None, df_with_labels_path=None, scaled_path=None):
    col = _labels_col(method,k)
    z = _load_latent(z_path or _latent_path(output_dir))
    df = _load_df_with_labels(df_with_labels_path or _df_with_labels_path(output_dir,method,k), col)

    latent_out = os.path.join(output_dir, f"latent_{method}_k{k}.png")
    plot_latent_scatter_2d(z, df[col].values, latent_out, f"VAE Latent Space — {method} (k={k})")

    boxplot_out, xs = None, _load_scaled_df(scaled_path or _scaled_path(output_dir))
    if xs is not None:
        boxplot_out = os.path.join(output_dir, f"boxplot_scaled_vars_{method}_k{k}.png")
        plot_boxplots(df, col, xs, boxplot_out)

    area_out = os.path.join(output_dir, f"area_by_cluster_{method}_k{k}.png")
    plot_area(df, col, area_out)

    return {"latent_path": latent_out, "boxplot_path": boxplot_out, "area_path": area_out}


def _build_argparser():
    p = argparse.ArgumentParser(description="Generate visualizations for clustering results.")
    p.add_argument("-o","--output-dir",required=True)
    p.add_argument("-m","--method",required=True,
                   choices=["KMeans","Agglomerative","Birch","GMM","FuzzyCMeans"])
    p.add_argument("-k","--k",type=int,required=True)
    p.add_argument("--z-path",default=None)
    p.add_argument("--df-with-labels-path",default=None)
    p.add_argument("--scaled-path",default=None)
    p.add_argument("-v","--verbose",action="count",default=1)
    return p


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        out = make_all_plots(args.output_dir, args.method, args.k,
                             z_path=args.z_path,
                             df_with_labels_path=args.df_with_labels_path,
                             scaled_path=args.scaled_path)
        print("✅ Visualization completed successfully.")
        print("Results saved in:", args.output_dir)
        for k_, v_ in out.items():
            if v_: print(f"  - {k_}: {v_}")
        return 0
    except Exception as e:
        logging.exception("Visualization failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
