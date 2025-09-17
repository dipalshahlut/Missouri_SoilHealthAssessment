#!/usr/bin/env python3
# latent_plots.py
"""
Stage 5 — Latent Space Visualization (2D scatter)

Inputs (from OUTPUT_DIR):
  - z_mean.npy
  - One of the following for labels (searched in this order):
      1) main_with_best_labels_allAlgo.parquet with column "{method}_best{k}"
      2) main_df_with_{method}_k{k}.csv with column "{method}_k{k}"
      3) labels_{method}_k{k}.csv with columns ["row_id","label"]
  - (optional) best_score_for.json + best_k_for.json if --method/--k not provided

Outputs (to OUTPUT_DIR/figures):
  - vae_{method_lower}_latent_2d_k{k}.png

CLI:
  python latent_plots.py \
    --output-dir /path/to/aggResult \
    --method KMeans --k 10 --title "KMeans (k=10) clustering plot"

__author__ = "Dipal Shah"
__email__  = "dipalshah@missouri.edu"
__license__ = "MIT"
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ───────────────────────── helpers ─────────────────────────

def _coerce_path(p) -> Path:
    return p if isinstance(p, Path) else Path(p)

def _ensure_fig_dir(output_dir: Path) -> Path:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir

def _discrete_cmap(n: int) -> ListedColormap:
    base = plt.cm.get_cmap("tab20", 20)
    if n <= 20:
        colors = base(np.linspace(0, 1, n))
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, n, endpoint=False))
    return ListedColormap(colors)

def _load_z(z_path: Path) -> np.ndarray:
    if not z_path.exists():
        raise FileNotFoundError(f"Missing latent file: {z_path}")
    Z = np.load(z_path)
    if Z.ndim != 2 or Z.shape[1] < 2:
        raise ValueError(f"z_mean must be 2D with >=2 columns; got {Z.shape}")
    return Z[:, :2]

def _auto_pick_method_k(output_dir: Path) -> Tuple[str, int]:
    best_score_for = output_dir / "best_score_for.json"
    best_k_for = output_dir / "best_k_for.json"
    if not best_score_for.exists() or not best_k_for.exists():
        raise FileNotFoundError(
            "Cannot auto-pick method/k: missing best_score_for.json or best_k_for.json. "
            "Run clustering_selection.py or provide --method and --k."
        )
    with open(best_score_for, "r") as f:
        score_map = json.load(f)  # {"KMeans": 0.41, ...}
    with open(best_k_for, "r") as f:
        k_map = json.load(f)      # {"KMeans": 10, ...}
    method = max(score_map, key=lambda m: (score_map[m] if score_map[m] is not None else -np.inf))
    k = int(k_map[method])
    return method, k

def _load_labels(output_dir: Path, method: str, k: int) -> np.ndarray:
    """
    Try, in order:
      1) main_with_best_labels_allAlgo.parquet (column f"{method}_best{k}")
      2) main_df_with_{method}_k{k}.csv (column f"{method}_k{k}")
      3) labels_{method}_k{k}.csv (columns row_id,label)
    Returns labels as a 1D np.ndarray.
    """
    # 1) “best” parquet with specific k (may not exist for user-chosen k)
    best_parq = output_dir / "main_with_best_labels_allAlgo.parquet"
    best_col = f"{method}_best{k}"
    if best_parq.exists():
        try:
            df = pd.read_parquet(best_parq)
            if best_col in df.columns:
                lab = df[best_col].to_numpy()
                logging.info("Loaded labels from %s[%s]", best_parq.name, best_col)
                return lab
            else:
                logging.debug("%s does not contain %s; will try other sources.", best_parq.name, best_col)
        except Exception as e:
            logging.warning("Failed reading %s: %s", best_parq, e)

    # 2) Single-join CSV for exact method/k
    single_csv = output_dir / f"main_df_with_{method}_k{k}.csv"
    single_col = f"{method}_k{k}"
    if single_csv.exists():
        try:
            df = pd.read_csv(single_csv)
            if single_col in df.columns:
                lab = df[single_col].to_numpy()
                logging.info("Loaded labels from %s[%s]", single_csv.name, single_col)
                return lab
            else:
                logging.debug("%s does not contain column %s.", single_csv.name, single_col)
        except Exception as e:
            logging.warning("Failed reading %s: %s", single_csv, e)

    # 3) Plain labels file
    labels_csv = output_dir / f"labels_{method}_k{k}.csv"
    if labels_csv.exists():
        try:
            df = pd.read_csv(labels_csv)
            if {"row_id", "label"}.issubset(df.columns):
                lab = df["label"].to_numpy()
                logging.info("Loaded labels from %s", labels_csv.name)
                return lab
            else:
                logging.debug("%s missing row_id/label columns.", labels_csv.name)
        except Exception as e:
            logging.warning("Failed reading %s: %s", labels_csv, e)

    # If we got here, we couldn’t resolve labels for that (method,k)
    # Provide a helpful message about available best-* columns if possible
    if best_parq.exists():
        try:
            df = pd.read_parquet(best_parq)
            avail = [c for c in df.columns if "_best" in c]
            raise FileNotFoundError(
                f"Could not locate labels for method={method}, k={k}. "
                f"Tried: {best_parq}[{best_col}], {single_csv}[{single_col}], and {labels_csv}. "
                f"Available best-label columns: {avail}"
            )
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not locate labels for method={method}, k={k}. "
        f"Tried: {best_parq}[{best_col}], {single_csv}[{single_col}], and {labels_csv}."
    )

def _plot_latent_space_2d(
    z2: np.ndarray,
    labels: Iterable,
    k: int,
    title: str,
    output_dir: Path,
    filename: str,
    point_size: float = 14.0,
    alpha: float = 0.7,
    show_centroids: bool = True,
) -> Path:
    z2 = np.asarray(z2)
    labels = np.asarray(list(labels))
    if len(labels) != len(z2):
        n = min(len(labels), len(z2))
        logging.warning("Latent rows (%d) != labels (%d); truncating to %d by position.", len(z2), len(labels), n)
        z2 = z2[:n, :]
        labels = labels[:n]

    uniq = pd.unique(labels)
    # sort numerically if possible; else lexicographically
    try:
        _ = uniq.astype(float)
        order = np.array(sorted(uniq, key=lambda x: float(x)))
    except Exception:
        order = np.array(sorted(uniq, key=lambda x: str(x)))
    label_to_idx = {lab: i for i, lab in enumerate(order)}
    idx = np.vectorize(label_to_idx.get)(labels)

    cmap = _discrete_cmap(max(len(order), k))
    fig_dir = _ensure_fig_dir(output_dir)
    out_path = fig_dir / filename

    plt.figure(figsize=(8, 6), dpi=150)
    ax = plt.gca()

    # draw clusters
    for lab, i in label_to_idx.items():
        sel = (idx == i)
        ax.scatter(
            z2[sel, 0], z2[sel, 1],
            s=point_size, alpha=alpha, c=[cmap(i)],
            label=f"{lab} (n={sel.sum()})", edgecolors="none"
        )

    if show_centroids:
        for lab, i in label_to_idx.items():
            sel = (idx == i)
            if np.any(sel):
                cx, cy = z2[sel, :].mean(axis=0)
                ax.scatter([cx], [cy], s=point_size*6, marker="X", c=[cmap(i)],
                           edgecolors="black", linewidths=0.8, alpha=0.95, zorder=5)

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ncol = 1 if len(order) <= 10 else 2
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, ncol=ncol, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    logging.info("Saved: %s", out_path)
    return out_path


# ───────────────────── public API ─────────────────────

def plot_latent(
    output_dir: str | Path,
    method: Optional[str] = None,
    k: Optional[int] = None,
    title: Optional[str] = None,
    z_path: Optional[str | Path] = None,
    df_path: Optional[str | Path] = None,  # unused; kept for symmetry
) -> Path:
    """
    Create and save a 2D latent scatter colored by cluster labels.
    Resolves labels from multiple artifacts so you can plot either “best” k or a custom k.
    Returns the path to the saved PNG.
    """
    output_dir = _coerce_path(output_dir)

    # 1) Decide method/k (if not provided)
    if method is None or k is None:
        method, k = _auto_pick_method_k(output_dir)
    k = int(k)

    # 2) Load z (first two dims)
    z_path_final = _coerce_path(z_path) if z_path else output_dir / "z_mean.npy"
    z2 = _load_z(z_path_final)

    # 3) Load labels for requested method/k using fallbacks
    labels = _load_labels(output_dir, method, k)

    # 4) Plot
    ttl = title if title else f"{method} (k={k}) — latent 2D"
    filename = f"vae_{method.lower()}_latent_2d_k{k}.png"
    return _plot_latent_space_2d(
        z2=z2, labels=labels, k=k, title=ttl, output_dir=output_dir, filename=filename
    )


# ───────────────────── CLI ─────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 5 — Latent Space Visualization (2D scatter)")
    p.add_argument("--output-dir", type=Path, default=None, help ="Directory to write intermediate artifacts." )
    p.add_argument("--method", type=str, default=None, help="Method (KMeans, Agglomerative, Birch, GMM).")
    p.add_argument("--k", type=int, default=None, help="Number of clusters.")
    p.add_argument("--title", type=str, default=None, help="Custom title.")
    p.add_argument("--z-path", type=Path, default=None, help="Optional explicit path to z_mean.npy.")
    p.add_argument("--df-path", type=Path, default=None, help="(Unused) kept for symmetry.")
    return p.parse_args()

def run(output_dir: Path, method: Optional[str], k: Optional[int], title: Optional[str],
        z_path: Optional[Path] = None, df_path: Optional[Path] = None) -> Path:
    return plot_latent(output_dir, method, k, title, z_path, df_path)

def main(argv: list[str] | None = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    return run(args.output_dir, args.method, args.k, args.title, args.z_path, args.df_path)


if __name__ == "__main__":
    main()
