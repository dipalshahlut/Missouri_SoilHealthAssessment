#!/usr/bin/env python3
# clustering_selection.py
"""
Stage 3 — Multi-Algorithm Clustering & Best Clustering Results (self-contained)

Inputs (from OUTPUT_DIR):
  - prepared_df.parquet
  - z_mean.npy

Outputs (to OUTPUT_DIR):
  - clustering_scores.csv                : tidy scores [method,k,metric,value]
  - best_k_for.json                      : {"KMeans": 10, "Agglomerative": 9, ...}
  - best_score_for.json                  : {"KMeans": 0.41, "Agglomerative": 0.38, ...}
  - main_with_best_labels_allAlgo.parquet: df + one best-label column per method
  - figures/silhouette_comparison.png    : line plot of silhouette vs k per method

CLI Usage:
  python clustering_selection.py \
    --output-dir /path/to/data/aggResult \
    --methods KMeans Agglomerative Birch GMM \
    --k-min 2 --k-max 20

__author__ = "Dipal Shah"
__email__  = "dipalshah@missouri.edu"
__license__ = "MIT"
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---- ML / metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# ---- Plotting (simple inline replacement for plotting_utils)
import matplotlib.pyplot as plt


RANDOM_STATE = 42
METHODS_ALLOWED = {"KMeans", "Agglomerative", "Birch", "GMM"}


# -------------------------------
# Minimal plotting
# -------------------------------
def plot_silhouette_comparison(scores: dict, k_range: range, methods: list[str], output_dir: Path) -> None:
    """
    Save a single comparison figure with silhouette vs k per method.
    """
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "silhouette_comparison.png"

    plt.figure()
    for m in methods:
        y = []
        for k in k_range:
            try:
                y.append(scores[m][k]["silhouette"])
            except Exception:
                y.append(np.nan)
        plt.plot(list(k_range), y, marker="o", label=m)

    plt.xlabel("k (number of clusters/components)")
    plt.ylabel("Silhouette score (higher is better)")
    plt.title("Silhouette vs k across methods")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------------
# Helper: run a method for a given k
# -------------------------------
def _fit_predict(method: str, Z: np.ndarray, k: int) -> np.ndarray:
    m = method.lower()
    if m == "kmeans":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)#, n_init="auto", random_state=RANDOM_STATE)
        return model.fit_predict(Z)
    elif m == "agglomerative":
        model = AgglomerativeClustering(n_clusters=k)
        return model.fit_predict(Z)
    elif m == "birch":
        model = Birch(n_clusters=k)
        return model.fit_predict(Z)
    elif m == "gmm":
        model = GaussianMixture(n_components=k, random_state=42) #covariance_type="full",
        model.fit(Z)
        return model.predict(Z)
    else:
        raise ValueError(f"Unknown method: {method}")


# -------------------------------
# Core evaluators
# -------------------------------
def evaluate_multiple_algorithms(
    Z: np.ndarray,
    methods: list[str],
    k_range: range,
) -> tuple[dict, dict, dict]:
    """
    Evaluate each method for each k using:
      - silhouette_score (primary; higher is better)
      - calinski_harabasz_score
      - davies_bouldin_score (lower is better)
    Returns:
      scores: dict[method][k] -> {"silhouette": float, "ch": float, "db": float}
      best_k_for: dict[method] -> k with best silhouette
      best_score_for: dict[method] -> best silhouette score
    """
    scores: dict = {}
    best_k_for: dict = {}
    best_score_for: dict = {}

    for m in methods:
        scores[m] = {}
        best_k = None
        best_s = -np.inf

        for k in k_range:
            try:
                labels = _fit_predict(m, Z, k)
                if len(set(labels)) < 2:
                    sil = float("nan"); ch = float("nan"); db = float("nan")
                else:
                    sil = float(silhouette_score(Z, labels))
                    ch = float(calinski_harabasz_score(Z, labels))
                    db = float(davies_bouldin_score(Z, labels))
                scores[m][k] = {"silhouette": sil, "ch": ch, "db": db}

                if not np.isnan(sil) and sil > best_s:
                    best_s = sil
                    best_k = k
            except Exception as e:
                logging.warning("Method %s failed for k=%d: %s", m, k, e)
                scores[m][k] = {"silhouette": float("nan"), "ch": float("nan"), "db": float("nan")}

        best_k_for[m] = best_k
        best_score_for[m] = best_s if best_k is not None else float("nan")

    return scores, best_k_for, best_score_for


def get_best_clustering_results(
    df: pd.DataFrame,
    Z: np.ndarray,
    best_k_for: dict,
    methods: list[str],
) -> pd.DataFrame:
    """
    For each method, fit the model with the selected best k and append a label column:
      <Method>_best<k>
    """
    out = df.copy()
    for m in methods:
        k = best_k_for.get(m)
        if k is None:
            logging.warning("No best k found for %s; skipping.", m)
            continue
        labels = _fit_predict(m, Z, int(k))
        col = f"{m}_best{int(k)}"
        out[col] = labels
    return out


def _scores_to_tidy_dataframe(scores: dict, methods: list[str], k_range: range) -> pd.DataFrame:
    rows = []
    for m in methods:
        block = scores.get(m, {})
        for k in k_range:
            if k not in block:
                continue
            vals = block[k]
            rows.append((m, int(k), "silhouette", float(vals.get("silhouette", np.nan))))
            rows.append((m, int(k), "calinski_harabasz", float(vals.get("ch", np.nan))))
            rows.append((m, int(k), "davies_bouldin", float(vals.get("db", np.nan))))
    df = pd.DataFrame(rows, columns=["method", "k", "metric", "value"])
    if not df.empty:
        df = df.sort_values(["method", "metric", "k"]).reset_index(drop=True)
    return df


# -------------------------------
# IO helpers
# -------------------------------
def _coerce_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _load_inputs(output_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df_path = output_dir / "prepared_df.parquet"
    z_path = output_dir / "z_mean.npy"

    if not df_path.exists():
        raise FileNotFoundError(f"Missing {df_path} (run Stage 1).")
    if not z_path.exists():
        raise FileNotFoundError(f"Missing {z_path} (run Stage 2).")

    df = pd.read_parquet(df_path)
    Z = np.load(z_path)
    if len(df) != len(Z):
        logging.warning(
            "Row count mismatch: prepared_df (%d) vs z_mean (%d). Ensure alignment.",
            len(df), len(Z)
        )
    return df, Z


def _write_artifacts(
    output_dir: Path,
    tidy_scores: pd.DataFrame,
    best_k_for: dict,
    best_score_for: dict,
    df_with_clusters: pd.DataFrame,
    k_range: range,
    methods: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    # plots
    try:
        # rebuild full dict structure expected by plotter
        scores = {}
        for m in methods:
            scores[m] = {}
        for _, row in tidy_scores.iterrows():
            m = row["method"]; k = int(row["k"]); metric = row["metric"]; val = float(row["value"])
            scores.setdefault(m, {}).setdefault(k, {})[metric if metric != "calinski_harabasz" else "ch"] = val
        plot_silhouette_comparison(scores, k_range, methods, output_dir)
    except Exception as e:
        logging.warning("Could not plot silhouette comparison: %s", e)

    # metadata
    with open(output_dir / "best_k_for.json", "w") as f:
        json.dump(best_k_for, f, indent=2)
    with open(output_dir / "best_score_for.json", "w") as f:
        json.dump(best_score_for, f, indent=2)

    # tidy scores
    if not tidy_scores.empty:
        tidy_scores.to_csv(output_dir / "clustering_scores.csv", index=False)

    # df with best labels
    df_with_clusters.to_parquet(output_dir / "main_with_best_labels_allAlgo.parquet", index=False)


# -------------------------------
# Public API
# -------------------------------
def score_many_clusterings(
    output_dir: str | Path,
    methods: list[str],
    k_min: int,
    k_max: int,
) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    """
    Programmatic entry point.
    Returns:
      tidy_scores, best_k_for, best_score_for, df_with_clusters
    (and writes standard artifacts to output_dir)
    """
    output_dir = _coerce_path(output_dir)
    # normalize/validate methods
    methods = [m if m in METHODS_ALLOWED else m.title() for m in methods]
    for m in methods:
        if m not in METHODS_ALLOWED:
            raise ValueError(f"Unsupported method: {m}. Allowed: {sorted(METHODS_ALLOWED)}")

    df, Z = _load_inputs(output_dir)
    k_range = range(int(k_min), int(k_max) + 1)

    scores, best_k_for, best_score_for = evaluate_multiple_algorithms(Z, methods, k_range)
    tidy_scores = _scores_to_tidy_dataframe(scores, methods, k_range)
    df_with_clusters = get_best_clustering_results(df, Z, best_k_for, methods)

    _write_artifacts(output_dir, tidy_scores, best_k_for, best_score_for, df_with_clusters, k_range, methods)
    logging.info("Saved: clustering_scores.csv, best_k_for.json, best_score_for.json, main_with_best_labels_allAlgo.parquet, figures/silhouette_comparison.png")

    return tidy_scores, best_k_for, best_score_for, df_with_clusters


# -------------------------------
# CLI runner
# -------------------------------
def run(output_dir: Path, methods: list[str], k_min: int, k_max: int) -> None:
    """
    CLI-style side-effecting runner (writes artifacts; no return).
    """
    # Delegate to programmatic API for a single source of truth
    score_many_clusterings(output_dir=output_dir, methods=methods, k_min=k_min, k_max=k_max)


def parse_args() -> argparse.Namespace:
    # Default: point to repo's standard output folder
    default_output = Path(__file__).resolve().parent / "data" / "aggResult"
    parser = argparse.ArgumentParser(description="Stage 3 — Multi-Algorithm Clustering & Best Results (self-contained)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=False,
        default=default_output,
        help="Directory with Stage-1/2 outputs and where Stage-3 outputs will be written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=False,
        default=["KMeans", "Agglomerative", "Birch", "GMM"],
        help=f"Clustering methods to evaluate. Allowed: {sorted(METHODS_ALLOWED)}",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        required=False,
        default=2,
        help="Minimum k (inclusive).",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        required=False,
        default=20,
        help="Maximum k (inclusive).",
    )
    return parser.parse_args()


def main(argv: list[str] | None = None):
    """
    Importable CLI-style entrypoint: allows calling `from clustering_selection import main as run_select; run_select([...])`.
    """
    # When called programmatically, we ignore argv and just parse sys.argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    return run(args.output_dir, args.methods, args.k_min, args.k_max)


if __name__ == "__main__":
    main()
